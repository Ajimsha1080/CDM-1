"""Dataset loader and PyTorch batch collation for CDM-1.
Supports candidate padding, explicit missing target masks, and leakage-free splits.
"""

from typing import Dict, List, Optional, Any, Tuple
import json
import torch
from torch.utils.data import Dataset, DataLoader
from cdm.data.data_contracts import DecisionExample
from cdm.data.preprocessor import DecisionPreprocessor


class DecisionDataset(Dataset):
    """PyTorch Dataset wrapping standardized CDM-1 decision examples."""

    def __init__(
        self,
        examples: List[DecisionExample],
        preprocessor: DecisionPreprocessor,
    ):
        self.examples = examples
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        return self.preprocessor.preprocess_example(ex)


def collate_decision_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function that batches examples, handles candidate padding,

    existence masks, and organizes multi-task targets with explicit missingness masks.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    numeric_values = torch.stack([item["numeric_values"] for item in batch], dim=0)
    missing_indicators = torch.stack([item["missing_indicators"] for item in batch], dim=0)

    # Categorical indices collation
    cat_keys = batch[0]["categorical_indices"].keys()
    categorical_indices = {
        k: torch.stack([item["categorical_indices"][k] for item in batch], dim=0)
        for k in cat_keys
    }

    # Explicitly masked multi-task targets
    targets: Dict[str, torch.Tensor] = {
        "binary_target": torch.tensor([item["binary_target"] for item in batch], dtype=torch.long),
        "binary_mask": torch.tensor([item["binary_mask"] for item in batch], dtype=torch.float32),
        "priority_target": torch.tensor([item["priority_target"] for item in batch], dtype=torch.long),
        "priority_mask": torch.tensor([item["priority_mask"] for item in batch], dtype=torch.float32),
        "risk_target": torch.tensor([item["risk_target"] for item in batch], dtype=torch.long),
        "risk_mask": torch.tensor([item["risk_mask"] for item in batch], dtype=torch.float32),
        "score_target": torch.tensor([item["score_target"] for item in batch], dtype=torch.float32),
        "score_mask": torch.tensor([item["score_mask"] for item in batch], dtype=torch.float32),
        "choice_target": torch.tensor(
            [item["choice_target"] if item["choice_target"] is not None else 0 for item in batch],
            dtype=torch.long,
        ),
        "choice_mask": torch.tensor([item["choice_mask"] for item in batch], dtype=torch.float32),
        "outcome_binary_target": torch.tensor([item["outcome_binary"] for item in batch], dtype=torch.float32),
        "outcome_binary_mask": torch.tensor([item["outcome_binary_mask"] for item in batch], dtype=torch.float32),
        "outcome_multi_target": torch.tensor([item["outcome_multi"] for item in batch], dtype=torch.long),
        "outcome_multi_mask": torch.tensor([item["outcome_multi_mask"] for item in batch], dtype=torch.float32),
        "outcome_numeric_target": torch.tensor([item["outcome_numeric"] for item in batch], dtype=torch.float32),
        "outcome_numeric_mask": torch.tensor([item["outcome_numeric_mask"] for item in batch], dtype=torch.float32),
        "risk_prob_target": torch.tensor([item["risk_prob"] for item in batch], dtype=torch.float32),
        "risk_prob_mask": torch.tensor([item["risk_prob_mask"] for item in batch], dtype=torch.float32),
    }

    # Candidate collation: support variable candidate counts with padding
    max_cands = max(item["num_candidates"] for item in batch)
    candidate_input_ids = None
    candidate_attention_mask = None
    candidate_constraint_mask = None
    candidate_existence_mask = None

    if max_cands > 0:
        cand_seq_len = 64
        batch_size = len(batch)
        candidate_input_ids = torch.zeros((batch_size, max_cands, cand_seq_len), dtype=torch.long)
        candidate_attention_mask = torch.zeros((batch_size, max_cands, cand_seq_len), dtype=torch.long)
        candidate_constraint_mask = torch.zeros((batch_size, max_cands), dtype=torch.float32)
        candidate_existence_mask = torch.zeros((batch_size, max_cands), dtype=torch.float32)

        for b_idx, item in enumerate(batch):
            n_c = item["num_candidates"]
            if n_c > 0 and item["cand_input_ids"] is not None:
                candidate_input_ids[b_idx, :n_c, :] = item["cand_input_ids"]
                candidate_attention_mask[b_idx, :n_c, :] = item["cand_attention_mask"]
                candidate_existence_mask[b_idx, :n_c] = 1.0
                if item["constraint_mask"] is not None:
                    candidate_constraint_mask[b_idx, :n_c] = item["constraint_mask"]
                else:
                    candidate_constraint_mask[b_idx, :n_c] = 1.0

        targets["candidate_existence_mask"] = candidate_existence_mask

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "numeric_values": numeric_values,
        "missing_indicators": missing_indicators,
        "categorical_indices": categorical_indices,
        "candidate_input_ids": candidate_input_ids,
        "candidate_attention_mask": candidate_attention_mask,
        "constraint_mask": candidate_constraint_mask,
        "existence_mask": candidate_existence_mask,
        "targets": targets,
        "example_ids": [item["example_id"] for item in batch],
    }


def split_dataset(
    examples: List[DecisionExample],
    split_type: str = "group_temporal",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[DecisionExample], List[DecisionExample], List[DecisionExample]]:
    """Splits dataset ensuring zero entity/customer leakage and temporal holdout.

    - 'group_temporal': sorts groups by earliest interaction timestamp, ensuring complete entity isolation.
    """
    if split_type == "group_temporal":
        # Group by domain and entity/customer prefix
        from collections import defaultdict
        groups = defaultdict(list)
        for ex in examples:
            prefix = ex.example_id.rsplit("-", 1)[0]
            groups[prefix].append(ex)

        # Sort groups by min timestamp
        sorted_groups = sorted(
            groups.values(),
            key=lambda grp: min(e.temporal_timestamp or "" for e in grp),
        )

        train_data: List[DecisionExample] = []
        val_data: List[DecisionExample] = []
        test_data: List[DecisionExample] = []

        total_examples = len(examples)
        train_target = int(total_examples * train_ratio)
        val_target = int(total_examples * (train_ratio + val_ratio))

        curr_count = 0
        for grp in sorted_groups:
            if curr_count + len(grp) <= train_target:
                train_data.extend(grp)
            elif curr_count + len(grp) <= val_target:
                val_data.extend(grp)
            else:
                test_data.extend(grp)
            curr_count += len(grp)

        # In case a small tail remains empty
        if not val_data and len(train_data) > 2:
            val_data.append(train_data.pop())
        if not test_data and len(val_data) > 2:
            test_data.append(val_data.pop())

        return train_data, val_data, test_data
    else:
        # Default temporal sort
        sorted_examples = sorted(
            examples,
            key=lambda x: x.temporal_timestamp or "",
        )
        n = len(sorted_examples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return sorted_examples[:train_end], sorted_examples[train_end:val_end], sorted_examples[val_end:]
