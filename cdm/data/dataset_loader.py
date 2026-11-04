"""Dataset loader and PyTorch batch collation for CDM-1.
Supports temporal splits, entity-isolated splits, and batch formatting.
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

    and organizes multi-task training targets.
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

    # Multi-task targets
    targets: Dict[str, torch.Tensor] = {}

    bin_targets = [item["binary_target"] for item in batch]
    if all(t is not None for t in bin_targets):
        targets["binary_target"] = torch.tensor(bin_targets, dtype=torch.long)

    prio_targets = [item["priority_target"] for item in batch]
    if all(t is not None for t in prio_targets):
        targets["priority_target"] = torch.tensor(prio_targets, dtype=torch.long)

    score_targets = [item["score_target"] for item in batch]
    if all(t is not None for t in score_targets):
        targets["score_target"] = torch.tensor(score_targets, dtype=torch.float32)

    choice_targets = [item["choice_target"] for item in batch]
    if all(t is not None for t in choice_targets):
        targets["choice_target"] = torch.tensor(choice_targets, dtype=torch.long)

    out_bin_targets = [item["outcome_binary"] for item in batch]
    if any(t is not None for t in out_bin_targets):
        clean_out_bin = [t if t is not None else 0 for t in out_bin_targets]
        targets["outcome_binary_target"] = torch.tensor(clean_out_bin, dtype=torch.float32)

    out_multi_targets = [item["outcome_multi"] for item in batch]
    if any(t is not None for t in out_multi_targets):
        clean_out_multi = [t if t is not None else 0 for t in out_multi_targets]
        targets["outcome_multi_target"] = torch.tensor(clean_out_multi, dtype=torch.long)

    out_num_targets = [item["outcome_numeric"] for item in batch]
    if any(t is not None for t in out_num_targets):
        clean_out_num = [t if t is not None else 0.0 for t in out_num_targets]
        targets["outcome_numeric_target"] = torch.tensor(clean_out_num, dtype=torch.float32)

    # Candidate collation (if all have candidates and same length)
    has_cands = all(item["constraint_mask"] is not None for item in batch)
    constraint_masks = None
    if has_cands:
        max_cands = max(len(item["constraint_mask"]) for item in batch)
        # Pad constraint masks if needed
        padded_masks = []
        for item in batch:
            mask = item["constraint_mask"]
            if len(mask) < max_cands:
                pad = torch.zeros(max_cands - len(mask))
                mask = torch.cat([mask, pad], dim=0)
            padded_masks.append(mask)
        constraint_masks = torch.stack(padded_masks, dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "numeric_values": numeric_values,
        "missing_indicators": missing_indicators,
        "categorical_indices": categorical_indices,
        "constraint_mask": constraint_masks,
        "targets": targets,
        "example_ids": [item["example_id"] for item in batch],
    }


def split_dataset(
    examples: List[DecisionExample],
    split_type: str = "temporal",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[List[DecisionExample], List[DecisionExample], List[DecisionExample]]:
    """Splits examples into Train, Validation, and Test partitions.

    - 'temporal': sorts by temporal_timestamp to prevent future leakage.
    - 'random': deterministic seed-based shuffle.
    """
    if split_type == "temporal":
        sorted_examples = sorted(
            examples,
            key=lambda x: x.temporal_timestamp or "",
        )
    else:
        import random
        rnd = random.Random(42)
        sorted_examples = list(examples)
        rnd.shuffle(sorted_examples)

    n = len(sorted_examples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = sorted_examples[:train_end]
    val_data = sorted_examples[train_end:val_end]
    test_data = sorted_examples[val_end:]

    return train_data, val_data, test_data
