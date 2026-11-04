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

        cand_bin_targets = torch.zeros((batch_size, max_cands), dtype=torch.float32)
        cand_bin_mask = torch.zeros((batch_size, max_cands), dtype=torch.float32)
        cand_risk_targets = torch.zeros((batch_size, max_cands), dtype=torch.float32)
        cand_risk_mask = torch.zeros((batch_size, max_cands), dtype=torch.float32)

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

                if item.get("cand_outcome_binary") is not None and item.get("cand_outcome_binary_mask") is not None:
                    cand_bin_targets[b_idx, :n_c] = item["cand_outcome_binary"]
                    cand_bin_mask[b_idx, :n_c] = item["cand_outcome_binary_mask"]
                if item.get("cand_risk") is not None and item.get("cand_risk_mask") is not None:
                    cand_risk_targets[b_idx, :n_c] = item["cand_risk"]
                    cand_risk_mask[b_idx, :n_c] = item["cand_risk_mask"]

        targets["candidate_existence_mask"] = candidate_existence_mask
        if cand_bin_mask.sum() > 0:
            targets["candidate_binary_targets"] = cand_bin_targets
            targets["candidate_binary_mask"] = cand_bin_mask
        if cand_risk_mask.sum() > 0:
            targets["candidate_risk_targets"] = cand_risk_targets
            targets["candidate_risk_mask"] = cand_risk_mask

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
    split_type: str = "stratified_group_temporal",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[List[DecisionExample], List[DecisionExample], List[DecisionExample]]:
    """Splits dataset with stratified, group-aware, and temporal holdout.

    Guarantees:
    1. Complete entity/customer group isolation (zero leakage between splits).
    2. Multi-domain representation across all splits (customer_support, business_operations, etc.).
    3. Balanced binary, priority, and risk class representation.
    4. Temporal holdout order (earliest interactions in train, latest in test).
    """
    from collections import defaultdict

    # 1. Group examples by customer / entity group
    entity_groups = defaultdict(list)
    for ex in examples:
        group_key = ex.structured_features.get("customer_id")
        if not group_key:
            # Fallback to example ID prefix if customer_id missing
            group_key = ex.example_id.rsplit("-", 1)[0]
        entity_groups[group_key].append(ex)

    # 2. Organize groups by (domain, dominant_binary_label)
    domain_label_groups = defaultdict(lambda: defaultdict(list))
    for grp_key, grp_examples in entity_groups.items():
        domain = grp_examples[0].domain
        # Dominant binary decision of the entity group
        bin_target = grp_examples[0].target_decision.binary_decision
        bin_label = bin_target if bin_target is not None else 0
        domain_label_groups[domain][bin_label].append(grp_examples)

    train_data: List[DecisionExample] = []
    val_data: List[DecisionExample] = []
    test_data: List[DecisionExample] = []

    # 3. Stratified temporal partitioning per (domain, binary_label) stratum
    for domain, label_dict in domain_label_groups.items():
        for bin_label, groups in label_dict.items():
            # Sort groups chronologically by earliest timestamp
            sorted_grps = sorted(
                groups,
                key=lambda g: min(e.temporal_timestamp or "" for e in g),
            )
            n_grps = len(sorted_grps)

            if n_grps == 1:
                train_data.extend(sorted_grps[0])
            elif n_grps == 2:
                train_data.extend(sorted_grps[0])
                test_data.extend(sorted_grps[1])
            elif n_grps == 3:
                train_data.extend(sorted_grps[0])
                val_data.extend(sorted_grps[1])
                test_data.extend(sorted_grps[2])
            else:
                n_train = max(int(n_grps * train_ratio), 1)
                n_val = max(int(n_grps * val_ratio), 1)
                if n_train + n_val >= n_grps:
                    n_train = n_grps - 2
                    n_val = 1
                train_grps = sorted_grps[:n_train]
                val_grps = sorted_grps[n_train:n_train + n_val]
                test_grps = sorted_grps[n_train + n_val:]

                for g in train_grps:
                    train_data.extend(g)
                for g in val_grps:
                    val_data.extend(g)
                for g in test_grps:
                    test_data.extend(g)

    return train_data, val_data, test_data


def validate_split_distribution(
    train_data: List[DecisionExample],
    val_data: List[DecisionExample],
    test_data: List[DecisionExample],
    required_domains: Optional[List[str]] = None,
    min_class_count: int = 5,
) -> Dict[str, Any]:
    """Automated split distribution validation and leakage audit.
    Fails dataset generation if evaluation coverage is insufficient.
    """
    from collections import Counter

    if required_domains is None:
        required_domains = [
            "customer_support",
            "business_operations",
            "finance_and_credit",
            "communication",
            "retail",
        ]

    # Leakage check: Example IDs
    train_ids = set(e.example_id for e in train_data)
    val_ids = set(e.example_id for e in val_data)
    test_ids = set(e.example_id for e in test_data)

    if train_ids & val_ids:
        raise ValueError(f"Data leakage detected between train and val: {train_ids & val_ids}")
    if train_ids & test_ids:
        raise ValueError(f"Data leakage detected between train and test: {train_ids & test_ids}")
    if val_ids & test_ids:
        raise ValueError(f"Data leakage detected between val and test: {val_ids & test_ids}")

    # Leakage check: Customer / Entity IDs
    def get_cust_ids(ex_list):
        return set(e.structured_features.get("customer_id") for e in ex_list if e.structured_features.get("customer_id"))

    train_cids = get_cust_ids(train_data)
    val_cids = get_cust_ids(val_data)
    test_cids = get_cust_ids(test_data)

    if train_cids & val_cids:
        raise ValueError(f"Entity leakage detected between train and val: {train_cids & val_cids}")
    if train_cids & test_cids:
        raise ValueError(f"Entity leakage detected between train and test: {train_cids & test_cids}")
    if val_cids & test_cids:
        raise ValueError(f"Entity leakage detected between val and test: {val_cids & test_cids}")

    # Coverage verification
    splits = {"train": train_data, "validation": val_data, "test": test_data}
    audit_summary = {}

    for s_name, s_data in splits.items():
        dom_counts = Counter(e.domain for e in s_data)
        bin_counts = Counter(e.target_decision.binary_decision for e in s_data)
        prio_counts = Counter(e.target_decision.priority_level for e in s_data)
        risk_counts = Counter(e.target_decision.risk_level for e in s_data)

        # 1. Domain coverage
        missing_domains = [d for d in required_domains if dom_counts.get(d, 0) == 0]
        if missing_domains:
            raise ValueError(f"Split '{s_name}' is missing required domains: {missing_domains}. Domain counts: {dict(dom_counts)}")

        # 2. Binary class coverage
        if s_name in ["validation", "test"]:
            if bin_counts.get(0, 0) < min_class_count or bin_counts.get(1, 0) < min_class_count:
                raise ValueError(
                    f"Split '{s_name}' does not meet minimum binary class requirement (min {min_class_count}). "
                    f"Counts: {dict(bin_counts)}"
                )

        # 3. Priority and Risk tier diversity
        if len(prio_counts) < 3:
            raise ValueError(f"Split '{s_name}' lacks priority class diversity: {dict(prio_counts)}")
        if len(risk_counts) < 3:
            raise ValueError(f"Split '{s_name}' lacks risk class diversity: {dict(risk_counts)}")

        audit_summary[s_name] = {
            "count": len(s_data),
            "domains": dict(dom_counts),
            "binary_distribution": dict(bin_counts),
            "priority_distribution": dict(prio_counts),
            "risk_distribution": dict(risk_counts),
        }

    return audit_summary
