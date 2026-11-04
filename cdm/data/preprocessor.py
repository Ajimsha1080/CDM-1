"""Data preprocessing module for CDM-1.
Tokenizes text using Hugging Face AutoTokenizer, normalizes structured features,
and compiles candidate matrices, constraint masks, and explicit loss masks.
"""

from typing import Dict, List, Optional, Any, Tuple
import torch
from transformers import AutoTokenizer
from cdm.config.feature_schema import FeatureSchema
from cdm.data.data_contracts import DecisionExample


class DecisionPreprocessor:
    """Transforms raw DecisionExample instances into PyTorch tensor batches."""

    def __init__(
        self,
        tokenizer_name_or_path: str = "answerdotai/ModernBERT-base",
        feature_schema: Optional[FeatureSchema] = None,
        max_seq_length: int = 256,
        local_files_only: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            local_files_only=local_files_only,
        )
        self.schema = feature_schema or FeatureSchema.get_default_schema()
        self.max_seq_length = max_seq_length

    def preprocess_example(
        self, example: DecisionExample
    ) -> Dict[str, Any]:
        """Preprocesses a single DecisionExample into model inputs with explicit masks."""
        # 1. Text tokenization
        text = example.input_text
        if example.context:
            text = f"{example.context} | {text}"

        tokenized = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 2. Structured tabular features
        num_vals = self.schema.transform_numeric(example.structured_features)
        missing_indicators = self.schema.get_missing_indicators(example.structured_features)
        cat_indices = self.schema.transform_categorical(example.structured_features)

        # 3. Candidate actions and policy constraints
        cand_texts = [f"{c.name}: {c.description}" for c in example.candidate_actions]
        num_cands = len(cand_texts)

        cand_input_ids = None
        cand_attention_mask = None
        if num_cands > 0:
            c_tokens = self.tokenizer(
                cand_texts,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            cand_input_ids = c_tokens["input_ids"]  # [num_cands, 64]
            cand_attention_mask = c_tokens["attention_mask"]  # [num_cands, 64]

        # Build hard constraint mask: 1.0 if candidate is allowed, 0.0 if prohibited
        prohibited_actions = set()
        for constraint in example.constraints:
            if constraint.is_hard_constraint:
                for act in constraint.prohibited_actions:
                    prohibited_actions.add(act)

        constraint_mask = [
            0.0 if c.action_id in prohibited_actions or c.name in prohibited_actions else 1.0
            for c in example.candidate_actions
        ]

        # Target candidate index if choice decision
        choice_target_idx = None
        if example.target_decision.selected_candidate_id:
            for idx, c in enumerate(example.candidate_actions):
                if c.action_id == example.target_decision.selected_candidate_id:
                    choice_target_idx = idx
                    break

        # Explicit target masks (1.0 if label exists, 0.0 if missing)
        td = example.target_decision
        ol = example.outcome_label

        cand_outcome_binary = [
            c.candidate_outcome_binary if c.candidate_outcome_binary is not None else 0
            for c in example.candidate_actions
        ]
        cand_outcome_binary_mask = [
            1.0 if c.candidate_outcome_binary is not None else 0.0
            for c in example.candidate_actions
        ]
        cand_risk = [
            c.candidate_risk if c.candidate_risk is not None else 0.0
            for c in example.candidate_actions
        ]
        cand_risk_mask = [
            1.0 if c.candidate_risk is not None else 0.0
            for c in example.candidate_actions
        ]

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "numeric_values": torch.tensor(num_vals, dtype=torch.float32),
            "missing_indicators": torch.tensor(missing_indicators, dtype=torch.float32),
            "categorical_indices": {k: torch.tensor(v, dtype=torch.long) for k, v in cat_indices.items()},
            "cand_input_ids": cand_input_ids,
            "cand_attention_mask": cand_attention_mask,
            "constraint_mask": torch.tensor(constraint_mask, dtype=torch.float32) if constraint_mask else None,
            "num_candidates": num_cands,
            "cand_outcome_binary": torch.tensor(cand_outcome_binary, dtype=torch.float32) if cand_outcome_binary else None,
            "cand_outcome_binary_mask": torch.tensor(cand_outcome_binary_mask, dtype=torch.float32) if cand_outcome_binary_mask else None,
            "cand_risk": torch.tensor(cand_risk, dtype=torch.float32) if cand_risk else None,
            "cand_risk_mask": torch.tensor(cand_risk_mask, dtype=torch.float32) if cand_risk_mask else None,
            "choice_target": choice_target_idx,
            "choice_mask": 1.0 if choice_target_idx is not None else 0.0,
            "binary_target": td.binary_decision if td.binary_decision is not None else 0,
            "binary_mask": 1.0 if td.binary_decision is not None else 0.0,
            "priority_target": td.priority_level if td.priority_level is not None else 0,
            "priority_mask": 1.0 if td.priority_level is not None else 0.0,
            "risk_target": td.risk_level if td.risk_level is not None else 0,
            "risk_mask": 1.0 if td.risk_level is not None else 0.0,
            "score_target": td.decision_score if td.decision_score is not None else 0.0,
            "score_mask": 1.0 if td.decision_score is not None else 0.0,
            "outcome_binary": ol.observed_outcome_binary if (ol and ol.observed_outcome_binary is not None) else 0,
            "outcome_binary_mask": 1.0 if (ol and ol.observed_outcome_binary is not None) else 0.0,
            "outcome_multi": ol.observed_outcome_multiclass if (ol and ol.observed_outcome_multiclass is not None) else 0,
            "outcome_multi_mask": 1.0 if (ol and ol.observed_outcome_multiclass is not None) else 0.0,
            "outcome_numeric": ol.observed_resolution_time_hours if (ol and ol.observed_resolution_time_hours is not None) else 0.0,
            "outcome_numeric_mask": 1.0 if (ol and ol.observed_resolution_time_hours is not None) else 0.0,
            "risk_prob": ol.observed_risk_realized if (ol and ol.observed_risk_realized is not None) else 0.0,
            "risk_prob_mask": 1.0 if (ol and ol.observed_risk_realized is not None) else 0.0,
            "example_id": example.example_id,
        }
