"""Data preprocessing module for CDM-1.
Tokenizes text using Hugging Face AutoTokenizer, normalizes structured features,
and compiles candidate matrices and constraint masks.
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
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self.schema = feature_schema or FeatureSchema.get_default_schema()
        self.max_seq_length = max_seq_length

    def preprocess_example(
        self, example: DecisionExample
    ) -> Dict[str, Any]:
        """Preprocesses a single DecisionExample into model inputs."""
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
        # Encode candidate action text
        cand_texts = [f"{c.name}: {c.description}" for c in example.candidate_actions]
        num_cands = len(cand_texts)

        # If candidates are present, tokenize them
        cand_tokens = None
        if num_cands > 0:
            cand_tokens = self.tokenizer(
                cand_texts,
                max_length=64,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

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

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "numeric_values": torch.tensor(num_vals, dtype=torch.float32),
            "missing_indicators": torch.tensor(missing_indicators, dtype=torch.float32),
            "categorical_indices": {k: torch.tensor(v, dtype=torch.long) for k, v in cat_indices.items()},
            "cand_tokens": cand_tokens,
            "constraint_mask": torch.tensor(constraint_mask, dtype=torch.float32) if constraint_mask else None,
            "choice_target": choice_target_idx,
            "binary_target": example.target_decision.binary_decision,
            "priority_target": example.target_decision.priority_level,
            "risk_target": example.target_decision.risk_level,
            "score_target": example.target_decision.decision_score,
            "outcome_binary": (
                example.outcome_label.observed_outcome_binary
                if example.outcome_label
                else None
            ),
            "outcome_multi": (
                example.outcome_label.observed_outcome_multiclass
                if example.outcome_label
                else None
            ),
            "outcome_numeric": (
                example.outcome_label.observed_resolution_time_hours
                if example.outcome_label
                else None
            ),
            "example_id": example.example_id,
        }
