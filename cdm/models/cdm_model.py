"""CDM-1 — Cognitive Decision Model Unified Architecture.
Integrates Language Encoder, Structured Feature Encoder, Cognitive Fusion,
Predictive Network, Multi-Head Decision Network, and Candidate Selection Scorer.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

from cdm.config.model_config import CDMModelConfig
from cdm.config.feature_schema import FeatureSchema
from cdm.models.language_encoder import LanguageEncoder
from cdm.models.structured_encoder import StructuredFeatureEncoder
from cdm.models.fusion import CognitiveFusion
from cdm.models.predictive_network import PredictiveNetwork
from cdm.models.decision_network import DecisionNetwork
from cdm.models.candidate_scorer import CandidateScorer


class CDMModel(nn.Module):
    """CDM-1 Unified Hybrid Neural Cognitive Decision Model."""

    def __init__(self, config: Optional[CDMModelConfig] = None):
        super().__init__()
        self.config = config or CDMModelConfig()

        # 1. Component A: Language Encoder
        self.language_encoder = LanguageEncoder(self.config.language_encoder)

        # 2. Component B: Structured Feature Encoder
        self.structured_encoder = StructuredFeatureEncoder(self.config.structured_encoder)

        # 3. Component C: Cognitive Representation Fusion
        self.fusion = CognitiveFusion(self.config.fusion)

        # 4. Component D: Predictive Network
        self.predictive_network = PredictiveNetwork(
            input_dim=self.config.fusion.fusion_dim,
            action_dim=self.config.fusion.fusion_dim,
            config=self.config.predictive,
        )

        # 5. Component E: Multi-Head Decision Network
        self.decision_network = DecisionNetwork(
            input_dim=self.config.fusion.fusion_dim,
            config=self.config.decision,
        )

        # 6. Candidate Scorer
        self.candidate_scorer = CandidateScorer(
            context_dim=self.config.fusion.fusion_dim,
            candidate_dim=self.config.fusion.fusion_dim,
        )

        # Candidate text encoder projection (for encoding textual candidate actions)
        self.candidate_text_proj = nn.Sequential(
            nn.Linear(self.config.language_encoder.projection_dim, self.config.fusion.fusion_dim),
            nn.LayerNorm(self.config.fusion.fusion_dim),
        )

    def count_parameters(self) -> Dict[str, int]:
        """Audit trainable, frozen, and total parameters of CDM-1."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "total_parameters": total,
        }

    def encode_context(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_values: torch.Tensor,
        missing_indicators: Optional[torch.Tensor] = None,
        categorical_indices: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Extracts text and tabular representations, then fuses them into the shared cognitive state."""
        text_emb = self.language_encoder(input_ids, attention_mask)
        struct_emb = self.structured_encoder(
            numeric_values=numeric_values,
            missing_indicators=missing_indicators,
            categorical_indices=categorical_indices,
        )
        cognitive_state = self.fusion(text_emb, struct_emb)
        return cognitive_state

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_values: torch.Tensor,
        missing_indicators: Optional[torch.Tensor] = None,
        categorical_indices: Optional[Dict[str, torch.Tensor]] = None,
        candidate_embeddings: Optional[torch.Tensor] = None,
        constraint_mask: Optional[torch.Tensor] = None,
        selected_action_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Unified forward pass returning cognitive state, decision outputs,

        predictive estimates, and candidate rankings.
        """
        # 1. Cognitive State Formation
        cognitive_state = self.encode_context(
            input_ids=input_ids,
            attention_mask=attention_mask,
            numeric_values=numeric_values,
            missing_indicators=missing_indicators,
            categorical_indices=categorical_indices,
        )

        # 2. Predictive Network Output
        predictive_out = self.predictive_network(
            cognitive_state=cognitive_state,
            action_embedding=selected_action_embedding,
        )

        # 3. Decision Network Output
        decision_out = self.decision_network(cognitive_state=cognitive_state)

        # 4. Candidate Scoring (if candidates provided)
        candidate_out: Optional[Dict[str, torch.Tensor]] = None
        if candidate_embeddings is not None:
            candidate_out = self.candidate_scorer(
                cognitive_context=cognitive_state,
                candidate_embeddings=candidate_embeddings,
                constraint_mask=constraint_mask,
            )

        return {
            "cognitive_state": cognitive_state,
            "decision": decision_out,
            "predictive": predictive_out,
            "candidates": candidate_out,
        }

    def compute_multitask_loss(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes balanced multi-task decision loss with configurable task weights."""
        total_loss = torch.tensor(0.0, device=outputs["cognitive_state"].device)
        loss_dict: Dict[str, float] = {}
        lw = self.config.loss_weights

        # 1. Binary Decision Loss (BCE / CrossEntropy)
        if "binary_target" in targets and targets["binary_target"] is not None:
            bin_target = targets["binary_target"].long()
            bin_logits = outputs["decision"]["binary_logits"]
            loss_bin = F.cross_entropy(bin_logits, bin_target)
            total_loss = total_loss + lw.lambda_binary * loss_bin
            loss_dict["loss_binary"] = loss_bin.item()

        # 2. Priority Decision Loss (CrossEntropy)
        if "priority_target" in targets and targets["priority_target"] is not None:
            prio_target = targets["priority_target"].long()
            prio_logits = outputs["decision"]["priority_logits"]
            loss_prio = F.cross_entropy(prio_logits, prio_target)
            total_loss = total_loss + lw.lambda_priority * loss_prio
            loss_dict["loss_priority"] = loss_prio.item()

        # 3. Decision Score Loss (Smooth L1 / Huber)
        if "score_target" in targets and targets["score_target"] is not None:
            score_target = targets["score_target"].float()
            pred_score = outputs["decision"]["decision_score"]
            loss_score = F.smooth_l1_loss(pred_score, score_target)
            total_loss = total_loss + lw.lambda_score * loss_score
            loss_dict["loss_score"] = loss_score.item()

        # 4. Choice & Ranking Loss
        if "choice_target" in targets and targets["choice_target"] is not None:
            if outputs.get("candidates") is not None:
                choice_target = targets["choice_target"].long()
                cand_scores = outputs["candidates"]["constrained_scores"]
                loss_choice = F.cross_entropy(cand_scores, choice_target)
                loss_rank = self.candidate_scorer.compute_ranking_loss(
                    cand_scores, choice_target, margin=1.0
                )
                total_loss = total_loss + (lw.lambda_choice * loss_choice) + (lw.lambda_ranking * loss_rank)
                loss_dict["loss_choice"] = loss_choice.item()
                loss_dict["loss_ranking"] = loss_rank.item()

        # 5. Outcome Binary Loss
        if "outcome_binary_target" in targets and targets["outcome_binary_target"] is not None:
            out_bin_target = targets["outcome_binary_target"].float().unsqueeze(-1)
            pred_out_logits = outputs["predictive"]["binary_outcome_logits"]
            loss_out_bin = F.binary_cross_entropy_with_logits(pred_out_logits, out_bin_target)
            total_loss = total_loss + lw.lambda_outcome_binary * loss_out_bin
            loss_dict["loss_outcome_binary"] = loss_out_bin.item()

        # 6. Outcome Multiclass Loss
        if "outcome_multi_target" in targets and targets["outcome_multi_target"] is not None:
            out_multi_target = targets["outcome_multi_target"].long()
            pred_multi_logits = outputs["predictive"]["multiclass_outcome_logits"]
            loss_out_multi = F.cross_entropy(pred_multi_logits, out_multi_target)
            total_loss = total_loss + lw.lambda_outcome_multiclass * loss_out_multi
            loss_dict["loss_outcome_multiclass"] = loss_out_multi.item()

        # 7. Outcome Numerical Loss
        if "outcome_numeric_target" in targets and targets["outcome_numeric_target"] is not None:
            out_num_target = targets["outcome_numeric_target"].float().unsqueeze(-1)
            pred_num = outputs["predictive"]["numeric_outcome"]
            loss_out_num = F.smooth_l1_loss(pred_num, out_num_target)
            total_loss = total_loss + lw.lambda_outcome_numeric * loss_out_num
            loss_dict["loss_outcome_numeric"] = loss_out_num.item()

        loss_dict["loss_total"] = total_loss.item()
        return total_loss, loss_dict

    def save_pretrained(self, save_directory: str) -> None:
        """Export CDM-1 weights in SafeTensors format with full configuration."""
        os.makedirs(save_directory, exist_ok=True)

        # 1. Save config JSON
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # 2. Save model weights using safetensors
        weights_path = os.path.join(save_directory, "model.safetensors")
        state_dict = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state_dict, weights_path)

    @classmethod
    def from_pretrained(cls, load_directory: str) -> "CDMModel":
        """Instantiate and load CDM-1 from a directory with config.json and model.safetensors."""
        config_path = os.path.join(load_directory, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config = CDMModelConfig.from_dict(config_dict)

        model = cls(config)
        weights_path = os.path.join(load_directory, "model.safetensors")
        if os.path.exists(weights_path):
            state_dict = load_file(weights_path)
            model.load_state_dict(state_dict, strict=False)
        return model
