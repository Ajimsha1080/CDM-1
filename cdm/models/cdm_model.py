"""CDM-1 — Cognitive Decision Model Unified Architecture.
Integrates Language Encoder, Structured Feature Encoder, Cognitive Fusion,
Predictive Network, Multi-Head Decision Network, and Candidate Selection Scorer.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import os
import sys
import time
import shutil
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from safetensors import safe_open

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

        # Candidate text encoder projection (for projected candidate actions)
        self.candidate_text_proj = nn.Sequential(
            nn.Linear(self.config.language_encoder.projection_dim, self.config.fusion.fusion_dim),
            nn.LayerNorm(self.config.fusion.fusion_dim),
            nn.GELU(),
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

    def encode_candidates_from_tokens(
        self,
        candidate_input_ids: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes candidate text tokens [batch_size, num_cands, seq_len]

        into candidate representations [batch_size, num_cands, dim].
        """
        batch_size, num_cands, seq_len = candidate_input_ids.shape
        flat_ids = candidate_input_ids.reshape(-1, seq_len)
        flat_mask = candidate_attention_mask.reshape(-1, seq_len)

        # Encode via language encoder
        flat_emb = self.language_encoder(flat_ids, flat_mask)
        flat_proj = self.candidate_text_proj(flat_emb)

        candidate_embeddings = flat_proj.reshape(batch_size, num_cands, -1)
        return candidate_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        numeric_values: torch.Tensor,
        missing_indicators: Optional[torch.Tensor] = None,
        categorical_indices: Optional[Dict[str, torch.Tensor]] = None,
        candidate_embeddings: Optional[torch.Tensor] = None,
        candidate_input_ids: Optional[torch.Tensor] = None,
        candidate_attention_mask: Optional[torch.Tensor] = None,
        constraint_mask: Optional[torch.Tensor] = None,
        existence_mask: Optional[torch.Tensor] = None,
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

        # 2. Candidate Embeddings (either provided directly or encoded from candidate tokens)
        if candidate_embeddings is None and candidate_input_ids is not None and candidate_attention_mask is not None:
            candidate_embeddings = self.encode_candidates_from_tokens(
                candidate_input_ids=candidate_input_ids,
                candidate_attention_mask=candidate_attention_mask,
            )

        # 3. Action-Conditioned Predictive Network Output
        predictive_out = self.predictive_network(
            cognitive_state=cognitive_state,
            action_embedding=selected_action_embedding,
            candidate_embeddings=candidate_embeddings,
        )

        # 4. Decision Network Output
        decision_out = self.decision_network(cognitive_state=cognitive_state)

        # 5. Candidate Scoring conditioned on cognitive state and predicted candidate outcomes/risks
        candidate_out: Optional[Dict[str, torch.Tensor]] = None
        if candidate_embeddings is not None:
            cand_outcomes = (
                torch.sigmoid(predictive_out["candidate_binary_outcome_logits"])
                if "candidate_binary_outcome_logits" in predictive_out
                else None
            )
            cand_risks = predictive_out.get("candidate_estimated_risk")

            candidate_out = self.candidate_scorer(
                cognitive_context=cognitive_state,
                candidate_embeddings=candidate_embeddings,
                candidate_outcomes=cand_outcomes,
                candidate_risks=cand_risks,
                constraint_mask=constraint_mask,
                existence_mask=existence_mask,
            )

        return {
            "cognitive_state": cognitive_state,
            "decision": decision_out,
            "predictive": predictive_out,
            "candidates": candidate_out,
            "candidate_embeddings": candidate_embeddings,
        }

    def compute_multitask_loss(
        self,
        outputs: Dict[str, Any],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes supervised multi-task decision losses with explicit masking

        for missing targets. Never replaces missing labels with artificial zeros.
        """
        device = outputs["cognitive_state"].device
        total_loss = torch.tensor(0.0, device=device)
        loss_dict: Dict[str, float] = {}
        lw = self.config.loss_weights

        # 1. Binary Decision Loss
        if "binary_target" in targets and targets["binary_target"] is not None:
            bin_target = targets["binary_target"].long().to(device)
            bin_mask = targets.get("binary_mask", torch.ones_like(bin_target, dtype=torch.float32)).to(device)
            if bin_mask.sum() > 0:
                bin_logits = outputs["decision"]["binary_logits"]
                raw_loss_bin = F.cross_entropy(bin_logits, bin_target, reduction="none")
                loss_bin = (raw_loss_bin * bin_mask).sum() / torch.clamp(bin_mask.sum(), min=1.0)
                total_loss = total_loss + lw.lambda_binary * loss_bin
                loss_dict["loss_binary"] = loss_bin.item()

        # 2. Priority Decision Loss (CrossEntropy with balanced class weighting for P2)
        if "priority_target" in targets and targets["priority_target"] is not None:
            prio_target = targets["priority_target"].long().to(device)
            prio_mask = targets.get("priority_mask", torch.ones_like(prio_target, dtype=torch.float32)).to(device)
            if prio_mask.sum() > 0:
                prio_logits = outputs["decision"]["priority_logits"]
                prio_class_weights = torch.tensor([1.0, 1.0, 2.0, 1.0], device=device)
                raw_loss_prio = F.cross_entropy(prio_logits, prio_target, weight=prio_class_weights, reduction="none")
                loss_prio = (raw_loss_prio * prio_mask).sum() / torch.clamp(prio_mask.sum(), min=1.0)
                total_loss = total_loss + lw.lambda_priority * loss_prio
                loss_dict["loss_priority"] = loss_prio.item()

        # 3. Decision Score Loss (Smooth L1 regression)
        if "score_target" in targets and targets["score_target"] is not None:
            score_target = targets["score_target"].float().to(device)
            score_mask = targets.get("score_mask", torch.ones_like(score_target, dtype=torch.float32)).to(device)
            if score_mask.sum() > 0:
                pred_score = outputs["decision"]["decision_score"]
                # Target normalization: scale 0-100 range to [0, 1] to prevent gradient scale mismatch
                raw_loss_score = F.smooth_l1_loss(pred_score / 100.0, score_target / 100.0, reduction="none")
                loss_score = (raw_loss_score * score_mask).sum() / torch.clamp(score_mask.sum(), min=1.0)
                total_loss = total_loss + lw.lambda_score * loss_score
                loss_dict["loss_score"] = loss_score.item()

        # 4. Risk Classification Head Loss
        if "risk_target" in targets and targets["risk_target"] is not None:
            risk_target = targets["risk_target"].long().to(device)
            risk_mask = targets.get("risk_mask", torch.ones_like(risk_target, dtype=torch.float32)).to(device)
            if risk_mask.sum() > 0:
                risk_logits = outputs["decision"]["risk_logits"]
                raw_loss_risk = F.cross_entropy(risk_logits, risk_target, reduction="none")
                loss_risk_cls = (raw_loss_risk * risk_mask).sum() / torch.clamp(risk_mask.sum(), min=1.0)
                total_loss = total_loss + lw.lambda_risk * loss_risk_cls
                loss_dict["loss_risk_classification"] = loss_risk_cls.item()

        # 5. Risk Probability Estimation Loss (Continuous observed risk probability)
        if "risk_prob_target" in targets and targets["risk_prob_target"] is not None:
            risk_prob_target = targets["risk_prob_target"].float().unsqueeze(-1).to(device)
            risk_prob_mask = targets.get("risk_prob_mask", torch.ones(risk_prob_target.shape[0], device=device)).unsqueeze(-1)
            if risk_prob_mask.sum() > 0:
                pred_risk = outputs["predictive"]["estimated_risk"]
                raw_loss_risk_prob = F.binary_cross_entropy(pred_risk, risk_prob_target, reduction="none")
                loss_risk_prob = (raw_loss_risk_prob * risk_prob_mask).sum() / torch.clamp(risk_prob_mask.sum(), min=1.0)
                total_loss = total_loss + (lw.lambda_risk * 0.5) * loss_risk_prob
                loss_dict["loss_risk_probability"] = loss_risk_prob.item()

        # 6. Candidate Choice and Pairwise Ranking Loss
        if "choice_target" in targets and targets["choice_target"] is not None:
            if outputs.get("candidates") is not None:
                choice_target = targets["choice_target"].long().to(device)
                choice_mask = targets.get("choice_mask", torch.ones_like(choice_target, dtype=torch.float32)).to(device)
                if choice_mask.sum() > 0:
                    cand_scores = outputs["candidates"]["constrained_scores"]
                    raw_loss_choice = F.cross_entropy(cand_scores, choice_target, reduction="none")
                    loss_choice = (raw_loss_choice * choice_mask).sum() / torch.clamp(choice_mask.sum(), min=1.0)

                    existence_mask = targets.get("candidate_existence_mask")
                    loss_rank = self.candidate_scorer.compute_ranking_loss(
                        scores=cand_scores,
                        target_idx=choice_target,
                        choice_mask=choice_mask,
                        existence_mask=existence_mask,
                        margin=1.0,
                    )

                    total_loss = total_loss + (lw.lambda_choice * loss_choice) + (lw.lambda_ranking * loss_rank)
                    loss_dict["loss_choice"] = loss_choice.item()
                    loss_dict["loss_ranking"] = loss_rank.item()

        # 7. Outcome Binary Prediction Loss
        if "outcome_binary_target" in targets and targets["outcome_binary_target"] is not None:
            out_bin_target = targets["outcome_binary_target"].float().unsqueeze(-1).to(device)
            out_bin_mask = targets.get("outcome_binary_mask", torch.ones(out_bin_target.shape[0], device=device)).unsqueeze(-1)
            if out_bin_mask.sum() > 0:
                pred_out_logits = outputs["predictive"]["binary_outcome_logits"]
                raw_loss_out_bin = F.binary_cross_entropy_with_logits(pred_out_logits, out_bin_target, reduction="none")
                loss_out_bin = (raw_loss_out_bin * out_bin_mask).sum() / torch.clamp(out_bin_mask.sum(), min=1.0)
                total_loss = total_loss + lw.lambda_outcome_binary * loss_out_bin
                loss_dict["loss_outcome_binary"] = loss_out_bin.item()

        # 8. Outcome Multiclass Prediction Loss
        if "outcome_multi_target" in targets and targets["outcome_multi_target"] is not None:
            out_multi_target = targets["outcome_multi_target"].long().to(device)
            out_multi_mask = targets.get("outcome_multi_mask", torch.ones_like(out_multi_target, dtype=torch.float32)).to(device)
            if out_multi_mask.sum() > 0:
                pred_multi_logits = outputs["predictive"]["multiclass_outcome_logits"]
                raw_loss_out_multi = F.cross_entropy(pred_multi_logits, out_multi_target, reduction="none")
                loss_out_multi = (raw_loss_out_multi * out_multi_mask).sum() / torch.clamp(out_multi_mask.sum(), min=1.0)
                total_loss = total_loss + lw.lambda_outcome_multiclass * loss_out_multi
                loss_dict["loss_outcome_multiclass"] = loss_out_multi.item()

        # 9. Outcome Numerical (Resolution Time) Loss
        if "outcome_numeric_target" in targets and targets["outcome_numeric_target"] is not None:
            out_num_target = targets["outcome_numeric_target"].float().unsqueeze(-1).to(device)
            out_num_mask = targets.get("outcome_numeric_mask", torch.ones(out_num_target.shape[0], device=device)).unsqueeze(-1)
            if out_num_mask.sum() > 0:
                pred_num = outputs["predictive"]["numeric_outcome"]
                # Target normalization: scale resolution hours to smooth loss scale
                raw_loss_out_num = F.smooth_l1_loss(pred_num / 10.0, out_num_target / 10.0, reduction="none")
                loss_out_num = (raw_loss_out_num * out_num_mask).sum() / torch.clamp(out_num_mask.sum(), min=1.0)
                total_loss = total_loss + lw.lambda_outcome_numeric * loss_out_num
                loss_dict["loss_outcome_numeric"] = loss_out_num.item()

        # 10. Candidate-Conditioned Predictive Outcome Losses (Priority 1)
        if outputs.get("predictive") and "candidate_binary_outcome_logits" in outputs["predictive"]:
            cand_bin_logits = outputs["predictive"]["candidate_binary_outcome_logits"]  # [B, K, 1]
            cand_risk_preds = outputs["predictive"]["candidate_estimated_risk"]        # [B, K, 1]
            cand_num_preds = outputs["predictive"]["candidate_numeric_outcome"]        # [B, K, 1]

            # (a) Per-candidate binary outcome supervision
            if "candidate_binary_targets" in targets and targets["candidate_binary_targets"] is not None:
                cb_target = targets["candidate_binary_targets"].float().unsqueeze(-1).to(device)
                cb_mask = targets.get("candidate_binary_mask")
                if cb_mask is not None:
                    cb_mask = cb_mask.float().unsqueeze(-1).to(device)
                else:
                    cb_mask = torch.ones_like(cb_target)
                if cb_mask.sum() > 0:
                    raw_cb_loss = F.binary_cross_entropy_with_logits(cand_bin_logits, cb_target, reduction="none")
                    loss_cb = (raw_cb_loss * cb_mask).sum() / torch.clamp(cb_mask.sum(), min=1.0)
                    total_loss = total_loss + (lw.lambda_outcome_binary * 0.5) * loss_cb
                    loss_dict["loss_cand_outcome_binary"] = loss_cb.item()
            elif "choice_target" in targets and targets["choice_target"] is not None and "outcome_binary_target" in targets:
                c_idx = torch.clamp(targets["choice_target"].long(), 0, cand_bin_logits.shape[1] - 1).to(device)
                c_mask = targets.get("choice_mask", torch.ones_like(c_idx, dtype=torch.float32)).to(device)
                out_b_mask = targets.get("outcome_binary_mask", torch.ones_like(c_mask)).to(device) * c_mask

                if out_b_mask.sum() > 0:
                    chosen_logits = cand_bin_logits.gather(1, c_idx.view(-1, 1, 1)).squeeze(1)  # [B, 1]
                    out_b_target = targets["outcome_binary_target"].float().unsqueeze(-1).to(device)
                    raw_loss_cand_bin = F.binary_cross_entropy_with_logits(chosen_logits, out_b_target, reduction="none")
                    loss_cand_bin = (raw_loss_cand_bin * out_b_mask.unsqueeze(-1)).sum() / torch.clamp(out_b_mask.sum(), min=1.0)
                    total_loss = total_loss + (lw.lambda_outcome_binary * 0.5) * loss_cand_bin
                    loss_dict["loss_cand_outcome_binary"] = loss_cand_bin.item()

            # (b) Per-candidate risk estimation supervision
            if "candidate_risk_targets" in targets and targets["candidate_risk_targets"] is not None:
                cr_target = targets["candidate_risk_targets"].float().unsqueeze(-1).to(device)
                cr_mask = targets.get("candidate_risk_mask")
                if cr_mask is not None:
                    cr_mask = cr_mask.float().unsqueeze(-1).to(device)
                else:
                    cr_mask = torch.ones_like(cr_target)
                if cr_mask.sum() > 0:
                    raw_cr_loss = F.binary_cross_entropy(cand_risk_preds, cr_target, reduction="none")
                    loss_cr = (raw_cr_loss * cr_mask).sum() / torch.clamp(cr_mask.sum(), min=1.0)
                    total_loss = total_loss + (lw.lambda_risk * 0.25) * loss_cr
                    loss_dict["loss_cand_risk"] = loss_cr.item()
            elif "choice_target" in targets and targets["choice_target"] is not None and "risk_prob_target" in targets:
                c_idx = torch.clamp(targets["choice_target"].long(), 0, cand_risk_preds.shape[1] - 1).to(device)
                c_mask = targets.get("choice_mask", torch.ones_like(c_idx, dtype=torch.float32)).to(device)
                r_mask = targets.get("risk_prob_mask", torch.ones_like(c_mask)).to(device) * c_mask

                if r_mask.sum() > 0:
                    chosen_risk = cand_risk_preds.gather(1, c_idx.view(-1, 1, 1)).squeeze(1)
                    r_target = targets["risk_prob_target"].float().unsqueeze(-1).to(device)
                    raw_loss_cr = F.binary_cross_entropy(chosen_risk, r_target, reduction="none")
                    loss_cr = (raw_loss_cr * r_mask.unsqueeze(-1)).sum() / torch.clamp(r_mask.sum(), min=1.0)
                    total_loss = total_loss + (lw.lambda_risk * 0.25) * loss_cr
                    loss_dict["loss_cand_risk"] = loss_cr.item()

        loss_dict["loss_total"] = total_loss.item()
        return total_loss, loss_dict

    def save_pretrained(self, save_directory: str) -> None:
        """Export CDM-1 weights in SafeTensors format with atomic staging and verification.
        Ensures a temporary staging directory is used, the serialized tensors are strictly
        validated before atomic replacement, and existing valid checkpoints are never corrupted.
        """
        save_path = os.path.abspath(save_directory)
        os.makedirs(save_path, exist_ok=True)
        stage_dir = f"{save_path}.tmp_stage_{os.getpid()}_{int(time.time() * 1000)}"
        os.makedirs(stage_dir, exist_ok=True)

        try:
            # 1. Serialize config with strict Unix newlines
            stage_config = os.path.join(stage_dir, "config.json")
            with open(stage_config, "w", encoding="utf-8", newline="\n") as f:
                json.dump(self.config.to_dict(), f, indent=2)

            # 2. Serialize weights in SafeTensors format
            stage_weights = os.path.join(stage_dir, "model.safetensors")
            state_dict = {k: v.contiguous() for k, v in self.state_dict().items()}
            save_file(state_dict, stage_weights)

            # 3. Validate staged weights before atomic move
            if not os.path.isfile(stage_weights) or os.path.getsize(stage_weights) == 0:
                raise RuntimeError(f"Staged checkpoint serialization failed: {stage_weights} is empty or missing.")

            with safe_open(stage_weights, framework="pt") as sf:
                staged_keys = set(sf.keys())
                expected_keys = set(state_dict.keys())
                if staged_keys != expected_keys:
                    missing = expected_keys - staged_keys
                    raise RuntimeError(f"Staged checkpoint validation mismatch: missing {len(missing)} tensors.")

            # 4. Atomically move staged files to final destination
            target_config = os.path.join(save_path, "config.json")
            target_weights = os.path.join(save_path, "model.safetensors")

            os.replace(stage_config, target_config)
            os.replace(stage_weights, target_weights)

        finally:
            if os.path.isdir(stage_dir):
                shutil.rmtree(stage_dir, ignore_errors=True)

    @classmethod
    def from_pretrained(cls, load_directory: str, strict: bool = True) -> "CDMModel":
        """Instantiate and strictly load CDM-1 from a directory with config.json and model.safetensors.

        Raises FileNotFoundError or ValueError if checkpoint files are missing, corrupt,
        or state dictionaries fail verification.
        """
        if not os.path.exists(load_directory):
            raise FileNotFoundError(f"Checkpoint directory does not exist: {load_directory}")

        config_path = os.path.join(load_directory, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Missing config.json in checkpoint directory: {load_directory}")

        weights_path = os.path.join(load_directory, "model.safetensors")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(
                f"Missing model.safetensors in checkpoint directory: {load_directory}. "
                "CDM-1 refuses to silently use random uninitialized weights."
            )

        if os.path.getsize(weights_path) == 0:
            raise ValueError(f"Checkpoint file model.safetensors is empty (0 bytes): {weights_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config = CDMModelConfig.from_dict(config_dict)

        model = cls(config)
        state_dict = load_file(weights_path)

        load_result = model.load_state_dict(state_dict, strict=False)
        
        # Rigorously verify load_state_dict results - never ignore missing or unexpected keys
        critical_missing = [k for k in load_result.missing_keys if not k.startswith("language_encoder.transformer.")]
        if critical_missing:
            raise ValueError(
                f"Checkpoint verification failed! Critical keys missing from state_dict: {critical_missing[:5]}"
            )
        if strict and load_result.missing_keys:
            raise ValueError(
                f"Checkpoint verification failed! Missing keys in strict mode: {load_result.missing_keys[:5]}"
            )
        if len(load_result.unexpected_keys) > 0:
            raise ValueError(
                f"Checkpoint verification failed! Unexpected keys found in state_dict: {load_result.unexpected_keys[:5]}"
            )

        return model
