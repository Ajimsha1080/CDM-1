"""STAGE 7: Complete Multi-Task & Candidate-Conditioned Training Pipeline for CDM-1.
Performs end-to-end multi-task optimization across text context, candidate actions,
predictive forward outcomes, risk estimation, and multi-head decisions.
Logs every individual loss and ensures candidate scorer receives gradient updates.
"""

import os
import sys
import json
import time
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cdm.config.model_config import CDMModelConfig
from cdm.models.cdm_model import CDMModel
from cdm.data.dataset_loader import DecisionDataset, collate_decision_batch
from cdm.data.curated_corpus import load_dataset_from_disk
from cdm.data.preprocessor import DecisionPreprocessor
from cdm.utils.training_utils import set_seed, EarlyStopping


def cache_multitask_batches(model: CDMModel, loader: DataLoader, device: torch.device):
    """Pre-computes sequence token pooling through the frozen transformer backbone,
    allowing the language projection layer, structured encoder, fusion, predictive network,
    decision heads, and candidate scorer to be trained end-to-end with live backpropagation.
    """
    model.eval()
    cached_batches = []
    print("Pre-computing context and candidate representations with frozen backbone...", flush=True)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Extract sequence output from transformer backbone and pool
            seq_out = model.language_encoder.transformer(input_ids=input_ids, attention_mask=attention_mask)[0]
            pooled_text = model.language_encoder.pool_tokens(seq_out, attention_mask)

            # Pre-extract candidate embeddings through transformer backbone
            c_input_ids = batch["candidate_input_ids"]
            c_attention_mask = batch["candidate_attention_mask"]
            candidate_pooled = None

            if c_input_ids is not None and c_attention_mask is not None:
                b_size, n_cands, seq_len = c_input_ids.shape
                flat_ids = c_input_ids.to(device).reshape(-1, seq_len)
                flat_mask = c_attention_mask.to(device).reshape(-1, seq_len)
                flat_out = model.language_encoder.transformer(input_ids=flat_ids, attention_mask=flat_mask)[0]
                flat_pooled = model.language_encoder.pool_tokens(flat_out, flat_mask)
                candidate_pooled = flat_pooled.reshape(b_size, n_cands, -1)

            cached_batches.append({
                "pooled_text": pooled_text.detach(),
                "candidate_pooled": candidate_pooled.detach() if candidate_pooled is not None else None,
                "numeric_values": batch["numeric_values"].to(device),
                "missing_indicators": batch["missing_indicators"].to(device),
                "categorical_indices": {k: v.to(device) for k, v in batch["categorical_indices"].items()},
                "targets": {k: v.to(device) for k, v in batch["targets"].items()},
                "constraint_mask": batch["constraint_mask"].to(device) if batch["constraint_mask"] is not None else None,
                "existence_mask": batch["existence_mask"].to(device) if batch["existence_mask"] is not None else None,
            })
    return cached_batches


def train_multitask(
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 16,
    checkpoint_dir: str = "experiments/checkpoints/cdm-1-general",
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STAGE 7: Multi-Task & Candidate-Conditioned Training on {device} ===", flush=True)

    train_data = load_dataset_from_disk("datasets/train.jsonl")
    val_data = load_dataset_from_disk("datasets/val.jsonl")

    preprocessor = DecisionPreprocessor()
    train_ds = DecisionDataset(train_data, preprocessor)
    val_ds = DecisionDataset(val_data, preprocessor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_decision_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_decision_batch)

    cfg = CDMModelConfig()
    model = CDMModel(cfg).to(device)
    model.language_encoder.freeze_backbone()

    param_stats = model.count_parameters()
    print(f"Trainable Parameters: {param_stats['trainable_parameters']:,}", flush=True)
    print(f"Frozen Parameters:    {param_stats['frozen_parameters']:,}", flush=True)
    print(f"Total Parameters:     {param_stats['total_parameters']:,}", flush=True)

    # Record initial weight snapshots to prove parameters update
    initial_proj_weight = model.language_encoder.projection[0].weight.clone().detach()
    initial_scorer_weight = model.candidate_scorer.scorer_mlp[0].weight.clone().detach()

    # Pre-compute representations
    cached_train = cache_multitask_batches(model, train_loader, device)
    cached_val = cache_multitask_batches(model, val_loader, device)

    # Train all custom neural components (including candidate scorer & projection)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stopping = EarlyStopping(patience=5, delta=1e-3, mode="min")

    history = []
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        epoch_task_losses = {}
        epoch_grad_norms = []

        for batch in cached_train:
            text_emb = model.language_encoder.projection(batch["pooled_text"])
            cand_pooled = batch["candidate_pooled"]
            cand_raw = (
                model.language_encoder.projection(cand_pooled)
                if cand_pooled is not None
                else None
            )
            num_vals = batch["numeric_values"]
            missing = batch["missing_indicators"]
            cat_idx = batch["categorical_indices"]
            targets = batch["targets"]
            constraint_mask = batch["constraint_mask"]
            existence_mask = batch["existence_mask"]

            optimizer.zero_grad()

            # 1. Structured & Fusion representation
            struct_emb = model.structured_encoder(num_vals, missing, cat_idx)
            cognitive_state = model.fusion(text_emb, struct_emb)

            # 2. Action-Conditioned Predictive Modeling & Candidate Scoring
            cand_embs = model.candidate_text_proj(cand_raw) if cand_raw is not None else None
            predictive_out = model.predictive_network(cognitive_state, candidate_embeddings=cand_embs)
            decision_out = model.decision_network(cognitive_state)

            candidate_out = None
            if cand_embs is not None:
                cand_outcomes = (
                    torch.sigmoid(predictive_out["candidate_binary_outcome_logits"])
                    if "candidate_binary_outcome_logits" in predictive_out
                    else None
                )
                cand_risks = predictive_out.get("candidate_estimated_risk")
                candidate_out = model.candidate_scorer(
                    cognitive_context=cognitive_state,
                    candidate_embeddings=cand_embs,
                    candidate_outcomes=cand_outcomes,
                    candidate_risks=cand_risks,
                    constraint_mask=constraint_mask,
                    existence_mask=existence_mask,
                )

            outputs = {
                "cognitive_state": cognitive_state,
                "decision": decision_out,
                "predictive": predictive_out,
                "candidates": candidate_out,
            }

            loss, loss_dict = model.compute_multitask_loss(outputs, targets)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
            optimizer.step()

            train_loss += loss.item()
            epoch_grad_norms.append(grad_norm)
            for k, v in loss_dict.items():
                epoch_task_losses[k] = epoch_task_losses.get(k, 0.0) + (v / len(cached_train))

        scheduler.step()
        avg_train_loss = train_loss / len(cached_train)
        avg_grad_norm = float(np.mean(epoch_grad_norms))

        # Validation step across all modalities and domains
        model.eval()
        val_loss = 0.0
        val_correct_bin, val_total_bin = 0, 0
        val_correct_choice, val_total_choice = 0, 0
        val_correct_prio, val_total_prio = 0, 0
        val_correct_risk, val_total_risk = 0, 0
        val_score_diffs = []
        val_domain_correct = defaultdict(lambda: {"correct": 0, "total": 0})

        with torch.no_grad():
            for b_idx, batch in enumerate(cached_val):
                text_emb = model.language_encoder.projection(batch["pooled_text"])
                cand_pooled = batch["candidate_pooled"]
                cand_raw = (
                    model.language_encoder.projection(cand_pooled)
                    if cand_pooled is not None
                    else None
                )
                num_vals = batch["numeric_values"]
                missing = batch["missing_indicators"]
                cat_idx = batch["categorical_indices"]
                targets = batch["targets"]
                constraint_mask = batch["constraint_mask"]
                existence_mask = batch["existence_mask"]

                struct_emb = model.structured_encoder(num_vals, missing, cat_idx)
                cognitive_state = model.fusion(text_emb, struct_emb)

                cand_embs = model.candidate_text_proj(cand_raw) if cand_raw is not None else None
                predictive_out = model.predictive_network(cognitive_state, candidate_embeddings=cand_embs)
                decision_out = model.decision_network(cognitive_state)

                candidate_out = None
                if cand_embs is not None:
                    cand_outcomes = (
                        torch.sigmoid(predictive_out["candidate_binary_outcome_logits"])
                        if "candidate_binary_outcome_logits" in predictive_out
                        else None
                    )
                    cand_risks = predictive_out.get("candidate_estimated_risk")
                    candidate_out = model.candidate_scorer(
                        cognitive_context=cognitive_state,
                        candidate_embeddings=cand_embs,
                        candidate_outcomes=cand_outcomes,
                        candidate_risks=cand_risks,
                        constraint_mask=constraint_mask,
                        existence_mask=existence_mask,
                    )

                outputs = {
                    "cognitive_state": cognitive_state,
                    "decision": decision_out,
                    "predictive": predictive_out,
                    "candidates": candidate_out,
                }

                v_loss, _ = model.compute_multitask_loss(outputs, targets)
                val_loss += v_loss.item()

                bin_preds = decision_out["binary_prediction"]
                bin_mask = targets["binary_mask"]
                val_correct_bin += ((bin_preds == targets["binary_target"]) * bin_mask).sum().item()
                val_total_bin += bin_mask.sum().item()

                prio_preds = decision_out["priority_prediction"]
                prio_mask = targets["priority_mask"]
                val_correct_prio += ((prio_preds == targets["priority_target"]) * prio_mask).sum().item()
                val_total_prio += prio_mask.sum().item()

                risk_preds = decision_out["risk_prediction"]
                risk_mask = targets["risk_mask"]
                val_correct_risk += ((risk_preds == targets["risk_target"]) * risk_mask).sum().item()
                val_total_risk += risk_mask.sum().item()

                score_preds = decision_out["decision_score"]
                score_mask = targets["score_mask"]
                diffs = torch.abs(score_preds - targets["score_target"]) * score_mask
                val_score_diffs.extend(diffs[score_mask.bool()].cpu().tolist())

                if candidate_out is not None:
                    choice_preds = candidate_out["selected_candidate_idx"]
                    choice_mask = targets["choice_mask"]
                    val_correct_choice += ((choice_preds == targets["choice_target"]) * choice_mask).sum().item()
                    val_total_choice += choice_mask.sum().item()

        avg_val_loss = val_loss / len(cached_val)
        val_bin_acc = val_correct_bin / max(val_total_bin, 1)
        val_choice_acc = val_correct_choice / max(val_total_choice, 1)
        val_prio_acc = val_correct_prio / max(val_total_prio, 1)
        val_risk_acc = val_correct_risk / max(val_total_risk, 1)
        val_score_mae = float(np.mean(val_score_diffs)) if val_score_diffs else 0.0

        gen_gap = round(avg_val_loss - avg_train_loss, 4)
        overfitting_flag = "STABLE" if gen_gap < 1.0 else ("MODERATE_GAP" if gen_gap < 3.0 else "OVERFITTING")

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"GradNorm: {avg_grad_norm:.3f} | "
            f"BinAcc: {val_bin_acc:.4f} | "
            f"ChoiceAcc: {val_choice_acc:.4f} | "
            f"PrioAcc: {val_prio_acc:.4f} | "
            f"Diag: {overfitting_flag}",
            flush=True,
        )

        history_entry = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "generalization_gap": gen_gap,
            "overfitting_diagnostic": overfitting_flag,
            "average_grad_norm": round(avg_grad_norm, 4),
            "validation_metrics": {
                "binary_accuracy": round(val_bin_acc, 4),
                "choice_top1_accuracy": round(val_choice_acc, 4),
                "priority_accuracy": round(val_prio_acc, 4),
                "risk_accuracy": round(val_risk_acc, 4),
                "score_mae": round(val_score_mae, 4),
            },
            "individual_task_losses": {k: round(v, 4) for k, v in epoch_task_losses.items()},
        }
        history.append(history_entry)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"--> Val loss improved to {best_val_loss:.4f}. Saving checkpoint to {checkpoint_dir}", flush=True)
            model.save_pretrained(checkpoint_dir)
            release_dir = "experiments/release/cdm-1-general"
            os.makedirs(release_dir, exist_ok=True)
            model.save_pretrained(release_dir)

        if early_stopping(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}.", flush=True)
            break

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f} seconds. Best Val Loss: {best_val_loss:.4f}", flush=True)

    # Prove all custom trainable parameters actually updated during training
    final_proj_weight = model.language_encoder.projection[0].weight.clone().detach()
    proj_diff = torch.norm(final_proj_weight - initial_proj_weight).item()
    print(f"Language Projection Weight Delta L2 Norm: {proj_diff:.6f} (>0 verifies parameter learning)", flush=True)
    assert proj_diff > 1e-5, "Language encoder projection parameters failed to update during training!"

    final_scorer_weight = model.candidate_scorer.scorer_mlp[0].weight.clone().detach()
    weight_diff = torch.norm(final_scorer_weight - initial_scorer_weight).item()
    print(f"Candidate Scorer Weight Delta L2 Norm: {weight_diff:.6f} (>0 verifies parameter learning)", flush=True)
    assert weight_diff > 1e-5, "Candidate scorer parameters failed to update during training!"

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "CDM-1",
            "parameters": param_stats,
            "language_projection_delta_norm": round(proj_diff, 6),
            "candidate_scorer_parameter_delta_norm": round(weight_diff, 6),
            "history": history,
            "best_val_loss": round(best_val_loss, 4),
            "duration_seconds": round(elapsed, 2),
        }, f, indent=2)

    return model


if __name__ == "__main__":
    train_multitask()
