"""Multi-Task Loss & Gradient Report Generator.
Computes:
1. Head-specific gradient norms across all trainable modules.
2. Loss-weight sensitivity across 4 different weighting schemes.
3. Multi-task head ablation study (isolated impact of candidate scorer, predictive network, decision network).
4. Saves verified output to research/multitask_gradient_report.json.
"""

import os
import sys
import json
import time
import numpy as np
from collections import defaultdict

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, os.path.abspath("."))

import torch
from torch.utils.data import DataLoader
from cdm.models.cdm_model import CDMModel
from cdm.data.dataset_loader import DecisionDataset, collate_decision_batch
from cdm.data.curated_corpus import load_dataset_from_disk
from cdm.data.preprocessor import DecisionPreprocessor
from research.training.train_multitask import (
    cache_multitask_batches,
    run_loss_weight_sensitivity,
    run_head_ablation_study,
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Multi-Task Gradient & Ablation Report Generation on {device} ===")

    val_data = load_dataset_from_disk("datasets/val.jsonl")
    preprocessor = DecisionPreprocessor(local_files_only=True)
    val_ds = DecisionDataset(val_data, preprocessor)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_decision_batch)

    model = CDMModel.from_pretrained("experiments/release/cdm-1-general").to(device)
    model.language_encoder.freeze_backbone()

    print("Pre-computing cached representations...")
    cached_val = cache_multitask_batches(model, val_loader, device)

    # 1. Compute head-specific gradient norms on a validation batch with backward pass
    print("Computing per-head gradient norms...")
    model.train()
    batch = cached_val[0]
    text_emb = model.language_encoder.projection(batch["pooled_text"])
    cand_pooled = batch["candidate_pooled"]
    cand_raw = model.language_encoder.projection(cand_pooled) if cand_pooled is not None else None
    struct_emb = model.structured_encoder(batch["numeric_values"], batch["missing_indicators"], batch["categorical_indices"])
    cog_state = model.fusion(text_emb, struct_emb)
    cand_embs = model.candidate_text_proj(cand_raw) if cand_raw is not None else None
    pred_out = model.predictive_network(cog_state, candidate_embeddings=cand_embs)
    dec_out = model.decision_network(cog_state)
    cand_out = None
    if cand_embs is not None:
        cand_outcomes = torch.sigmoid(pred_out["candidate_binary_outcome_logits"]) if "candidate_binary_outcome_logits" in pred_out else None
        cand_out = model.candidate_scorer(cog_state, cand_embs, cand_outcomes, pred_out.get("candidate_estimated_risk"), batch["constraint_mask"], batch["existence_mask"])

    outputs = {"cognitive_state": cog_state, "decision": dec_out, "predictive": pred_out, "candidates": cand_out}
    loss, loss_dict = model.compute_multitask_loss(outputs, batch["targets"])
    loss.backward()

    head_submodules = [
        ("language_projection", model.language_encoder.projection),
        ("candidate_text_proj", model.candidate_text_proj),
        ("structured_encoder", model.structured_encoder),
        ("fusion", model.fusion),
        ("predictive_network", model.predictive_network),
        ("decision_network", model.decision_network),
        ("candidate_scorer", model.candidate_scorer),
    ]

    head_gradients = {}
    for name, submod in head_submodules:
        grads = [p.grad.detach() for p in submod.parameters() if p.grad is not None]
        if grads:
            h_norm = torch.norm(torch.stack([torch.norm(g) for g in grads])).item()
        else:
            h_norm = 0.0
        head_gradients[name] = round(h_norm, 6)

    print("Head gradients:", head_gradients)

    # 2. Run loss weight sensitivity
    print("Running loss-weight sensitivity experiments...")
    sensitivity_results = run_loss_weight_sensitivity(model, cached_val, device)

    # 3. Run ablation study
    print("Running head ablation study...")
    ablation_results = run_head_ablation_study(model, cached_val, device)

    report = {
        "report_metadata": {
            "checkpoint": "experiments/release/cdm-1-general",
            "device": str(device),
            "timestamp": time.time(),
        },
        "head_gradient_norms": head_gradients,
        "all_heads_active_gradients": all(v > 1e-6 for v in head_gradients.values()),
        "loss_weight_sensitivity": sensitivity_results,
        "head_ablation_study": ablation_results,
    }

    out_path = "research/multitask_gradient_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    main()
