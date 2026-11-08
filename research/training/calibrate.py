"""STAGE 8: Probability Calibration and Abstention Threshold Tuning for CDM-1.
Implements Temperature Scaling to minimize Negative Log-Likelihood (NLL) and Expected
Calibration Error (ECE), and sweeps abstention thresholds for selective classification.
"""

import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cdm.models.cdm_model import CDMModel
from cdm.data.dataset_loader import DecisionDataset, collate_decision_batch
from cdm.data.curated_corpus import load_dataset_from_disk
from cdm.data.preprocessor import DecisionPreprocessor


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) across probability bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return float(ece)


def calibrate_model(
    checkpoint_dir: str = "experiments/checkpoints/cdm-1-general",
    output_report: str = "research/calibration_report.json",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STAGE 8: Calibrating Model Probabilities & Abstention on {device} ===")

    val_data = load_dataset_from_disk("datasets/val.jsonl")
    preprocessor = DecisionPreprocessor()
    val_ds = DecisionDataset(val_data, preprocessor)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_decision_batch)

    model = CDMModel.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            num_vals = batch["numeric_values"].to(device)
            missing = batch["missing_indicators"].to(device)
            cat_idx = {k: v.to(device) for k, v in batch["categorical_indices"].items()}
            targets = batch["targets"]["binary_target"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numeric_values=num_vals,
                missing_indicators=missing,
                categorical_indices=cat_idx,
            )
            all_logits.append(out["decision"]["binary_logits"])
            all_targets.append(targets)

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Initial uncalibrated metrics
    uncal_probs = F.softmax(logits, dim=-1).cpu().numpy()
    targets_np = targets.cpu().numpy()
    uncal_ece = compute_ece(uncal_probs, targets_np)

    # Optimize Temperature parameter T via L-BFGS or Adam on NLL
    temp_param = nn.Parameter(torch.ones(1, device=device) * 1.5)
    optimizer = torch.optim.LBFGS([temp_param], lr=0.01, max_iter=50)

    def eval_nll():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / torch.clamp(temp_param, min=0.1, max=10.0), targets)
        loss.backward()
        return loss

    optimizer.step(eval_nll)
    optimal_temp = float(temp_param.detach().cpu().item())

    # Calibrated probabilities
    cal_logits = (logits / optimal_temp).cpu()
    cal_probs = F.softmax(cal_logits, dim=-1).numpy()
    cal_ece = compute_ece(cal_probs, targets_np)

    print(f"Uncalibrated ECE: {uncal_ece:.4f}")
    print(f"Optimal Temperature: {optimal_temp:.4f}")
    print(f"Calibrated ECE:   {cal_ece:.4f}")

    # Abstention Analysis: sweep confidence threshold tau in [0.5, 0.95]
    abstention_sweeps = []
    max_confs = np.max(cal_probs, axis=1)
    preds = np.argmax(cal_probs, axis=1)
    is_correct = (preds == targets_np).astype(float)

    best_thresh = 0.60
    for tau in np.arange(0.50, 0.95, 0.05):
        accepted = max_confs >= tau
        coverage = float(np.mean(accepted))
        selective_acc = float(np.mean(is_correct[accepted])) if coverage > 0 else 1.0
        abstention_sweeps.append({
            "threshold": round(float(tau), 2),
            "coverage": round(coverage, 3),
            "selective_accuracy": round(selective_acc, 3),
            "selective_risk": round(1.0 - selective_acc, 3),
        })
        if selective_acc >= 0.95 and coverage >= 0.70:
            best_thresh = float(tau)

    # Update model temperature parameter and save
    model.decision_network.temperature.data = torch.tensor([optimal_temp], device=device)
    model.config.decision.abstention_threshold = best_thresh
    model.save_pretrained(checkpoint_dir)

    report = {
        "temperature": optimal_temp,
        "uncalibrated_ece": uncal_ece,
        "calibrated_ece": cal_ece,
        "optimal_abstention_threshold": best_thresh,
        "abstention_sweeps": abstention_sweeps,
    }

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Calibration completed and saved to {output_report}")
    return report


if __name__ == "__main__":
    calibrate_model()
