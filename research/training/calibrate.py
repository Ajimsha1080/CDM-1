"""STAGE 8: Rigorous Probability Calibration and Abstention Threshold Tuning for CDM-1.
Separates validation calibration from final held-out test evaluation.
Tuning of temperature and selection of abstention threshold occurs strictly on calibration data.
Computes Expected Calibration Error (ECE), Brier Score, Negative Log-Likelihood (NLL),
coverage, selective risk, and reliability diagrams with binomial confidence intervals.
"""

import os
import sys
import json
import math
import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from cdm.models.cdm_model import CDMModel
from cdm.data.dataset_loader import DecisionDataset, collate_decision_batch
from cdm.data.curated_corpus import load_dataset_from_disk
from cdm.data.preprocessor import DecisionPreprocessor


def compute_reliability_diagram(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict:
    """Computes Expected Calibration Error (ECE), bin statistics, and 95% Wilson confidence intervals."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    ece = 0.0
    n = len(labels)
    bins_data = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > low) & (confidences <= high) if i > 0 else (confidences >= low) & (confidences <= high)
        count = int(np.sum(in_bin))
        prop_in_bin = count / max(n, 1)

        if count > 0:
            acc_in_bin = float(np.mean(accuracies[in_bin]))
            conf_in_bin = float(np.mean(confidences[in_bin]))
            ece += abs(acc_in_bin - conf_in_bin) * prop_in_bin

            # 95% Wilson score interval for binomial accuracy
            z = 1.96
            denom = 1.0 + (z**2) / count
            center = (acc_in_bin + (z**2) / (2 * count)) / denom
            half_width = (z / denom) * math.sqrt((acc_in_bin * (1.0 - acc_in_bin) / count) + (z**2) / (4 * (count**2)))
            ci_low = max(0.0, center - half_width)
            ci_high = min(1.0, center + half_width)
        else:
            acc_in_bin = 0.0
            conf_in_bin = (low + high) / 2.0
            ci_low, ci_high = 0.0, 0.0

        bins_data.append({
            "bin_index": i,
            "bin_range": [round(float(low), 2), round(float(high), 2)],
            "sample_count": count,
            "mean_confidence": round(float(conf_in_bin), 4),
            "empirical_accuracy": round(float(acc_in_bin), 4),
            "confidence_interval_95": [round(float(ci_low), 4), round(float(ci_high), 4)],
        })

    return {
        "ece": float(ece),
        "reliability_bins": bins_data,
    }


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted class probabilities and one-hot true targets."""
    num_classes = probs.shape[1]
    one_hot = np.eye(num_classes)[labels]
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(brier)


def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Negative Log-Likelihood (NLL)."""
    eps = 1e-9
    clipped = np.clip(probs, eps, 1.0 - eps)
    n = len(labels)
    log_probs = np.log(clipped[np.arange(n), labels])
    return float(-np.mean(log_probs))


def collect_model_predictions(model: CDMModel, data_loader: DataLoader, device: torch.device):
    """Collects raw decision logits, targets, and metadata from data loader."""
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
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
    return logits, targets


def sweep_abstention(probs: np.ndarray, labels: np.ndarray) -> tuple:
    """Sweeps confidence thresholds on calibration data to select optimal threshold."""
    max_confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    is_correct = (preds == labels).astype(float)

    sweeps = []
    best_thresh = 0.55
    for tau in np.arange(0.50, 0.95, 0.05):
        accepted = max_confs >= tau
        coverage = float(np.mean(accepted))
        selective_acc = float(np.mean(is_correct[accepted])) if coverage > 0 else 1.0
        selective_risk = float(1.0 - selective_acc)
        sweeps.append({
            "threshold": round(float(tau), 2),
            "coverage": round(coverage, 3),
            "selective_accuracy": round(selective_acc, 3),
            "selective_risk": round(selective_risk, 3),
        })
        if selective_acc >= 0.95 and coverage >= 0.70:
            best_thresh = float(tau)

    return best_thresh, sweeps


def evaluate_split_metrics(logits: torch.Tensor, targets: torch.Tensor, temperature: float, threshold: float) -> dict:
    """Computes comprehensive calibration and selective classification metrics for any dataset split."""
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1).cpu().numpy()
    labels = targets.cpu().numpy()

    rel_diag = compute_reliability_diagram(probs, labels)
    brier = compute_brier_score(probs, labels)
    nll = compute_nll(probs, labels)

    max_confs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    accepted = max_confs >= threshold
    coverage = float(np.mean(accepted))
    accuracy_overall = float(np.mean(preds == labels))
    selective_acc = float(np.mean((preds == labels)[accepted])) if coverage > 0 else 1.0
    selective_risk = float(1.0 - selective_acc)

    return {
        "sample_count": len(labels),
        "overall_accuracy": round(accuracy_overall, 4),
        "ece": round(rel_diag["ece"], 4),
        "brier_score": round(brier, 4),
        "nll": round(nll, 4),
        "abstention_threshold": round(threshold, 2),
        "coverage": round(coverage, 4),
        "selective_accuracy": round(selective_acc, 4),
        "selective_risk": round(selective_risk, 4),
        "reliability_diagram": rel_diag["reliability_bins"],
    }


def calibrate_and_evaluate(
    checkpoint_dir: str = "experiments/release/cdm-1-general",
    output_report: str = "research/calibration_report.json",
    save_updated_checkpoint: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== STAGE 8: Calibrating Model Probabilities & Evaluating on Unseen Data on {device} ===")

    preprocessor = DecisionPreprocessor(local_files_only=True)
    val_data = load_dataset_from_disk("datasets/val.jsonl")
    val_ds = DecisionDataset(val_data, preprocessor)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_decision_batch)

    test_data = load_dataset_from_disk("datasets/test.jsonl")
    test_ds = DecisionDataset(test_data, preprocessor)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_decision_batch)

    model = CDMModel.from_pretrained(checkpoint_dir).to(device)
    model.eval()

    # --- Phase 1: Collect calibration predictions (validation split only) ---
    val_logits, val_targets = collect_model_predictions(model, val_loader, device)

    # Uncalibrated baseline metrics on validation data
    uncal_val_metrics = evaluate_split_metrics(val_logits, val_targets, temperature=1.0, threshold=0.55)

    # --- Phase 2: Optimize Temperature T strictly on validation calibration data ---
    temp_param = nn.Parameter(torch.ones(1, device=device) * 1.5)
    optimizer = torch.optim.LBFGS([temp_param], lr=0.01, max_iter=50)

    def eval_nll():
        optimizer.zero_grad()
        loss = F.cross_entropy(val_logits / torch.clamp(temp_param, min=0.1, max=10.0), val_targets)
        loss.backward()
        return loss

    optimizer.step(eval_nll)
    optimal_temp = float(temp_param.detach().cpu().item())

    # --- Phase 3: Tune abstention threshold on calibrated validation data ---
    cal_val_probs = F.softmax(val_logits / optimal_temp, dim=-1).cpu().numpy()
    best_thresh, val_abstention_sweeps = sweep_abstention(cal_val_probs, val_targets.cpu().numpy())

    cal_val_metrics = evaluate_split_metrics(val_logits, val_targets, temperature=optimal_temp, threshold=best_thresh)

    print(f"Validation (Calib Set) - Uncalibrated ECE: {uncal_val_metrics['ece']:.4f} | Calibrated ECE: {cal_val_metrics['ece']:.4f}")
    print(f"Optimal Temperature: {optimal_temp:.4f} | Selected Threshold: {best_thresh:.2f}")

    # --- Phase 4: Evaluate frozen calibrated model on held-out test data ---
    test_logits, test_targets = collect_model_predictions(model, test_loader, device)
    uncal_test_metrics = evaluate_split_metrics(test_logits, test_targets, temperature=1.0, threshold=best_thresh)
    cal_test_metrics = evaluate_split_metrics(test_logits, test_targets, temperature=optimal_temp, threshold=best_thresh)

    print(f"Held-Out Test Set - Uncalibrated ECE: {uncal_test_metrics['ece']:.4f} | Calibrated ECE: {cal_test_metrics['ece']:.4f}")
    print(f"Held-Out Test Set - Brier Score: {cal_test_metrics['brier_score']:.4f} | NLL: {cal_test_metrics['nll']:.4f}")
    print(f"Held-Out Test Set - Coverage: {cal_test_metrics['coverage']:.2%} | Selective Risk: {cal_test_metrics['selective_risk']:.2%}")

    # --- Phase 5: Evaluate on external/OOD datasets if present ---
    external_evaluations = {}
    for ext_name, ext_path in [
        ("heldout_domain_test", "datasets/heldout_domain_test.jsonl"),
        ("ood_test", "datasets/ood_test.jsonl"),
    ]:
        try:
            ext_data = load_dataset_from_disk(ext_path)
            ext_ds = DecisionDataset(ext_data, preprocessor)
            ext_loader = DataLoader(ext_ds, batch_size=16, shuffle=False, collate_fn=collate_decision_batch)
            ext_logits, ext_targets = collect_model_predictions(model, ext_loader, device)
            external_evaluations[ext_name] = evaluate_split_metrics(
                ext_logits, ext_targets, temperature=optimal_temp, threshold=best_thresh
            )
        except Exception as e:
            external_evaluations[ext_name] = {"status": "skipped", "error": str(e)}

    report = {
        "calibration_provenance": {
            "checkpoint_evaluated": checkpoint_dir,
            "calibration_split": "datasets/val.jsonl",
            "held_out_test_split": "datasets/test.jsonl",
            "temperature_tuning_method": "NLL minimization via L-BFGS on validation logits",
            "threshold_selection_method": "Sweep on calibrated validation probabilities (tau >= 0.50)",
        },
        "optimal_temperature": round(optimal_temp, 4),
        "selected_abstention_threshold": round(best_thresh, 2),
        "validation_calibration_metrics": {
            "uncalibrated": uncal_val_metrics,
            "calibrated": cal_val_metrics,
            "threshold_sweeps": val_abstention_sweeps,
        },
        "held_out_test_metrics": {
            "uncalibrated": uncal_test_metrics,
            "calibrated": cal_test_metrics,
        },
        "external_out_of_distribution_evaluations": external_evaluations,
    }

    if save_updated_checkpoint:
        model.decision_network.temperature.data = torch.tensor([optimal_temp], device=device)
        model.config.decision.abstention_threshold = best_thresh
        model.save_pretrained(checkpoint_dir)

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Rigorous calibration report saved to {output_report}")
    return report


if __name__ == "__main__":
    calibrate_and_evaluate()
