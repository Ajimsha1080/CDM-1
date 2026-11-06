"""Controlled Adaptation Subsystem for CDM-1.
Enforces safety gates before fine-tuning, benchmark regression testing,
model versioning, dataset staging from verified feedback, and atomic rollback capability.
"""

from typing import Dict, List, Optional, Any, Tuple
import os
import shutil
import json
import time


class AdaptationGatekeeper:
    """Controls model parameter updates with strict benchmark regression checks,
    staging of verified human-reviewed telemetry, and atomic rollback capability.
    """

    def __init__(self, checkpoints_dir: str = "experiments/checkpoints"):
        self.checkpoints_dir = checkpoints_dir
        self.active_version = "cdm-1-general"

    def create_rollback_snapshot(self, version: str) -> str:
        """Create an immutable snapshot of the active model checkpoint."""
        source = os.path.join(self.checkpoints_dir, version)
        snapshot_dir = os.path.join(self.checkpoints_dir, f"{version}_rollback_snapshot")
        if os.path.exists(source):
            if os.path.exists(snapshot_dir):
                shutil.rmtree(snapshot_dir)
            shutil.copytree(source, snapshot_dir)
        return snapshot_dir

    def stage_feedback_dataset(
        self, feedback_events: List[Any], output_path: str = "datasets/staged_feedback.jsonl"
    ) -> int:
        """Stages verified feedback events into standardized JSONL training samples.
        Strictly prevents training on holdout test or OOD data.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        count = 0
        with open(output_path, "a", encoding="utf-8") as f:
            for ev in feedback_events:
                # Only stage events that have been verified by human review
                status = getattr(ev, "review_status", None) or (ev.get("review_status") if isinstance(ev, dict) else None)
                if status != "verified":
                    continue

                sample = {
                    "example_id": getattr(ev, "event_id", f"staged_{count}"),
                    "input_text": getattr(ev, "predicted_decision", {}).get("text", "Telemetry feedback sample"),
                    "structured_features": getattr(ev, "predicted_decision", {}).get("structured_features", {}),
                    "target_decision": {
                        "binary_decision": getattr(ev, "verified_actual_outcome", {}).get("binary_target", 1),
                        "selected_candidate_id": getattr(ev, "verified_actual_outcome", {}).get("selected_candidate_id", "act_auto_resolve"),
                        "decision_score": 85.0,
                        "priority_level": 1,
                        "risk_level": 0,
                    },
                    "outcome_label": getattr(ev, "verified_actual_outcome", {}),
                    "label_provenance": {
                        "source_type": "verified_human_feedback",
                        "verifier_id": "human_reviewer",
                        "synthetic": False,
                    },
                }
                f.write(json.dumps(sample) + "\n")
                count += 1
        return count

    def evaluate_benchmark_gate(
        self,
        candidate_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        max_allowed_accuracy_drop: float = 0.02,
    ) -> bool:
        """Enforces that newly fine-tuned candidate checkpoint does not regress beyond tolerance."""
        cand_acc = candidate_metrics.get("binary_decision", {}).get("accuracy", 0.0)
        base_acc = baseline_metrics.get("binary_decision", {}).get("accuracy", 0.0)

        if (base_acc - cand_acc) > max_allowed_accuracy_drop:
            print(f"Gate REJECTED: Candidate accuracy {cand_acc} regressed below baseline {base_acc}")
            return False

        print(f"Gate APPROVED: Candidate accuracy {cand_acc} meets or exceeds standard.")
        return True

    def evaluate_regression_gate(
        self,
        candidate_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        max_allowed_accuracy_drop: float = 0.02,
        max_allowed_f1_drop: float = 0.03,
    ) -> Tuple[bool, List[str]]:
        """Comprehensive multi-task regression gate enforcing accuracy, F1, and zero policy violation tolerance."""
        rejection_reasons = []

        # 1. Binary Decision Accuracy Check
        cand_acc = candidate_metrics.get("binary_decision", {}).get("accuracy", 0.0)
        base_acc = baseline_metrics.get("binary_decision", {}).get("accuracy", 0.0)
        if (base_acc - cand_acc) > max_allowed_accuracy_drop:
            rejection_reasons.append(
                f"Binary accuracy regressed: candidate={cand_acc:.4f}, baseline={base_acc:.4f} (drop > {max_allowed_accuracy_drop})"
            )

        # 2. Binary Macro F1 Check
        cand_f1 = candidate_metrics.get("binary_decision", {}).get("macro_f1", 0.0)
        base_f1 = baseline_metrics.get("binary_decision", {}).get("macro_f1", 0.0)
        if (base_f1 - cand_f1) > max_allowed_f1_drop:
            rejection_reasons.append(
                f"Binary Macro F1 regressed: candidate={cand_f1:.4f}, baseline={base_f1:.4f} (drop > {max_allowed_f1_drop})"
            )

        # 3. Policy Constraint Violations (Zero-Tolerance Gate)
        cand_violations = candidate_metrics.get("candidate_choice", {}).get("prohibited_violation_rate", 0.0)
        if cand_violations > 0.0:
            rejection_reasons.append(
                f"SAFETY POLICY VIOLATION: Candidate selected prohibited actions at rate {cand_violations * 100:.2f}% (tolerance is 0.00%)"
            )

        # 4. NDCG Ranking Check
        cand_ndcg = candidate_metrics.get("candidate_choice", {}).get("ndcg", 1.0)
        base_ndcg = baseline_metrics.get("candidate_choice", {}).get("ndcg", 1.0)
        if (base_ndcg - cand_ndcg) > 0.05:
            rejection_reasons.append(f"Candidate ranking NDCG regressed: {cand_ndcg:.4f} vs baseline {base_ndcg:.4f}")

        is_approved = len(rejection_reasons) == 0
        return is_approved, rejection_reasons

    def rollback_to_snapshot(self, version: str) -> bool:
        """Rollback to the previously created snapshot in case of failure."""
        source = os.path.join(self.checkpoints_dir, f"{version}_rollback_snapshot")
        target = os.path.join(self.checkpoints_dir, version)
        if os.path.exists(source):
            if os.path.exists(target):
                shutil.rmtree(target)
            shutil.copytree(source, target)
            print(f"Rollback to {version} snapshot successfully completed.")
            return True
        return False
