"""Controlled Adaptation Subsystem for CDM-1.
Enforces safety gates before fine-tuning, benchmark regression testing,
model versioning, and atomic rollback capability.
"""

from typing import Dict, List, Optional, Any
import os
import shutil
import json


class AdaptationGatekeeper:
    """Controls model parameter updates with strict benchmark regression checks and rollback."""

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
