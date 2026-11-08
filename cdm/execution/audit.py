"""Layer 4: Tamper-Evident Signed Provenance Audit Trail for CDM-1 Agent Runs.
Persists every proposal, gate decision, execution result, and verification result
as a cryptographically signed provenance record.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import time
import json
import uuid
import hashlib
from pydantic import BaseModel, Field

from cdm.inference.provenance import PredictionProvenance, _load_runtime_secrets


class AgentStepAuditRecord(BaseModel):
    run_id: str
    step_number: int
    tenant_id: str
    timestamp: float
    proposed_action: Dict[str, Any]
    gate_decision: Dict[str, Any]
    execution_result: Optional[Dict[str, Any]] = None
    verification_result: Optional[Dict[str, Any]] = None
    provenance_record: Dict[str, Any]


class AuditTrailManager:
    """Manages appending and verifying signed audit trails."""

    def __init__(self, audit_dir: str = "experiments/agent_runs"):
        self.audit_dir = audit_dir
        os.makedirs(self.audit_dir, exist_ok=True)

    def _get_run_file(self, run_id: str) -> str:
        return os.path.join(self.audit_dir, f"run_{run_id}.jsonl")

    def record_step(
        self,
        run_id: str,
        step_number: int,
        tenant_id: str,
        proposed_action: Dict[str, Any],
        gate_decision: Dict[str, Any],
        execution_result: Optional[Dict[str, Any]] = None,
        verification_result: Optional[Dict[str, Any]] = None,
    ) -> AgentStepAuditRecord:
        """Creates a signed provenance record for the step and persists to disk."""
        now = time.time()
        step_digest_payload = {
            "run_id": run_id,
            "step_number": step_number,
            "action": proposed_action,
            "gate": gate_decision,
            "execution": execution_result,
            "verification": verification_result,
        }

        # Generate cryptographic provenance signature
        prov = PredictionProvenance.create(
            model_id="CDM-1-Control-Loop",
            model_version="1.0.0",
            artifact_hash="agent_loop_provenance",
            execution_id=f"{run_id}_step_{step_number}",
            tenant_id=tenant_id,
            task="agent_control_step",
            prediction_output=step_digest_payload,
            estimated_risk=gate_decision.get("risk_score", 0.0),
        )

        record = AgentStepAuditRecord(
            run_id=run_id,
            step_number=step_number,
            tenant_id=tenant_id,
            timestamp=now,
            proposed_action=proposed_action,
            gate_decision=gate_decision,
            execution_result=execution_result,
            verification_result=verification_result,
            provenance_record=prov,
        )

        # Append to run audit log file
        run_path = self._get_run_file(run_id)
        with open(run_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.model_dump()) + "\n")

        return record

    def get_run_history(self, run_id: str) -> List[AgentStepAuditRecord]:
        """Loads and returns all step records for a run."""
        run_path = self._get_run_file(run_id)
        if not os.path.isfile(run_path):
            return []

        records = []
        with open(run_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(AgentStepAuditRecord.model_validate(json.loads(line)))
        return records

    def verify_run_audit(self, run_id: str) -> Tuple[bool, List[str]]:
        """Cryptographically verifies every step in a run's audit trail against its signature."""
        records = self.get_run_history(run_id)
        if not records:
            return False, ["Run audit log does not exist or is empty."]

        issues = []
        for rec in records:
            step_digest_payload = {
                "run_id": rec.run_id,
                "step_number": rec.step_number,
                "action": rec.proposed_action,
                "gate": rec.gate_decision,
                "execution": rec.execution_result,
                "verification": rec.verification_result,
            }
            valid, source, details = PredictionProvenance.validate(
                provenance_dict=rec.provenance_record,
                expected_artifact_hash="agent_loop_provenance",
                expected_tenant_id=rec.tenant_id,
                expected_task="agent_control_step",
                prediction_output=step_digest_payload,
            )
            if not valid:
                issues.append(f"Step {rec.step_number} signature verification failed: {details.get('reason')}")

        return (len(issues) == 0), issues
