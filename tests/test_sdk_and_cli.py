"""Tests for CDM-1 Python SDK (v1.2.0 contract) and Unified CLI."""

import subprocess
import sys
import pytest
from sdk.cdm_sdk import CDM, CognitiveEngineResponse, CDMResponse


def test_cli_version():
    """Verify 'cdm version' prints architecture details."""
    res = subprocess.run([sys.executable, "-m", "cdm.cli", "version"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "CDM-1 Cognitive Decision Architecture" in res.stdout
    assert "ModernBERT-base (158M parameters)" in res.stdout


def test_cli_verify():
    """Verify 'cdm verify' executes Ed25519 signature checks."""
    res = subprocess.run([sys.executable, "-m", "cdm.cli", "verify"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Checkpoint Authenticated Successfully" in res.stdout
    assert "9fab839b607312d4" in res.stdout


def test_sdk_response_schema_v12():
    """Verify CognitiveEngineResponse validates v1.2.0 payload correctly."""
    payload = {
        "schema_version": "v1.2.0",
        "answer": "The evaluated decision is 'act_reboot'.",
        "decision": "act_reboot",
        "reasoning": {"summary": "Safe to reboot worker node."},
        "confidence": 0.94,
        "risk_assessment": {"assessed_risk": 0.05, "priority": "LOW"},
        "safety_status": "verified",
        "abstention_reason": None,
        "required_approval": {"requires_human_approval": False},
        "evidence": {"citations": [], "retrieved_passages": []},
        "execution_status": "authorized",
        "cognitive_trace": [{"stage_name": "Perception", "execution_status": "success"}],
        "latency": {"latency_ms": 14.2},
    }
    resp = CognitiveEngineResponse.model_validate(payload)
    assert resp.schema_version == "v1.2.0"
    assert resp.answer == "The evaluated decision is 'act_reboot'."
    assert resp.decision == "act_reboot"
    assert resp.safety_status == "verified"
    assert resp.execution_status == "authorized"
    assert resp.confidence == 0.94
    assert resp.latency["latency_ms"] == 14.2


def test_sdk_local_process():
    """Verify SDK local process populates v1.2.0 response fields."""
    client = CDM(mode="local")
    res = client.process(
        text="Cluster memory alert in staging node",
        task="choice",
        candidates=["act_restart_pod", "act_ignore"],
        session_id="test_sess_sdk",
    )
    assert isinstance(res, CognitiveEngineResponse)
    assert res.schema_version == "v1.2.0"
    assert len(res.answer) > 0
    assert res.decision in ["act_restart_pod", "act_ignore", "ABSTAIN"]
    assert res.safety_status in ["verified", "needs_approval", "abstained"]
    assert "latency_ms" in res.latency
