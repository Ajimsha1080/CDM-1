"""Test for Provenance HMAC persistence across simulated process restarts and key rotation."""

import os
import tempfile
import pytest
from cdm.inference.provenance import (
    PredictionProvenance,
    _load_runtime_secrets,
    reload_runtime_secrets,
)


@pytest.fixture(autouse=True)
def clean_provenance_env():
    old_env = {
        k: os.environ.get(k)
        for k in ["CDM_PROVENANCE_SECRET", "CDM_PROVENANCE_KEY_FILE", "CDM_PROVENANCE_KEY_ID", "CDM_PROVENANCE_KEY_RING"]
    }
    yield
    for k, v in old_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    reload_runtime_secrets()


def test_provenance_persists_across_restart_with_env_secret():
    """Verify that provenance records remain valid across simulated process restarts using CDM_PROVENANCE_SECRET."""
    test_secret = "test_provenance_hmac_secret_32bytes_long!!"
    os.environ["CDM_PROVENANCE_SECRET"] = test_secret
    os.environ["CDM_PROVENANCE_KEY_ID"] = "prov-key-v1"
    reload_runtime_secrets()

    # Process 1: Create provenance record
    pred_data = {"decision": "YES", "risk": 0.12, "priority": "LOW"}
    record = PredictionProvenance.create(
        model_id="CDM-1-General",
        model_version="1.0.0",
        artifact_hash="9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a",
        execution_id="exec_restart_001",
        tenant_id="tenant_finance",
        task="binary",
        prediction_output=pred_data,
        estimated_risk=0.12,
    )
    assert record["provenance_signature"]
    assert record["key_id"] == "prov-key-v1"

    # Simulate Process Restart: reload runtime secrets in new state
    reload_runtime_secrets()

    # Process 2: Validate the provenance record created by Process 1
    valid, source, details = PredictionProvenance.validate(
        record,
        expected_artifact_hash="9fab839b607312d4697cbc4af95993832f793c7b5a3640f5dc7579e29319495a",
        expected_tenant_id="tenant_finance",
        expected_task="binary",
        prediction_output=pred_data,
    )
    assert valid is True
    assert source == "trained_model_prediction"
    assert details["key_id"] == "prov-key-v1"


def test_provenance_persists_across_restart_with_keyfile():
    """Verify that provenance records persist across restarts via persistent keyfile on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        key_file = os.path.join(tmpdir, "prov_secret.key")
        os.environ.pop("CDM_PROVENANCE_SECRET", None)
        os.environ["CDM_PROVENANCE_KEY_FILE"] = key_file
        os.environ["CDM_PROVENANCE_KEY_ID"] = "disk-key-v1"
        reload_runtime_secrets()

        # Secret file should have been automatically generated and persisted
        assert os.path.isfile(key_file)
        with open(key_file, "rb") as f:
            saved_secret = f.read()

        pred_data = {"decision": "ALLOW", "risk": 0.05}
        record = PredictionProvenance.create(
            model_id="CDM-1-General",
            model_version="1.0.0",
            artifact_hash="abc123hash",
            execution_id="exec_disk_001",
            tenant_id="tenant_ops",
            task="binary",
            prediction_output=pred_data,
        )

        # Simulate fresh process restart reading same keyfile
        reload_runtime_secrets()
        valid, _, _ = PredictionProvenance.validate(record, expected_artifact_hash="abc123hash", prediction_output=pred_data)
        assert valid is True


def test_provenance_key_rotation_support():
    """Verify that provenance validation supports rotated keys from the keyring."""
    # Setup key ring with old key v1 and current key v2
    key_v1 = "1111111111111111111111111111111111111111111111111111111111111111"
    key_v2 = "2222222222222222222222222222222222222222222222222222222222222222"

    import json
    os.environ["CDM_PROVENANCE_KEY_ID"] = "key-v2"
    os.environ["CDM_PROVENANCE_SECRET"] = "active_secret_for_v2_signing_here!!"
    os.environ["CDM_PROVENANCE_KEY_RING"] = json.dumps({
        "key-v1": key_v1,
        "key-v2": key_v2,
    })
    reload_runtime_secrets()

    # Create record using old key-v1
    pred_data = {"decision": "NO", "risk": 0.88}
    record_old = PredictionProvenance.create(
        model_id="CDM-1-General",
        model_version="1.0.0",
        artifact_hash="hash999",
        task="binary",
        prediction_output=pred_data,
        key_id="key-v1",
    )
    assert record_old["key_id"] == "key-v1"

    # Validation should succeed for old key through keyring rotation
    valid, _, _ = PredictionProvenance.validate(record_old, expected_artifact_hash="hash999", prediction_output=pred_data)
    assert valid is True
