"""Tests for persistent cognitive memory storage, tenant isolation, and policy constraint adherence."""

import os
import tempfile
import time
import pytest
from cdm.memory.storage import SQLiteMemoryStorage
from cdm.memory.episodic import EpisodicMemory, MemoryRecord
from cdm.memory.semantic import SemanticMemory, PolicyRule


def test_sqlite_persistence_and_similarity():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_memory.db")
        storage = SQLiteMemoryStorage(db_path=db_path)
        mem = EpisodicMemory(storage=storage)

        rec1 = MemoryRecord(
            record_id="rec_01",
            tenant_id="tenant_alpha",
            context_text="Server high memory alert",
            decision_made={"selected_candidate": "act_escalate_tier2"},
            vector=[1.0, 0.0, 0.0],
        )
        rec2 = MemoryRecord(
            record_id="rec_02",
            tenant_id="tenant_alpha",
            context_text="Refund request under 30 days",
            decision_made={"selected_candidate": "act_issue_refund"},
            vector=[0.0, 1.0, 0.0],
        )
        mem.insert(rec1)
        mem.insert(rec2)

        # Retrieve for alpha
        results = mem.retrieve_similar(query_vector=[0.9, 0.1, 0.0], tenant_id="tenant_alpha", top_k=1)
        assert len(results) == 1
        assert results[0]["record_id"] == "rec_01"


def test_tenant_isolation_in_memory():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_iso.db")
        storage = SQLiteMemoryStorage(db_path=db_path)
        mem = EpisodicMemory(storage=storage)

        # Insert record for tenant A
        rec_a = MemoryRecord(
            record_id="rec_tenant_a",
            tenant_id="tenant_a",
            context_text="Confidential enterprise contract renewal",
            decision_made={"selected_candidate": "act_request_info"},
            vector=[0.5, 0.5, 0.0],
        )
        mem.insert(rec_a)

        # Query from tenant B: must return empty (tenant isolation)
        results_b = mem.retrieve_similar(query_vector=[0.5, 0.5, 0.0], tenant_id="tenant_b")
        assert len(results_b) == 0, "Tenant B must NOT access Tenant A memories!"


def test_memory_retrieval_does_not_bypass_policy_constraints():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_policy.db")
        storage = SQLiteMemoryStorage(db_path=db_path)
        mem = EpisodicMemory(storage=storage)

        # Past interaction resulted in cash refund
        rec = MemoryRecord(
            record_id="rec_refund_past",
            tenant_id="tenant_retail",
            context_text="Refund requested for late package",
            decision_made={"selected_candidate": "act_issue_refund"},
            vector=[0.8, 0.2, 0.0],
        )
        mem.insert(rec)

        # Query with prohibited actions = ["act_issue_refund"]
        results = mem.retrieve_similar(
            query_vector=[0.8, 0.2, 0.0],
            tenant_id="tenant_retail",
            prohibited_actions=["act_issue_refund"],
        )
        # Must be suppressed because act_issue_refund is prohibited by current policy
        assert len(results) == 0, "Prohibited actions must not be surfaced by episodic retrieval!"
