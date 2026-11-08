"""Unit tests for Layer 1 LLM Provider, strict schema validation, JSON repair, and tenant accounting."""

import pytest
from pydantic import BaseModel, Field
from cdm.llm import (
    LLMMessage,
    LLMProviderConfig,
    MockLLMProvider,
    validate_json_against_schema,
    validate_and_repair_json,
    get_llm_provider,
)


class ProposedActionSchema(BaseModel):
    action: str = Field(..., description="Action name")
    arguments: dict = Field(default_factory=dict, description="Action arguments")
    reasoning: str = Field(..., description="Action justification")


def test_llm_provider_valid_schema():
    """Verify that valid structured JSON parses and validates against Pydantic schema."""
    raw = '```json\n{"action": "query_database", "arguments": {"user_id": 42}, "reasoning": "Need account history"}\n```'
    is_valid, parsed, err = validate_json_against_schema(raw, ProposedActionSchema)
    assert is_valid is True
    assert parsed["action"] == "query_database"
    assert parsed["arguments"]["user_id"] == 42
    assert err is None


def test_llm_provider_schema_rejection_on_invalid():
    """Verify that invalid schema or missing required fields are rejected."""
    raw = '{"action": "query_database"}' # Missing required 'reasoning'
    is_valid, parsed, err = validate_json_against_schema(raw, ProposedActionSchema)
    assert is_valid is False
    assert err is not None


def test_llm_provider_json_repair_success():
    """Verify that malformed JSON triggers exactly one repair prompt which succeeds."""
    repair_call_count = 0

    def mock_repair_call(messages):
        nonlocal repair_call_count
        repair_call_count += 1
        repaired_json = '{"action": "view_ticket", "arguments": {"ticket_id": "TICK-101"}, "reasoning": "Repaired JSON successfully"}'
        return repaired_json, 150, 45

    malformed_raw = 'Here is the action: { action: view_ticket, ticket_id: TICK-101 (unquoted json) }'
    parsed, is_valid, err, was_repaired, r_in, r_out = validate_and_repair_json(
        raw_text=malformed_raw,
        schema=ProposedActionSchema,
        messages=[LLMMessage(role="user", content="What should we do next?")],
        call_fn=mock_repair_call,
    )
    assert repair_call_count == 1
    assert was_repaired is True
    assert is_valid is True
    assert parsed["action"] == "view_ticket"
    assert err is None


def test_llm_provider_json_repair_failure_fails_cleanly():
    """Verify that if repair output is still broken, it fails cleanly without raising into the loop."""
    def mock_broken_repair_call(messages):
        return "still totally broken text without any json", 100, 20

    broken_raw = "broken text"
    parsed, is_valid, err, was_repaired, r_in, r_out = validate_and_repair_json(
        raw_text=broken_raw,
        schema=ProposedActionSchema,
        messages=[],
        call_fn=mock_broken_repair_call,
    )
    assert was_repaired is True
    assert is_valid is False
    assert "Output unrepairable" in err


def test_tenant_token_and_cost_accounting():
    """Verify that provider tracks input tokens, output tokens, and dollar cost per tenant."""
    provider = MockLLMProvider()
    
    resp1 = provider.complete(
        messages=[LLMMessage(role="user", content="Hello tenant A")],
        tenant_id="tenant_alpha",
    )
    assert resp1.tenant_id == "tenant_alpha"
    assert resp1.cost_usd >= 0.0

    resp2 = provider.complete(
        messages=[LLMMessage(role="user", content="Hello tenant B with longer query text")],
        tenant_id="tenant_beta",
    )
    assert resp2.tenant_id == "tenant_beta"

    usage_a = provider.get_tenant_usage("tenant_alpha")
    usage_b = provider.get_tenant_usage("tenant_beta")
    assert usage_a["request_count"] == 1
    assert usage_b["request_count"] == 1
    assert usage_a["input_tokens"] > 0
    assert usage_b["input_tokens"] > 0


def test_llm_provider_factory_fallback():
    """Verify that factory falls back gracefully when API keys are missing."""
    import os
    os.environ.pop("ANTHROPIC_API_KEY", None)
    cfg = LLMProviderConfig(provider_type="anthropic", api_key=None)
    provider = get_llm_provider(cfg)
    assert isinstance(provider, MockLLMProvider)
