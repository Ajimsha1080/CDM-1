"""Layer 1: Foundation Model Provider Interface for CDM-1 Agent Loop.
Defines provider abstractions, message formats, structured response envelopes,
and per-tenant token and cost accounting.
"""

from typing import List, Dict, Any, Optional, Union, Type, Tuple
from abc import ABC, abstractmethod
import time
import json
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger("cdm.llm")


class LLMMessage(BaseModel):
    role: str = Field(..., description="'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message text payload")


class LLMResponse(BaseModel):
    raw_text: str
    parsed_json: Optional[Dict[str, Any]] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model_name: str
    latency_ms: float = 0.0
    tenant_id: str = "default_tenant"
    repaired: bool = False
    validation_error: Optional[str] = None


class LLMProviderConfig(BaseModel):
    provider_type: str = Field(default="anthropic", description="'anthropic', 'openai_compatible', or 'mock'")
    model_name: str = Field(default="claude-3-5-sonnet-20241022", description="Model identifier")
    api_key: Optional[str] = Field(default=None, description="API secret key")
    api_base: Optional[str] = Field(default=None, description="Base URL for local/OpenAI compatible server")
    timeout_seconds: float = Field(default=30.0, description="Network timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry count for transient rate limits or 5xx errors")
    backoff_factor: float = Field(default=1.5, description="Exponential backoff multiplier")
    temperature: float = Field(default=0.0, description="Sampling temperature")


class BaseLLMProvider(ABC):
    """Abstract Foundation Model Provider interface with tenant metering and schema enforcement."""

    def __init__(self, config: Optional[LLMProviderConfig] = None):
        self.config = config or LLMProviderConfig()
        self._tenant_usage: Dict[str, Dict[str, Any]] = {}

    def record_usage(self, tenant_id: str, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        """Records token counts and accrued cost per tenant."""
        if tenant_id not in self._tenant_usage:
            self._tenant_usage[tenant_id] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "request_count": 0,
            }
        usage = self._tenant_usage[tenant_id]
        usage["input_tokens"] += input_tokens
        usage["output_tokens"] += output_tokens
        usage["total_tokens"] += (input_tokens + output_tokens)
        usage["cost_usd"] += cost_usd
        usage["request_count"] += 1

    def get_tenant_usage(self, tenant_id: str) -> Dict[str, Any]:
        """Returns cumulative usage statistics for a specific tenant."""
        return self._tenant_usage.get(
            tenant_id,
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0, "request_count": 0},
        )

    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Estimates cost based on standard pricing tables per 1k tokens."""
        # Standard pricing per million tokens
        pricing_per_million = {
            "claude-3-5-sonnet": (3.0, 15.0),
            "claude-3-haiku": (0.25, 1.25),
            "claude-3-opus": (15.0, 75.0),
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.60),
            "mock": (0.0, 0.0),
        }
        for prefix, (in_price, out_price) in pricing_per_million.items():
            if prefix in model_name.lower():
                return round((input_tokens * in_price + output_tokens * out_price) / 1_000_000.0, 6)
        # Default baseline ($1 / $3 per million)
        return round((input_tokens * 1.0 + output_tokens * 3.0) / 1_000_000.0, 6)

    @abstractmethod
    def _call_model(self, messages: List[LLMMessage], temperature: float) -> Tuple[str, int, int]:
        """Low-level model invocation returning (raw_text, input_tokens, output_tokens).
        Must be implemented by concrete providers with retry and timeout handling.
        """
        pass

    def complete(
        self,
        messages: List[LLMMessage],
        schema: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        tenant_id: str = "default_tenant",
    ) -> LLMResponse:
        """Executes LLM completion, validates structured JSON against schema,
        attempts one automatic JSON repair prompt on malformed output,
        and logs tenant token and cost accounting.
        """
        from cdm.llm.validator import validate_and_repair_json

        start_time = time.time()
        raw_text, in_tok, out_tok = self._call_model(messages, temperature=self.config.temperature)
        cost = self.calculate_cost(self.config.model_name, in_tok, out_tok)
        self.record_usage(tenant_id, in_tok, out_tok, cost)

        if not schema:
            latency_ms = (time.time() - start_time) * 1000.0
            return LLMResponse(
                raw_text=raw_text,
                parsed_json=None,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                model_name=self.config.model_name,
                latency_ms=round(latency_ms, 2),
                tenant_id=tenant_id,
            )

        # Validate against schema with one single repair attempt if malformed
        parsed_json, is_valid, err_msg, was_repaired, repair_in, repair_out = validate_and_repair_json(
            raw_text=raw_text,
            schema=schema,
            messages=messages,
            call_fn=lambda msgs: self._call_model(msgs, temperature=0.0),
        )

        if was_repaired:
            in_tok += repair_in
            out_tok += repair_out
            add_cost = self.calculate_cost(self.config.model_name, repair_in, repair_out)
            cost += add_cost
            self.record_usage(tenant_id, repair_in, repair_out, add_cost)

        latency_ms = (time.time() - start_time) * 1000.0

        return LLMResponse(
            raw_text=raw_text,
            parsed_json=parsed_json if is_valid else None,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 6),
            model_name=self.config.model_name,
            latency_ms=round(latency_ms, 2),
            tenant_id=tenant_id,
            repaired=was_repaired,
            validation_error=err_msg if not is_valid else None,
        )
