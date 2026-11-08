"""Layer 1: Local / OpenAI-Compatible Foundation Model Provider.
Supports local models (Ollama, vLLM, LocalAI) and deterministic mock mode for testing.
"""

from typing import List, Tuple, Optional, Dict, Any
import os
import time
import json
import logging
import requests
from cdm.llm.base import BaseLLMProvider, LLMMessage, LLMProviderConfig

logger = logging.getLogger("cdm.llm.local")


class LocalOpenAIProvider(BaseLLMProvider):
    """Local OpenAI-compatible API Provider (vLLM, Ollama, LM Studio, etc.)."""

    def __init__(self, config: Optional[LLMProviderConfig] = None):
        super().__init__(config=config)
        self.api_base = self.config.api_base or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
        self.api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY", "dummy-local-key")

    def _call_model(self, messages: List[LLMMessage], temperature: float) -> Tuple[str, int, int]:
        url = f"{self.api_base.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": 2048,
        }

        retries = 0
        backoff = 1.0

        while True:
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.config.timeout_seconds)
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")

                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                in_tok = usage.get("prompt_tokens", len(json.dumps(payload)) // 4)
                out_tok = usage.get("completion_tokens", len(raw_text) // 4)
                return raw_text, in_tok, out_tok

            except Exception as err:
                retries += 1
                if retries > self.config.max_retries:
                    raise RuntimeError(f"Local LLM API failed after {retries} attempts: {err}") from err

                wait_time = backoff * (self.config.backoff_factor ** (retries - 1))
                logger.warning(f"Local LLM error ({err}). Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)


class MockLLMProvider(BaseLLMProvider):
    """Deterministic Mock LLM Provider for offline evaluation and unit testing."""

    def __init__(self, config: Optional[LLMProviderConfig] = None, response_canned: Optional[str] = None):
        super().__init__(config=config or LLMProviderConfig(provider_type="mock", model_name="mock-deterministic"))
        self.response_canned = response_canned
        self.history: List[List[LLMMessage]] = []

    def _call_model(self, messages: List[LLMMessage], temperature: float) -> Tuple[str, int, int]:
        self.history.append(messages)
        if self.response_canned:
            text = self.response_canned
        else:
            # Default deterministic response proposing an action
            last_msg = messages[-1].content if messages else ""
            if "repair" in last_msg.lower():
                text = json.dumps({
                    "action": "view_ticket",
                    "arguments": {"ticket_id": "TICK-101"},
                    "reasoning": "Repaired JSON following validation failure.",
                })
            else:
                text = json.dumps({
                    "action": "check_status",
                    "arguments": {"resource": "payment_gateway"},
                    "reasoning": "Inspecting operational health of payment system.",
                })

        in_tok = sum(len(m.content) for m in messages) // 4
        out_tok = len(text) // 4
        return text, in_tok, out_tok
