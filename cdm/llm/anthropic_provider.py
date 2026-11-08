"""Layer 1: Anthropic Foundation Model Provider.
Integrates Claude models with retry backoff, timeout enforcement,
and per-tenant token and cost accounting.
"""

from typing import List, Tuple, Optional
import os
import time
import logging
from cdm.llm.base import BaseLLMProvider, LLMMessage, LLMProviderConfig

logger = logging.getLogger("cdm.llm.anthropic")


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API Provider with exponential backoff and timeout guards."""

    def __init__(self, config: Optional[LLMProviderConfig] = None):
        super().__init__(config=config)
        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.api_key = api_key
        self.client = None

        if api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(
                    api_key=api_key,
                    timeout=self.config.timeout_seconds,
                    max_retries=0, # Retries handled explicitly with exponential backoff
                )
            except ImportError:
                logger.warning("anthropic package is not installed.")
        else:
            logger.info("AnthropicProvider initialized without ANTHROPIC_API_KEY.")

    def _call_model(self, messages: List[LLMMessage], temperature: float) -> Tuple[str, int, int]:
        """Executes Anthropic API call with exponential backoff retries."""
        if not self.client:
            raise RuntimeError(
                "Anthropic API client is not configured. Set ANTHROPIC_API_KEY environment variable "
                "or provide api_key in LLMProviderConfig."
            )

        # Convert to Anthropic messages format: extract system message if present
        system_prompt = ""
        user_assistant_msgs = []
        for m in messages:
            if m.role == "system":
                system_prompt += f"{m.content}\n"
            else:
                user_assistant_msgs.append({"role": m.role, "content": m.content})

        if not user_assistant_msgs:
            user_assistant_msgs.append({"role": "user", "content": "Begin."})

        retries = 0
        backoff = 1.0

        while True:
            try:
                kwargs = {
                    "model": self.config.model_name,
                    "max_tokens": 2048,
                    "temperature": temperature,
                    "messages": user_assistant_msgs,
                }
                if system_prompt:
                    kwargs["system"] = system_prompt.strip()

                response = self.client.messages.create(**kwargs)

                raw_text = ""
                for block in response.content:
                    if getattr(block, "type", "") == "text":
                        raw_text += block.text

                in_tok = getattr(response.usage, "input_tokens", 0)
                out_tok = getattr(response.usage, "output_tokens", 0)
                return raw_text, in_tok, out_tok

            except Exception as err:
                retries += 1
                if retries > self.config.max_retries:
                    logger.error(f"Anthropic API call failed after {retries} attempts: {err}")
                    raise RuntimeError(f"Anthropic API failed after retries: {err}") from err

                wait_time = backoff * (self.config.backoff_factor ** (retries - 1))
                logger.warning(f"Anthropic API error ({err}). Retrying in {wait_time:.2f}s (attempt {retries}/{self.config.max_retries})...")
                time.sleep(wait_time)
