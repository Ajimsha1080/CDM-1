"""Factory for selecting and instantiating Layer 1 Foundation Model Providers."""

import os
from typing import Optional
from cdm.llm.base import BaseLLMProvider, LLMProviderConfig
from cdm.llm.anthropic_provider import AnthropicProvider
from cdm.llm.local_provider import LocalOpenAIProvider, MockLLMProvider


def get_llm_provider(config: Optional[LLMProviderConfig] = None) -> BaseLLMProvider:
    """Instantiates LLM provider based on configuration or environment variables.
    Falls back gracefully to MockLLMProvider when offline or API keys are absent.
    """
    cfg = config or LLMProviderConfig()
    provider_type = os.environ.get("CDM_LLM_PROVIDER", cfg.provider_type).lower()

    if provider_type == "anthropic":
        key = cfg.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            # Fallback to MockLLMProvider with clear log notice
            return MockLLMProvider(cfg)
        return AnthropicProvider(cfg)

    elif provider_type in ("openai", "openai_compatible", "local"):
        return LocalOpenAIProvider(cfg)

    elif provider_type == "mock":
        return MockLLMProvider(cfg)

    # Default fallback
    return MockLLMProvider(cfg)
