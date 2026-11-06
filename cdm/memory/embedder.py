"""Semantic & Structured Representation Embedder for CDM-1 Cognitive Memory.
Produces grounded, normalized vector representations from text and structured features
for semantic episodic retrieval and state matching, replacing hardcoded mock vectors.
"""

from typing import Dict, List, Optional, Any, Tuple
import math
import hashlib
import logging
import torch

logger = logging.getLogger("cdm.memory.embedder")


class SemanticMemoryEmbedder:
    """Produces normalized vector representations for cognitive memory search.
    Enables transformer-based semantic embeddings when the language encoder and tokenizer
    are available, while maintaining a deterministic, statistically normalized structured+lexical
    representation as an explicitly identified fallback.
    """

    def __init__(
        self,
        language_encoder: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        dim: int = 16,
        use_neural: Optional[bool] = None,
    ):
        self.language_encoder = language_encoder
        self.tokenizer = tokenizer
        self.dim = dim
        # Automatically enable transformer neural embeddings if both encoder and tokenizer are present
        if use_neural is None:
            self.use_neural = bool(language_encoder is not None and tokenizer is not None)
        else:
            self.use_neural = use_neural
        self.last_embedding_mode: str = "uninitialized"

    def embed_with_metadata(
        self,
        text: str,
        structured_features: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[float], str]:
        """Generates embedding vector and returns it along with the explicit embedding mode."""
        vec = self.embed(text=text, structured_features=structured_features)
        return vec, self.last_embedding_mode

    def embed(
        self,
        text: str,
        structured_features: Optional[Dict[str, Any]] = None,
    ) -> List[float]:
        """Generates a normalized dense vector embedding (L2 unit vector)."""
        struct = structured_features or {}

        # Path 1: Transformer-based semantic embedding if available
        if self.use_neural and self.language_encoder is not None and self.tokenizer is not None:
            try:
                device = torch.device("cpu")
                if hasattr(self.language_encoder, "parameters"):
                    try:
                        device = next(self.language_encoder.parameters()).device
                    except Exception:
                        pass

                tokens = self.tokenizer(
                    text,
                    max_length=64,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}

                with torch.no_grad():
                    input_ids = tokens["input_ids"]
                    attention_mask = tokens["attention_mask"]
                    # Extract sequence output from transformer
                    if hasattr(self.language_encoder, "transformer"):
                        seq_out = self.language_encoder.transformer(input_ids=input_ids, attention_mask=attention_mask)[0]
                        if hasattr(self.language_encoder, "pool_tokens"):
                            pooled = self.language_encoder.pool_tokens(seq_out, attention_mask)[0]
                        else:
                            pooled = seq_out[:, 0, :][0]
                    elif hasattr(self.language_encoder, "forward"):
                        pooled = self.language_encoder(input_ids, attention_mask)[0]
                    else:
                        raise AttributeError("Language encoder has no recognizable forward or transformer module")

                    vec = pooled[: self.dim].cpu().float().numpy().tolist()
                    norm = math.sqrt(sum(x * x for x in vec))
                    if norm > 1e-6:
                        self.last_embedding_mode = "transformer"
                        return [round(x / norm, 6) for x in vec]
            except Exception as e:
                # Do NOT silently suppress neural embedding failures
                logger.warning(
                    f"Transformer neural embedding failed: {e}. "
                    f"Falling back to deterministic structured embedding."
                )

        # Path 2: Deterministic structured + lexical feature projection (Explicit Fallback)
        self.last_embedding_mode = "structured_fallback"
        vec = [0.0] * self.dim

        # Extract normalized structured signals
        # Dimension 0: Sentiment (-1.0 to 1.0)
        vec[0] = float(struct.get("sentiment_score", 0.0))

        # Dimension 1: Urgency level
        urgency = str(struct.get("urgency_level", "medium")).lower()
        urg_map = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
        vec[1] = urg_map.get(urgency, 0.5)

        # Dimension 2: Normalized monetary impact (sigmoid scale)
        monetary = float(struct.get("monetary_value_usd", 50.0))
        vec[2] = 2.0 / (1.0 + math.exp(-monetary / 200.0)) - 1.0

        # Dimension 3: Normalized interaction frequency
        interactions = float(struct.get("interaction_count", 1.0))
        vec[3] = min(1.0, interactions / 10.0)

        # Dimension 4: SLA pressure
        sla_hours = float(struct.get("sla_remaining_hours", 24.0))
        vec[4] = max(0.0, 1.0 - (sla_hours / 48.0))

        # Dimension 5: Domain identifier hash bucket
        domain = str(struct.get("domain", "general"))
        domain_hash = int(hashlib.md5(domain.encode()).hexdigest(), 16) % 1000 / 1000.0
        vec[5] = domain_hash

        # Dimensions 6..15: Word n-gram feature hashing over text
        words = text.lower().split()
        for idx, w in enumerate(words):
            slot = 6 + (int(hashlib.md5(w.encode()).hexdigest(), 16) % (self.dim - 6))
            vec[slot] += 1.0 / (idx + 1.0)

        # L2 Normalization to ensure valid cosine similarity calculations
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-6:
            vec = [round(x / norm, 6) for x in vec]

        return vec
