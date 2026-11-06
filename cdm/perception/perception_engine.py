"""Perception Intelligence Subsystem for CDM-1 Cognitive AI Engine.
Converts incoming unstructured text, structured features, and external stimuli
into structured representations, extracted entities, signals, and perception confidence.
"""

from typing import Dict, List, Optional, Any
import re
import time
from cdm.cognitive.state import PerceptionResult, EntityMention


class PerceptionEngine:
    """Perception processor extracting structured signals, entities, and validation metrics."""

    def __init__(self):
        # Common entity pattern matchers
        self.patterns = {
            "monetary": r"\$([0-9]+(?:\.[0-9]{2})?)",
            "account_id": r"\b(ACC-[0-9]{4,8}|USR-[0-9]{4,8}|CUST-[0-9]{4,8})\b",
            "incident_id": r"\b(INC-[0-9]{4,8}|TICK-[0-9]{4,8})\b",
            "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "error_code": r"\b(ERR-[0-9]{3,5}|HTTP [45][0-9]{2}|50[0-4] Server Error)\b",
        }

    def process(
        self,
        text: str,
        structured_features: Optional[Dict[str, Any]] = None,
    ) -> PerceptionResult:
        """Processes raw text and tabular features into a validated PerceptionResult."""
        structured = structured_features or {}
        extracted_entities: List[EntityMention] = []

        # 1. Entity extraction
        for ent_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for m in matches:
                matched_val = m.group(0)
                extracted_entities.append(
                    EntityMention(
                        entity_id=f"{ent_type}_{len(extracted_entities) + 1}",
                        entity_type=ent_type,
                        name=matched_val,
                        confidence=0.95,
                        attributes={"span": m.span()},
                    )
                )

        # 2. Extract operational signals
        signals: Dict[str, float] = {}

        # Urgency signal
        urgency_words = ["urgent", "critical", "immediately", "outage", "emergency", "deadline", "severe"]
        text_lower = text.lower()
        urgency_score = sum(1 for w in urgency_words if w in text_lower) / max(1, len(urgency_words))
        signals["text_urgency"] = min(1.0, urgency_score * 2.5)

        # Sentiment / distress signal
        negative_words = ["broken", "fail", "failed", "crash", "loss", "angry", "dispute", "unauthorized", "breach"]
        distress_score = sum(1 for w in negative_words if w in text_lower) / max(1, len(negative_words))
        signals["distress_level"] = min(1.0, distress_score * 2.0)

        # 3. Tabular validation & perception confidence
        validated_tabular: Dict[str, Any] = {}
        missing_count = 0
        total_expected = 5  # Standard core features

        for k, v in structured.items():
            if v is None:
                missing_count += 1
            validated_tabular[k] = v

        modality_confidence = max(0.4, 1.0 - (missing_count * 0.15))
        if len(text.strip()) < 10:
            modality_confidence *= 0.7

        return PerceptionResult(
            raw_text=text,
            validated_tabular=validated_tabular,
            extracted_entities=extracted_entities,
            signals=signals,
            modality_confidence=round(modality_confidence, 4),
            timestamp=time.time(),
        )
