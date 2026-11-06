"""Contextual Intelligence Subsystem for CDM-1 Cognitive AI Engine.
Tracks multi-turn conversation context, temporal deltas, entity relationships,
contextual contradictions, and relevance scoring.
"""

from typing import Dict, List, Optional, Any, Tuple
import time
import threading
from cdm.cognitive.state import ContextFrame, PerceptionResult


class ContextEngine:
    """Manages situation tracking across time, conversation history, and detects contextual contradictions.
    Enforces strict tenant isolation and thread-safe session ownership.
    """

    def __init__(self):
        # Key: (tenant_id, session_id)
        self.sessions: Dict[Tuple[str, str], List[ContextFrame]] = {}
        self._lock = threading.Lock()

    def update_context(
        self,
        session_id: str,
        tenant_id: str,
        perception: PerceptionResult,
        user_message: str,
        system_response: Optional[str] = None,
    ) -> ContextFrame:
        """Appends a new turn to the session dialogue history and calculates temporal/contradiction metrics."""
        with self._lock:
            key = (tenant_id, session_id)
            history = self.sessions.setdefault(key, [])

            now = time.time()
            if history:
                last_turn_time = getattr(history[-1], "timestamp", now)
                temporal_delta = max(0.0, now - last_turn_time)
            else:
                temporal_delta = 0.0

            turn_index = len(history) + 1

            # Build dialogue history
            dialogue = []
            if history:
                dialogue = list(history[-1].dialogue_history)
            dialogue.append({"role": "user", "content": user_message})
            if system_response:
                dialogue.append({"role": "assistant", "content": system_response})

            # Check for contradictions
            contradictions = self._detect_contradictions(dialogue, perception)

            # Calculate relevance scores
            relevance: Dict[str, float] = {}
            for ent in perception.extracted_entities:
                relevance[ent.name] = ent.confidence

            frame = ContextFrame(
                session_id=session_id,
                tenant_id=tenant_id,
                turn_index=turn_index,
                dialogue_history=dialogue,
                temporal_delta_seconds=temporal_delta,
                timestamp=now,
                detected_contradictions=contradictions,
                relevance_scores=relevance,
            )

            history.append(frame)
            return frame

    def _detect_contradictions(self, dialogue: List[Dict[str, str]], perception: PerceptionResult) -> List[str]:
        """Detects factual contradictions between recent turns and current statements."""
        contradictions = []
        if len(dialogue) < 2:
            return contradictions

        # Simple semantic contradiction check (e.g., claiming never contacted vs earlier interaction count > 0)
        recent_user_texts = [d["content"].lower() for d in dialogue if d["role"] == "user"]
        current_text = perception.raw_text.lower()

        if "never contacted" in current_text or "first time reporting" in current_text:
            interactions = perception.validated_tabular.get("interaction_count", 0)
            if interactions and interactions > 1:
                contradictions.append(
                    f"Contradiction: user reports 'first time' or 'never contacted', but records indicate interaction_count={interactions}"
                )

        if "immediately" in current_text and "urgent" in current_text:
            urgency = perception.validated_tabular.get("urgency_level")
            if urgency == "low":
                contradictions.append("Contradiction: text conveys critical urgency but tabular urgency_level is 'low'")

        return contradictions

    def get_latest_context(self, session_id: str, tenant_id: str = "default_tenant") -> Optional[ContextFrame]:
        """Retrieves the most recent context frame for a tenant's session.
        Strictly requires exact (tenant_id, session_id) matching; no fallback scanning across other tenants.
        """
        with self._lock:
            key = (tenant_id, session_id)
            frames = self.sessions.get(key)
            if frames:
                return frames[-1]
            return None

    def clear_session(self, session_id: str, tenant_id: str = "default_tenant") -> bool:
        """Clears session context for a specific tenant."""
        with self._lock:
            key = (tenant_id, session_id)
            if key in self.sessions:
                del self.sessions[key]
                return True
            return False
