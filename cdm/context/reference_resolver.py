"""Conversation Context & Reference Resolution for CDM-1.
Resolves anaphora and contextual references (e.g., 'it', 'this option', 'the previous decision',
'the budget', 'that action') to explicit entities, candidates, constraints, and decisions
recorded in prior turns of a tenant-isolated session.
"""

from typing import Dict, List, Optional, Any
import re
from cdm.cognitive.state import ContextFrame


class ResolvedReference(BaseModel if False else object):
    pass


class ReferenceResolver:
    """Resolves conversational references to prior context entities and decisions."""

    PRONOUN_CUES = [
        r"\b(it|this|that|this option|that candidate|the selected option|the chosen action)\b",
        r"\b(why did you choose it|why choose that|compare it with|what if we do it)\b",
    ]
    DECISION_CUES = [
        r"\b(the decision|the recommendation|the choice|previous decision)\b",
    ]
    ENTITY_CUES = [
        r"\b(the customer|the user|the account|the server|the ticket)\b",
    ]

    def resolve_references(
        self,
        current_text: str,
        context: Optional[ContextFrame],
        prior_decisions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Resolves references in current_text using context history.
        Returns:
            Dict containing resolved entities, target_decision, resolved_text, and resolved_constraints.
        """
        resolutions: Dict[str, Any] = {
            "resolved_text": current_text,
            "resolved_decision": None,
            "resolved_entities": {},
            "resolved_constraints": [],
            "referenced_turn_index": None,
            "anaphora_detected": False,
        }

        if not context or not context.dialogue_history or len(context.dialogue_history) < 1:
            return resolutions

        text_lower = current_text.lower()
        has_pronoun = any(re.search(pat, text_lower) for pat in self.PRONOUN_CUES)
        has_decision_ref = any(re.search(pat, text_lower) for pat in self.DECISION_CUES)

        if not (has_pronoun or has_decision_ref):
            return resolutions

        resolutions["anaphora_detected"] = True

        # Scan previous dialogue turns from most recent backwards
        if context.dialogue_history and context.dialogue_history[-1].get("content") == current_text:
            prior_turns = context.dialogue_history[:-1]
        else:
            prior_turns = context.dialogue_history

        for turn in reversed(prior_turns):
            content = str(turn.get("content", ""))
            role = turn.get("role", "")
            feats = turn.get("features", {})
            if isinstance(feats, dict):
                for k, v in feats.items():
                    if k not in resolutions["resolved_entities"]:
                        resolutions["resolved_entities"][k] = v

            # If previous assistant turn contains a decision
            if role == "assistant":
                # Look for decision mentions e.g. 'auto_resolve', 'issue_refund', 'escalate_to_tier2'
                dec_match = re.search(r"'(auto_resolve|issue_refund|escalate_to_tier2|terminate_service|standard_healthcheck)'", content)
                if dec_match and not resolutions["resolved_decision"]:
                    resolutions["resolved_decision"] = dec_match.group(1)
                    resolutions["referenced_turn_index"] = context.turn_index - 1

            # Look for account / ticket entity mentions
            acc_match = re.search(r"\b(ACC-[0-9]+|USR-[0-9]+|CUST-[0-9]+)\b", content)
            if acc_match and "account_id" not in resolutions["resolved_entities"]:
                resolutions["resolved_entities"]["account_id"] = acc_match.group(1)

            # Look for monetary amounts
            money_match = re.search(r"\$([0-9]+(?:\.[0-9]{2})?)", content)
            if money_match and "monetary_value_usd" not in resolutions["resolved_entities"]:
                resolutions["resolved_entities"]["monetary_value_usd"] = float(money_match.group(1))

        # Check context session features
        if hasattr(context, "session_features") and context.session_features:
            for k, v in context.session_features.items():
                if k not in resolutions["resolved_entities"]:
                    resolutions["resolved_entities"][k] = v

        # Check prior decisions passed explicitly
        if not resolutions["resolved_decision"] and prior_decisions:
            last_dec = prior_decisions[-1].get("decision") or prior_decisions[-1].get("proposed_decision")
            if last_dec:
                resolutions["resolved_decision"] = str(last_dec)

        # Build augmented text incorporating resolved references
        if resolutions["resolved_decision"]:
            resolved_str = current_text
            for p in ["this option", "that option", "the chosen action", "it"]:
                if re.search(rf"\b{p}\b", resolved_str, re.IGNORECASE):
                    resolved_str = re.sub(
                        rf"\b{p}\b",
                        f"'{resolutions['resolved_decision']}'",
                        resolved_str,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                    break
            resolutions["resolved_text"] = resolved_str

        return resolutions
