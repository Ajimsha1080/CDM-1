"""Intent and Task Routing Subsystem for CDM-1 Cognitive AI Engine.
Classifies natural-language inputs into task categories:
question, explanation, decision, prediction, planning, comparison, execution, clarification.
Provides dynamic routing plans to invoke only the necessary cognitive components,
and generates structured clarification prompts when essential parameters are missing.
"""

from typing import Dict, List, Optional, Any, Set
import re
from pydantic import BaseModel, Field


class CognitiveRoutePlan(BaseModel):
    """Execution plan identifying which cognitive components are needed for a task."""
    intent_task: str
    run_perception: bool = True
    run_context: bool = True
    run_memory_retrieval: bool = True
    run_reasoning: bool = True
    run_neural_inference: bool = True
    run_planning: bool = False
    run_tools: bool = False
    run_synthesis: bool = True
    needs_clarification: bool = False
    missing_parameters: List[str] = Field(default_factory=list)
    clarification_prompt: Optional[str] = None


class IntentTaskRouter:
    """Classifies user inquiries and determines optimal cognitive component execution paths."""

    TASK_TYPES: Set[str] = {
        "question",
        "explanation",
        "decision",
        "prediction",
        "planning",
        "comparison",
        "execution",
        "clarification",
    }

    def __init__(self):
        # Patterns for routing
        self.question_patterns = [
            r"^(what|where|who|when|which|is|are|does|can)\b",
            r"\b(information|policy|sla|definition|status of)\b",
        ]
        self.explanation_patterns = [
            r"^(why|how come|explain|what is the reason)\b",
            r"\b(why did you|reason for|justify)\b",
        ]
        self.decision_patterns = [
            r"\b(should we|choose between|select|recommend|which action|decide)\b",
        ]
        self.prediction_patterns = [
            r"\b(predict|forecast|projection|what will happen|estimate resolution time|future)\b",
        ]
        self.planning_patterns = [
            r"\b(plan|steps|workflow|how do we execute|prerequisites|procedure for)\b",
        ]
        self.comparison_patterns = [
            r"\b(compare|versus|vs|difference between|trade-off)\b",
        ]
        self.execution_patterns = [
            r"\b(execute|run|apply action|dispatch|perform|trigger now|do it)\b",
        ]

    def classify_intent(
        self,
        text: str,
        structured_features: Optional[Dict[str, Any]] = None,
        candidates: Optional[List[str]] = None,
        task_hint: Optional[str] = None,
    ) -> str:
        """Classifies text into one of the 7 primary intent tasks or clarification."""
        q = text.strip().lower()

        # Check for execution requests
        if any(re.search(p, q) for p in self.execution_patterns) and not q.startswith("how"):
            return "execution"

        # Check for comparison
        if any(re.search(p, q) for p in self.comparison_patterns):
            return "comparison"

        # Check for prediction
        if any(re.search(p, q) for p in self.prediction_patterns):
            return "prediction"

        # Check for planning
        if any(re.search(p, q) for p in self.planning_patterns):
            return "planning"

        # Check for explanation
        if any(re.search(p, q) for p in self.explanation_patterns):
            return "explanation"

        # Check for decision
        if any(re.search(p, q) for p in self.decision_patterns) or (candidates and len(candidates) > 1 and "what" not in q):
            return "decision"

        # Check for general question
        if any(re.search(p, q) for p in self.question_patterns) or q.endswith("?"):
            return "question"

        if task_hint in self.TASK_TYPES:
            return task_hint

        return "decision" if candidates else "question"

    def check_missing_parameters(
        self,
        text: str,
        intent: str,
        structured_features: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Identifies missing essential parameters needed to safely process the intent."""
        missing = []
        q = text.strip().lower()
        feats = structured_features or {}

        # If execution or decision task involves sensitive financial/account action
        if intent in ("execution", "decision"):
            if "refund" in q or "compensat" in q:
                if "monetary_value_usd" not in feats and not re.search(r"\$[0-9]+", text):
                    missing.append("monetary_value_usd")
                if "account_id" not in feats and not re.search(r"\b(ACC|USR|CUST)-[0-9]+", text):
                    missing.append("account_id")

            if "database" in q or "cluster" in q or "server" in q:
                if "target_host" not in feats and not re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text):
                    missing.append("target_host_or_cluster_id")

        return missing

    def plan_route(
        self,
        text: str,
        structured_features: Optional[Dict[str, Any]] = None,
        candidates: Optional[List[str]] = None,
        task_hint: Optional[str] = None,
    ) -> CognitiveRoutePlan:
        """Determines the exact cognitive components required for the request."""
        intent = self.classify_intent(
            text=text,
            structured_features=structured_features,
            candidates=candidates,
            task_hint=task_hint,
        )

        missing_params = self.check_missing_parameters(
            text=text,
            intent=intent,
            structured_features=structured_features,
        )

        if missing_params:
            param_list = ", ".join(f"'{p}'" for p in missing_params)
            clarification_msg = (
                f"Essential operational information is missing to proceed with this {intent} request: {param_list}. "
                f"Please provide these details before execution can be evaluated."
            )
            return CognitiveRoutePlan(
                intent_task="clarification",
                run_perception=True,
                run_context=True,
                run_memory_retrieval=False,
                run_reasoning=False,
                run_neural_inference=False,
                run_planning=False,
                run_tools=False,
                run_synthesis=True,
                needs_clarification=True,
                missing_parameters=missing_params,
                clarification_prompt=clarification_msg,
            )

        # Dynamic routing table
        if intent == "question":
            # Informational questions require perception, context, memory retrieval, synthesis
            # Neural candidate scoring is optional unless comparing actions
            return CognitiveRoutePlan(
                intent_task=intent,
                run_perception=True,
                run_context=True,
                run_memory_retrieval=True,
                run_reasoning=True,
                run_neural_inference=bool(candidates),
                run_planning=False,
                run_tools=False,
                run_synthesis=True,
            )
        elif intent == "explanation":
            return CognitiveRoutePlan(
                intent_task=intent,
                run_perception=True,
                run_context=True,
                run_memory_retrieval=True,
                run_reasoning=True,
                run_neural_inference=bool(candidates),
                run_planning=False,
                run_tools=False,
                run_synthesis=True,
            )
        elif intent == "planning":
            return CognitiveRoutePlan(
                intent_task=intent,
                run_perception=True,
                run_context=True,
                run_memory_retrieval=True,
                run_reasoning=True,
                run_neural_inference=True,
                run_planning=True,
                run_tools=False,
                run_synthesis=True,
            )
        elif intent == "execution":
            return CognitiveRoutePlan(
                intent_task=intent,
                run_perception=True,
                run_context=True,
                run_memory_retrieval=True,
                run_reasoning=True,
                run_neural_inference=True,
                run_planning=True,
                run_tools=True,
                run_synthesis=True,
            )
        elif intent in ("decision", "comparison", "prediction"):
            return CognitiveRoutePlan(
                intent_task=intent,
                run_perception=True,
                run_context=True,
                run_memory_retrieval=True,
                run_reasoning=True,
                run_neural_inference=True,
                run_planning=False,
                run_tools=False,
                run_synthesis=True,
            )

        return CognitiveRoutePlan(
            intent_task=intent,
            run_perception=True,
            run_context=True,
            run_memory_retrieval=True,
            run_reasoning=True,
            run_neural_inference=True,
            run_planning=True,
            run_tools=False,
            run_synthesis=True,
        )
