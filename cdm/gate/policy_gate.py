"""Layer 3: Deterministic Policy Gate and Risk Assessment Core for CDM-1.
Enforces multi-stage policy verification in strict order:
(a) Hard prohibited-action and tenant policy rules
(b) Planning engine precondition and constraint checks
(c) Deterministic risk scoring
(d) Optional neural model risk scoring as one input only.

Outputs one of: ALLOW, REQUIRE_HUMAN_APPROVAL, BLOCK with explicit reasons.
LLMs can NEVER override a BLOCK or REQUIRE_HUMAN_APPROVAL decision.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from decimal import Decimal, InvalidOperation
import math
import logging
from pydantic import BaseModel, Field

from cdm.cognitive.state import WorldState
from cdm.inference.pipeline import CognitiveDecisionModel

logger = logging.getLogger("cdm.gate")


class GateDecisionType(str, Enum):
    ALLOW = "allow"
    REQUIRE_HUMAN_APPROVAL = "require_human_approval"
    BLOCK = "block"


class ProposedAction(BaseModel):
    tool: str = Field(..., description="Tool or action identifier to execute")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for tool invocation")
    reason: str = Field(default="", description="LLM justification or intent")


class GateDecision(BaseModel):
    decision: GateDecisionType
    reason: str
    risk_score: float = 0.0
    evaluated_rules: List[str] = Field(default_factory=list)
    requires_human: bool = False
    is_blocked: bool = False
    neural_score: Optional[float] = None


class PolicyGate:
    """Deterministic Multi-Stage Policy Gate."""

    def __init__(
        self,
        decision_model: Optional[CognitiveDecisionModel] = None,
        human_approval_risk_threshold: float = 0.50,
        block_risk_threshold: float = 0.80,
    ):
        self.decision_model = decision_model
        self.human_approval_risk_threshold = human_approval_risk_threshold
        self.block_risk_threshold = block_risk_threshold

        # Global hard-prohibited commands / tools that are never allowed
        self.prohibited_commands = {
            "rm -rf", "drop table", "mkfs", "shutdown", "reboot",
            ":(){ :|:& };:", "dd if=/dev/zero", "chmod -r 777 /",
            "format c:", "del /f /s /q",
        }

    def evaluate(
        self,
        action: ProposedAction,
        tenant_id: str = "default_tenant",
        world_state: Optional[WorldState] = None,
        tenant_policies: Optional[List[str]] = None,
        prohibited_tools: Optional[List[str]] = None,
        prohibited_actions: Optional[List[str]] = None,
        monetary_limit: float = 500.0,
        authoritative_risk: Optional[float] = None,
    ) -> GateDecision:
        """Evaluates proposed action in strict deterministic sequence."""
        evaluated_rules: List[str] = []
        action_name = action.tool.lower()
        args_str = str(action.args).lower()
        reason_str = action.reason.lower()

        # =====================================================================
        # Stage (a): Hard Prohibited-Action and Tenant Policy Rules
        # =====================================================================
        evaluated_rules.append("stage_a:hard_prohibited_checks")

        # 1. Prohibited tools check
        proh_tools = set(t.lower() for t in (prohibited_tools or []))
        if action_name in proh_tools:
            return GateDecision(
                decision=GateDecisionType.BLOCK,
                reason=f"Action '{action.tool}' is explicitly listed in prohibited tools.",
                risk_score=1.0,
                evaluated_rules=evaluated_rules,
                is_blocked=True,
            )

        # 2. Prohibited actions check
        proh_actions = set(a.lower() for a in (prohibited_actions or []))
        if action_name in proh_actions:
            return GateDecision(
                decision=GateDecisionType.BLOCK,
                reason=f"Action '{action.tool}' is explicitly prohibited by policy.",
                risk_score=1.0,
                evaluated_rules=evaluated_rules,
                is_blocked=True,
            )

        # 3. Dangerous system command injection checks
        for danger in self.prohibited_commands:
            if danger in args_str:
                return GateDecision(
                    decision=GateDecisionType.BLOCK,
                    reason=f"Action contains prohibited dangerous pattern: '{danger}'.",
                    risk_score=1.0,
                    evaluated_rules=evaluated_rules,
                    is_blocked=True,
                )

        # 4. Financial transaction bound and numerical safety check
        amount_decimal = Decimal("0.0")
        has_monetary_arg = False
        raw_currency = action.args.get("currency") or action.args.get("curr")
        if raw_currency:
            curr_str = str(raw_currency).strip().upper()
            allowed_currencies = {"USD", "EUR", "GBP", "CAD", "AUD", "JPY"}
            if curr_str not in allowed_currencies:
                evaluated_rules.append("stage_a:unsupported_currency")
                return GateDecision(
                    decision=GateDecisionType.BLOCK,
                    reason=f"Financial policy violation: Unsupported or unapproved currency '{curr_str}'.",
                    risk_score=1.0,
                    evaluated_rules=evaluated_rules,
                    is_blocked=True,
                )

        for k in ("amount", "refund_amount", "monetary_value", "usd", "payment_amount", "transaction_value"):
            if k in action.args:
                val = action.args[k]
                has_monetary_arg = True
                # Check for NaN or Inf in float values
                if isinstance(val, (float, int)):
                    if math.isnan(val) or math.isinf(val):
                        evaluated_rules.append("stage_a:nan_inf_amount_blocked")
                        return GateDecision(
                            decision=GateDecisionType.BLOCK,
                            reason="Financial security violation: NaN or infinite monetary amount detected.",
                            risk_score=1.0,
                            evaluated_rules=evaluated_rules,
                            is_blocked=True,
                        )
                    amount_decimal = Decimal(str(val))
                elif isinstance(val, str):
                    clean_str = val.strip().replace("$", "").replace("€", "").replace("£", "").replace(",", "")
                    if clean_str.lower() in ("nan", "inf", "-inf", "+inf", "infinity"):
                        evaluated_rules.append("stage_a:nan_inf_amount_blocked")
                        return GateDecision(
                            decision=GateDecisionType.BLOCK,
                            reason="Financial security violation: NaN or infinite monetary amount detected.",
                            risk_score=1.0,
                            evaluated_rules=evaluated_rules,
                            is_blocked=True,
                        )
                    try:
                        amount_decimal = Decimal(clean_str)
                    except (InvalidOperation, ValueError):
                        evaluated_rules.append("stage_a:malformed_numeric_string")
                        return GateDecision(
                            decision=GateDecisionType.BLOCK,
                            reason=f"Financial validation error: Malformed numeric string input '{val}'.",
                            risk_score=1.0,
                            evaluated_rules=evaluated_rules,
                            is_blocked=True,
                        )
                else:
                    evaluated_rules.append("stage_a:invalid_numeric_type")
                    return GateDecision(
                        decision=GateDecisionType.BLOCK,
                        reason=f"Financial validation error: Invalid non-numeric type for monetary argument '{k}'.",
                        risk_score=1.0,
                        evaluated_rules=evaluated_rules,
                        is_blocked=True,
                    )
                break

        if has_monetary_arg:
            # Reject negative transaction amounts
            if amount_decimal < Decimal("0"):
                evaluated_rules.append("stage_a:negative_amount_blocked")
                return GateDecision(
                    decision=GateDecisionType.BLOCK,
                    reason=f"Financial safety violation: Negative transaction amount ({amount_decimal}) is strictly prohibited.",
                    risk_score=1.0,
                    evaluated_rules=evaluated_rules,
                    is_blocked=True,
                )

            # Hard maximum threshold check
            hard_max_limit = Decimal("10000.00")
            if amount_decimal > hard_max_limit:
                evaluated_rules.append("stage_a:hard_maximum_limit_exceeded")
                return GateDecision(
                    decision=GateDecisionType.BLOCK,
                    reason=f"Financial hard limit violation: Transaction amount (${amount_decimal}) exceeds hard boundary (${hard_max_limit}).",
                    risk_score=1.0,
                    evaluated_rules=evaluated_rules,
                    is_blocked=True,
                )

            # Autonomous threshold check
            limit_decimal = Decimal(str(monetary_limit))
            if amount_decimal > limit_decimal:
                evaluated_rules.append("stage_a:financial_limit_exceeded")
                return GateDecision(
                    decision=GateDecisionType.REQUIRE_HUMAN_APPROVAL,
                    reason=f"Financial action amount (${amount_decimal}) exceeds autonomous limit (${limit_decimal}).",
                    risk_score=0.75,
                    evaluated_rules=evaluated_rules,
                    requires_human=True,
                )

        amount = float(amount_decimal)

        # 5. Tenant explicit policy rules
        if tenant_policies:
            for policy in tenant_policies:
                evaluated_rules.append(f"tenant_policy:{policy}")
                p_lower = policy.lower()
                if "no_external_network" in p_lower and action_name in ("http_request", "curl", "wget"):
                    return GateDecision(
                        decision=GateDecisionType.BLOCK,
                        reason=f"Tenant policy strictly forbids external HTTP networking: '{policy}'",
                        risk_score=1.0,
                        evaluated_rules=evaluated_rules,
                        is_blocked=True,
                    )
                if "require_approval_for_refunds" in p_lower and "refund" in action_name:
                    return GateDecision(
                        decision=GateDecisionType.REQUIRE_HUMAN_APPROVAL,
                        reason=f"Tenant policy requires human approval for all refunds: '{policy}'",
                        risk_score=0.60,
                        evaluated_rules=evaluated_rules,
                        requires_human=True,
                    )

        # =====================================================================
        # Stage (b): Precondition and Constraint Checks
        # =====================================================================
        evaluated_rules.append("stage_b:precondition_checks")
        if world_state:
            # Verify required target entity exists in world state
            target_id = action.args.get("ticket_id") or action.args.get("order_id") or action.args.get("user_id")
            if target_id and hasattr(world_state, "entities"):
                # If entities tracked, ensure entity is known
                known_entities = getattr(world_state, "entities", {})
                if known_entities and target_id not in known_entities:
                    # Target not in world state
                    evaluated_rules.append("stage_b:unknown_target_entity")
                    return GateDecision(
                        decision=GateDecisionType.BLOCK,
                        reason=f"Target entity '{target_id}' does not exist in current world state.",
                        risk_score=0.85,
                        evaluated_rules=evaluated_rules,
                        is_blocked=True,
                    )

            # Check if ticket already resolved / closed
            if target_id and hasattr(world_state, "state_metrics"):
                metrics = getattr(world_state, "state_metrics", {})
                if metrics.get(f"{target_id}_status") in ("closed", "resolved") and "resolve" in action_name:
                    return GateDecision(
                        decision=GateDecisionType.BLOCK,
                        reason=f"Entity '{target_id}' is already closed in world state.",
                        risk_score=0.90,
                        evaluated_rules=evaluated_rules,
                        is_blocked=True,
                    )

        # =====================================================================
        # Stage (c): Deterministic Risk Scoring
        # =====================================================================
        evaluated_rules.append("stage_c:deterministic_risk_scoring")
        base_risk = 0.15

        # Destructive or write actions carry elevated baseline risk
        write_actions = {"write_file", "edit_file", "shell_exec", "issue_refund", "delete_ticket", "reset_password"}
        if action_name in write_actions:
            base_risk += 0.25

        if "critical" in reason_str or "urgent" in reason_str:
            base_risk += 0.15

        if amount > 100.0:
            base_risk += 0.20

        calculated_risk = min(max(base_risk, 0.05), 0.95)

        # =====================================================================
        # Stage (d): Authoritative Neural Model Risk Scoring (One input only)
        # =====================================================================
        neural_risk: Optional[float] = None
        if authoritative_risk is not None:
            evaluated_rules.append("stage_d:authoritative_neural_scoring")
            neural_risk = float(authoritative_risk)
            calculated_risk = max(calculated_risk, neural_risk)
        elif self.decision_model is not None:
            evaluated_rules.append("stage_d:neural_scoring")
            try:
                model_res = self.decision_model.decide(
                    text=f"Action: {action.tool}. Reason: {action.reason}",
                    structured_features={"monetary_value_usd": amount, "urgency_level": "medium"},
                    task="risk",
                    tenant_id=tenant_id,
                )
                neural_risk = float(model_res.risk)
                # Combine deterministic and neural risk: taking maximum conservative risk
                calculated_risk = max(calculated_risk, neural_risk)
            except Exception as e:
                logger.warning(f"Neural scorer evaluation skipped due to exception: {e}")

        # Final threshold determination
        if calculated_risk >= self.block_risk_threshold:
            return GateDecision(
                decision=GateDecisionType.BLOCK,
                reason=f"Calculated risk ({calculated_risk:.2f}) exceeds blocking threshold ({self.block_risk_threshold:.2f}).",
                risk_score=round(calculated_risk, 4),
                evaluated_rules=evaluated_rules,
                is_blocked=True,
                neural_score=neural_risk,
            )

        if calculated_risk >= self.human_approval_risk_threshold:
            return GateDecision(
                decision=GateDecisionType.REQUIRE_HUMAN_APPROVAL,
                reason=f"Calculated risk ({calculated_risk:.2f}) requires human confirmation.",
                risk_score=round(calculated_risk, 4),
                evaluated_rules=evaluated_rules,
                requires_human=True,
                neural_score=neural_risk,
            )

        return GateDecision(
            decision=GateDecisionType.ALLOW,
            reason="Action passed all policy rules, preconditions, and risk thresholds.",
            risk_score=round(calculated_risk, 4),
            evaluated_rules=evaluated_rules,
            requires_human=False,
            is_blocked=False,
            neural_score=neural_risk,
        )
