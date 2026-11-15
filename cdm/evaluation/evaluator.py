"""Cognitive Evaluation Framework for CDM-1.
Evaluates end-to-end cognitive response quality across 10 foundational dimensions:
1. QA Relevance
2. Factual Correctness
3. Evidence Grounding
4. Decision Quality
5. Prediction Quality
6. Calibration Error (ECE)
7. Abstention Quality
8. Planning Consistency
9. Tool Execution Correctness
10. Latency and Resource Consumption
"""

from typing import Dict, List, Optional, Any
import math
import time
from pydantic import BaseModel, Field

from cdm.cognitive.state import CognitiveEngineResult


class MetricScore(BaseModel):
    name: str
    score: float  # 0.0 to 1.0 (or normalized)
    passed: bool
    explanation: str
    details: Dict[str, Any] = Field(default_factory=dict)


class ComprehensiveEvaluationReport(BaseModel):
    """Unified evaluation scorecard for a cognitive run or evaluation batch."""
    evaluation_id: str
    overall_quality_score: float
    metrics: Dict[str, MetricScore]
    summary_verdict: str  # 'EXCELLENT', 'ACCEPTABLE', 'DEGRADED', 'FAILED'
    timestamp: float = Field(default_factory=time.time)


class CognitiveEvaluator:
    """Evaluates CDM-1 cognitive responses across all mandatory reliability axes."""

    def __init__(self):
        pass

    def evaluate(
        self,
        inquiry_text: str,
        result: CognitiveEngineResult,
        ground_truth: Optional[Dict[str, Any]] = None,
        tool_results: Optional[List[Any]] = None,
    ) -> ComprehensiveEvaluationReport:
        """Evaluates a CognitiveEngineResult across all 10 dimensions."""
        gt = ground_truth or {}
        metrics: Dict[str, MetricScore] = {}

        # 1. QA Relevance
        ans = (result.answer or "").lower()
        q_words = set(inquiry_text.lower().split())
        overlap = sum(1 for w in q_words if w in ans and len(w) > 3)
        relevance_score = min(1.0, (overlap / max(1, len(q_words))) * 2.5) if ans else 0.0
        metrics["qa_relevance"] = MetricScore(
            name="QA Relevance",
            score=round(relevance_score, 4),
            passed=relevance_score >= 0.3,
            explanation=f"Relevance of synthesized answer to user inquiry text (term overlap: {overlap}).",
        )

        # 2. Factual Correctness
        # Check against ground truth target or verified external facts
        has_gt_decision = "expected_decision" in gt
        if has_gt_decision:
            is_correct = (result.decision == gt["expected_decision"]) or (result.proposed_decision == gt["expected_decision"])
            fact_score = 1.0 if is_correct else 0.0
        else:
            # Score based on absence of contradictions and presence of verified facts
            fact_score = 1.0 if not result.reasoning_evidence.contradictions_found else 0.5
        metrics["factual_correctness"] = MetricScore(
            name="Factual Correctness",
            score=round(fact_score, 4),
            passed=fact_score >= 0.7,
            explanation="Agreement with verified operational facts or target labels.",
        )

        # 3. Evidence Grounding
        has_evidence = bool(
            result.supporting_evidence.get("semantic_rules_applied")
            or result.retrieved_external_facts
            or result.reasoning_evidence.steps
        )
        grounding_score = 1.0 if has_evidence else 0.4
        metrics["evidence_grounding"] = MetricScore(
            name="Evidence Grounding",
            score=round(grounding_score, 4),
            passed=grounding_score >= 0.5,
            explanation="Evidence backing from semantic rules, external telemetry, or reasoning traces.",
        )

        # 4. Decision Quality
        # Policy compliance and safety verifier status
        policy_passed = result.policy_check.get("passed", True)
        dec_score = 1.0 if policy_passed else 0.0
        metrics["decision_quality"] = MetricScore(
            name="Decision Quality",
            score=round(dec_score, 4),
            passed=policy_passed,
            explanation="Compliance with active governance constraints and policy rules.",
        )

        # 5. Prediction Quality
        # Evaluates resolution time and binary outcome prediction plausibility
        preds = result.predicted_outcomes.get("predicted_outcome", {})
        pred_valid = (
            preds
            and not math.isnan(preds.get("binary_outcome_probability", 0.0))
            and 0.0 <= preds.get("binary_outcome_probability", 0.0) <= 1.0
        )
        pred_score = 1.0 if pred_valid else 0.5
        metrics["prediction_quality"] = MetricScore(
            name="Prediction Quality",
            score=round(pred_score, 4),
            passed=pred_score >= 0.7,
            explanation="Probabilistic validity of outcome predictions.",
        )

        # 6. Calibration Error (ECE)
        # Difference between confidence and calibration status
        calibrated = result.is_calibrated
        conf = result.confidence
        cal_conf = result.calibrated_confidence
        ece_diff = abs(conf - cal_conf)
        cal_score = max(0.0, 1.0 - ece_diff)
        metrics["calibration"] = MetricScore(
            name="Calibration",
            score=round(cal_score, 4),
            passed=cal_score >= 0.7,
            explanation=f"Calibration discrepancy: {round(ece_diff, 4)}.",
            details={"is_calibrated": calibrated, "ece_discrepancy": round(ece_diff, 4)},
        )

        # 7. Abstention Quality
        # Appropriately abstains when uncertainty is high or contradictions exist
        should_abstain = (
            gt.get("should_abstain", False)
            or result.calibrated_confidence < 0.35
            or bool(result.reasoning_evidence.contradictions_found and result.calibrated_confidence < 0.75)
        )
        if should_abstain:
            abst_score = 1.0 if result.abstain else 0.0
        else:
            abst_score = 1.0 if not result.abstain else 0.7
        metrics["abstention_quality"] = MetricScore(
            name="Abstention Quality",
            score=round(abst_score, 4),
            passed=abst_score >= 0.7,
            explanation="Metacognitive abstention appropriateness under uncertainty or contradictions.",
        )

        # 8. Planning Consistency
        plan_consistent = result.verification.get("checks", [{}])
        p_score = 1.0 if result.verification.get("verified", False) or result.abstain else 0.5
        metrics["planning_consistency"] = MetricScore(
            name="Planning Consistency",
            score=round(p_score, 4),
            passed=p_score >= 0.6,
            explanation="Consistency between selected action, plan steps, and constraints.",
        )

        # 9. Tool Execution Correctness
        if tool_results is not None:
            tool_success_rate = (
                sum(1 for t in tool_results if getattr(t, "success", False)) / max(1, len(tool_results))
            )
        else:
            tool_success_rate = 1.0  # N/A
        metrics["tool_execution_correctness"] = MetricScore(
            name="Tool Execution Correctness",
            score=round(tool_success_rate, 4),
            passed=tool_success_rate >= 0.8,
            explanation="Success proportion of executed tool calls without security violations.",
        )

        # 10. Latency and Resource Consumption
        lat = result.latency_ms
        lat_score = 1.0 if lat < 200.0 else (0.8 if lat < 1000.0 else 0.5)
        metrics["latency_and_resources"] = MetricScore(
            name="Latency & Resources",
            score=round(lat_score, 4),
            passed=lat < 2000.0,
            explanation=f"Processing latency ({lat:.2f} ms).",
            details={"latency_ms": lat},
        )

        overall = sum(m.score for m in metrics.values()) / len(metrics)
        verdict = "EXCELLENT" if overall >= 0.85 else ("ACCEPTABLE" if overall >= 0.70 else "DEGRADED")

        return ComprehensiveEvaluationReport(
            evaluation_id=f"eval_{int(time.time())}_{result.execution_id[:8]}",
            overall_quality_score=round(overall, 4),
            metrics=metrics,
            summary_verdict=verdict,
        )
