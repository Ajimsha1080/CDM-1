"""Tests for CDM-1 Phase 3: Real Image Multimodal Processing, Adaptation Gatekeeper, and Cognitive Evaluation."""

import io
from PIL import Image
import pytest
from cdm.perception.multimodal_processor import MultimodalProcessor
from cdm.learning.adaptation import AdaptationGatekeeper
from cdm.evaluation.evaluator import CognitiveEvaluator
from cdm.cognitive.state import CognitiveEngineResult, ReasoningTrace


def test_multimodal_processor_real_image_bytes():
    """Verify MultimodalProcessor parses raw PNG image bytes using PIL."""
    # Create an in-memory 120x80 RGB test image
    img = Image.new("RGB", (120, 80), color=(73, 109, 137))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_png_bytes = buf.getvalue()

    processor = MultimodalProcessor()
    res = processor.process_image(
        image_data=raw_png_bytes,
        image_name="test_screenshot.png",
        image_metadata={"ocr_text": "Error Code 500: Internal Server Outage"},
    )

    assert res.modality == "image"
    assert res.extraction_status == "success"
    assert res.extracted_properties["resolution"] == "120x80"
    assert res.extracted_properties["format"] == "PNG"
    assert res.extracted_properties["color_mode"] == "RGB"
    assert "Error Code 500" in res.extracted_text
    assert res.extraction_confidence >= 0.95


def test_adaptation_gatekeeper_multi_task_regression_gate():
    """Verify AdaptationGatekeeper approves compliant checkpoints and rejects policy/accuracy regressions."""
    gatekeeper = AdaptationGatekeeper()

    baseline = {
        "binary_decision": {"accuracy": 0.95, "macro_f1": 0.94},
        "candidate_choice": {"prohibited_violation_rate": 0.0, "ndcg": 0.98},
    }

    # Case 1: Healthy candidate
    candidate_good = {
        "binary_decision": {"accuracy": 0.955, "macro_f1": 0.945},
        "candidate_choice": {"prohibited_violation_rate": 0.0, "ndcg": 0.985},
    }
    approved, reasons = gatekeeper.evaluate_regression_gate(candidate_good, baseline)
    assert approved is True
    assert len(reasons) == 0

    # Case 2: Policy violation candidate (strictly rejected)
    candidate_bad_policy = {
        "binary_decision": {"accuracy": 0.96, "macro_f1": 0.95},
        "candidate_choice": {"prohibited_violation_rate": 0.02, "ndcg": 0.98},
    }
    approved_bad, reasons_bad = gatekeeper.evaluate_regression_gate(candidate_bad_policy, baseline)
    assert approved_bad is False
    assert any("SAFETY POLICY VIOLATION" in r for r in reasons_bad)

    # Case 3: Accuracy regression candidate
    candidate_regressed = {
        "binary_decision": {"accuracy": 0.91, "macro_f1": 0.90},
        "candidate_choice": {"prohibited_violation_rate": 0.0, "ndcg": 0.98},
    }
    approved_reg, reasons_reg = gatekeeper.evaluate_regression_gate(candidate_regressed, baseline, max_allowed_accuracy_drop=0.02)
    assert approved_reg is False
    assert any("accuracy regressed" in r for r in reasons_reg)


def test_cognitive_evaluator_10_dimensions():
    """Verify CognitiveEvaluator scores cognitive responses across core reliability dimensions."""
    evaluator = CognitiveEvaluator()

    dummy_result = CognitiveEngineResult(
        decision="standard_healthcheck",
        confidence=0.92,
        calibrated_confidence=0.91,
        priority="LOW",
        risk=0.05,
        execution_status="authorized",
        safety_status="verified",
        answer="Executing standard_healthcheck to verify cluster status.",
        reasoning_evidence=ReasoningTrace(summary="Routine diagnostic healthcheck"),
        latency_ms=15.0,
    )

    report = evaluator.evaluate(
        inquiry_text="Should we execute standard_healthcheck on cluster node?",
        result=dummy_result,
        ground_truth={"target_decision": "standard_healthcheck"},
    )

    assert report.overall_quality_score > 0.70
    assert report.summary_verdict in ["EXCELLENT", "ACCEPTABLE"]
    assert "qa_relevance" in report.metrics
    assert "decision_quality" in report.metrics
    assert report.metrics["decision_quality"].passed is True
