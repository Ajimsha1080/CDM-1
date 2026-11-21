"""Automated Test Suite for CDM-1 Cognitive Capability Benchmark Framework.
Validates all 8 capability dimensions:
1. Perception Benchmark
2. Memory Benchmark
3. Reasoning Benchmark
4. Causal Reasoning Benchmark
5. Planning Benchmark
6. Metacognition Benchmark
7. Learning Benchmark
8. Integrated Cognitive Benchmark
"""

import pytest
import os
import tempfile

from cdm.evaluation.benchmark import (
    PerceptionBenchmark,
    MemoryBenchmark,
    ReasoningBenchmark,
    CausalBenchmark,
    PlanningBenchmark,
    MetacognitionBenchmark,
    LearningBenchmark,
    IntegratedBenchmark,
    CDM1BenchmarkRunner,
)


@pytest.fixture(scope="module")
def shared_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        path = tf.name
    yield path
    if os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass


def test_1_perception_benchmark():
    """Validates perception benchmark: entity F1, intent accuracy, context signals."""
    b = PerceptionBenchmark()
    res = b.run()
    assert res.passed is True
    assert res.overall_score >= 0.80
    assert "entity_extraction" in res.metrics
    assert res.metrics["entity_extraction"].score >= 0.85
    assert res.metrics["intent_understanding"].score >= 0.85


def test_2_memory_benchmark(shared_db):
    """Validates memory benchmark: episodic recall, tenant isolation, semantic rules."""
    b = MemoryBenchmark(db_path=shared_db)
    res = b.run()
    assert res.passed is True
    assert res.overall_score >= 0.80
    # Invariant: Tenant isolation must be 100% verified
    assert res.metrics["episodic_recall"].details["tenant_isolation_preserved"] is True
    assert res.metrics["semantic_memory"].passed is True
    assert res.metrics["procedural_knowledge"].passed is True


def test_3_reasoning_benchmark():
    """Validates reasoning benchmark: multi-step deductions, logical consistency, constraint compliance."""
    b = ReasoningBenchmark()
    res = b.run()
    assert res.passed is True
    assert res.overall_score >= 0.80
    assert res.metrics["multi_step_reasoning"].details["deduction_steps_generated"] >= 2
    assert res.metrics["constraint_satisfaction"].passed is True


def test_4_causal_reasoning_benchmark():
    """Validates causal benchmark: mechanism identification, counterfactuals, confounder backdoor adjustment."""
    b = CausalBenchmark()
    res = b.run()
    assert res.passed is True
    assert res.overall_score >= 0.80
    assert res.metrics["cause_effect_identification"].score >= 0.80
    assert res.metrics["intervention_reasoning"].passed is True
    assert res.metrics["temporal_invariance"].passed is True


def test_5_planning_benchmark():
    """Validates planning benchmark: goal decomposition, constraint compliance, execution."""
    b = PlanningBenchmark()
    res = b.run()
    assert res.passed is True
    assert res.overall_score >= 0.80
    # Invariant: 0% prohibited steps in plan
    assert res.metrics["constraint_compliance"].details["prohibited_excluded"] is True
    assert res.metrics["plan_execution_success"].passed is True


def test_6_metacognition_benchmark():
    """Validates metacognition benchmark: abstention accuracy and uncertainty scaling."""
    b = MetacognitionBenchmark()
    res = b.run()
    assert res.passed is True
    assert res.overall_score >= 0.80
    assert res.metrics["uncertainty_recognition"].passed is True


def test_7_learning_benchmark(shared_db):
    """Validates learning benchmark: user opinion segregation, failure retention, dataset staging."""
    b = LearningBenchmark(db_path=shared_db)
    res = b.run()
    assert res.passed is True
    assert res.overall_score >= 0.80
    # Invariant: User opinions must NEVER be eligible for automatic fine-tuning
    assert res.metrics["feedback_segregation"].details["opinion_eligible_for_training"] is False
    assert res.metrics["mistake_recording"].passed is True
    assert res.metrics["memory_consolidation"].passed is True


def test_8_integrated_cognitive_benchmark():
    """Validates integrated cognitive benchmark across multi-turn scenarios."""
    b = IntegratedBenchmark()
    res = b.run()
    assert res.passed is True
    assert res.overall_score >= 0.70
    assert res.metrics["safety_invariance"].passed is True


def test_9_master_benchmark_runner(shared_db):
    """Validates full master runner output structure and execution."""
    runner = CDM1BenchmarkRunner(use_neural=False, db_path=shared_db)
    report = runner.run_all()
    assert report["all_benchmarks_passed"] is True
    assert report["aggregate_cognitive_score"] >= 0.80
    assert len(report["categories"]) == 8
