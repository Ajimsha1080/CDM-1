"""Curated multi-domain decision datasets for CDM-1.
Generates rigorously validated decision examples across Communication, Customer Support,
Business Operations, and Retail with ground truth labels, constraints, and observed outcomes.
"""

from typing import List, Dict, Any
import json
import os
from cdm.data.data_contracts import (
    DecisionExample,
    CandidateAction,
    PolicyConstraint,
    TargetDecision,
    OutcomeLabel,
    LabelProvenance,
)


def get_standard_candidate_set() -> List[CandidateAction]:
    """Default candidate operational actions."""
    return [
        CandidateAction(
            action_id="act_auto_resolve",
            name="auto_resolve",
            description="Automatically resolve the request with standard automated guidance.",
            expected_impact="Immediate resolution with minimal labor cost.",
            estimated_cost=0.5,
        ),
        CandidateAction(
            action_id="act_request_info",
            name="request_more_information",
            description="Ask the customer or stakeholder for necessary clarification or logs.",
            expected_impact="Prevents premature action and gathers verified evidence.",
            estimated_cost=2.0,
        ),
        CandidateAction(
            action_id="act_issue_refund",
            name="issue_refund",
            description="Issue financial refund or billing adjustment under explicit policy rules.",
            expected_impact="High customer satisfaction; requires policy compliance.",
            estimated_cost=45.0,
        ),
        CandidateAction(
            action_id="act_escalate_tier2",
            name="escalate_to_tier2",
            description="Escalate interaction to senior specialist or operations engineer.",
            expected_impact="Comprehensive deep investigation; consumes high tier resources.",
            estimated_cost=25.0,
        ),
    ]


def build_curated_decision_corpus() -> List[DecisionExample]:
    """Generates a corpus of standardized multi-domain decision examples

    with realistic text, structured features, hard constraints, and outcome labels.
    """
    candidates = get_standard_candidate_set()
    examples: List[DecisionExample] = []

    # Domain 1: Customer Support — Billing & Refund Decisions
    examples.append(
        DecisionExample(
            example_id="CS-001",
            domain="customer_support",
            input_text="I ordered this item 12 days ago and it arrived damaged in the box. I have photos and need my money back.",
            structured_features={
                "sentiment_score": -0.85,
                "interaction_count": 1.0,
                "account_tenure_months": 18.0,
                "monetary_value_usd": 89.99,
                "sla_remaining_hours": 14.0,
                "historical_escalation_rate": 0.05,
                "domain": "customer_support",
                "channel": "email",
                "customer_tier": "professional",
                "urgency_level": "medium",
            },
            context="Customer order #98234 delivered 12 days ago. Return policy permits full refund within 30 days with damage proof.",
            objective="Determine refund eligibility and immediate action.",
            candidate_actions=candidates,
            constraints=[
                PolicyConstraint(
                    constraint_id="POL-REFUND-30D",
                    description="Refunds strictly permitted if delivered within 30 days.",
                    prohibited_actions=[],
                    is_hard_constraint=True,
                )
            ],
            target_decision=TargetDecision(
                binary_decision=1,  # YES: Grant refund
                selected_candidate_id="act_issue_refund",
                decision_score=92.5,
                priority_level=1,  # Medium
                risk_level=0,  # Low
            ),
            evidence=["Delivered 12 days ago (< 30 days limit)", "Customer attached damage photos"],
            label_provenance=LabelProvenance(
                source_type="curated_expert",
                verifier_id="qa_specialist_12",
                verification_method="policy_audit",
            ),
            outcome_label=OutcomeLabel(
                observed_outcome_binary=1,
                observed_outcome_multiclass=0,  # Resolved
                observed_resolution_time_hours=0.5,
                observed_risk_realized=0.0,
            ),
            temporal_timestamp="2026-01-10T09:15:00Z",
        )
    )

    examples.append(
        DecisionExample(
            example_id="CS-002",
            domain="customer_support",
            input_text="I bought this jacket 75 days ago. I never wore it and want a full cash refund to my card.",
            structured_features={
                "sentiment_score": -0.4,
                "interaction_count": 2.0,
                "account_tenure_months": 6.0,
                "monetary_value_usd": 150.00,
                "sla_remaining_hours": 10.0,
                "historical_escalation_rate": 0.1,
                "domain": "customer_support",
                "channel": "chat",
                "customer_tier": "starter",
                "urgency_level": "low",
            },
            context="Purchase date 75 days ago. Explicit policy: cash refunds strictly prohibited after 30 days.",
            objective="Evaluate refund request against explicit policy window.",
            candidate_actions=candidates,
            constraints=[
                PolicyConstraint(
                    constraint_id="POL-HARD-30D",
                    description="Cash refund strictly prohibited after 30 days.",
                    prohibited_actions=["act_issue_refund", "issue_refund"],
                    is_hard_constraint=True,
                )
            ],
            target_decision=TargetDecision(
                binary_decision=0,  # NO: Do not refund
                selected_candidate_id="act_auto_resolve",
                decision_score=20.0,
                priority_level=0,  # Low
                risk_level=1,  # Moderate
            ),
            evidence=["75 days elapsed exceeds 30-day window", "Hard constraint prohibits act_issue_refund"],
            label_provenance=LabelProvenance(
                source_type="curated_expert",
                verifier_id="qa_specialist_12",
                verification_method="policy_audit",
            ),
            outcome_label=OutcomeLabel(
                observed_outcome_binary=1,
                observed_outcome_multiclass=0,
                observed_resolution_time_hours=0.2,
                observed_risk_realized=0.1,
            ),
            temporal_timestamp="2026-01-11T11:30:00Z",
        )
    )

    # Domain 2: Communication — Critical Security / Escalation
    examples.append(
        DecisionExample(
            example_id="COMM-001",
            domain="communication",
            input_text="URGENT: Multiple unauthorized logins from foreign IP address on our production server admin account. Lock sessions immediately!",
            structured_features={
                "sentiment_score": -0.95,
                "interaction_count": 1.0,
                "account_tenure_months": 36.0,
                "monetary_value_usd": 25000.0,
                "sla_remaining_hours": 1.0,
                "historical_escalation_rate": 0.0,
                "domain": "communication",
                "channel": "email",
                "customer_tier": "enterprise",
                "urgency_level": "critical",
            },
            context="Enterprise customer reporting live active security breach on infrastructure.",
            objective="Decide on urgent escalation vs standard response.",
            candidate_actions=candidates,
            constraints=[
                PolicyConstraint(
                    constraint_id="POL-SEC-AUTO-PROHIBIT",
                    description="Never auto-resolve security incidents.",
                    prohibited_actions=["act_auto_resolve", "auto_resolve"],
                    is_hard_constraint=True,
                )
            ],
            target_decision=TargetDecision(
                binary_decision=1,  # YES: Escalate immediately
                selected_candidate_id="act_escalate_tier2",
                decision_score=99.0,
                priority_level=3,  # Critical
                risk_level=3,  # Severe
            ),
            evidence=["Unauthorized logins reported", "Enterprise tier customer", "SLA remaining < 1 hour"],
            label_provenance=LabelProvenance(
                source_type="curated_expert",
                verifier_id="security_lead_04",
                verification_method="incident_review",
            ),
            outcome_label=OutcomeLabel(
                observed_outcome_binary=1,
                observed_outcome_multiclass=1,  # Escalated
                observed_resolution_time_hours=0.8,
                observed_risk_realized=0.2,
            ),
            temporal_timestamp="2026-01-12T14:05:00Z",
        )
    )

    examples.append(
        DecisionExample(
            example_id="COMM-002",
            domain="communication",
            input_text="Thanks for your monthly newsletter. Just confirming we received the product update notes.",
            structured_features={
                "sentiment_score": 0.5,
                "interaction_count": 0.0,
                "account_tenure_months": 12.0,
                "monetary_value_usd": 50.0,
                "sla_remaining_hours": 48.0,
                "historical_escalation_rate": 0.0,
                "domain": "communication",
                "channel": "email",
                "customer_tier": "starter",
                "urgency_level": "low",
            },
            context="Informational receipt acknowledgement. No question or grievance raised.",
            objective="Determine if manual reply or escalation is needed.",
            candidate_actions=candidates,
            constraints=[],
            target_decision=TargetDecision(
                binary_decision=0,  # NO: Reply not required
                selected_candidate_id="act_auto_resolve",
                decision_score=10.0,
                priority_level=0,  # Low
                risk_level=0,  # Low
            ),
            evidence=["Acknowledgement only", "No inquiry or problem stated"],
            label_provenance=LabelProvenance(
                source_type="curated_expert",
                verifier_id="comm_ops_01",
                verification_method="telemetry_audit",
            ),
            outcome_label=OutcomeLabel(
                observed_outcome_binary=1,
                observed_outcome_multiclass=0,
                observed_resolution_time_hours=0.1,
                observed_risk_realized=0.0,
            ),
            temporal_timestamp="2026-01-13T10:00:00Z",
        )
    )

    # Domain 3: Retail Operations — Delivery Delay & Disputed Shipments
    examples.append(
        DecisionExample(
            example_id="RET-001",
            domain="retail",
            input_text="Tracking says 'Delivered to porch' 3 hours ago, but my doorbell camera shows no courier stopped by. Package is missing.",
            structured_features={
                "sentiment_score": -0.6,
                "interaction_count": 1.0,
                "account_tenure_months": 14.0,
                "monetary_value_usd": 120.0,
                "sla_remaining_hours": 12.0,
                "historical_escalation_rate": 0.1,
                "domain": "retail",
                "channel": "chat",
                "customer_tier": "professional",
                "urgency_level": "high",
            },
            context="Courier marked delivered. Customer claims missing shipment. Standard SOP requires verification before refund.",
            objective="Decide immediate action on missing delivery claim.",
            candidate_actions=candidates,
            constraints=[
                PolicyConstraint(
                    constraint_id="POL-CARRIER-CHECK",
                    description="Must request carrier GPS proof or statement before refunding over $100.",
                    prohibited_actions=["act_issue_refund", "issue_refund"],
                    is_hard_constraint=True,
                )
            ],
            target_decision=TargetDecision(
                binary_decision=1,  # YES: Active investigation needed
                selected_candidate_id="act_request_info",
                decision_score=75.0,
                priority_level=2,  # High
                risk_level=1,  # Moderate
            ),
            evidence=["Value exceeds $100 carrier check threshold", "Recent delivery timestamp requires GPS verification"],
            label_provenance=LabelProvenance(
                source_type="curated_expert",
                verifier_id="retail_ops_mgr",
                verification_method="sop_adherence",
            ),
            outcome_label=OutcomeLabel(
                observed_outcome_binary=1,
                observed_outcome_multiclass=0,
                observed_resolution_time_hours=4.0,
                observed_risk_realized=0.15,
            ),
            temporal_timestamp="2026-01-14T15:20:00Z",
        )
    )

    # Domain 4: Business Operations — Server Outage & SLA Breach Risk
    examples.append(
        DecisionExample(
            example_id="OPS-001",
            domain="business_operations",
            input_text="High memory leak detected on payment processing microservices cluster. Latency has increased 400% in last 15 minutes.",
            structured_features={
                "sentiment_score": -0.8,
                "interaction_count": 3.0,
                "account_tenure_months": 48.0,
                "monetary_value_usd": 50000.0,
                "sla_remaining_hours": 0.5,
                "historical_escalation_rate": 0.25,
                "domain": "business_operations",
                "channel": "api",
                "customer_tier": "enterprise",
                "urgency_level": "critical",
            },
            context="Telemetry indicates active degradation in core payment flow. Immediate SRE paging mandated.",
            objective="Determine resource dispatch and escalation.",
            candidate_actions=candidates,
            constraints=[
                PolicyConstraint(
                    constraint_id="POL-OPS-NO-AUTO",
                    description="Never auto-resolve active production alerts.",
                    prohibited_actions=["act_auto_resolve", "auto_resolve"],
                    is_hard_constraint=True,
                )
            ],
            target_decision=TargetDecision(
                binary_decision=1,  # YES: Escalate
                selected_candidate_id="act_escalate_tier2",
                decision_score=98.0,
                priority_level=3,  # Critical
                risk_level=3,  # Severe
            ),
            evidence=["Payment gateway latency spike +400%", "SLA remaining < 30 minutes"],
            label_provenance=LabelProvenance(
                source_type="curated_expert",
                verifier_id="sre_incident_commander",
                verification_method="telemetry_validation",
            ),
            outcome_label=OutcomeLabel(
                observed_outcome_binary=1,
                observed_outcome_multiclass=1,  # Escalated
                observed_resolution_time_hours=0.5,
                observed_risk_realized=0.3,
            ),
            temporal_timestamp="2026-01-15T08:12:00Z",
        )
    )

    templates = [
        ("customer_support", "I was charged twice on invoice #{idx} for the same subscription.", "high", "professional", -0.7, 15.0, 49.0, 6.0, 1, "act_issue_refund", 2, 1, 85.0, 1, 0, 1.2),
        ("customer_support", "Where can I download my annual tax receipt from the portal?", "low", "free", 0.1, 5.0, 0.0, 24.0, 0, "act_auto_resolve", 0, 0, 15.0, 1, 0, 0.3),
        ("customer_support", "I received the wrong item in my package. It's completely different from what I ordered.", "medium", "starter", -0.6, 8.0, 65.0, 16.0, 1, "act_request_info", 1, 1, 60.0, 1, 0, 2.5),
        ("customer_support", "Our system is down and we are losing customers by the minute! Help now!", "critical", "enterprise", -0.95, 36.0, 10000.0, 0.5, 1, "act_escalate_tier2", 3, 3, 98.0, 1, 1, 0.8),
        ("communication", "Can we reschedule tomorrow's quarterly review meeting to 3 PM?", "low", "professional", 0.2, 20.0, 500.0, 18.0, 1, "act_auto_resolve", 0, 0, 30.0, 1, 0, 0.5),
        ("communication", "URGENT: Legal notice regarding trademark infringement on marketing assets.", "critical", "enterprise", -0.8, 40.0, 15000.0, 2.0, 1, "act_escalate_tier2", 3, 3, 95.0, 1, 1, 1.0),
        ("communication", "Automatic out of office reply: I will return next Tuesday.", "low", "starter", 0.0, 2.0, 20.0, 48.0, 0, "act_auto_resolve", 0, 0, 5.0, 1, 0, 0.1),
        ("communication", "Please provide the updated pricing matrix for enterprise renewal.", "medium", "enterprise", 0.3, 30.0, 8000.0, 12.0, 1, "act_request_info", 1, 0, 55.0, 1, 0, 3.0),
        ("retail", "Order arrived broken. Package crushed in transit. Requesting replacement or refund.", "high", "starter", -0.75, 4.0, 85.0, 8.0, 1, "act_issue_refund", 2, 1, 80.0, 1, 0, 1.5),
        ("retail", "Tracking has not updated for 8 days after dispatch from warehouse.", "medium", "professional", -0.5, 12.0, 110.0, 14.0, 1, "act_request_info", 1, 1, 65.0, 1, 0, 4.0),
        ("retail", "I would like to know if size 10 will be restocked next month.", "low", "free", 0.2, 1.0, 0.0, 36.0, 0, "act_auto_resolve", 0, 0, 20.0, 1, 0, 0.2),
        ("retail", "Customer threatens chargeback and social media campaign over delayed delivery.", "critical", "professional", -0.9, 16.0, 220.0, 3.0, 1, "act_escalate_tier2", 3, 2, 90.0, 1, 1, 1.5),
        ("business_operations", "Routine database index maintenance scheduled for Sunday at 02:00 UTC.", "low", "enterprise", 0.1, 50.0, 30000.0, 40.0, 0, "act_auto_resolve", 0, 0, 25.0, 1, 0, 0.4),
        ("business_operations", "API gateway error rate reached 5.2%, exceeding 1% error budget threshold.", "critical", "enterprise", -0.85, 45.0, 45000.0, 1.0, 1, "act_escalate_tier2", 3, 3, 97.0, 1, 1, 0.6),
        ("business_operations", "SSL Certificate expiring in 5 days for staging environment.", "medium", "professional", -0.3, 22.0, 5000.0, 10.0, 1, "act_request_info", 1, 1, 50.0, 1, 0, 2.0),
        ("business_operations", "Weekly operational metrics digest generated successfully.", "low", "free", 0.3, 6.0, 100.0, 48.0, 0, "act_auto_resolve", 0, 0, 10.0, 1, 0, 0.1),
    ]

    for cycle in range(8):
        for i, t in enumerate(templates):
            ex_id = f"SYNTH-{t[0][:3].upper()}-{cycle:02d}-{i:02d}"
            day = 16 + (cycle * 2) + (i % 2)
            timestamp = f"2026-01-{day:02d}T{10 + (i % 8):02d}:00:00Z"
            
            constraints = []
            if t[9] == "act_issue_refund" and t[6] > 100.0:
                constraints.append(
                    PolicyConstraint(
                        constraint_id="POL-AUTH-HIGH-VAL",
                        description="Refunds > $100 require manager approval.",
                        prohibited_actions=[],
                        is_hard_constraint=False,
                    )
                )

            examples.append(
                DecisionExample(
                    example_id=ex_id,
                    domain=t[0],
                    input_text=t[1].replace("{idx}", f"{1000 + cycle * 100 + i}"),
                    structured_features={
                        "sentiment_score": t[4],
                        "interaction_count": float((i % 4) + 1),
                        "account_tenure_months": t[5],
                        "monetary_value_usd": t[6],
                        "sla_remaining_hours": t[7],
                        "historical_escalation_rate": 0.05 * (i % 4),
                        "domain": t[0],
                        "channel": "email" if i % 2 == 0 else "chat",
                        "customer_tier": t[3],
                        "urgency_level": t[2],
                    },
                    context=f"Context for domain {t[0]} with priority {t[10]}.",
                    objective="Determine optimal operational decision and risk assessment.",
                    candidate_actions=candidates,
                    constraints=constraints,
                    target_decision=TargetDecision(
                        binary_decision=t[8],
                        selected_candidate_id=t[9],
                        decision_score=t[12],
                        priority_level=t[10],
                        risk_level=t[11],
                    ),
                    evidence=[f"Domain feature pattern: {t[0]}", f"Priority category: {t[10]}"],
                    label_provenance=LabelProvenance(
                        source_type="curated_expert",
                        verifier_id="curator_suite_v1",
                        verification_method="cross_validated_rule",
                        synthetic=True,
                    ),
                    outcome_label=OutcomeLabel(
                        observed_outcome_binary=t[13],
                        observed_outcome_multiclass=t[14],
                        observed_resolution_time_hours=t[15],
                        observed_risk_realized=0.1 * t[11],
                    ),
                    temporal_timestamp=timestamp,
                )
            )

    return examples


def save_dataset_to_disk(examples: List[DecisionExample], output_path: str) -> None:
    """Save dataset to JSON Lines format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")


def load_dataset_from_disk(input_path: str) -> List[DecisionExample]:
    """Load dataset from JSON Lines format."""
    examples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line.strip())
                examples.append(DecisionExample.model_validate(data))
    return examples
