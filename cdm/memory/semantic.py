"""Semantic Memory Subsystem for CDM-1.
Stores domain facts, policy rules, and conceptual knowledge graphs.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class PolicyRule(BaseModel):
    rule_id: str
    domain: str
    condition_description: str
    permitted_actions: List[str]
    prohibited_actions: List[str]
    version: str = "1.0.0"
    active: bool = True


class SemanticMemory:
    """Manages explicit verified operational rules and domain knowledge."""

    def __init__(self):
        self.rules: Dict[str, PolicyRule] = {}

    def register_rule(self, rule: PolicyRule) -> None:
        self.rules[rule.rule_id] = rule

    def query_rules_for_domain(self, domain: str) -> List[PolicyRule]:
        return [r for r in self.rules.values() if r.active and (r.domain == domain or r.domain == "global")]

    def get_prohibited_actions(self, domain: str) -> List[str]:
        prohibited = set()
        for r in self.query_rules_for_domain(domain):
            for act in r.prohibited_actions:
                prohibited.add(act)
        return list(prohibited)
