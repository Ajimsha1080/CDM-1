"""Semantic Memory Subsystem for CDM-1.
Stores and queries domain facts, policy rules, and prohibited action constraints
backed by persistent storage with tenant isolation.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from cdm.memory.storage import BaseMemoryStorage, SQLiteMemoryStorage


class PolicyRule(BaseModel):
    rule_id: str
    tenant_id: str = "default_tenant"
    domain: str
    condition_description: str
    permitted_actions: List[str]
    prohibited_actions: List[str]
    version: str = "1.0.0"
    active: bool = True


class SemanticMemory:
    """Manages explicit verified operational rules and domain knowledge with persistent storage."""

    def __init__(self, storage: Optional[BaseMemoryStorage] = None):
        self.storage = storage or SQLiteMemoryStorage()

    def register_rule(self, rule: PolicyRule) -> None:
        self.storage.insert_rule(
            rule_id=rule.rule_id,
            tenant_id=rule.tenant_id,
            domain=rule.domain,
            rule_data=rule.model_dump(),
        )

    def query_rules_for_domain(self, domain: str, tenant_id: str = "default_tenant") -> List[PolicyRule]:
        raw_rules = self.storage.get_rules_for_domain(tenant_id=tenant_id, domain=domain)
        return [PolicyRule.model_validate(r) for r in raw_rules if r.get("active", True)]

    def get_prohibited_actions(self, domain: str, tenant_id: str = "default_tenant") -> List[str]:
        prohibited = set()
        for r in self.query_rules_for_domain(domain=domain, tenant_id=tenant_id):
            for act in r.prohibited_actions:
                prohibited.add(act)
        return list(prohibited)
