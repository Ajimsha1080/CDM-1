"""Procedural Memory Subsystem for CDM-1.
Stores and retrieves verified operational standard operating procedures (SOPs)
and workflows backed by persistent storage with tenant isolation.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from cdm.memory.storage import BaseMemoryStorage, SQLiteMemoryStorage


class OperationalProcedure(BaseModel):
    procedure_id: str
    tenant_id: str = "default_tenant"
    domain: str
    trigger_condition: str
    action_sequence: List[str]
    required_validations: List[str]
    version: str = "1.0.0"


class ProceduralMemory:
    """Stores validated multi-step operational runbooks and workflows with persistent storage."""

    def __init__(self, storage: Optional[BaseMemoryStorage] = None):
        self.storage = storage or SQLiteMemoryStorage()

    def register_procedure(self, proc: OperationalProcedure) -> None:
        self.storage.insert_procedure(
            proc_id=proc.procedure_id,
            tenant_id=proc.tenant_id,
            domain=proc.domain,
            proc_data=proc.model_dump(),
        )

    def get_procedure(self, proc_id: str, tenant_id: str = "default_tenant") -> Optional[OperationalProcedure]:
        data = self.storage.get_procedure(proc_id=proc_id, tenant_id=tenant_id)
        return OperationalProcedure.model_validate(data) if data else None

    def get_workflows_for_domain(self, domain: str, tenant_id: str = "default_tenant") -> List[Dict[str, Any]]:
        if hasattr(self.storage, "get_procedures_for_domain"):
            return self.storage.get_procedures_for_domain(tenant_id=tenant_id, domain=domain)
        return []
