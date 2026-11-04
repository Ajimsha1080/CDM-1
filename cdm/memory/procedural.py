"""Procedural Memory Subsystem for CDM-1.
Stores verified operational standard operating procedures (SOPs) and decision sequences.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class OperationalProcedure(BaseModel):
    procedure_id: str
    domain: str
    trigger_condition: str
    action_sequence: List[str]
    required_validations: List[str]
    version: str = "1.0.0"


class ProceduralMemory:
    """Stores validated multi-step operational runbooks and workflows."""

    def __init__(self):
        self.procedures: Dict[str, OperationalProcedure] = {}

    def register_procedure(self, proc: OperationalProcedure) -> None:
        self.procedures[proc.procedure_id] = proc

    def get_procedure(self, proc_id: str) -> Optional[OperationalProcedure]:
        return self.procedures.get(proc_id)
