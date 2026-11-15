"""CDM-1 Cognitive Answer Synthesis Layer."""

from cdm.synthesis.answer_synthesizer import (
    AnswerSynthesizer,
    SynthesizedAnswer,
    ExternalInformationRetriever,
)
from cdm.synthesis.router import IntentTaskRouter, CognitiveRoutePlan
from cdm.synthesis.document_retriever import DocumentKnowledgeRetriever, DocumentPassage
from cdm.synthesis.tool_orchestrator import ControlledToolOrchestrator, ToolExecutionRecord
from cdm.synthesis.citation_manager import CitationManager, GroundedCitation, CitationVerificationReport
from cdm.synthesis.answer_validator import AnswerValidator, AnswerValidationResult
from cdm.context.reference_resolver import ReferenceResolver

__all__ = [
    "AnswerSynthesizer",
    "SynthesizedAnswer",
    "ExternalInformationRetriever",
    "IntentTaskRouter",
    "CognitiveRoutePlan",
    "DocumentKnowledgeRetriever",
    "DocumentPassage",
    "ControlledToolOrchestrator",
    "ToolExecutionRecord",
    "CitationManager",
    "GroundedCitation",
    "CitationVerificationReport",
    "AnswerValidator",
    "AnswerValidationResult",
    "ReferenceResolver",
]
