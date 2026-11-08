"""Layer 1: Strict LLM Output Validation and JSON Repair Gate.
Treats all foundation model output as untrusted caller data.
Validates structured output against expected schemas and performs at most
one automatic repair prompt on malformed JSON before cleanly rejecting.
"""

from typing import Dict, Any, Tuple, Optional, Union, Type, List, Callable
import json
import re
import logging
from pydantic import BaseModel, ValidationError

logger = logging.getLogger("cdm.llm.validator")


class LLMValidationError(ValueError):
    """Raised when LLM output fails schema validation or produces unrepairable JSON."""
    pass


def extract_json_candidate(raw_text: str) -> Optional[str]:
    """Extracts JSON substring from raw model output, handling markdown fences or leading prose."""
    if not raw_text:
        return None

    # Check for markdown code blocks ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if match:
        return match.group(1).strip()

    # Find outermost curly braces
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw_text[start : end + 1].strip()

    # Find outermost square brackets for array responses
    start_arr = raw_text.find("[")
    end_arr = raw_text.rfind("]")
    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        return raw_text[start_arr : end_arr + 1].strip()

    return raw_text.strip()


def validate_json_against_schema(
    raw_text: str,
    schema: Union[Dict[str, Any], Type[BaseModel]],
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """Strictly parses and validates raw text against a Pydantic model or dict schema.
    Returns:
        (is_valid, parsed_dict, error_message)
    """
    candidate = extract_json_candidate(raw_text)
    if not candidate:
        return False, None, "Empty response or no JSON structure detected."

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as err:
        return False, None, f"JSON parse error: {err}"

    if not isinstance(parsed, dict) and not isinstance(parsed, list):
        return False, None, f"Expected JSON object or array, got {type(parsed).__name__}"

    # Validate against Pydantic schema
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        try:
            if isinstance(parsed, dict):
                model_inst = schema.model_validate(parsed)
                return True, model_inst.model_dump(), None
            else:
                return False, None, "Expected dict matching Pydantic model schema, got list."
        except ValidationError as err:
            return False, parsed if isinstance(parsed, dict) else None, f"Schema validation error: {err}"

    # Validate against dict of required field types
    if isinstance(schema, dict) and "required" in schema:
        req_fields = schema.get("required", [])
        if isinstance(parsed, dict):
            missing = [f for f in req_fields if f not in parsed]
            if missing:
                return False, parsed, f"Missing required fields: {missing}"

    return True, parsed if isinstance(parsed, dict) else {"items": parsed}, None


def validate_and_repair_json(
    raw_text: str,
    schema: Union[Dict[str, Any], Type[BaseModel]],
    messages: List[Any],
    call_fn: Callable[[List[Any]], Tuple[str, int, int]],
) -> Tuple[Optional[Dict[str, Any]], bool, Optional[str], bool, int, int]:
    """Validates raw output against schema. If malformed, executes exactly ONE repair prompt.
    Returns:
        (parsed_dict, is_valid, error_message, was_repaired, repair_in_tok, repair_out_tok)
    """
    is_valid, parsed, err = validate_json_against_schema(raw_text, schema)
    if is_valid:
        return parsed, True, None, False, 0, 0

    logger.warning(f"Initial LLM output failed validation ({err}). Attempting single JSON repair...")

    schema_repr = schema.model_json_schema() if (isinstance(schema, type) and issubclass(schema, BaseModel)) else str(schema)

    repair_prompt = (
        f"Your previous response was malformed or failed schema validation with error:\n"
        f"\"{err}\"\n\n"
        f"Previous output:\n{raw_text}\n\n"
        f"Please repair this output and output ONLY valid, RFC-8259 compliant JSON matching this schema:\n"
        f"{schema_repr}\n\n"
        f"Do NOT include conversational prose, markdown explanation, or preamble. Output raw JSON only."
    )

    from cdm.llm.base import LLMMessage
    repair_messages = list(messages) + [
        LLMMessage(role="assistant", content=raw_text),
        LLMMessage(role="user", content=repair_prompt),
    ]

    try:
        repaired_text, r_in, r_out = call_fn(repair_messages)
        rep_valid, rep_parsed, rep_err = validate_json_against_schema(repaired_text, schema)
        if rep_valid:
            logger.info("JSON repair prompt succeeded.")
            return rep_parsed, True, None, True, r_in, r_out
        else:
            logger.error(f"JSON repair failed second check: {rep_err}")
            return rep_parsed, False, f"Output unrepairable after retry: {rep_err}", True, r_in, r_out
    except Exception as exc:
        logger.error(f"Exception during LLM JSON repair execution: {exc}")
        return None, False, f"Repair call failed: {exc}", True, 0, 0
