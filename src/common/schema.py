"""Strict input/output contract, JSON parser, and defensive prompt templates."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator


class AuditStatus(str, Enum):
    """Allowed security audit verdict states."""

    VULNERABLE = "VULNERABLE"
    SAFE = "SAFE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class VerificationAssertion(BaseModel):
    """Non-destructive HTTP status/schema diff and CI/CD pytest regression assertion."""

    expected_http_status: int = Field(..., ge=100, le=599)
    observed_http_status: int = Field(..., ge=100, le=599)
    status_diff_summary: str = Field(..., min_length=1)
    response_body_evidence: str = Field(..., min_length=1)
    pytest_regression_test: str = Field(..., min_length=1)


class SecurityAuditOutput(BaseModel):
    """Strict JSON output contract for the defensive API/Web security auditor."""

    status: Literal["VULNERABLE", "SAFE", "NEEDS_REVIEW"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    cwe_id: str = Field(..., min_length=1)
    owasp_category: str = Field(..., min_length=1)
    root_cause: str = Field(..., min_length=1)
    architectural_evidence: List[str] = Field(..., min_length=1)
    next_verification_step: str = Field(..., min_length=1)
    remediation: str = Field(..., min_length=1)
    verification_assertion: VerificationAssertion

    @field_validator("cwe_id")
    @classmethod
    def validate_cwe_id(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if cleaned in {"NONE", "N/A", "CWE-0"}:
            return "NONE"
        if not re.match(r"^CWE-\d+$", cleaned):
            raise ValueError(f"cwe_id must match 'CWE-<digits>' or 'NONE', got: {value!r}")
        return cleaned

    @field_validator("architectural_evidence")
    @classmethod
    def validate_evidence(cls, value: List[str]) -> List[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("architectural_evidence must contain at least one non-empty item")
        return cleaned


class SecuritySampleRecord(BaseModel):
    """Normalized training/evaluation record flowing through the data factory."""

    sample_id: str
    source: str
    domain: Literal[
        "adapter_auth_bola",
        "adapter_injection_validation",
        "adapter_logic_verifier",
    ]
    input_type: Literal["http_pair", "multi_turn_session", "openapi_schema", "graphql_schema", "code_snippet"]
    prompt_input: str
    is_vulnerable: bool
    cwe_id: str = "NONE"
    owasp_category: str = "NONE"
    title: str = ""
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)
    reasoning_trace: Optional[str] = None
    target_output: Optional[SecurityAuditOutput] = None


SYSTEM_PROMPT_DEFENSIVE_AUDITOR = (
    "You are an elite defensive web application security and API architecture auditor. "
    "Analyze the provided HTTP request/response pairs, multi-turn session traces, or OpenAPI/GraphQL schemas "
    "strictly from a defensive architectural perspective. Do NOT generate exploit payloads or attack instructions. "
    "First, write your step-by-step architectural reasoning inside <think>...</think> tags, evaluating "
    "authentication boundaries, authorization enforcement, input validation, state transitions, and potential "
    "false-positive indicators. Immediately after </think>, output ONLY a single valid JSON object matching "
    "the exact schema: "
    '{"status": "VULNERABLE | SAFE | NEEDS_REVIEW", "confidence": float, "cwe_id": "CWE-XXX or NONE", '
    '"owasp_category": "string", "root_cause": "string", "architectural_evidence": ["string"], '
    '"next_verification_step": "string", "remediation": "string", '
    '"verification_assertion": {"expected_http_status": int, "observed_http_status": int, '
    '"status_diff_summary": "string", "response_body_evidence": "string", "pytest_regression_test": "string"}}.'
)

THINK_AND_JSON_PATTERN = re.compile(
    r"<think>(?P<think>.*?)</think>\s*(?:```json\s*)?(?P<json>\{.*\})\s*(?:```)?\s*$",
    re.DOTALL,
)


def format_assistant_response(think_text: str, audit_output: SecurityAuditOutput) -> str:
    """Format a validated reasoning trace and JSON output into the strict assistant target."""
    clean_think = think_text.strip()
    json_str = audit_output.model_dump_json(indent=2)
    return f"<think>\n{clean_think}\n</think>\n{json_str}"


def _extract_balanced_json(text: str, start_pos: int = 0) -> Optional[str]:
    """Extract the first balanced JSON object `{...}` starting at or after `start_pos`."""
    idx = text.find("{", start_pos)
    while idx != -1:
        depth = 0
        in_string = False
        escape = False
        for pos in range(idx, len(text)):
            ch = text[pos]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return text[idx : pos + 1]
        idx = text.find("{", idx + 1)
    return None


def parse_model_response(raw_text: str) -> Tuple[Optional[str], Optional[SecurityAuditOutput], Optional[str]]:
    """Extract and validate <think>...</think> and strict JSON from model output.

    Rescues schema-valid JSON even if <think> tags are missing or trailing prose is present.

    Returns:
        (think_text, validated_output, error_message)
    """
    if not raw_text or not raw_text.strip():
        return None, None, "Empty model output"

    text = raw_text.strip()
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""
    search_from = think_match.end() if think_match else 0

    json_candidate = _extract_balanced_json(text, search_from)
    if json_candidate is None and search_from > 0:
        json_candidate = _extract_balanced_json(text, 0)
    if json_candidate is None:
        return think_text or None, None, "Missing valid JSON object"

    try:
        parsed_dict = json.loads(json_candidate)
    except json.JSONDecodeError as exc:
        return think_text or None, None, f"JSONDecodeError: {exc}"

    try:
        validated = SecurityAuditOutput.model_validate(parsed_dict)
    except ValidationError as exc:
        return think_text or None, None, f"SchemaValidationError: {exc}"

    if not think_text:
        think_text = "Reasoning trace omitted by model; JSON schema validated."

    return think_text, validated, None
