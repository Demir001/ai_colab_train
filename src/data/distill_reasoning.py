"""Dual-mode reasoning generator: Async OpenAI-compatible distillation & Offline deterministic synthesis."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

from src.common.schema import (
    SYSTEM_PROMPT_DEFENSIVE_AUDITOR,
    SecurityAuditOutput,
    SecuritySampleRecord,
    VerificationAssertion,
    parse_model_response,
)

logger = logging.getLogger(__name__)

EXPERT_DOMAINS = (
    "adapter_auth_bola",
    "adapter_injection_validation",
    "adapter_logic_verifier",
)

REMEDIATION_TEMPLATES: Dict[str, str] = {
    "CWE-639": (
        "Enforce server-side object ownership and tenant scoping on every repository query "
        "(e.g., `WHERE id = :resource_id AND tenant_id = :auth_tenant_id`). Never trust client-supplied "
        "IDs without verifying the authenticated subject's RBAC/ABAC policy."
    ),
    "CWE-285": (
        "Apply explicit role-based authorization middleware on administrative routes, deny by default, "
        "and declare non-empty security schemes in OpenAPI specifications."
    ),
    "CWE-287": (
        "Validate JWT signatures against pinned asymmetric keys (RS256/EdDSA), enforce `iss`/`aud`/`exp` "
        "claims, and reject `alg=none` or unauthenticated fallback paths."
    ),
    "CWE-918": (
        "Implement an outbound URL allowlist, resolve DNS prior to connection, and block loopback, "
        "link-local (169.254.0.0/16), and RFC1918 private IP ranges at the network egress layer."
    ),
    "CWE-89": (
        "Use parameterized prepared statements or ORM bind variables exclusively; enforce strict schema "
        "validation and least-privilege database roles."
    ),
    "CWE-79": (
        "Apply context-aware output encoding, enforce a strict Content-Security-Policy (CSP), "
        "and avoid raw HTML interpolation in HTTP responses."
    ),
    "CWE-362": (
        "Wrap state-changing business flows in serializable database transactions with row-level locks "
        "(`SELECT ... FOR UPDATE`) and enforce idempotency keys with unique constraints."
    ),
}


def _extract_http_telemetry(prompt_input: str, is_vulnerable: bool, cwe_id: str) -> Dict[str, Any]:
    """Extract actual HTTP method, path, parameter names, and status codes from input text."""
    req_match = re.search(
        r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/[^\s?\"']+)(?:\?([^\s\"']+))?",
        prompt_input,
    )
    method = req_match.group(1) if req_match else "POST" if "graphql" in prompt_input.lower() else "GET"
    path = req_match.group(2) if req_match else "/graphql" if "graphql" in prompt_input.lower() else "/api/v1/items"
    query_str = req_match.group(3) if (req_match and req_match.group(3)) else ""

    query_params = [kv.split("=")[0] for kv in query_str.split("&") if "=" in kv]
    json_keys = re.findall(r'"([a-zA-Z_][a-zA-Z0-9_]{1,32})"\s*:', prompt_input)
    param_names = list(dict.fromkeys(query_params + json_keys))[:4]

    statuses = [int(s) for s in re.findall(r"\bHTTP/\d\.\d\s+(\d{3})\b", prompt_input)]
    observed_status = statuses[-1] if statuses else (200 if is_vulnerable else 403)

    if not is_vulnerable:
        expected_status = observed_status
    elif cwe_id in {"CWE-20", "CWE-79", "CWE-89", "CWE-918", "CWE-22"}:
        expected_status = 422
    elif cwe_id in {"CWE-362", "CWE-840"}:
        expected_status = 409
    elif cwe_id in {"CWE-287", "CWE-347"}:
        expected_status = 401
    else:
        expected_status = 403

    return {
        "method": method,
        "path": path,
        "params": param_names,
        "observed_status": observed_status,
        "expected_status": expected_status,
    }


def _build_dynamic_pytest(
    telemetry: Dict[str, Any],
    is_vulnerable: bool,
    cwe_id: str,
) -> str:
    """Construct a non-destructive, input-specific pytest CI/CD regression assertion."""
    method_lower = telemetry["method"].lower()
    path = telemetry["path"]
    params = telemetry["params"]
    expected = telemetry["expected_status"]
    safe_fn_suffix = re.sub(r"[^a-z0-9_]+", "_", f"{method_lower}_{path}".lower()).strip("_")[:42]

    if method_lower in {"post", "put", "patch"} and params:
        payload_dict = {p: "test_boundary_value" for p in params[:3]}
        call_expr = f"client.{method_lower}({path!r}, json={payload_dict!r}, headers=test_headers)"
    elif params:
        query_dict = {p: "test_boundary_value" for p in params[:2]}
        call_expr = f"client.{method_lower}({path!r}, params={query_dict!r}, headers=test_headers)"
    else:
        call_expr = f"client.{method_lower}({path!r}, headers=test_headers)"

    if not is_vulnerable:
        return (
            f"def test_safe_{safe_fn_suffix}(client, test_headers):\n"
            f"    resp = {call_expr}\n"
            f"    assert resp.status_code == {expected}"
        )

    return (
        f"def test_remediated_{cwe_id.lower().replace('-', '_')}_{safe_fn_suffix}(client, test_headers):\n"
        f"    resp = {call_expr}\n"
        f"    assert resp.status_code in ({expected}, 401, 403, 409, 422)"
    )


def build_deterministic_reasoning(record: SecuritySampleRecord) -> Tuple[str, SecurityAuditOutput]:
    """Generate deterministic step-by-step defensive architectural reasoning and strict JSON."""
    snippet_preview = record.prompt_input.replace("\n", " ")[:180].strip()
    telemetry = _extract_http_telemetry(record.prompt_input, record.is_vulnerable, record.cwe_id)
    pytest_code = _build_dynamic_pytest(telemetry, record.is_vulnerable, record.cwe_id)

    if not record.is_vulnerable:
        think_text = (
            f"1. Scope & Input Classification: Analyzing {record.input_type} sample ({record.sample_id}) "
            f"targeting {telemetry['method']} {telemetry['path']} under domain '{record.domain}'.\n"
            f"2. Boundary & Control Inspection: Observed input snippet '{snippet_preview}...'. "
            f"Observed HTTP status {telemetry['observed_status']} with parameters {telemetry['params'] or ['none']}.\n"
            f"3. False-Positive Elimination: Server response and architectural controls demonstrate "
            f"proper enforcement (status {telemetry['observed_status']} without unauthorized data leakage).\n"
            f"4. Verdict Synthesis: Classify as SAFE with high confidence to suppress false positives."
        )
        output = SecurityAuditOutput(
            status="SAFE",
            confidence=0.94,
            cwe_id="NONE",
            owasp_category="NONE",
            root_cause=f"Defensive server-side controls on {telemetry['method']} {telemetry['path']} properly enforce boundaries.",
            architectural_evidence=[
                f"Endpoint {telemetry['method']} {telemetry['path']} returned HTTP {telemetry['observed_status']}.",
                f"Context summary: {record.title or 'Hardened control verified'}.",
            ],
            next_verification_step=(
                f"Run automated regression tests for {telemetry['method']} {telemetry['path']} "
                "across sibling methods and roles."
            ),
            remediation="No remediation required; maintain existing server-side validation and regression tests.",
            verification_assertion=VerificationAssertion(
                expected_http_status=telemetry["expected_status"],
                observed_http_status=telemetry["observed_status"],
                status_diff_summary=(
                    f"Observed HTTP {telemetry['observed_status']} on {telemetry['method']} {telemetry['path']} "
                    "matches expected defensive boundary behavior."
                ),
                response_body_evidence=f"Server response for {telemetry['path']} enforced expected control without leakage.",
                pytest_regression_test=pytest_code,
            ),
        )
        return think_text, output

    cwe = record.cwe_id if record.cwe_id and record.cwe_id != "NONE" else "CWE-840"
    owasp = record.owasp_category if record.owasp_category and record.owasp_category != "NONE" else "API8:2023-Security Misconfiguration"
    remediation = REMEDIATION_TEMPLATES.get(
        cwe,
        "Enforce strict server-side input validation, explicit authorization checks, and fail-closed error handling.",
    )

    think_text = (
        f"1. Scope & Input Classification: Auditing {record.input_type} sample ({record.sample_id}) "
        f"for {telemetry['method']} {telemetry['path']} routed to '{record.domain}'.\n"
        f"2. Architectural Risk Analysis: Inspecting '{snippet_preview}...'. "
        f"Parameters {telemetry['params'] or ['path/headers']} and status {telemetry['observed_status']} align with {cwe} ({owasp}).\n"
        f"3. Root Cause Evaluation: Server-side enforcement is missing or bypassed at the trust boundary, "
        f"allowing unauthorized state access, unvalidated data flow, or non-atomic state transitions.\n"
        f"4. Defensive Verification & Remediation: Formulate a safe architectural verification check "
        f"and server-side code/configuration remediation without generating exploit payloads."
    )
    output = SecurityAuditOutput(
        status="VULNERABLE",
        confidence=0.93,
        cwe_id=cwe,
        owasp_category=owasp,
        root_cause=record.title or f"Missing server-side architectural control ({cwe}) on {telemetry['method']} {telemetry['path']}.",
        architectural_evidence=[
            f"Identified {cwe} pattern on {telemetry['method']} {telemetry['path']} (source '{record.source}').",
            f"Observed HTTP {telemetry['observed_status']} where HTTP {telemetry['expected_status']} is expected for {owasp}.",
        ],
        next_verification_step=(
            f"Inspect route handler for {telemetry['method']} {telemetry['path']}, middleware ordering, "
            "and parameter validation before business logic execution."
        ),
        remediation=remediation,
        verification_assertion=VerificationAssertion(
            expected_http_status=telemetry["expected_status"],
            observed_http_status=telemetry["observed_status"],
            status_diff_summary=(
                f"Expected HTTP {telemetry['expected_status']} rejection on {telemetry['method']} {telemetry['path']}, "
                f"observed HTTP {telemetry['observed_status']} ({cwe})."
            ),
            response_body_evidence=f"Response/schema for {telemetry['path']} indicates missing {cwe} control.",
            pytest_regression_test=pytest_code,
        ),
    )
    return think_text, output


async def _distill_single_async(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    record: SecuritySampleRecord,
    api_base: str,
    api_key: str,
    model_name: str,
    max_retries: int = 2,
) -> SecuritySampleRecord:
    """Distill <think>...</think> + JSON via an OpenAI-compatible async endpoint with deterministic fallback."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "temperature": 0.2,
        "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_DEFENSIVE_AUDITOR},
            {
                "role": "user",
                "content": (
                    f"Ground-truth classification hint: is_vulnerable={record.is_vulnerable}, "
                    f"expected_cwe={record.cwe_id}.\n\n{record.prompt_input}"
                ),
            },
        ],
    }

    async with semaphore:
        for attempt in range(max_retries + 1):
            try:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                think_text, parsed_json, err = parse_model_response(content)
                if think_text and parsed_json and err is None:
                    return record.model_copy(
                        update={"reasoning_trace": think_text, "target_output": parsed_json}
                    )
                logger.debug("Distillation parse failed for %s (attempt %d): %s", record.sample_id, attempt, err)
            except Exception as exc:
                logger.debug("API distillation error for %s (attempt %d): %s", record.sample_id, attempt, exc)
            await asyncio.sleep(1.0 * (attempt + 1))

    think_text, det_output = build_deterministic_reasoning(record)
    return record.model_copy(update={"reasoning_trace": think_text, "target_output": det_output})


async def _distill_batch_async(
    records: List[SecuritySampleRecord],
    api_base: Optional[str],
    api_key: Optional[str],
    model_name: str,
    concurrency: int,
) -> List[SecuritySampleRecord]:
    if not api_base or not api_key:
        logger.info("No API endpoint/key provided -> running offline deterministic reasoning synthesis")
        results: List[SecuritySampleRecord] = []
        for rec in records:
            think_text, det_output = build_deterministic_reasoning(rec)
            results.append(rec.model_copy(update={"reasoning_trace": think_text, "target_output": det_output}))
        return results

    logger.info("Running async API reasoning distillation against %s (model=%s)", api_base, model_name)
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=45.0) as client:
        tasks = [
            _distill_single_async(client, semaphore, rec, api_base, api_key, model_name)
            for rec in records
        ]
        return await asyncio.gather(*tasks)


def run_distillation(
    input_dir: str | Path = "data/balanced",
    output_dir: str | Path = "data/distilled",
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: str = "gpt-4o-mini",
    concurrency: int = 8,
) -> Dict[str, int]:
    """Distill <think>...</think> + strict JSON targets for all balanced expert datasets."""
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    summary: Dict[str, int] = {}

    for domain in EXPERT_DOMAINS:
        in_file = in_root / f"{domain}.jsonl"
        if not in_file.exists():
            logger.warning("Balanced file missing for %s (%s); skipping", domain, in_file)
            continue

        records: List[SecuritySampleRecord] = []
        with in_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(SecuritySampleRecord.model_validate_json(line))

        distilled = asyncio.run(
            _distill_batch_async(
                records=records,
                api_base=api_base,
                api_key=resolved_key,
                model_name=model_name,
                concurrency=concurrency,
            )
        )

        out_file = out_root / f"{domain}.jsonl"
        with out_file.open("w", encoding="utf-8") as handle:
            for rec in distilled:
                handle.write(rec.model_dump_json() + "\n")

        summary[domain] = len(distilled)
        logger.info("Distilled %d records for %s -> %s", len(distilled), domain, out_file)

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Dual-mode <think>...</think> reasoning distillation.")
    parser.add_argument("--input-dir", default="data/balanced", help="Balanced JSONL directory")
    parser.add_argument("--output-dir", default="data/distilled", help="Distilled JSONL output directory")
    parser.add_argument("--api-base", default=None, help="Optional OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None, help="Optional API key (or set OPENAI_API_KEY)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Teacher model name for online mode")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent async API requests")
    args = parser.parse_args()
    run_distillation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()
