"""Synthesize multi-turn black-box HTTP audit scenarios (Turn 1 baseline vs Turn 2 diff) with 40% SAFE cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.common.schema import SecuritySampleRecord
from src.data.http_normalizer import normalize_http_exchange

logger = logging.getLogger(__name__)


def _make_id(prefix: str, key: str) -> str:
    digest = hashlib.sha256(f"{prefix}:{key}".encode("utf-8")).hexdigest()[:14]
    return f"{prefix}_{digest}"


def _build_multi_turn_scenario(
    scenario_id: str,
    domain: str,
    is_vulnerable: bool,
    cwe_id: str,
    owasp_category: str,
    title: str,
    turn1_req: str,
    turn1_resp: str,
    turn2_req: str,
    turn2_resp: str,
    diff_summary: str,
    source_ref: str,
) -> SecuritySampleRecord:
    raw_trace = (
        "Multi-Turn Black-Box HTTP Audit Trace:\n"
        f"=== Turn 1 (Authorized Baseline) ===\n{turn1_req}\n\n{turn1_resp}\n\n"
        f"=== Turn 2 (Boundary / Alternate Context Check) ===\n{turn2_req}\n\n{turn2_resp}\n\n"
        f"=== Non-Destructive Response Diff ===\n{diff_summary}"
    )
    return SecuritySampleRecord(
        sample_id=_make_id("bb_synth", scenario_id),
        source=f"blackbox_synthesizer:{source_ref}",
        domain=domain,  # type: ignore[arg-type]
        input_type="multi_turn_session",
        prompt_input=normalize_http_exchange(raw_trace),
        is_vulnerable=is_vulnerable,
        cwe_id=cwe_id if is_vulnerable else "NONE",
        owasp_category=owasp_category if is_vulnerable else "NONE",
        title=title,
        raw_metadata={"scenario_id": scenario_id, "synthesized": True},
    )


def synthesize_from_openapi_spec(spec: Dict[str, Any], spec_name: str) -> List[SecuritySampleRecord]:
    """Convert an OpenAPI 3.0 / Swagger dict into Turn-1/Turn-2 black-box HTTP audit scenarios."""
    records: List[SecuritySampleRecord] = []
    paths = spec.get("paths", {})
    if not isinstance(paths, dict):
        return records

    for path, methods in list(paths.items())[:15]:
        if not isinstance(methods, dict):
            continue
        for method, op in methods.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete"} or not isinstance(op, dict):
                continue

            has_security = bool(op.get("security") or spec.get("security"))
            has_id_param = "{" in path and "}" in path
            is_admin_route = any(k in path.lower() for k in ("admin", "internal", "role", "config", "users"))
            upper_m = method.upper()

            if is_admin_route and not has_security:
                records.append(
                    _build_multi_turn_scenario(
                        scenario_id=f"{spec_name}_{upper_m}_{path}_bfla_vuln",
                        domain="adapter_auth_bola",
                        is_vulnerable=True,
                        cwe_id="CWE-285",
                        owasp_category="API5:2023-Broken Function Level Authorization",
                        title=f"Unprotected administrative route {upper_m} {path}",
                        turn1_req=f"{upper_m} {path} HTTP/1.1\nAuthorization: Bearer <ADMIN_JWT>",
                        turn1_resp='HTTP/1.1 200 OK\nContent-Type: application/json\n{"status": "ok"}',
                        turn2_req=f"{upper_m} {path} HTTP/1.1\nAuthorization: Bearer <STANDARD_USER_JWT>",
                        turn2_resp='HTTP/1.1 200 OK\nContent-Type: application/json\n{"status": "ok"}',
                        diff_summary="Status Diff: Expected 403 Forbidden for standard user, observed 200 OK.",
                        source_ref=spec_name,
                    )
                )
            elif has_id_param:
                sample_path_a = path.replace("{", "").replace("}", "") + "_101"
                sample_path_b = path.replace("{", "").replace("}", "") + "_102"
                records.append(
                    _build_multi_turn_scenario(
                        scenario_id=f"{spec_name}_{upper_m}_{path}_bola_safe",
                        domain="adapter_auth_bola",
                        is_vulnerable=False,
                        cwe_id="NONE",
                        owasp_category="NONE",
                        title=f"Tenant-isolated resource route {upper_m} {path}",
                        turn1_req=f"{upper_m} {sample_path_a} HTTP/1.1\nAuthorization: Bearer <TENANT_A_JWT>",
                        turn1_resp='HTTP/1.1 200 OK\nContent-Type: application/json\n{"tenant_id": "t_A"}',
                        turn2_req=f"{upper_m} {sample_path_b} HTTP/1.1\nAuthorization: Bearer <TENANT_A_JWT>",
                        turn2_resp='HTTP/1.1 403 Forbidden\nContent-Type: application/json\n{"error": "forbidden"}',
                        diff_summary="Status Diff: Turn 1=200 OK (own tenant), Turn 2=403 Forbidden (cross-tenant). Boundary enforced.",
                        source_ref=spec_name,
                    )
                )
    return records


def _infer_endpoint_from_code_context(rec: SecuritySampleRecord) -> Tuple[str, str, str, str]:
    """Derive a realistic REST/GraphQL method, path, and parameter context from source/advisory text."""
    text = rec.prompt_input
    lower = text.lower()
    route_match = re.search(r"['\"](/[a-zA-Z0-9_/{}-]{3,48})['\"]", text)
    if "graphql" in lower:
        return "POST", "/graphql", "query", "application/json"
    if rec.domain == "adapter_auth_bola":
        path = route_match.group(1) if route_match else "/api/v1/accounts/402/profile"
        return "GET", path, "account_id", "application/json"
    if rec.domain == "adapter_injection_validation":
        if "url" in lower or "webhook" in lower or rec.cwe_id == "CWE-918":
            path = route_match.group(1) if route_match else "/api/v1/webhooks/dispatch"
            return "POST", path, "callback_url", "application/json"
        if "file" in lower or "path" in lower or rec.cwe_id == "CWE-22":
            path = route_match.group(1) if route_match else "/api/v1/files/download"
            return "GET", path, "filename", "application/octet-stream"
        path = route_match.group(1) if route_match else "/api/v1/catalog/search"
        return "GET", path, "q", "application/json"
    path = route_match.group(1) if route_match else "/api/v1/orders/checkout"
    return "POST", path, "idempotency_key", "application/json"


def synthesize_blackbox_corpus(
    input_jsonl: str | Path = "data/raw/ingested_samples.jsonl",
    output_jsonl: str | Path = "data/raw/blackbox_synthesized.jsonl",
    safe_ratio: float = 0.40,
    seed: int = 42,
) -> Path:
    """Transform ingested OpenAPI/source records into multi-turn black-box scenarios (40% SAFE)."""
    rng = random.Random(seed)
    synthesized: List[SecuritySampleRecord] = []

    in_path = Path(input_jsonl)
    if in_path.exists():
        with in_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = SecuritySampleRecord.model_validate_json(line)
                except Exception:
                    continue

                if rec.input_type == "openapi_schema":
                    try:
                        start = rec.prompt_input.find("{")
                        if start != -1:
                            spec = json.loads(rec.prompt_input[start:])
                            synthesized.extend(synthesize_from_openapi_spec(spec, rec.sample_id))
                            continue
                    except Exception:
                        pass

                method, path, param_name, content_type = _infer_endpoint_from_code_context(rec)
                code_excerpt = rec.prompt_input.replace("\n", " ")[:220]

                if rec.is_vulnerable:
                    if method == "GET":
                        t1_req = f"GET {path}?{param_name}=normal_100 HTTP/1.1\nAuthorization: Bearer <TENANT_A_JWT>"
                        t2_req = f"GET {path}?{param_name}=boundary_999 HTTP/1.1\nAuthorization: Bearer <TENANT_A_JWT>"
                    else:
                        t1_req = f"{method} {path} HTTP/1.1\nAuthorization: Bearer <TENANT_A_JWT>\nContent-Type: {content_type}\n{json.dumps({param_name: 'valid_input'})}"
                        t2_req = f"{method} {path} HTTP/1.1\nAuthorization: Bearer <TENANT_A_JWT>\nContent-Type: {content_type}\n{json.dumps({param_name: 'boundary_input'})}"

                    synthesized.append(
                        _build_multi_turn_scenario(
                            scenario_id=f"{rec.sample_id}_vuln",
                            domain=rec.domain,
                            is_vulnerable=True,
                            cwe_id=rec.cwe_id,
                            owasp_category=rec.owasp_category,
                            title=rec.title or f"Multi-turn audit for {rec.cwe_id} on {path}",
                            turn1_req=t1_req,
                            turn1_resp=f'HTTP/1.1 200 OK\nContent-Type: {content_type}\n{{"status": "ok", "context": "baseline"}}',
                            turn2_req=t2_req,
                            turn2_resp=f'HTTP/1.1 200 OK\nContent-Type: {content_type}\n{{"status": "ok", "unvalidated_param": {json.dumps(param_name)}}}',
                            diff_summary=(
                                f"Status/Schema Diff on {method} {path}: Turn 2 boundary check returned 200 OK instead of 403/422; "
                                f"source context ({rec.cwe_id}): {code_excerpt}"
                            ),
                            source_ref=rec.source,
                        )
                    )
                else:
                    if method == "GET":
                        t1_req = f"GET {path}?{param_name}=own_100 HTTP/1.1\nAuthorization: Bearer <TENANT_A_JWT>"
                        t2_req = f"GET {path}?{param_name}=other_999 HTTP/1.1\nAuthorization: Bearer <TENANT_A_JWT>"
                    else:
                        t1_req = f"{method} {path} HTTP/1.1\nAuthorization: Bearer <TENANT_A_JWT>\nContent-Type: {content_type}\n{json.dumps({param_name: 'valid_input'})}"
                        t2_req = f"{method} {path} HTTP/1.1\nAuthorization: Bearer <TENANT_A_JWT>\nContent-Type: {content_type}\n{json.dumps({param_name: 'boundary_input'})}"

                    reject_code = 422 if rec.domain == "adapter_injection_validation" else (409 if rec.domain == "adapter_logic_verifier" else 403)
                    synthesized.append(
                        _build_multi_turn_scenario(
                            scenario_id=f"{rec.sample_id}_safe",
                            domain=rec.domain,
                            is_vulnerable=False,
                            cwe_id="NONE",
                            owasp_category="NONE",
                            title=rec.title or f"Hardened SAFE multi-turn interaction on {path}",
                            turn1_req=t1_req,
                            turn1_resp=f'HTTP/1.1 200 OK\nContent-Type: {content_type}\n{{"status": "ok"}}',
                            turn2_req=t2_req,
                            turn2_resp=f'HTTP/1.1 {reject_code} Rejected\nContent-Type: application/json\n{{"error": "policy_enforced", "param": {json.dumps(param_name)}}}',
                            diff_summary=f"Status Diff on {method} {path}: Turn 1=200 OK, Turn 2={reject_code}. Server enforces strict boundary.",
                            source_ref=rec.source,
                        )
                    )

    vuln_list = [r for r in synthesized if r.is_vulnerable]
    safe_list = [r for r in synthesized if not r.is_vulnerable]
    if vuln_list and safe_list:
        needed_safe = max(1, int(round((safe_ratio / (1.0 - safe_ratio)) * len(vuln_list))))
        if len(safe_list) < needed_safe:
            max_vuln = max(1, int(round(len(safe_list) * ((1.0 - safe_ratio) / safe_ratio))))
            rng.shuffle(vuln_list)
            vuln_list = vuln_list[:max_vuln]
        elif len(safe_list) > needed_safe:
            rng.shuffle(safe_list)
            safe_list = safe_list[:needed_safe]

    final_records = vuln_list + safe_list
    rng.shuffle(final_records)

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for rec in final_records:
            handle.write(rec.model_dump_json() + "\n")

    logger.info("Synthesized %d multi-turn black-box scenarios -> %s", len(final_records), out_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Synthesize multi-turn black-box HTTP audit scenarios.")
    parser.add_argument("--input", default="data/raw/ingested_samples.jsonl", help="Ingested JSONL path")
    parser.add_argument("--output", default="data/raw/blackbox_synthesized.jsonl", help="Synthesized JSONL path")
    parser.add_argument("--safe-ratio", type=float, default=0.40, help="Target SAFE proportion")
    args = parser.parse_args()
    synthesize_blackbox_corpus(input_jsonl=args.input, output_jsonl=args.output, safe_ratio=args.safe_ratio)


if __name__ == "__main__":
    main()
