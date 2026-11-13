"""Multi-source defensive security data ingestion (OSV.dev, HuggingFace, Semgrep rules, OpenAPI/HTTP)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import yaml

from src.common.schema import SecuritySampleRecord

logger = logging.getLogger(__name__)

OSV_API_URL = "https://api.osv.dev/v1/query"
DEFAULT_ECOSYSTEMS: List[Tuple[str, str]] = [
    ("npm", "express"),
    ("npm", "jsonwebtoken"),
    ("PyPI", "django"),
    ("PyPI", "fastapi"),
    ("Go", "github.com/gin-gonic/gin"),
    ("Maven", "org.springframework.boot:spring-boot-starter-web"),
]

CWE_TO_OWASP: Dict[str, str] = {
    "CWE-639": "API1:2023-Broken Object Level Authorization",
    "CWE-287": "API2:2023-Broken Authentication",
    "CWE-306": "API2:2023-Broken Authentication",
    "CWE-798": "API2:2023-Broken Authentication",
    "CWE-285": "API5:2023-Broken Function Level Authorization",
    "CWE-862": "API5:2023-Broken Function Level Authorization",
    "CWE-863": "API1:2023-Broken Object Level Authorization",
    "CWE-347": "API2:2023-Broken Authentication",
    "CWE-384": "API2:2023-Broken Authentication",
    "CWE-613": "API2:2023-Broken Authentication",
    "CWE-915": "API3:2023-Broken Object Property Level Authorization",
    "CWE-200": "API3:2023-Broken Object Property Level Authorization",
    "CWE-770": "API4:2023-Unrestricted Resource Consumption",
    "CWE-799": "API4:2023-Unrestricted Resource Consumption",
    "CWE-400": "API4:2023-Unrestricted Resource Consumption",
    "CWE-362": "API6:2023-Unrestricted Access to Sensitive Business Flows",
    "CWE-840": "API6:2023-Unrestricted Access to Sensitive Business Flows",
    "CWE-918": "API7:2023-Server Side Request Forgery",
    "CWE-16": "API8:2023-Security Misconfiguration",
    "CWE-942": "API8:2023-Security Misconfiguration",
    "CWE-89": "API8:2023-Security Misconfiguration / Injection",
    "CWE-79": "A03:2021-Injection (XSS)",
    "CWE-78": "A03:2021-Injection (OS Command Injection)",
    "CWE-20": "A03:2021-Injection / Improper Input Validation",
    "CWE-22": "A01:2021-Broken Access Control (Path Traversal)",
    "CWE-94": "A03:2021-Injection (Code Injection)",
    "CWE-502": "A08:2021-Software and Data Integrity Failures",
    "CWE-352": "A01:2021-Broken Access Control (CSRF)",
}

AUTH_CWES = {"CWE-287", "CWE-285", "CWE-306", "CWE-347", "CWE-639", "CWE-798", "CWE-862", "CWE-863", "CWE-384", "CWE-613"}
INJECTION_CWES = {"CWE-89", "CWE-79", "CWE-78", "CWE-918", "CWE-22", "CWE-94", "CWE-502", "CWE-77", "CWE-1336", "CWE-20"}


def _make_sample_id(source: str, raw_key: str) -> str:
    digest = hashlib.sha256(f"{source}:{raw_key}".encode("utf-8")).hexdigest()[:16]
    return f"{source}_{digest}"


def classify_domain(cwe_id: str, text_blob: str) -> str:
    """Route a sample into one of the 3 specialist LoRA domains."""
    cwe_norm = cwe_id.strip().upper()
    lower = text_blob.lower()
    if cwe_norm in AUTH_CWES or any(k in lower for k in ("bola", "idor", "jwt", "oauth", "rbac", "authorization", "session")):
        return "adapter_auth_bola"
    if cwe_norm in INJECTION_CWES or any(k in lower for k in ("sql", "xss", "ssrf", "traversal", "sanitize", "parameterized", "template")):
        return "adapter_injection_validation"
    return "adapter_logic_verifier"


def fetch_osv_advisories(
    ecosystems: Optional[List[Tuple[str, str]]] = None,
    min_year: int = 2024,
    max_year: int = 2026,
    limit_per_pkg: int = 25,
    timeout_sec: float = 15.0,
) -> List[SecuritySampleRecord]:
    """Fetch 2024-2026 web ecosystem advisories from Google OSV.dev API."""
    targets = ecosystems or DEFAULT_ECOSYSTEMS
    records: List[SecuritySampleRecord] = []

    with httpx.Client(timeout=timeout_sec) as client:
        for ecosystem, package_name in targets:
            try:
                resp = client.post(
                    OSV_API_URL,
                    json={"package": {"ecosystem": ecosystem, "name": package_name}},
                )
                resp.raise_for_status()
                vulns = resp.json().get("vulns", [])
            except Exception as exc:
                logger.warning("OSV.dev query failed for %s/%s: %s", ecosystem, package_name, exc)
                continue

            count = 0
            for vuln in vulns:
                if count >= limit_per_pkg:
                    break
                published = str(vuln.get("published", ""))
                year_str = published[:4]
                if year_str.isdigit() and not (min_year <= int(year_str) <= max_year):
                    continue

                osv_id = vuln.get("id", "UNKNOWN")
                summary = vuln.get("summary") or vuln.get("details", "")[:240]
                details = vuln.get("details", summary)
                cwes = vuln.get("database_specific", {}).get("cwe_ids", [])
                cwe_id = cwes[0] if cwes else "CWE-840"
                owasp = CWE_TO_OWASP.get(cwe_id, "API8:2023-Security Misconfiguration")

                prompt_input = (
                    f"Dependency & API Architecture Audit Request:\n"
                    f"Ecosystem: {ecosystem} | Package: {package_name} | Advisory: {osv_id}\n"
                    f"Published: {published}\n"
                    f"Observed Behavior / Advisory Context:\n{details[:1500]}"
                )
                domain = classify_domain(cwe_id, f"{summary} {details}")
                records.append(
                    SecuritySampleRecord(
                        sample_id=_make_sample_id("osv", f"{osv_id}_{package_name}"),
                        source="osv.dev",
                        domain=domain,  # type: ignore[arg-type]
                        input_type="code_snippet",
                        prompt_input=prompt_input,
                        is_vulnerable=True,
                        cwe_id=cwe_id,
                        owasp_category=owasp,
                        title=summary[:160],
                        raw_metadata={"osv_id": osv_id, "ecosystem": ecosystem, "package": package_name},
                    )
                )
                count += 1

    logger.info("Ingested %d records from OSV.dev", len(records))
    return records


def fetch_huggingface_security_pairs(
    dataset_name: str = "CyberNative/Code_Vulnerability_Security_DPO",
    max_rows: int = 300,
) -> List[SecuritySampleRecord]:
    """Ingest paired vulnerable/safe code samples from Hugging Face."""
    records: List[SecuritySampleRecord] = []
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset(dataset_name, split="train", streaming=True)
        for idx, row in enumerate(ds):
            if idx >= max_rows:
                break
            question = str(row.get("question", "") or row.get("prompt", "")).strip()
            chosen = str(row.get("chosen", "")).strip()
            rejected = str(row.get("rejected", "")).strip()
            lang = str(row.get("lang", "web"))
            if not question or (not chosen and not rejected):
                continue

            if rejected:
                domain_v = classify_domain("NONE", f"{question} {rejected}")
                records.append(
                    SecuritySampleRecord(
                        sample_id=_make_sample_id("hf_vuln", f"{idx}_{lang}"),
                        source=f"hf:{dataset_name}",
                        domain=domain_v,  # type: ignore[arg-type]
                        input_type="code_snippet",
                        prompt_input=f"Language: {lang}\nContext: {question}\n\nCode Under Review:\n{rejected[:2000]}",
                        is_vulnerable=True,
                        cwe_id="CWE-20",
                        owasp_category="A03:2021-Injection / Input Validation",
                        title=question[:140],
                        raw_metadata={"lang": lang, "polarity": "vulnerable"},
                    )
                )

            if chosen:
                domain_s = classify_domain("NONE", f"{question} {chosen}")
                records.append(
                    SecuritySampleRecord(
                        sample_id=_make_sample_id("hf_safe", f"{idx}_{lang}"),
                        source=f"hf:{dataset_name}",
                        domain=domain_s,  # type: ignore[arg-type]
                        input_type="code_snippet",
                        prompt_input=f"Language: {lang}\nContext: {question}\n\nCode Under Review:\n{chosen[:2000]}",
                        is_vulnerable=False,
                        cwe_id="NONE",
                        owasp_category="NONE",
                        title=f"Hardened implementation: {question[:120]}",
                        raw_metadata={"lang": lang, "polarity": "safe"},
                    )
                )
    except Exception as exc:
        logger.warning("HuggingFace ingestion failed for %s: %s", dataset_name, exc)

    logger.info("Ingested %d records from HuggingFace (%s)", len(records), dataset_name)
    return records


def fetch_semgrep_web_rules(timeout_sec: float = 15.0) -> List[SecuritySampleRecord]:
    """Fetch public Semgrep web security rule metadata from GitHub as architectural test cases."""
    rule_urls = [
        ("python/django/security/audit/xss/direct-use-of-httpresponse.yaml", "CWE-79"),
        ("javascript/express/security/audit/express-check-csurf-middleware-usage.yaml", "CWE-352"),
        ("python/fastapi/security/audit/fastapi-sql-injection.yaml", "CWE-89"),
    ]
    base_raw = "https://raw.githubusercontent.com/semgrep/semgrep-rules/develop/"
    records: List[SecuritySampleRecord] = []

    with httpx.Client(timeout=timeout_sec) as client:
        for rel_path, default_cwe in rule_urls:
            try:
                resp = client.get(base_raw + rel_path)
                if resp.status_code != 200:
                    continue
                content = resp.text[:2200]
                domain = classify_domain(default_cwe, content)
                records.append(
                    SecuritySampleRecord(
                        sample_id=_make_sample_id("semgrep", rel_path),
                        source="semgrep-rules",
                        domain=domain,  # type: ignore[arg-type]
                        input_type="code_snippet",
                        prompt_input=f"Semgrep Web Rule Reference ({rel_path}):\n{content}",
                        is_vulnerable=True,
                        cwe_id=default_cwe,
                        owasp_category=CWE_TO_OWASP.get(default_cwe, "A03:2021-Injection"),
                        title=f"Semgrep rule: {Path(rel_path).name}",
                        raw_metadata={"rule_path": rel_path},
                    )
                )
            except Exception as exc:
                logger.warning("Semgrep rule fetch failed for %s: %s", rel_path, exc)

    logger.info("Ingested %d records from Semgrep rules", len(records))
    return records


def fetch_hf_advisory_datasets(max_rows_per_ds: int = 150) -> List[SecuritySampleRecord]:
    """Stream public vulnerability advisory & code-fix datasets with schema-aware field extraction."""
    candidate_datasets = [
        "seyyedaliayati/cwe-vulnerability-dataset",
        "lemon42-ai/PrimeVul",
        "msc-lab/MegaVul",
    ]
    records: List[SecuritySampleRecord] = []

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        logger.warning("datasets library unavailable; skipping HF advisory datasets")
        return records

    for ds_name in candidate_datasets:
        try:
            ds = load_dataset(ds_name, split="train", streaming=True)
            for idx, row in enumerate(ds):
                if idx >= max_rows_per_ds:
                    break
                vuln_code = str(
                    row.get("func_before")
                    or row.get("vulnerable_code")
                    or row.get("func")
                    or row.get("code")
                    or row.get("description")
                    or ""
                ).strip()
                safe_code = str(row.get("func_after") or row.get("fixed_code") or "").strip()
                if not vuln_code and not safe_code:
                    continue

                raw_cwe = str(row.get("cwe") or row.get("cwe_id") or "CWE-20").strip().upper()
                if not raw_cwe.startswith("CWE-"):
                    raw_cwe = "CWE-20"
                title = str(row.get("title") or row.get("cve") or row.get("commit_msg") or f"{ds_name} #{idx}")[:140]

                if vuln_code:
                    is_vuln = bool(row.get("target", 1)) if ("target" in row and not safe_code) else True
                    domain = classify_domain(raw_cwe, f"{title} {vuln_code[:400]}")
                    records.append(
                        SecuritySampleRecord(
                            sample_id=_make_sample_id("hf_adv_v", f"{ds_name}_{idx}"),
                            source=f"hf:{ds_name}",
                            domain=domain,  # type: ignore[arg-type]
                            input_type="code_snippet",
                            prompt_input=f"Security Advisory / Code Audit ({ds_name}):\n{vuln_code[:2000]}",
                            is_vulnerable=is_vuln,
                            cwe_id=raw_cwe if is_vuln else "NONE",
                            owasp_category=CWE_TO_OWASP.get(raw_cwe, "API8:2023-Security Misconfiguration") if is_vuln else "NONE",
                            title=title,
                            raw_metadata={"dataset": ds_name, "index": idx},
                        )
                    )

                if safe_code:
                    domain_s = classify_domain(raw_cwe, f"{title} {safe_code[:400]}")
                    records.append(
                        SecuritySampleRecord(
                            sample_id=_make_sample_id("hf_adv_s", f"{ds_name}_{idx}"),
                            source=f"hf:{ds_name}",
                            domain=domain_s,  # type: ignore[arg-type]
                            input_type="code_snippet",
                            prompt_input=f"Patched / Hardened Code Audit ({ds_name}):\n{safe_code[:2000]}",
                            is_vulnerable=False,
                            cwe_id="NONE",
                            owasp_category="NONE",
                            title=f"Patched: {title[:120]}",
                            raw_metadata={"dataset": ds_name, "index": idx, "patched": True},
                        )
                    )
        except Exception as exc:
            logger.warning("Skipping HF advisory dataset %s: %s", ds_name, exc)

    logger.info("Ingested %d records from HF advisory/disclosed datasets", len(records))
    return records


def fetch_openapi_directory_specs(
    cache_dir: str | Path = "data/cache/openapi-directory",
    max_specs: int = 30,
) -> List[SecuritySampleRecord]:
    """Shallow-clone APIs-guru/openapi-directory and extract endpoint authorization/schema audit records."""
    records: List[SecuritySampleRecord] = []
    repo_dir = Path(cache_dir)

    try:
        if not repo_dir.exists():
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/APIs-guru/openapi-directory.git",
                    str(repo_dir),
                ],
                check=True,
                timeout=90,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except Exception as exc:
        logger.warning("Shallow clone of APIs-guru/openapi-directory failed: %s", exc)
        return records

    public_route_allowlist = (
        "login", "signin", "signup", "register", "token", "oauth", "auth/",
        "health", "status", "ping", "docs", "swagger", "openapi", "public",
    )
    spec_files = (
        list((repo_dir / "APIs").rglob("openapi.yaml"))
        + list((repo_dir / "APIs").rglob("swagger.yaml"))
    )[:max_specs]
    for spec_path in spec_files:
        try:
            if spec_path.stat().st_size > 1_000_000:
                continue
            spec_data = yaml.safe_load(spec_path.read_text(encoding="utf-8", errors="ignore"))
            if not isinstance(spec_data, dict) or "paths" not in spec_data:
                continue

            paths = spec_data.get("paths", {})
            global_sec = bool(spec_data.get("security"))
            trimmed_paths: Dict[str, Any] = {}
            missing_sec_count = 0

            for p_key, p_val in list(paths.items())[:6]:
                if not isinstance(p_val, dict):
                    continue
                trimmed_paths[p_key] = {}
                p_lower = p_key.lower()
                is_public_route = any(pub in p_lower for pub in public_route_allowlist)
                for method, op in p_val.items():
                    if method.lower() in {"get", "post", "put", "patch", "delete"} and isinstance(op, dict):
                        op_sec = op.get("security")
                        if (
                            not is_public_route
                            and not global_sec
                            and not op_sec
                            and any(k in p_lower for k in ("admin", "internal", "role", "billing", "invoice", "delete"))
                        ):
                            missing_sec_count += 1
                        trimmed_paths[p_key][method] = {
                            "summary": str(op.get("summary", ""))[:100],
                            "security": op_sec if op_sec is not None else ([] if not global_sec else ["inherited"]),
                        }

            if not trimmed_paths:
                continue

            is_vuln = missing_sec_count > 0
            cwe = "CWE-862" if is_vuln else "NONE"
            spec_json = json.dumps({"info": spec_data.get("info", {}).get("title", spec_path.parent.name), "paths": trimmed_paths}, indent=2)
            records.append(
                SecuritySampleRecord(
                    sample_id=_make_sample_id("openapi_guru", str(spec_path.relative_to(repo_dir))),
                    source="APIs-guru/openapi-directory",
                    domain="adapter_auth_bola",
                    input_type="openapi_schema",
                    prompt_input=f"OpenAPI 3.0 Endpoint Security Schema:\n{spec_json[:2200]}",
                    is_vulnerable=is_vuln,
                    cwe_id=cwe,
                    owasp_category=CWE_TO_OWASP.get(cwe, "NONE") if is_vuln else "NONE",
                    title=f"OpenAPI spec audit: {spec_path.parent.name}",
                    raw_metadata={"spec_path": str(spec_path.name), "missing_sec_count": missing_sec_count},
                )
            )
        except Exception as exc:
            logger.debug("Failed to parse OpenAPI spec %s: %s", spec_path, exc)

    logger.info("Ingested %d OpenAPI schema records from APIs-guru", len(records))
    return records


def generate_curated_api_and_safe_corpus() -> List[SecuritySampleRecord]:
    """Built-in defensive HTTP/OpenAPI/GraphQL corpus (vulnerable + hardened SAFE pairs)."""
    seeds: List[Dict[str, Any]] = [
        {
            "id": "bola_vuln_01",
            "domain": "adapter_auth_bola",
            "input_type": "http_pair",
            "is_vulnerable": True,
            "cwe_id": "CWE-639",
            "owasp_category": "API1:2023-Broken Object Level Authorization",
            "title": "Invoice endpoint returns another tenant's record via sequential ID",
            "prompt_input": (
                "HTTP Request:\n"
                "GET /api/v2/invoices/10482 HTTP/1.1\n"
                "Host: api.example.internal\n"
                "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyXzEwMSIsInRlbmFudF9pZCI6InRfQSJ9.sig\n\n"
                "HTTP Response:\n"
                "HTTP/1.1 200 OK\n"
                "Content-Type: application/json\n"
                '{"invoice_id": 10482, "tenant_id": "t_B", "owner_id": "user_889", "amount_usd": 4200.00}'
            ),
        },
        {
            "id": "bola_safe_01",
            "domain": "adapter_auth_bola",
            "input_type": "http_pair",
            "is_vulnerable": False,
            "cwe_id": "NONE",
            "owasp_category": "NONE",
            "title": "Hardened tenant-scoped invoice endpoint rejects cross-tenant access",
            "prompt_input": (
                "HTTP Request:\n"
                "GET /api/v2/invoices/10482 HTTP/1.1\n"
                "Host: api.example.internal\n"
                "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1c2VyXzEwMSIsInRlbmFudF9pZCI6InRfQSJ9.sig\n\n"
                "HTTP Response:\n"
                "HTTP/1.1 403 Forbidden\n"
                "Content-Type: application/json\n"
                '{"error": "forbidden", "detail": "Resource does not belong to authenticated tenant scope"}'
            ),
        },
        {
            "id": "bfla_openapi_vuln_01",
            "domain": "adapter_auth_bola",
            "input_type": "openapi_schema",
            "is_vulnerable": True,
            "cwe_id": "CWE-285",
            "owasp_category": "API5:2023-Broken Function Level Authorization",
            "title": "Admin user-role elevation route lacks security scheme and role requirement",
            "prompt_input": (
                "OpenAPI 3.0 Schema:\n"
                "{\n"
                '  "paths": {\n'
                '    "/api/admin/users/{userId}/role": {\n'
                '      "patch": {\n'
                '        "summary": "Update user role",\n'
                '        "security": [],\n'
                '        "requestBody": {"content": {"application/json": {"schema": {"properties": {"role": {"type": "string"}}}}}}\n'
                "      }\n"
                "    }\n"
                "  }\n"
                "}"
            ),
        },
        {
            "id": "ssrf_vuln_01",
            "domain": "adapter_injection_validation",
            "input_type": "http_pair",
            "is_vulnerable": True,
            "cwe_id": "CWE-918",
            "owasp_category": "API7:2023-Server Side Request Forgery",
            "title": "Webhook preview fetches arbitrary internal/metadata URL without egress allowlist",
            "prompt_input": (
                "HTTP Request:\n"
                "POST /api/v1/webhooks/preview HTTP/1.1\n"
                "Content-Type: application/json\n"
                '{"callback_url": "http://169.254.169.254/latest/meta-data/iam/security-credentials/"}\n\n'
                "HTTP Response:\n"
                "HTTP/1.1 200 OK\n"
                '{"upstream_status": 200, "body_preview": "AccessKeyId..."}'
            ),
        },
        {
            "id": "ssrf_safe_01",
            "domain": "adapter_injection_validation",
            "input_type": "http_pair",
            "is_vulnerable": False,
            "cwe_id": "NONE",
            "owasp_category": "NONE",
            "title": "Hardened webhook preview enforces DNS resolution check and private-IP denylist",
            "prompt_input": (
                "HTTP Request:\n"
                "POST /api/v1/webhooks/preview HTTP/1.1\n"
                "Content-Type: application/json\n"
                '{"callback_url": "http://169.254.169.254/latest/meta-data/"}\n\n'
                "HTTP Response:\n"
                "HTTP/1.1 422 Unprocessable Entity\n"
                '{"error": "invalid_callback_url", "reason": "Link-local and RFC1918 destinations are blocked by egress policy"}'
            ),
        },
        {
            "id": "sqli_safe_01",
            "domain": "adapter_injection_validation",
            "input_type": "http_pair",
            "is_vulnerable": False,
            "cwe_id": "NONE",
            "owasp_category": "NONE",
            "title": "Parameterized search endpoint treats SQL metacharacters as literal string",
            "prompt_input": (
                "HTTP Request:\n"
                "GET /api/v1/products?q=widget%27%20OR%201%3D1-- HTTP/1.1\n\n"
                "HTTP Response:\n"
                "HTTP/1.1 200 OK\n"
                "Content-Type: application/json\n"
                '{"items": [], "total": 0, "query_echo": "widget\' OR 1=1--"}'
            ),
        },
        {
            "id": "logic_race_vuln_01",
            "domain": "adapter_logic_verifier",
            "input_type": "multi_turn_session",
            "is_vulnerable": True,
            "cwe_id": "CWE-362",
            "owasp_category": "API6:2023-Unrestricted Access to Sensitive Business Flows",
            "title": "Coupon redemption lacks atomic state lock allowing concurrent double-credit",
            "prompt_input": (
                "Multi-Turn Session Trace:\n"
                "Turn 1 (t=0.000s): POST /api/v1/coupons/redeem {\"code\": \"WELCOME50\"} -> 200 OK {\"balance\": 50}\n"
                "Turn 2 (t=0.002s): POST /api/v1/coupons/redeem {\"code\": \"WELCOME50\"} -> 200 OK {\"balance\": 100}\n"
                "Server DB note: SELECT used_at ... followed by non-atomic UPDATE outside serializable transaction."
            ),
        },
        {
            "id": "logic_safe_fp_01",
            "domain": "adapter_logic_verifier",
            "input_type": "multi_turn_session",
            "is_vulnerable": False,
            "cwe_id": "NONE",
            "owasp_category": "NONE",
            "title": "Idempotency-Key + DB unique constraint prevents duplicate coupon redemption",
            "prompt_input": (
                "Multi-Turn Session Trace:\n"
                "Turn 1 (t=0.000s): POST /api/v1/coupons/redeem {\"code\": \"WELCOME50\"} -> 200 OK {\"balance\": 50}\n"
                "Turn 2 (t=0.002s): POST /api/v1/coupons/redeem {\"code\": \"WELCOME50\"} -> 409 Conflict {\"error\": \"coupon_already_redeemed\"}"
            ),
        },
        {
            "id": "graphql_safe_01",
            "domain": "adapter_logic_verifier",
            "input_type": "graphql_schema",
            "is_vulnerable": False,
            "cwe_id": "NONE",
            "owasp_category": "NONE",
            "title": "GraphQL endpoint enforces query depth limit and persisted-query allowlist",
            "prompt_input": (
                "GraphQL Request:\n"
                '{"query": "query { viewer { id email orders(first: 10) { id total } } }"}\n\n'
                "HTTP Response:\n"
                "HTTP/1.1 200 OK\n"
                "X-GraphQL-Depth: 3\n"
                "X-GraphQL-Complexity: 18\n"
                '{"data": {"viewer": {"id": "u_101", "email": "alice@example.com", "orders": []}}}'
            ),
        },
    ]

    records: List[SecuritySampleRecord] = []
    for item in seeds:
        records.append(
            SecuritySampleRecord(
                sample_id=_make_sample_id("curated", item["id"]),
                source="curated_defensive_corpus",
                domain=item["domain"],
                input_type=item["input_type"],
                prompt_input=item["prompt_input"],
                is_vulnerable=item["is_vulnerable"],
                cwe_id=item["cwe_id"],
                owasp_category=item["owasp_category"],
                title=item["title"],
                raw_metadata={"seed_id": item["id"]},
            )
        )
    return records


def save_records_jsonl(records: Iterable[SecuritySampleRecord], output_path: str | Path) -> Path:
    """Persist normalized SecuritySampleRecord items to JSONL."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(rec.model_dump_json() + "\n")
            count += 1
    logger.info("Saved %d records to %s", count, out)
    return out


def run_ingestion(
    output_path: str | Path = "data/raw/ingested_samples.jsonl",
    include_remote: bool = True,
    hf_max_rows: int = 300,
) -> Path:
    """Execute full multi-source ingestion with graceful fallback."""
    all_records: List[SecuritySampleRecord] = []
    all_records.extend(generate_curated_api_and_safe_corpus())

    if include_remote:
        all_records.extend(fetch_osv_advisories())
        all_records.extend(fetch_huggingface_security_pairs(max_rows=hf_max_rows))
        all_records.extend(fetch_hf_advisory_datasets(max_rows_per_ds=min(hf_max_rows, 150)))
        all_records.extend(fetch_semgrep_web_rules())
        all_records.extend(fetch_openapi_directory_specs())

    return save_records_jsonl(all_records, output_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Ingest defensive security datasets.")
    parser.add_argument("--output", default="data/raw/ingested_samples.jsonl", help="Output JSONL path")
    parser.add_argument("--offline", action="store_true", help="Skip remote APIs and use curated corpus only")
    parser.add_argument("--hf-max-rows", type=int, default=300, help="Max rows to stream from HuggingFace")
    args = parser.parse_args()
    run_ingestion(output_path=args.output, include_remote=not args.offline, hf_max_rows=args.hf_max_rows)


if __name__ == "__main__":
    main()
