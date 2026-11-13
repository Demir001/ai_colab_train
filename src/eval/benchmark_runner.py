"""Static benchmark runner measuring Accuracy, CWE Accuracy, FPR, Schema Compliance, and Token Efficiency."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.schema import SecurityAuditOutput, VerificationAssertion, format_assistant_response, parse_model_response
from src.data.http_normalizer import normalize_http_exchange
from src.inference.multi_adapter_best_of_n import MultiAdapterAuditor, route_primary_adapter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkCase:
    """Isolated, non-overlapping static test scenario for defensive security evaluation."""

    case_id: str
    expected_status: str  # "VULNERABLE" or "SAFE"
    expected_cwe: str
    input_type: str
    prompt_input: str


STATIC_BENCHMARK_CASES: List[BenchmarkCase] = [
    BenchmarkCase(
        case_id="eval_jwt_alg_none_vuln",
        expected_status="VULNERABLE",
        expected_cwe="CWE-347",
        input_type="http_pair",
        prompt_input=(
            "HTTP Request:\n"
            "GET /api/v3/billing/statements HTTP/1.1\n"
            "Authorization: Bearer eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJhZG1pbiIsInJvbGUiOiJzdXBlcnVzZXIifQ.\n\n"
            "HTTP Response:\n"
            "HTTP/1.1 200 OK\n"
            '{"account": "enterprise_root", "statements": [{"id": "st_901", "balance_due": 9800.0}]}'
        ),
    ),
    BenchmarkCase(
        case_id="eval_jwt_alg_none_safe",
        expected_status="SAFE",
        expected_cwe="NONE",
        input_type="http_pair",
        prompt_input=(
            "HTTP Request:\n"
            "GET /api/v3/billing/statements HTTP/1.1\n"
            "Authorization: Bearer eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJhZG1pbiJ9.\n\n"
            "HTTP Response:\n"
            "HTTP/1.1 401 Unauthorized\n"
            '{"error": "invalid_token", "reason": "Unsigned JWT algorithm none is rejected"}'
        ),
    ),
    BenchmarkCase(
        case_id="eval_mass_assignment_vuln",
        expected_status="VULNERABLE",
        expected_cwe="CWE-915",
        input_type="http_pair",
        prompt_input=(
            "HTTP Request:\n"
            "PATCH /api/v2/profile/me HTTP/1.1\n"
            "Authorization: Bearer <STANDARD_USER_JWT>\n"
            '{"display_name": "Bob", "is_verified": true, "subscription_tier": "enterprise"}\n\n'
            "HTTP Response:\n"
            "HTTP/1.1 200 OK\n"
            '{"user_id": "u_44", "display_name": "Bob", "is_verified": true, "subscription_tier": "enterprise"}'
        ),
    ),
    BenchmarkCase(
        case_id="eval_mass_assignment_safe",
        expected_status="SAFE",
        expected_cwe="NONE",
        input_type="http_pair",
        prompt_input=(
            "HTTP Request:\n"
            "PATCH /api/v2/profile/me HTTP/1.1\n"
            "Authorization: Bearer <STANDARD_USER_JWT>\n"
            '{"display_name": "Bob", "is_verified": true, "subscription_tier": "enterprise"}\n\n'
            "HTTP Response:\n"
            "HTTP/1.1 200 OK\n"
            '{"user_id": "u_44", "display_name": "Bob", "is_verified": false, "subscription_tier": "free"}'
        ),
    ),
    BenchmarkCase(
        case_id="eval_path_traversal_vuln",
        expected_status="VULNERABLE",
        expected_cwe="CWE-22",
        input_type="http_pair",
        prompt_input=(
            "HTTP Request:\n"
            "GET /api/v1/exports/download?file=../../config/database.yml HTTP/1.1\n"
            "Authorization: Bearer <USER_JWT>\n\n"
            "HTTP Response:\n"
            "HTTP/1.1 200 OK\n"
            "Content-Type: text/plain\n"
            "production:\n  adapter: postgresql\n  host: db.internal"
        ),
    ),
    BenchmarkCase(
        case_id="eval_path_traversal_safe",
        expected_status="SAFE",
        expected_cwe="NONE",
        input_type="http_pair",
        prompt_input=(
            "HTTP Request:\n"
            "GET /api/v1/exports/download?file=../../config/database.yml HTTP/1.1\n"
            "Authorization: Bearer <USER_JWT>\n\n"
            "HTTP Response:\n"
            "HTTP/1.1 400 Bad Request\n"
            '{"error": "invalid_filename", "detail": "Path separators are not permitted"}'
        ),
    ),
    BenchmarkCase(
        case_id="eval_cors_misconfig_vuln",
        expected_status="VULNERABLE",
        expected_cwe="CWE-942",
        input_type="http_pair",
        prompt_input=(
            "HTTP Request:\n"
            "GET /api/v1/user/secrets HTTP/1.1\n"
            "Origin: https://untrusted-external.example\n"
            "Cookie: session=abc123\n\n"
            "HTTP Response:\n"
            "HTTP/1.1 200 OK\n"
            "Access-Control-Allow-Origin: https://untrusted-external.example\n"
            "Access-Control-Allow-Credentials: true\n"
            '{"api_key_prefix": "sk_live_..."}'
        ),
    ),
    BenchmarkCase(
        case_id="eval_public_login_openapi_safe",
        expected_status="SAFE",
        expected_cwe="NONE",
        input_type="openapi_schema",
        prompt_input=(
            "OpenAPI 3.0 Schema:\n"
            '{"paths": {"/api/v1/auth/login": {"post": {"security": [], "summary": "Public user login endpoint"}}}}'
        ),
    ),
    BenchmarkCase(
        case_id="eval_bfla_delete_tenant_vuln",
        expected_status="VULNERABLE",
        expected_cwe="CWE-285",
        input_type="multi_turn_session",
        prompt_input=(
            "Multi-Turn Session Trace:\n"
            "Turn 1: DELETE /api/v1/tenants/t_22 (Authorization: Bearer <ORG_OWNER_JWT>) -> HTTP/1.1 204 No Content\n"
            "Turn 2: DELETE /api/v1/tenants/t_23 (Authorization: Bearer <READONLY_VIEWER_JWT>) -> HTTP/1.1 204 No Content"
        ),
    ),
    BenchmarkCase(
        case_id="eval_ssrf_dns_rebind_safe",
        expected_status="SAFE",
        expected_cwe="NONE",
        input_type="http_pair",
        prompt_input=(
            "HTTP Request:\n"
            "POST /api/v2/integrations/ping HTTP/1.1\n"
            "Authorization: Bearer <USER_JWT>\n"
            '{"endpoint": "http://127.0.0.1:8080/metrics"}\n\n'
            "HTTP Response:\n"
            "HTTP/1.1 422 Unprocessable Entity\n"
            '{"error": "egress_blocked", "detail": "Resolved IP falls within loopback CIDR"}'
        ),
    ),
    BenchmarkCase(
        case_id="eval_race_withdrawal_vuln",
        expected_status="VULNERABLE",
        expected_cwe="CWE-362",
        input_type="multi_turn_session",
        prompt_input=(
            "Multi-Turn Session Trace:\n"
            "Turn 1 (t=0ms): POST /api/v2/payouts {\"amount\": 500} -> HTTP/1.1 201 Created {\"remaining\": 0}\n"
            "Turn 2 (t=2ms): POST /api/v2/payouts {\"amount\": 500} -> HTTP/1.1 201 Created {\"remaining\": -500}"
        ),
    ),
    BenchmarkCase(
        case_id="eval_race_withdrawal_safe",
        expected_status="SAFE",
        expected_cwe="NONE",
        input_type="multi_turn_session",
        prompt_input=(
            "Multi-Turn Session Trace:\n"
            "Turn 1 (t=0ms): POST /api/v2/payouts {\"amount\": 500, \"idempotency_key\": \"tx_91\"} -> HTTP/1.1 201 Created\n"
            "Turn 2 (t=2ms): POST /api/v2/payouts {\"amount\": 500, \"idempotency_key\": \"tx_91\"} -> HTTP/1.1 409 Conflict"
        ),
    ),
]


def _run_dry_eval(case: BenchmarkCase) -> Dict[str, Any]:
    """Pipeline/schema/normalizer smoke check without cheating on ground-truth labels."""
    normalized = normalize_http_exchange(case.prompt_input)
    routed_adapter = route_primary_adapter(normalized)
    dummy_out = SecurityAuditOutput(
        status="NEEDS_REVIEW",
        confidence=0.5,
        cwe_id="NONE",
        owasp_category="NONE",
        root_cause="Dry-run pipeline check (no model weights loaded).",
        architectural_evidence=[f"Normalized length={len(normalized)}, routed={routed_adapter}"],
        next_verification_step="Run with trained LoRA adapters for real evaluation.",
        remediation="N/A in dry-run.",
        verification_assertion=VerificationAssertion(
            expected_http_status=403,
            observed_http_status=200,
            status_diff_summary="Dry-run schema validation check.",
            response_body_evidence="N/A",
            pytest_regression_test="def test_dry_run(): assert True",
        ),
    )
    rendered = format_assistant_response("Dry-run pipeline check.", dummy_out)
    _, parsed, err = parse_model_response(rendered)
    return {
        "case_id": case.case_id,
        "expected_status": case.expected_status,
        "expected_cwe": case.expected_cwe,
        "predicted_status": parsed.status if parsed else "PARSE_ERROR",
        "predicted_cwe": parsed.cwe_id if parsed else "NONE",
        "routed_adapter": routed_adapter,
        "schema_valid": err is None and parsed is not None,
        "generated_tokens": len(rendered.split()),
        "valid_candidates": 1 if err is None else 0,
        "total_candidates": 1,
        "verifier_override": False,
    }


def run_benchmark(
    adapters_dir: Optional[str | Path] = "/content/outputs/adapters",
    best_of_n: int = 4,
    config_path: str | Path = "configs/hardware_profiles.yaml",
    report_out: str | Path = "/content/outputs/benchmark_report.json",
    dry_run: bool = False,
    force_profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute static benchmark suite and compute Accuracy, CWE Accuracy, FPR, Schema Compliance, and Token Efficiency."""
    auditor: Optional[MultiAdapterAuditor] = None
    if not dry_run:
        auditor = MultiAdapterAuditor(
            adapters_dir=adapters_dir or "/content/outputs/adapters",
            config_path=config_path,
            force_profile=force_profile,
        )

    case_results: List[Dict[str, Any]] = []
    tp = fp = tn = fn = 0
    unclassified_safe = 0
    cwe_correct = 0
    vuln_total = 0
    schema_valid_count = 0
    total_tokens = 0

    for case in STATIC_BENCHMARK_CASES:
        if dry_run or auditor is None:
            res = _run_dry_eval(case)
        else:
            inf = auditor.audit(case.prompt_input, best_of_n=best_of_n)
            res = {
                "case_id": case.case_id,
                "expected_status": case.expected_status,
                "expected_cwe": case.expected_cwe,
                "predicted_status": inf.final_output.status,
                "predicted_cwe": inf.final_output.cwe_id,
                "confidence": inf.final_output.confidence,
                "schema_valid": inf.valid_candidates > 0,
                "generated_tokens": inf.total_generated_tokens,
                "valid_candidates": inf.valid_candidates,
                "total_candidates": inf.total_candidates,
                "verifier_override": inf.verifier_override,
            }

        case_results.append(res)
        if res["schema_valid"]:
            schema_valid_count += 1
        total_tokens += int(res["generated_tokens"])

        exp = case.expected_status
        pred = res["predicted_status"]
        if exp == "VULNERABLE":
            vuln_total += 1
            if pred == "VULNERABLE":
                tp += 1
                if res.get("predicted_cwe") == case.expected_cwe:
                    cwe_correct += 1
            else:
                fn += 1
        elif exp == "SAFE":
            if pred == "VULNERABLE":
                fp += 1
            elif pred == "SAFE":
                tn += 1
            else:
                unclassified_safe += 1

    total = len(STATIC_BENCHMARK_CASES)
    safe_total = fp + tn + unclassified_safe
    accuracy = (tp + tn) / total if total else 0.0
    fpr = fp / safe_total if safe_total > 0 else 0.0
    cwe_accuracy = cwe_correct / vuln_total if vuln_total > 0 else 0.0
    schema_compliance = schema_valid_count / total if total else 0.0
    avg_tokens = total_tokens / total if total else 0.0

    report = {
        "total_cases": total,
        "best_of_n": best_of_n,
        "dry_run": dry_run,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "cwe_accuracy": round(cwe_accuracy, 4),
            "false_positive_rate": round(fpr, 4),
            "json_schema_compliance": round(schema_compliance, 4),
            "avg_tokens_per_audit": round(avg_tokens, 2),
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "unclassified_safe": unclassified_safe,
            },
        },
        "cases": case_results,
    }

    out_path = Path(report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Benchmark Report: %s", json.dumps(report["metrics"], indent=2))
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run defensive security LLM benchmark.")
    parser.add_argument("--adapters-dir", default="/content/outputs/adapters", help="Trained LoRA adapters directory")
    parser.add_argument("--best-of-n", type=int, default=4, choices=[4, 8], help="Best-of-N sampling count")
    parser.add_argument("--config", default="configs/hardware_profiles.yaml", help="Hardware profiles YAML")
    parser.add_argument("--report-out", default="/content/outputs/benchmark_report.json", help="Output JSON report")
    parser.add_argument("--dry-run", action="store_true", help="Run deterministic pipeline validation without model weights")
    parser.add_argument("--force-profile", default=None, help="Optional hardware profile override")
    args = parser.parse_args()

    run_benchmark(
        adapters_dir=args.adapters_dir,
        best_of_n=args.best_of_n,
        config_path=args.config,
        report_out=args.report_out,
        dry_run=args.dry_run,
        force_profile=args.force_profile,
    )


if __name__ == "__main__":
    main()
