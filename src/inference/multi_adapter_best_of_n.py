"""Multi-Adapter Best-of-N (N=4/8) defensive security inference & logic-verifier adjudication."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

from src.common.hardware import detect_hardware_profile
from src.common.schema import (
    SYSTEM_PROMPT_DEFENSIVE_AUDITOR,
    SecurityAuditOutput,
    VerificationAssertion,
    parse_model_response,
)
from src.data.http_normalizer import normalize_http_exchange

logger = logging.getLogger(__name__)

ADAPTER_NAMES = (
    "adapter_auth_bola",
    "adapter_injection_validation",
    "adapter_logic_verifier",
)


@dataclass
class InferenceResult:
    """Structured Best-of-N adjudication result with token & consensus telemetry."""

    final_output: SecurityAuditOutput
    reasoning_trace: str
    total_generated_tokens: int
    valid_candidates: int
    total_candidates: int
    verifier_override: bool


def route_primary_adapter(raw_input: str) -> str:
    """Select the primary domain specialist adapter using weighted multi-cue scoring."""
    lower = raw_input.lower()
    inj_cues = (
        "callback_url", "http://169.254", "http://127.0.0.1", "http://localhost",
        "select ", "union ", "<script", "file://", "../", "..\\", "query=", "ssrf", "sql", "xss",
    )
    auth_cues = (
        "tenant", "owner_id", "user_id", "role", "admin", "rbac", "jwt",
        "alg\": \"none\"", "alg=none", "security\": []", "403 forbidden", "invoice", "order_id",
    )
    logic_cues = (
        "turn 1", "turn 2", "idempotency", "409 conflict", "coupon", "balance", "graphql", "x-graphql-depth",
    )

    inj_score = sum(2 for k in inj_cues if k in lower)
    auth_score = sum(2 for k in auth_cues if k in lower)
    logic_score = sum(2 for k in logic_cues if k in lower)

    if inj_score >= auth_score and inj_score > 0:
        return "adapter_injection_validation"
    if auth_score >= logic_score and auth_score > 0:
        return "adapter_auth_bola"
    if logic_score > 0:
        return "adapter_logic_verifier"
    if "authorization:" in lower or "bearer " in lower:
        return "adapter_auth_bola"
    return "adapter_logic_verifier"


class MultiAdapterAuditor:
    """Loads base model + 3 specialist LoRA adapters and runs Best-of-N + Logic Verifier."""

    def __init__(
        self,
        adapters_dir: str | Path = "/content/outputs/adapters",
        config_path: str | Path = "configs/hardware_profiles.yaml",
        force_profile: Optional[str] = None,
    ) -> None:
        self.hw = detect_hardware_profile(config_path=config_path, force_profile=force_profile)
        self.adapters_dir = Path(adapters_dir)
        self.model_id = self.hw.resolved_model_id
        self.tokenizer: Any = None
        self.model: Any = None
        self.loaded_adapters: List[str] = []
        self._load_models()

    def _load_models(self) -> None:
        cfg = self.hw.config
        torch_dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True, "torch_dtype": torch_dtype}
        if self.hw.backend == "cuda":
            model_kwargs["device_map"] = "auto"
            if cfg.get("use_4bit_qlora", False):
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                )

        base_model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        if self.hw.backend == "xla":
            import torch_xla.core.xla_model as xm  # type: ignore

            base_model = base_model.to(xm.xla_device())

        peft_model: Optional[PeftModel] = None
        for adapter_name in ADAPTER_NAMES:
            dpo_candidate = self.adapters_dir / f"{adapter_name}_dpo"
            adapter_path = dpo_candidate if (adapter_name == "adapter_logic_verifier" and dpo_candidate.exists()) else (self.adapters_dir / adapter_name)
            if not adapter_path.exists():
                logger.warning("Adapter directory not found: %s (skipping)", adapter_path)
                continue
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(base_model, str(adapter_path), adapter_name=adapter_name)
            else:
                peft_model.load_adapter(str(adapter_path), adapter_name=adapter_name)
            self.loaded_adapters.append(adapter_name)
            logger.info("Loaded LoRA adapter: %s", adapter_name)

        self.model = peft_model if peft_model is not None else base_model
        if self.hw.backend == "xla":
            import torch_xla.core.xla_model as xm  # type: ignore

            self.model = self.model.to(xm.xla_device())
        self.model.eval()

    def _generate_candidates(
        self,
        user_prompt: str,
        adapter_name: str,
        num_samples: int = 4,
        max_new_tokens: int = 768,
        temperature: float = 0.4,
    ) -> Tuple[List[Tuple[str, SecurityAuditOutput]], int]:
        """Generate `num_samples` candidates using the specified adapter."""
        if isinstance(self.model, PeftModel) and adapter_name in self.loaded_adapters:
            self.model.set_adapter(adapter_name)

        prompt_str = (
            f"<|im_start|>system\n{SYSTEM_PROMPT_DEFENSIVE_AUDITOR}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(prompt_str, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        valid: List[Tuple[str, SecurityAuditOutput]] = []
        total_tokens = 0
        mini_batch = 2

        remaining = num_samples
        while remaining > 0:
            batch_n = min(mini_batch, remaining)
            remaining -= batch_n
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=(temperature > 0 and num_samples > 1),
                    temperature=temperature if num_samples > 1 else 1.0,
                    top_p=0.92,
                    num_return_sequences=batch_n,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            for seq in outputs:
                gen_ids = seq[input_len:]
                total_tokens += int(gen_ids.shape[0])
                decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                think_text, parsed, err = parse_model_response(decoded)
                if parsed and err is None:
                    valid.append((think_text or "", parsed))

        return valid, total_tokens

    def audit(
        self,
        raw_input: str,
        best_of_n: int = 4,
        max_new_tokens: int = 768,
    ) -> InferenceResult:
        """Run Best-of-N sampling on primary expert + false-positive verification via logic_verifier."""
        if best_of_n not in (4, 8):
            logger.info("Using custom best_of_n=%d (recommended: 4 or 8)", best_of_n)

        normalized_input = normalize_http_exchange(raw_input) or raw_input
        primary_adapter = route_primary_adapter(normalized_input)
        candidates, tokens_primary = self._generate_candidates(
            user_prompt=normalized_input,
            adapter_name=primary_adapter,
            num_samples=best_of_n,
            max_new_tokens=max_new_tokens,
        )

        if not candidates:
            fallback = SecurityAuditOutput(
                status="NEEDS_REVIEW",
                confidence=0.50,
                cwe_id="NONE",
                owasp_category="NONE",
                root_cause="Model outputs did not satisfy strict JSON schema across Best-of-N samples.",
                architectural_evidence=["Insufficient schema-valid candidates during Best-of-N sampling."],
                next_verification_step="Manually inspect the HTTP/OpenAPI artifact with a security architect.",
                remediation="Review endpoint authorization and input validation rules manually.",
                verification_assertion=VerificationAssertion(
                    expected_http_status=403,
                    observed_http_status=200,
                    status_diff_summary="Automated schema consensus not reached; manual HTTP diff verification required.",
                    response_body_evidence="N/A (fallback due to schema parse failure).",
                    pytest_regression_test=(
                        "def test_manual_security_review_placeholder(client):\n"
                        "    # Replace with endpoint-specific authorization assertion after review\n"
                        "    assert client is not None"
                    ),
                ),
            )
            return InferenceResult(
                final_output=fallback,
                reasoning_trace="Fallback triggered due to 0 schema-valid candidates.",
                total_generated_tokens=tokens_primary,
                valid_candidates=0,
                total_candidates=best_of_n,
                verifier_override=False,
            )

        pair_counts = Counter((out.status, out.cwe_id) for _, out in candidates)
        (majority_status, majority_cwe), _ = pair_counts.most_common(1)[0]
        matching = [(t, o) for t, o in candidates if o.status == majority_status and o.cwe_id == majority_cwe]
        best_think, best_output = max(matching, key=lambda pair: pair[1].confidence)

        verifier_override = False
        tokens_verifier = 0
        if "adapter_logic_verifier" in self.loaded_adapters and best_output.status == "VULNERABLE":
            # Keep prompt format in-distribution with SFT/DPO training data.
            ver_cands, tokens_verifier = self._generate_candidates(
                user_prompt=normalized_input,
                adapter_name="adapter_logic_verifier",
                num_samples=2,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
            )
            if ver_cands:
                ver_status_counts = Counter(out.status for _, out in ver_cands)
                if ver_status_counts.get("SAFE", 0) == len(ver_cands):
                    ver_think, ver_out = max(ver_cands, key=lambda pair: pair[1].confidence)
                    best_output = ver_out
                    best_think = ver_think
                    verifier_override = True

        return InferenceResult(
            final_output=best_output,
            reasoning_trace=best_think,
            total_generated_tokens=tokens_primary + tokens_verifier,
            valid_candidates=len(candidates),
            total_candidates=best_of_n,
            verifier_override=verifier_override,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Run Multi-Adapter Best-of-N defensive security audit.")
    parser.add_argument("--input-file", required=True, help="Path to HTTP/OpenAPI/GraphQL text file")
    parser.add_argument("--adapters-dir", default="/content/outputs/adapters", help="Trained LoRA adapters directory")
    parser.add_argument("--best-of-n", type=int, default=4, choices=[4, 8], help="Best-of-N sample count")
    parser.add_argument("--config", default="configs/hardware_profiles.yaml", help="Hardware profiles YAML")
    args = parser.parse_args()

    raw_text = Path(args.input_file).read_text(encoding="utf-8")
    auditor = MultiAdapterAuditor(adapters_dir=args.adapters_dir, config_path=args.config)
    result = auditor.audit(raw_text, best_of_n=args.best_of_n)
    print(json.dumps(result.final_output.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
