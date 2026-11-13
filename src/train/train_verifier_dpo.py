"""Post-SFT DPO trainer for adapter_logic_verifier to eliminate false positives and CWE errors."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset  # type: ignore
from peft import PeftModel, prepare_model_for_kbit_training  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

from src.common.hardware import detect_hardware_profile
from src.common.schema import (
    SYSTEM_PROMPT_DEFENSIVE_AUDITOR,
    SecurityAuditOutput,
    SecuritySampleRecord,
    VerificationAssertion,
    format_assistant_response,
)

logger = logging.getLogger(__name__)


def _build_rejected_response(rec: SecuritySampleRecord) -> str:
    """Synthesize a realistic false-positive or wrong-CWE rejected response for DPO."""
    if not rec.is_vulnerable:
        fp_think = (
            "1. Surface pattern match detected an identifier or URL parameter.\n"
            "2. Assuming missing authorization without verifying the 403/409/422 rejection in response."
        )
        fp_out = SecurityAuditOutput(
            status="VULNERABLE",
            confidence=0.81,
            cwe_id="CWE-639",
            owasp_category="API1:2023-Broken Object Level Authorization",
            root_cause="False positive: flagged parameter presence without checking server-side 403 enforcement.",
            architectural_evidence=["Parameter observed in request path."],
            next_verification_step="Check response status code.",
            remediation="Add authorization check.",
            verification_assertion=VerificationAssertion(
                expected_http_status=403,
                observed_http_status=200,
                status_diff_summary="Assumed 200 OK bypass without verifying actual 403 response.",
                response_body_evidence="Ignored rejection body.",
                pytest_regression_test="def test_fp(): assert True",
            ),
        )
        return format_assistant_response(fp_think, fp_out)

    wrong_cwe = "CWE-79" if rec.cwe_id != "CWE-79" else "CWE-89"
    bad_think = (
        f"1. Misclassified architectural flaw as {wrong_cwe} without verifying trust boundary.\n"
        "2. Incomplete root-cause analysis."
    )
    bad_out = SecurityAuditOutput(
        status="SAFE",
        confidence=0.62,
        cwe_id="NONE",
        owasp_category="NONE",
        root_cause="False negative: overlooked missing server-side validation.",
        architectural_evidence=["No obvious error message in response."],
        next_verification_step="None.",
        remediation="No action.",
        verification_assertion=VerificationAssertion(
            expected_http_status=200,
            observed_http_status=200,
            status_diff_summary="Missed boundary violation.",
            response_body_evidence="N/A",
            pytest_regression_test="def test_fn(): assert True",
        ),
    )
    return format_assistant_response(bad_think, bad_out)


def build_dpo_preference_dataset(distilled_files: List[Path]) -> Dataset:
    """Construct (prompt, chosen, rejected) pairs from distilled JSONL files."""
    rows: List[Dict[str, str]] = []
    for path in distilled_files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = SecuritySampleRecord.model_validate_json(line)
                if not rec.reasoning_trace or not rec.target_output:
                    continue
                prompt_str = (
                    f"<|im_start|>system\n{SYSTEM_PROMPT_DEFENSIVE_AUDITOR}<|im_end|>\n"
                    f"<|im_start|>user\n{rec.prompt_input}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                chosen_str = format_assistant_response(rec.reasoning_trace, rec.target_output) + "<|im_end|>\n"
                rejected_str = _build_rejected_response(rec) + "<|im_end|>\n"
                rows.append({"prompt": prompt_str, "chosen": chosen_str, "rejected": rejected_str})

    if not rows:
        raise ValueError("No valid distilled records found to build DPO preference dataset.")
    logger.info("Built %d DPO preference pairs", len(rows))
    return Dataset.from_list(rows)


def train_verifier_dpo(
    distilled_dir: str | Path = "data/distilled",
    sft_adapter_dir: str | Path = "/content/outputs/adapters/adapter_logic_verifier",
    output_dir: str | Path = "/content/outputs/adapters/adapter_logic_verifier_dpo",
    config_path: str | Path = "configs/hardware_profiles.yaml",
    force_profile: Optional[str] = None,
    beta: float = 0.1,
    max_steps: int = -1,
) -> Path:
    """Run DPO fine-tuning on adapter_logic_verifier to suppress false positives."""
    try:
        from trl import DPOConfig, DPOTrainer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("trl is required for DPO training. Install via `pip install trl>=0.9.6`.") from exc

    hw = detect_hardware_profile(config_path=config_path, force_profile=force_profile)
    cfg = hw.config
    model_id = hw.resolved_model_id
    torch_dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float32

    dist_root = Path(distilled_dir)
    dpo_dataset = build_dpo_preference_dataset(
        [
            dist_root / "adapter_logic_verifier.jsonl",
            dist_root / "adapter_auth_bola.jsonl",
            dist_root / "adapter_injection_validation.jsonl",
        ]
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"trust_remote_code": True, "torch_dtype": torch_dtype}
    if hw.backend == "cuda":
        model_kwargs["device_map"] = "auto"
        if cfg.get("use_4bit_qlora", False):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )

    base_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if hw.backend == "cuda" and cfg.get("use_4bit_qlora", False):
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        )
    elif cfg.get("gradient_checkpointing", True):
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base_model.enable_input_require_grads()

    base_model.config.use_cache = False

    sft_path = Path(sft_adapter_dir)
    if sft_path.exists():
        logger.info("Loading SFT adapter from %s for DPO refinement", sft_path)
        model = PeftModel.from_pretrained(base_model, str(sft_path), is_trainable=True)
    else:
        raise FileNotFoundError(f"SFT adapter not found at {sft_path}. Train SFT adapter first.")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dpo_args = DPOConfig(
        output_dir=str(out_path / "dpo_checkpoints"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        max_steps=max_steps,
        beta=beta,
        bf16=(cfg.get("precision", "bf16") == "bf16" and hw.backend in {"cuda", "xla"}),
        optim=str(cfg.get("optim", "adamw_torch")),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.get("gradient_checkpointing", True) else None,
        dataloader_drop_last=bool(cfg.get("dataloader_drop_last", hw.backend == "xla")),
        max_length=min(2048, int(cfg.get("seq_len", 4096))),
        max_prompt_length=1024,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
    )
    logger.info("Starting DPO false-positive elimination training on %s", hw.device_name)
    trainer.train()

    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    logger.info("Saved DPO-refined adapter_logic_verifier -> %s", out_path)
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Post-SFT DPO training for adapter_logic_verifier.")
    parser.add_argument("--distilled-dir", default="data/distilled", help="Distilled JSONL directory")
    parser.add_argument(
        "--sft-adapter-dir",
        default="/content/outputs/adapters/adapter_logic_verifier",
        help="Input SFT adapter directory",
    )
    parser.add_argument(
        "--output-dir",
        default="/content/outputs/adapters/adapter_logic_verifier_dpo",
        help="Output DPO-refined adapter directory",
    )
    parser.add_argument("--config", default="configs/hardware_profiles.yaml", help="Hardware profiles YAML")
    parser.add_argument("--force-profile", default=None, help="Optional hardware profile override")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO KL penalty beta")
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional max DPO steps")
    args = parser.parse_args()

    train_verifier_dpo(
        distilled_dir=args.distilled_dir,
        sft_adapter_dir=args.sft_adapter_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        force_profile=args.force_profile,
        beta=args.beta,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
