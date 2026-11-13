"""Hybrid TPU v5e-1 (PyTorch-XLA) & CUDA (A100/L4) LoRA trainer for Qwen2.5-Coder-14B-Instruct."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_from_disk  # type: ignore
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from src.common.hardware import HardwareSelection, detect_hardware_profile

logger = logging.getLogger(__name__)


def _build_model_and_tokenizer(hw: HardwareSelection) -> tuple[Any, Any]:
    """Load base model and tokenizer with hardware-specific memory optimizations."""
    model_id = hw.resolved_model_id
    cfg = hw.config
    precision = cfg.get("precision", "bf16")
    torch_dtype = torch.bfloat16 if precision == "bf16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
    }

    if hw.backend == "cuda":
        if cfg.get("use_4bit_qlora", False):
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=cfg.get("bnb_4bit_use_double_quant", True),
                bnb_4bit_compute_dtype=torch_dtype,
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = "auto"
        if cfg.get("flash_attention_2", False):
            model_kwargs["attn_implementation"] = "flash_attention_2"

    logger.info("Loading base model %s on backend=%s (%s)", model_id, hw.backend, hw.device_name)
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    if hw.backend == "cuda" and cfg.get("use_4bit_qlora", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True))
    elif cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    model.config.use_cache = False

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=list(lora_cfg["target_modules"]),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


def train_expert(
    expert_name: str,
    packed_data_dir: str | Path = "/content/data_packed",
    output_dir: str | Path = "/content/outputs/adapters",
    config_path: str | Path = "configs/hardware_profiles.yaml",
    force_profile: Optional[str] = None,
    max_steps: int = -1,
) -> Path:
    """Train a specialist LoRA adapter on the packed dataset for `expert_name`."""
    hw = detect_hardware_profile(config_path=config_path, force_profile=force_profile)
    if hw.backend == "xla":
        os.environ.setdefault("XLA_USE_BF16", "1")
    if expert_name not in hw.experts:
        raise ValueError(f"Unknown expert {expert_name!r}. Valid: {list(hw.experts.keys())}")

    dataset_path = Path(packed_data_dir) / expert_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Packed dataset not found at {dataset_path}")

    train_dataset = load_from_disk(str(dataset_path))
    logger.info("Loaded packed dataset for %s: %d blocks", expert_name, len(train_dataset))

    model, tokenizer = _build_model_and_tokenizer(hw)
    cfg = hw.config

    adapter_out = Path(output_dir) / expert_name
    adapter_out.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(adapter_out / "checkpoints"),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=str(cfg.get("lr_scheduler_type", "cosine")),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.05)),
        num_train_epochs=float(cfg.get("num_train_epochs", 2)),
        max_steps=max_steps,
        bf16=(cfg.get("precision", "bf16") == "bf16" and hw.backend == "cuda"),
        optim=str(cfg.get("optim", "adamw_torch")),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.get("gradient_checkpointing", True) else None,
        dataloader_drop_last=bool(cfg.get("dataloader_drop_last", hw.backend == "xla")),
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    logger.info("Starting LoRA training for expert=%s on %s", expert_name, hw.device_name)
    trainer.train()

    model.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))
    logger.info("Saved trained LoRA adapter for %s -> %s", expert_name, adapter_out)
    return adapter_out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Train specialist LoRA adapter on TPU-XLA or CUDA GPU.")
    parser.add_argument(
        "--expert",
        required=True,
        choices=["adapter_auth_bola", "adapter_injection_validation", "adapter_logic_verifier"],
        help="Specialist adapter name",
    )
    parser.add_argument("--packed-dir", default="/content/data_packed", help="Packed Arrow datasets root")
    parser.add_argument("--output-dir", default="/content/outputs/adapters", help="Output adapters root")
    parser.add_argument("--config", default="configs/hardware_profiles.yaml", help="Hardware profiles YAML")
    parser.add_argument("--force-profile", default=None, help="Optional profile override")
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional max training steps")
    args = parser.parse_args()

    train_expert(
        expert_name=args.expert,
        packed_data_dir=args.packed_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        force_profile=args.force_profile,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
