"""ChatML formatting and zero-padding 4096-token sequence packing for TPU XLA & CUDA efficiency."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset  # type: ignore
from transformers import AutoTokenizer  # type: ignore

from src.common.hardware import detect_hardware_profile
from src.common.schema import (
    SYSTEM_PROMPT_DEFENSIVE_AUDITOR,
    SecuritySampleRecord,
    format_assistant_response,
)

logger = logging.getLogger(__name__)

EXPERT_DOMAINS = (
    "adapter_auth_bola",
    "adapter_injection_validation",
    "adapter_logic_verifier",
)


def encode_chatml_sample(
    tokenizer: Any,
    record: SecuritySampleRecord,
) -> Dict[str, List[int]]:
    """Tokenize a SecuritySampleRecord in Multi-Turn ChatML with exact token-boundary loss masking."""
    if not record.reasoning_trace or not record.target_output:
        raise ValueError(f"Record {record.sample_id} is missing reasoning_trace or target_output")

    assistant_text = format_assistant_response(record.reasoning_trace, record.target_output)
    messages_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT_DEFENSIVE_AUDITOR},
        {"role": "user", "content": record.prompt_input},
    ]
    messages_full = messages_prompt + [{"role": "assistant", "content": assistant_text}]

    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        prompt_str = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)
        full_str = tokenizer.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
    else:
        prompt_str = (
            f"<|im_start|>system\n{SYSTEM_PROMPT_DEFENSIVE_AUDITOR}<|im_end|>\n"
            f"<|im_start|>user\n{record.prompt_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_str = f"{prompt_str}{assistant_text}<|im_end|>\n"

    full_ids = tokenizer(full_str, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not full_ids or full_ids[-1] != eos_id):
        full_ids.append(eos_id)

    # Locate exact assistant header token boundary inside full_ids to avoid BPE boundary off-by-one.
    marker_ids = tokenizer("<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
    prompt_len = None
    if marker_ids:
        m_len = len(marker_ids)
        for pos in range(len(full_ids) - m_len + 1):
            if full_ids[pos : pos + m_len] == marker_ids:
                prompt_len = pos + m_len
    if prompt_len is None:
        prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        prompt_len = min(len(prompt_ids), len(full_ids))

    labels = ([-100] * prompt_len) + full_ids[prompt_len:]
    return {
        "input_ids": full_ids,
        "labels": labels,
    }


def pack_token_streams(
    encoded_samples: List[Dict[str, List[int]]],
    seq_len: int = 4096,
    pad_remainder: bool = True,
    pad_token_id: int = 0,
) -> Dict[str, List[List[int]]]:
    """Concatenate variable-length samples into fixed seq_len blocks with per-sample position_ids.

    Samples exceeding seq_len are skipped rather than truncated mid-JSON.
    """
    stream_ids: List[int] = []
    stream_labels: List[int] = []
    stream_pos_ids: List[int] = []
    skipped_oversized = 0

    for sample in encoded_samples:
        ids = sample["input_ids"]
        lbls = sample["labels"]
        if len(ids) > seq_len:
            skipped_oversized += 1
            continue
        stream_ids.extend(ids)
        stream_labels.extend(lbls)
        stream_pos_ids.extend(range(len(ids)))

    if skipped_oversized:
        logger.warning("Skipped %d samples exceeding seq_len=%d to avoid mid-JSON truncation", skipped_oversized, seq_len)

    packed_input_ids: List[List[int]] = []
    packed_attention_mask: List[List[int]] = []
    packed_labels: List[List[int]] = []
    packed_position_ids: List[List[int]] = []

    total_tokens = len(stream_ids)
    num_full_blocks = total_tokens // seq_len

    for idx in range(num_full_blocks):
        start = idx * seq_len
        end = start + seq_len
        packed_input_ids.append(stream_ids[start:end])
        packed_attention_mask.append([1] * seq_len)
        packed_labels.append(stream_labels[start:end])
        packed_position_ids.append(stream_pos_ids[start:end])

    remainder = total_tokens % seq_len
    if remainder > 0 and (num_full_blocks == 0 or pad_remainder):
        tail_ids = stream_ids[num_full_blocks * seq_len :]
        tail_labels = stream_labels[num_full_blocks * seq_len :]
        tail_pos = stream_pos_ids[num_full_blocks * seq_len :]
        pad_len = seq_len - len(tail_ids)
        packed_input_ids.append(tail_ids + [pad_token_id] * pad_len)
        packed_attention_mask.append([1] * len(tail_ids) + [0] * pad_len)
        packed_labels.append(tail_labels + [-100] * pad_len)
        packed_position_ids.append(tail_pos + [0] * pad_len)

    return {
        "input_ids": packed_input_ids,
        "attention_mask": packed_attention_mask,
        "labels": packed_labels,
        "position_ids": packed_position_ids,
    }


def run_packing(
    input_dir: str | Path = "data/distilled",
    output_dir: str | Path = "/content/data_packed",
    model_id: str | None = None,
    config_path: str | Path = "configs/hardware_profiles.yaml",
    seq_len: int | None = None,
) -> Dict[str, int]:
    """Tokenize and pack all distilled expert datasets to disk in Arrow format."""
    hw = detect_hardware_profile(config_path=config_path)
    resolved_model_id = model_id or hw.resolved_model_id
    resolved_seq_len = seq_len or int(hw.config.get("seq_len", 4096))

    logger.info("Loading tokenizer: %s (seq_len=%d)", resolved_model_id, resolved_seq_len)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_id, trust_remote_code=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)

    in_root = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, int] = {}

    for domain in EXPERT_DOMAINS:
        in_file = in_root / f"{domain}.jsonl"
        if not in_file.exists():
            logger.warning("Distilled file missing for %s (%s); skipping", domain, in_file)
            continue

        encoded: List[Dict[str, List[int]]] = []
        with in_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = SecuritySampleRecord.model_validate_json(line)
                encoded.append(encode_chatml_sample(tokenizer, rec))

        packed = pack_token_streams(
            encoded_samples=encoded,
            seq_len=resolved_seq_len,
            pad_remainder=True,
            pad_token_id=pad_token_id,
        )
        ds = Dataset.from_dict(packed)
        target_path = out_root / domain
        ds.save_to_disk(str(target_path))
        summary[domain] = len(ds)
        logger.info("Packed %d blocks (seq_len=%d) for %s -> %s", len(ds), resolved_seq_len, domain, target_path)

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Pack distilled ChatML datasets into static XLA blocks.")
    parser.add_argument("--input-dir", default="data/distilled", help="Distilled JSONL directory")
    parser.add_argument("--output-dir", default="/content/data_packed", help="Output Arrow dataset directory")
    parser.add_argument("--model-id", default=None, help="Optional tokenizer model ID override")
    parser.add_argument("--config", default="configs/hardware_profiles.yaml", help="Hardware profiles YAML")
    parser.add_argument("--seq-len", type=int, default=None, help="Optional static sequence block length")
    args = parser.parse_args()
    run_packing(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        config_path=args.config,
        seq_len=args.seq_len,
    )


if __name__ == "__main__":
    main()
