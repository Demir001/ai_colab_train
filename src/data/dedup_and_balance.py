"""MinHash LSH deduplication (threshold=0.85), 40% SAFE balancing, and 3-expert domain routing."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from src.common.schema import SecuritySampleRecord

logger = logging.getLogger(__name__)

EXPERT_DOMAINS = (
    "adapter_auth_bola",
    "adapter_injection_validation",
    "adapter_logic_verifier",
)


def _shingle_text(text: str, k: int = 5) -> Set[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if len(normalized) <= k:
        return {normalized} if normalized else {""}
    return {normalized[i : i + k] for i in range(len(normalized) - k + 1)}


def deduplicate_minhash(
    records: List[SecuritySampleRecord],
    threshold: float = 0.85,
    num_perm: int = 128,
) -> List[SecuritySampleRecord]:
    """Deduplicate near-identical samples using MinHash LSH (with pure-Python fallback)."""
    try:
        from datasketch import MinHash, MinHashLSH  # type: ignore

        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        kept: List[SecuritySampleRecord] = []
        for idx, rec in enumerate(records):
            mh = MinHash(num_perm=num_perm)
            for shingle in _shingle_text(rec.prompt_input):
                mh.update(shingle.encode("utf-8"))
            if not lsh.query(mh):
                lsh.insert(f"rec_{idx}_{rec.sample_id}", mh)
                kept.append(rec)
        logger.info("MinHash LSH (datasketch) reduced %d -> %d records", len(records), len(kept))
        return kept
    except ImportError:
        logger.warning("datasketch not installed; using deterministic Jaccard shingle deduplication")
        kept = []
        seen_shingles: List[Set[str]] = []
        for rec in records:
            shingles = _shingle_text(rec.prompt_input)
            is_dup = False
            for prev in seen_shingles:
                union_len = len(shingles | prev)
                if union_len > 0 and (len(shingles & prev) / union_len) >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                seen_shingles.append(shingles)
                kept.append(rec)
        logger.info("Fallback Jaccard reduced %d -> %d records", len(records), len(kept))
        return kept


def balance_domain_dataset(
    records: List[SecuritySampleRecord],
    target_safe_ratio: float = 0.40,
    seed: int = 42,
) -> List[SecuritySampleRecord]:
    """Balance ~40% genuine SAFE examples without static shortcut banners."""
    rng = random.Random(seed)
    vuln_recs = [r for r in records if r.is_vulnerable]
    safe_recs = [r for r in records if not r.is_vulnerable]

    if not vuln_recs and not safe_recs:
        return []

    if vuln_recs and safe_recs:
        max_vuln_for_ratio = int(round(len(safe_recs) * ((1.0 - target_safe_ratio) / target_safe_ratio)))
        if len(vuln_recs) > max_vuln_for_ratio and max_vuln_for_ratio >= 1:
            # Cap oversized vulnerable majority or oversample genuine SAFE records up to 2x before capping.
            if len(safe_recs) * 2 >= int(round((target_safe_ratio / (1.0 - target_safe_ratio)) * len(vuln_recs))):
                needed_safe = int(round((target_safe_ratio / (1.0 - target_safe_ratio)) * len(vuln_recs)))
                oversampled = [safe_recs[i % len(safe_recs)].model_copy(update={"sample_id": f"{safe_recs[i % len(safe_recs)].sample_id}_os_{i}"}) for i in range(needed_safe)]
                safe_recs = oversampled
            else:
                rng.shuffle(vuln_recs)
                vuln_recs = vuln_recs[:max_vuln_for_ratio]
        else:
            needed_safe = max(1, int(round((target_safe_ratio / (1.0 - target_safe_ratio)) * len(vuln_recs))))
            if len(safe_recs) > needed_safe:
                rng.shuffle(safe_recs)
                safe_recs = safe_recs[:needed_safe]

    combined = vuln_recs + safe_recs
    rng.shuffle(combined)
    return combined


def run_dedup_and_balance(
    input_path: str | Path = "data/raw/ingested_samples.jsonl",
    output_dir: str | Path = "data/balanced",
    threshold: float = 0.85,
    safe_ratio: float = 0.40,
) -> Dict[str, int]:
    """Load raw JSONL, deduplicate via MinHash LSH, balance SAFE ratio, and split by expert."""
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {in_path}")

    raw_records: List[SecuritySampleRecord] = []
    with in_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_records.append(SecuritySampleRecord.model_validate_json(line))
            except Exception as exc:
                logger.warning("Skipping malformed record at line %d: %s", line_num, exc)

    deduped = deduplicate_minhash(raw_records, threshold=threshold)

    by_domain: Dict[str, List[SecuritySampleRecord]] = defaultdict(list)
    for rec in deduped:
        by_domain[rec.domain].append(rec)

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, int] = {}

    for domain in EXPERT_DOMAINS:
        domain_recs = by_domain.get(domain, [])
        if not domain_recs:
            logger.warning("Domain %-30s -> 0 records after dedup; skipping empty split", domain)
            summary[domain] = 0
            continue
        balanced = balance_domain_dataset(domain_recs, target_safe_ratio=safe_ratio)
        val_count = max(1, int(round(len(balanced) * 0.1))) if len(balanced) >= 10 else 0
        train_recs = balanced[val_count:]
        val_recs = balanced[:val_count]

        out_file = out_root / f"{domain}.jsonl"
        with out_file.open("w", encoding="utf-8") as handle:
            for rec in train_recs:
                handle.write(rec.model_dump_json() + "\n")

        if val_recs:
            val_file = out_root / f"{domain}_val.jsonl"
            with val_file.open("w", encoding="utf-8") as handle:
                for rec in val_recs:
                    handle.write(rec.model_dump_json() + "\n")

        safe_count = sum(1 for r in train_recs if not r.is_vulnerable)
        logger.info(
            "Domain %-30s -> train=%d (safe=%.1f%%), val=%d",
            domain,
            len(train_recs),
            (100.0 * safe_count / len(train_recs)) if train_recs else 0.0,
            len(val_recs),
        )
        summary[domain] = len(train_recs)

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="MinHash LSH deduplication and 40% SAFE balancing.")
    parser.add_argument("--input", default="data/raw/ingested_samples.jsonl", help="Raw ingested JSONL")
    parser.add_argument("--output-dir", default="data/balanced", help="Output directory for expert JSONLs")
    parser.add_argument("--threshold", type=float, default=0.85, help="MinHash LSH Jaccard threshold")
    parser.add_argument("--safe-ratio", type=float, default=0.40, help="Target SAFE proportion (default 0.40)")
    args = parser.parse_args()
    run_dedup_and_balance(
        input_path=args.input,
        output_dir=args.output_dir,
        threshold=args.threshold,
        safe_ratio=args.safe_ratio,
    )


if __name__ == "__main__":
    main()
