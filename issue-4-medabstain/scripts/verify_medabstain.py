"""Check dataset integrity and overlap with MedQA splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.medabstain import (
    FILES, check_medqa_overlap, load_medabstain,
)


def verify(data_dir=None, medqa_splits=None):
    report = {
        "datasets": {},
        "medqa_overlap": {
            "status": "pending", "reason": "MedQA train/val records were not supplied",
        },
    }
    loaded = {}
    for split, variants in FILES.items():
        for variant in variants:
            rows = load_medabstain(variant, split, data_dir)
            expected = 1007 if split == "test" else 100
            if len(rows) != expected:
                raise ValueError(f"{split}/{variant}: expected {expected} records, got {len(rows)}")
            loaded[split, variant] = rows
            report["datasets"][f"{split}/{variant}"] = {
                "count": len(rows),
                "abstention_options": sum(r["abstain_option"] is not None for r in rows),
                "valid_answers": sum(r["answer"] in r["options"] for r in rows),
            }
    if medqa_splits is not None:
        report["medqa_overlap"] = {
            "status": "checked",
            "normalizer": "question-only: lowercase, punctuation to spaces, collapse whitespace",
            "results": {f"{split}/{variant}": check_medqa_overlap(rows, medqa_splits)
                        for (split, variant), rows in loaded.items()},
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--medqa-splits", type=Path,
                        help='Common-format JSON export: {"train": [...], "val": [...]}')
    args = parser.parse_args()
    splits = json.loads(args.medqa_splits.read_text()) if args.medqa_splits else None
    report = verify(args.data_dir, splits)
    print(json.dumps(report, indent=2))
    if splits is not None and any(
        result[split]["overlapping_examples"]
        for result in report["medqa_overlap"]["results"].values() for split in ("train", "val")
    ):
        raise SystemExit("MedQA question overlaps detected")


if __name__ == "__main__":
    main()
