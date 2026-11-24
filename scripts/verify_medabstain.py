"""Check dataset integrity and overlap with MedQA splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.medabstain import (
    FILES, check_checksums, check_medqa_overlap, load_medabstain,
)
from src.data.medqa_datasets import load_splits


def verify(data_dir=None, medqa_splits=None):
    report = {
        "checksums": check_checksums(data_dir),
        "datasets": {},
        "medqa_overlap": {
            "status": "pending", "reason": "MedQA splits were not checked",
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
    parser.add_argument("--skip-medqa-overlap", action="store_true",
                        help="Do not check overlap with the MedQA train/val splits from load_splits")
    args = parser.parse_args()
    splits = None if args.skip_medqa_overlap else load_splits("medqa")
    report = verify(args.data_dir, splits)
    print(json.dumps(report, indent=2))
    # Only the test files are excluded from MedQA; the val_100 files overlap MedQA train/val by design.
    if splits is not None and any(
        result[split]["overlapping_examples"]
        for key, result in report["medqa_overlap"]["results"].items() if key.startswith("test/")
        for split in ("train", "val")
    ):
        raise SystemExit("MedAbstain test questions found in MedQA train/val")


if __name__ == "__main__":
    main()
