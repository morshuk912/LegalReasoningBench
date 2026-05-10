#!/usr/bin/env python3
"""
Validate the final binary thesis dataset without rewriting it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prepare_final_binary_dataset import BINARY_LABELS, REQUIRED_COLUMNS, clean_text, normalize_frame, validate_columns
from prepare_final_binary_dataset import text_word_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the final binary legal outcome dataset.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--expect_rows", type=int, default=None)
    parser.add_argument(
        "--strict_text_fields",
        action="store_true",
        help="Fail if facts, reasoning, conclusion, or doc_id contain empty values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.csv)
    frame = pd.read_csv(path)
    validate_columns(frame)
    normalized = normalize_frame(frame)

    labels = set(normalized["binary_outcome_label"].dropna().astype(str))
    unexpected = sorted(labels - set(BINARY_LABELS))
    duplicate_doc_ids = int(normalized["doc_id"].duplicated().sum())
    empty_required_targets = {
        column: int(normalized[column].map(clean_text).eq("").sum())
        for column in ["facts", "reasoning", "conclusion", "doc_id", "binary_outcome_label"]
    }

    report = {
        "path": str(path),
        "rows": int(len(normalized)),
        "expected_rows": args.expect_rows,
        "columns": list(normalized.columns),
        "label_counts": {
            str(label): int(count)
            for label, count in normalized["binary_outcome_label"].value_counts(dropna=False).to_dict().items()
        },
        "unique_outcome_labels_before_binary_grouping": int(
            normalized["final_outcome_label"].map(clean_text).replace("", pd.NA).dropna().nunique()
        ),
        "text_word_statistics": text_word_statistics(normalized),
        "duplicate_doc_ids": duplicate_doc_ids,
        "empty_required_targets": empty_required_targets,
        "unexpected_labels": unexpected,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.expect_rows is not None and len(normalized) != args.expect_rows:
        raise SystemExit(f"Expected {args.expect_rows} rows, found {len(normalized)}.")
    if unexpected:
        raise SystemExit(f"Unexpected labels found: {unexpected}")
    if duplicate_doc_ids:
        raise SystemExit(f"Duplicate doc_id values found: {duplicate_doc_ids}")
    if args.strict_text_fields and (
        empty_required_targets["facts"] or empty_required_targets["reasoning"] or empty_required_targets["doc_id"]
    ):
        raise SystemExit(f"Required fields contain empty values: {empty_required_targets}")


if __name__ == "__main__":
    main()
