#!/usr/bin/env python3
"""
Prepare the final binary thesis dataset.

The final dataset is produced by taking the reviewed structured master file and
keeping only cases whose binary_outcome_label is one of:
  - Affirmed
  - Reversed/Vacated

The script validates schema, labels, missing values, and duplicate doc_id values.

Example:
  python prepare_final_binary_dataset.py \
    --input_csv ../../uslegalkit_structured_dataset_expert4_no_unknown.csv \
    --output_csv final_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "facts",
    "claim_plaintiff",
    "claim_defendant",
    "reasoning",
    "conclusion",
    "doc_id",
    "full_document",
    "final_outcome_label",
    "binary_outcome_label",
]

BINARY_LABELS = ["Affirmed", "Reversed/Vacated"]
LEGACY_BINARY_OUTCOME_COLUMNS = ["final_outcome_group_expert4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and validate the final binary legal outcome dataset.")
    parser.add_argument("--input_csv", required=True, help="Reviewed structured dataset with binary outcome labels.")
    parser.add_argument("--output_csv", required=True, help="Path for the final binary dataset CSV.")
    parser.add_argument("--report_json", default=None, help="Optional path for a JSON validation report.")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def validate_columns(frame: pd.DataFrame) -> None:
    available_columns = set(frame.columns)
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in available_columns
        and not (column == "binary_outcome_label" and any(alias in available_columns for alias in LEGACY_BINARY_OUTCOME_COLUMNS))
    ]
    if missing:
        raise ValueError(f"Input dataset is missing required columns: {missing}")


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    validate_columns(frame)
    normalized = frame.copy()
    if "binary_outcome_label" not in normalized.columns:
        for alias in LEGACY_BINARY_OUTCOME_COLUMNS:
            if alias in normalized.columns:
                normalized = normalized.rename(columns={alias: "binary_outcome_label"})
                break

    normalized = normalized[REQUIRED_COLUMNS].copy()
    normalized["doc_id"] = normalized["doc_id"].map(clean_text)
    normalized["binary_outcome_label"] = normalized["binary_outcome_label"].map(clean_text)
    return normalized


def build_final_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_frame(frame)
    final = normalized[normalized["binary_outcome_label"].isin(BINARY_LABELS)].copy()
    final = final.drop_duplicates(subset=["doc_id"], keep="first")
    final = final.reset_index(drop=True)
    return final


def validation_report(frame: pd.DataFrame) -> dict[str, Any]:
    normalized = normalize_frame(frame)
    duplicate_doc_ids = int(normalized["doc_id"].duplicated().sum())
    label_counts = normalized["binary_outcome_label"].value_counts(dropna=False).to_dict()
    empty_counts = {
        column: int(normalized[column].map(clean_text).eq("").sum())
        for column in REQUIRED_COLUMNS
    }
    unexpected_labels = sorted(set(normalized["binary_outcome_label"].dropna().astype(str)) - set(BINARY_LABELS))
    return {
        "rows": int(len(frame)),
        "columns": REQUIRED_COLUMNS,
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "duplicate_doc_ids": duplicate_doc_ids,
        "empty_counts": empty_counts,
        "unexpected_labels": unexpected_labels,
        "is_binary_only": unexpected_labels == [],
    }


def write_report(report: dict[str, Any], path: Path | None) -> None:
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    report_json = Path(args.report_json) if args.report_json else None

    source = pd.read_csv(input_csv)
    final = build_final_dataset(source)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)

    report = validation_report(final)
    if not report["is_binary_only"]:
        raise ValueError(f"Final dataset contains unexpected labels: {report['unexpected_labels']}")
    if report["duplicate_doc_ids"]:
        raise ValueError(f"Final dataset contains {report['duplicate_doc_ids']} duplicate doc_id values.")
    write_report(report, report_json)


if __name__ == "__main__":
    main()
