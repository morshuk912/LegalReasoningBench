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
import re
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
TEXT_STAT_COLUMNS = ["full_document", "facts", "reasoning", "claim_plaintiff", "claim_defendant"]
WORD_PATTERN = re.compile(r"[A-Za-z0-9_]+")


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


def word_count(value: Any) -> int:
    return len(WORD_PATTERN.findall(clean_text(value)))


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


def text_word_statistics(frame: pd.DataFrame) -> dict[str, dict[str, int | float]]:
    stats: dict[str, dict[str, int | float]] = {}
    for column in TEXT_STAT_COLUMNS:
        counts = frame[column].map(word_count)
        nonempty_counts = counts[counts > 0]
        stats[column] = {
            "nonempty_rows": int((counts > 0).sum()),
            "empty_rows": int((counts == 0).sum()),
            "avg_words_all_rows": round(float(counts.mean()), 1),
            "median_words_all_rows": round(float(counts.median()), 1),
            "avg_words_nonempty_rows": round(float(nonempty_counts.mean()), 1) if len(nonempty_counts) else 0.0,
            "median_words_nonempty_rows": round(float(nonempty_counts.median()), 1) if len(nonempty_counts) else 0.0,
            "max_words": int(counts.max()),
        }
    return stats


def validation_report(frame: pd.DataFrame, source_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    normalized = normalize_frame(frame)
    source = normalize_frame(source_frame) if source_frame is not None else normalized
    duplicate_doc_ids = int(normalized["doc_id"].duplicated().sum())
    label_counts = normalized["binary_outcome_label"].value_counts(dropna=False).to_dict()
    source_outcome_labels = source["final_outcome_label"].map(clean_text)
    empty_counts = {
        column: int(normalized[column].map(clean_text).eq("").sum())
        for column in REQUIRED_COLUMNS
    }
    unexpected_labels = sorted(set(normalized["binary_outcome_label"].dropna().astype(str)) - set(BINARY_LABELS))
    return {
        "rows": int(len(frame)),
        "columns": REQUIRED_COLUMNS,
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "unique_outcome_labels_before_binary_grouping": int(source_outcome_labels[source_outcome_labels != ""].nunique()),
        "outcome_label_counts_before_binary_grouping": {
            str(label): int(count)
            for label, count in source_outcome_labels[source_outcome_labels != ""].value_counts().to_dict().items()
        },
        "text_word_statistics": text_word_statistics(normalized),
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

    report = validation_report(final, source)
    if not report["is_binary_only"]:
        raise ValueError(f"Final dataset contains unexpected labels: {report['unexpected_labels']}")
    if report["duplicate_doc_ids"]:
        raise ValueError(f"Final dataset contains {report['duplicate_doc_ids']} duplicate doc_id values.")
    write_report(report, report_json)


if __name__ == "__main__":
    main()
