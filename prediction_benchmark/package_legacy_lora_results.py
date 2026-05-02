#!/usr/bin/env python3
"""
Package legacy LoRA prediction CSVs into the final thesis result format.

The legacy runs saved only gold/pred columns. This script verifies that each
legacy file is in the same row order as the final binary dataset, then adds
doc_id and deterministic StratifiedKFold fold assignments and writes metrics in
the same shape used by the cleaned benchmark scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


NEG_LABEL = "Affirmed"
POS_LABEL = "Reversed/Vacated"
BINARY_LABELS = [NEG_LABEL, POS_LABEL]
LABEL_COL = "binary_outcome_label"
LEGACY_LABEL_COLS = ["final_outcome_group_expert4"]

RUNS = [
    {
        "model": "Equall/Saul-7B-Instruct-v1",
        "model_key": "saullm",
        "text_variant": "facts",
        "legacy_predictions": "results_saullm_lora_cv_facts/predictions_facts.csv",
    },
    {
        "model": "Equall/Saul-7B-Instruct-v1",
        "model_key": "saullm",
        "text_variant": "facts_claims",
        "legacy_predictions": "results_saullm_lora_cv_facts_claims/predictions_facts_claims.csv",
    },
    {
        "model": "lexlms/legal-longformer-large",
        "model_key": "lexlm",
        "text_variant": "facts",
        "legacy_predictions": "results_lexlm_cv_facts/predictions_facts.csv",
    },
    {
        "model": "lexlms/legal-longformer-large",
        "model_key": "lexlm",
        "text_variant": "facts_claims",
        "legacy_predictions": "results_lexlm_cv_facts_claims/predictions_facts_claims.csv",
    },
    {
        "model": "AdaptLLM/law-LLM-13B",
        "model_key": "lawllm",
        "text_variant": "facts",
        "legacy_predictions": "results_lawllm_lora_cv_facts/predictions_facts.csv",
    },
    {
        "model": "AdaptLLM/law-LLM-13B",
        "model_key": "lawllm",
        "text_variant": "facts_claims",
        "legacy_predictions": "results_lawllm_lora_cv_facts_claims/predictions_facts_claims.csv",
    },
    {
        "model": "ricdomolm/lawma-8b",
        "model_key": "lawma",
        "text_variant": "facts",
        "legacy_predictions": "results_lawma_lora_cv_facts/predictions_facts.csv",
    },
    {
        "model": "ricdomolm/lawma-8b",
        "model_key": "lawma",
        "text_variant": "facts_claims",
        "legacy_predictions": "results_lawma_lora_cv_facts_claims/predictions_facts_claims.csv",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package old LoRA prediction CSVs into final result artifacts.")
    parser.add_argument("--data_path", default="data/final_dataset.csv")
    parser.add_argument("--output_dir", default="results/lora_benchmark_packaged")
    parser.add_argument("--legacy_results_dir", default="legacy_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--allow_lexlm", action="store_true", help="Include LexLM legacy outputs despite known all-Affirmed predictions.")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def load_binary_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if LABEL_COL not in df.columns:
        for legacy_col in LEGACY_LABEL_COLS:
            if legacy_col in df.columns:
                df = df.rename(columns={legacy_col: LABEL_COL})
                break
    required = ["doc_id", LABEL_COL]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df[LABEL_COL] = df[LABEL_COL].map(clean_text)
    df = df[df[LABEL_COL].isin(BINARY_LABELS)].copy()
    df = df.drop_duplicates(subset=["doc_id"], keep="first").reset_index(drop=True)
    return df


def fold_assignments(labels: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.empty(len(labels), dtype=int)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        folds[test_idx] = fold
    return folds


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=BINARY_LABELS, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=BINARY_LABELS, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=BINARY_LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=BINARY_LABELS, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=BINARY_LABELS).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=BINARY_LABELS, output_dict=True, zero_division=0),
    }


def package_run(
    run: dict[str, str],
    df: pd.DataFrame,
    folds: np.ndarray,
    output_dir: Path,
    legacy_results_dir: Path,
    seed: int,
    n_splits: int,
) -> dict[str, Any]:
    legacy_path = legacy_results_dir / run["legacy_predictions"]
    if not legacy_path.exists():
        raise FileNotFoundError(f"Missing legacy prediction file: {legacy_path}")

    legacy = pd.read_csv(legacy_path)
    if list(legacy.columns) != ["gold", "pred"]:
        raise ValueError(f"{legacy_path} must contain exactly gold,pred columns.")
    y_true = df[LABEL_COL].to_numpy(dtype=object)
    legacy_gold = legacy["gold"].map(clean_text).to_numpy(dtype=object)
    if len(legacy) != len(df) or not np.array_equal(legacy_gold, y_true):
        raise ValueError(f"{legacy_path} does not match final dataset row order.")

    y_pred = legacy["pred"].map(clean_text).to_numpy(dtype=object)
    metrics = metric_dict(y_true, y_pred)
    metrics.update(
        {
            "model": run["model"],
            "model_key": run["model_key"],
            "text_variant": run["text_variant"],
            "source": "legacy_lora_predictions_packaged_without_rerun",
            "legacy_predictions": str(legacy_path),
            "n_splits": n_splits,
            "seed": seed,
            "notes": "Legacy CSV had only gold/pred. doc_id and fold were reconstructed from final_dataset.csv and StratifiedKFold(seed=42).",
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_key = f"lora_{run['model_key']}_{run['text_variant']}"
    pd.DataFrame(
        {
            "doc_id": df["doc_id"].astype(str),
            "fold": folds,
            "gold": y_true,
            "pred": y_pred,
        }
    ).to_csv(output_dir / f"{artifact_key}_predictions.csv", index=False)
    (output_dir / f"{artifact_key}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def save_summary(output_dir: Path, metrics_list: list[dict[str, Any]]) -> None:
    rows = [
        {
            "model": metrics["model"],
            "model_key": metrics["model_key"],
            "text_variant": metrics["text_variant"],
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        }
        for metrics in metrics_list
    ]
    pd.DataFrame(rows).to_csv(output_dir / "metrics_summary.csv", index=False)
    (output_dir / "metrics_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    df = load_binary_dataset(Path(args.data_path))
    folds = fold_assignments(df[LABEL_COL].to_numpy(dtype=object), args.n_splits, args.seed)
    runs = RUNS if args.allow_lexlm else [run for run in RUNS if run["model_key"] != "lexlm"]
    metrics = [
        package_run(
            run,
            df,
            folds,
            Path(args.output_dir),
            Path(args.legacy_results_dir),
            args.seed,
            args.n_splits,
        )
        for run in runs
    ]
    save_summary(Path(args.output_dir), metrics)
    print(f"Packaged {len(metrics)} LoRA runs into {args.output_dir}")


if __name__ == "__main__":
    main()
