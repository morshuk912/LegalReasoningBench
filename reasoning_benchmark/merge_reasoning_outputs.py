#!/usr/bin/env python3
"""
Merge generated reasoning outputs into a single comparison table.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge reasoning benchmark outputs.")
    parser.add_argument("--base_csv", default="data/reasoning_minimal_same50.csv")
    parser.add_argument("--results_dir", default="results/reasoning_benchmark")
    parser.add_argument("--output_csv", default="results/reasoning_benchmark/reasoning_same50_merged.csv")
    parser.add_argument("--gemma_predictions_csv", default=None)
    parser.add_argument("--gemma_full_ft_predictions_csv", default=None)
    return parser.parse_args()


def clean_reasoning(value):
    if pd.isna(value):
        return pd.NA
    text = " ".join(str(value).split()).strip()
    text = re.sub(r"^(Output:\s*|\[/INST\]\s*|COURT'S REASONING:\s*)+", "", text).strip()
    if not text:
        return pd.NA
    if re.fullmatch(r"[0-9\s\-–.,]+", text):
        return pd.NA
    if re.search(r"\b(?:[A-T]\s+){8,}[A-T]\b", text):
        return pd.NA
    return text


def load_model_reasoning(path: Path, output_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"doc_id", "reasoning_summary"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")
    subset = frame[["doc_id", "reasoning_summary"]].copy()
    subset["reasoning_summary"] = subset["reasoning_summary"].apply(clean_reasoning)
    return subset.rename(columns={"reasoning_summary": output_col})


def load_gemma_reasoning(path: Path, output_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"doc_id", "pred_judicial_reasoning"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")
    subset = frame[["doc_id", "pred_judicial_reasoning"]].copy()
    subset["pred_judicial_reasoning"] = subset["pred_judicial_reasoning"].apply(clean_reasoning)
    subset = subset.dropna(subset=["doc_id"]).drop_duplicates(subset=["doc_id"], keep="first")
    return subset.rename(columns={"pred_judicial_reasoning": output_col})


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    merged = pd.read_csv(args.base_csv)
    if "doc_id" not in merged.columns:
        raise ValueError("Base CSV must contain doc_id.")

    for model_key in ["saullm", "lawma", "lawllm"]:
        path = results_dir / f"reasoning_{model_key}_same50.csv"
        if path.exists():
            merged = merged.merge(load_model_reasoning(path, f"{model_key}_reasoning"), on="doc_id", how="left")

    if args.gemma_predictions_csv:
        merged = merged.merge(
            load_gemma_reasoning(Path(args.gemma_predictions_csv), "gemma_instruction_tuned_reasoning"),
            on="doc_id",
            how="left",
        )
    if args.gemma_full_ft_predictions_csv:
        merged = merged.merge(
            load_gemma_reasoning(Path(args.gemma_full_ft_predictions_csv), "gemma_full_ft_reasoning"),
            on="doc_id",
            how="left",
        )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Rows: {len(merged)}")


if __name__ == "__main__":
    main()
