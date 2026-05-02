#!/usr/bin/env python3
"""
Run GPT binary outcome prediction on a 200-case stratified sample.

Uses Prompt 2 from the thesis LLM-prediction pilot and runs both text variants
by default: facts only and facts + claims.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm.auto import tqdm


NEG_LABEL = "Affirmed"
POS_LABEL = "Reversed/Vacated"
BINARY_LABELS = [NEG_LABEL, POS_LABEL]
LABEL_ALIASES = ["binary_outcome_label", "final_outcome_group_expert4", "outcome_label"]
FACTS_ALIASES = ["facts"]
PLAINTIFF_ALIASES = ["plaintiff_claims", "claim_plaintiff"]
DEFENDANT_ALIASES = ["defendant_claims", "claim_defendant"]
DOC_ID_ALIASES = ["doc_id", "case_id", "document_id", "id"]

SYSTEM_PROMPT = "You are a careful legal expert. Follow the instructions exactly."
PROMPT_2 = """You are a legal expert.

Task: Determine the FINAL DISPOSITION.

Based on the provided case information, answer the following question:
Did the appellate court ultimately leave the lower court's judgment in effect?

- If YES, output Affirmed.
- If NO, output Reversed/Vacated.

Important:
- Do NOT decide who is right based on the facts.
- Focus ONLY on what the court ultimately did.
- If the disposition is mixed, output Reversed/Vacated.
- If the case was reversed, vacated, remanded due to error, dismissed, or changed in part, output Reversed/Vacated.

Input:
{input_text}

Return ONLY valid JSON:
{{"predicted_label":"Affirmed"|"Reversed/Vacated"}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT Prompt-2 prediction on 200 cases.")
    parser.add_argument("--data_path", default="data/final_dataset.csv")
    parser.add_argument("--output_dir", default="results/prediction_benchmark/gpt_200")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--sample_n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text_variants", nargs="+", choices=["facts", "facts_claims"], default=["facts", "facts_claims"])
    parser.add_argument("--max_input_chars", type=int, default=4500)
    parser.add_argument("--max_tokens", type=int, default=120)
    parser.add_argument("--sleep_between_calls", type=float, default=0.2)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def resolve_column(frame: pd.DataFrame, aliases: list[str], required: bool = True) -> str | None:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    if required:
        raise ValueError(f"Missing required column. Tried: {aliases}")
    return None


def normalize_label(value: Any) -> str:
    lowered = clean_text(value).lower()
    if "affirm" in lowered:
        return NEG_LABEL
    if any(term in lowered for term in ["reverse", "revers", "vacat", "remand", "dismiss"]):
        return POS_LABEL
    return ""


def extract_json(text: str) -> dict[str, Any]:
    raw = clean_text(text)
    raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("No JSON object found.")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("JSON output is not an object.")
    return obj


def stratified_sample(frame: pd.DataFrame, label_col: str, sample_n: int, seed: int) -> pd.DataFrame:
    frame = frame[frame[label_col].isin(BINARY_LABELS)].copy()
    sample_n = min(sample_n, len(frame))
    proportions = frame[label_col].value_counts(normalize=True)
    counts = {label: int(round(float(proportions[label]) * sample_n)) for label in proportions.index}
    diff = sample_n - sum(counts.values())
    if diff:
        counts[proportions.idxmax()] += diff
    parts = [frame[frame[label_col] == label].sample(n=min(count, (frame[label_col] == label).sum()), random_state=seed) for label, count in counts.items() if count > 0]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def build_input(row: pd.Series, columns: dict[str, str | None], variant: str, max_chars: int) -> str:
    facts = clean_text(row.get(columns["facts"], ""))[:max_chars]
    if variant == "facts":
        return f"Facts:\n{facts}"
    plaintiff = clean_text(row.get(columns["plaintiff"], ""))[:max_chars] if columns["plaintiff"] else ""
    defendant = clean_text(row.get(columns["defendant"], ""))[:max_chars] if columns["defendant"] else ""
    return f"Facts:\n{facts}\n\nPlaintiff/Appellant Claims:\n{plaintiff}\n\nDefendant/Appellee Claims:\n{defendant}"


def call_model(client: OpenAI, model: str, prompt: str, max_tokens: int, max_retries: int = 4) -> tuple[dict[str, Any] | None, str, str]:
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content or ""
            return extract_json(raw), raw, ""
        except Exception as exc:
            last_error = str(exc)
            time.sleep(float(attempt))
    return None, "", last_error


def save_metrics(predictions: pd.DataFrame, output_path: Path) -> None:
    valid = predictions[predictions["pred"].isin(BINARY_LABELS)].copy()
    metrics = {
        "rows": int(len(predictions)),
        "valid_predictions": int(len(valid)),
        "valid_prediction_rate": float(len(valid) / len(predictions)) if len(predictions) else 0.0,
    }
    if len(valid):
        metrics.update(
            {
                "accuracy": float(accuracy_score(valid["gold"], valid["pred"])),
                "macro_f1": float(f1_score(valid["gold"], valid["pred"], labels=BINARY_LABELS, average="macro", zero_division=0)),
                "confusion_matrix": confusion_matrix(valid["gold"], valid["pred"], labels=BINARY_LABELS).tolist(),
                "classification_report": classification_report(valid["gold"], valid["pred"], labels=BINARY_LABELS, output_dict=True, zero_division=0),
            }
        )
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Missing OPENAI_API_KEY.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.data_path)
    columns = {
        "label": resolve_column(frame, LABEL_ALIASES),
        "facts": resolve_column(frame, FACTS_ALIASES),
        "plaintiff": resolve_column(frame, PLAINTIFF_ALIASES, required=False),
        "defendant": resolve_column(frame, DEFENDANT_ALIASES, required=False),
        "doc_id": resolve_column(frame, DOC_ID_ALIASES, required=False),
    }
    frame["_label"] = frame[columns["label"]].map(clean_text)
    sample = stratified_sample(frame, "_label", args.sample_n, args.seed)
    client = OpenAI()

    for variant in args.text_variants:
        rows = []
        for row_index, row in tqdm(sample.iterrows(), total=len(sample), desc=f"gpt:{variant}"):
            input_text = build_input(row, columns, variant, args.max_input_chars)
            parsed, raw, error = call_model(client, args.model, PROMPT_2.format(input_text=input_text), args.max_tokens)
            raw_pred = (parsed or {}).get("predicted_label", "")
            rows.append(
                {
                    "row_index": row_index,
                    "doc_id": clean_text(row.get(columns["doc_id"], row_index)) if columns["doc_id"] else str(row_index),
                    "text_variant": variant,
                    "gold": row["_label"],
                    "pred": normalize_label(raw_pred),
                    "raw_pred": raw_pred,
                    "error": error,
                    "raw_response": raw,
                }
            )
            time.sleep(args.sleep_between_calls)
        predictions = pd.DataFrame(rows)
        predictions.to_csv(output_dir / f"gpt_{variant}_predictions.csv", index=False)
        save_metrics(predictions, output_dir / f"gpt_{variant}_metrics.json")


if __name__ == "__main__":
    main()
