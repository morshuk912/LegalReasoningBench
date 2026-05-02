from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
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
CASE_ID_ALIASES = ["doc_id", "case_id", "document_id", "id"]

SYSTEM_PROMPT = "You are a careful legal expert. Follow the instructions exactly."

PROMPTS = {
    "cot": """You are a legal expert.

Task: Predict the final appellate disposition.

Think internally through:
1. Whether the lower-court judgment remained in effect.
2. Whether the appellate court reversed, vacated, remanded, dismissed, or changed any part of the judgment.
3. Whether the outcome is mixed.

Do not reveal your chain of thought. Return only the final label.

Input:
Facts:
{facts}

Plaintiff/Appellant Claims:
{plaintiff}

Defendant/Appellee Claims:
{defendant}

Return ONLY valid JSON:
{{"predicted_label":"Affirmed"|"Reversed/Vacated"}}
""",
    "story_facts": """You are a legal expert.

Task: Determine the FINAL DISPOSITION (binary) of the appellate court from the facts/story summary only.

Rule hierarchy:
1. If the text explicitly states the appellate disposition, decide strictly from that language.
2. If no explicit disposition language appears, do not infer reversal. Output Affirmed.

Labels:
- Affirmed
- Reversed/Vacated, including reversal, vacatur, remand, dismissal, or mixed outcome.

Input facts/story:
{facts}

Return ONLY valid JSON:
{{"predicted_label":"Affirmed"|"Reversed/Vacated"}}
""",
}

DEFAULT_MODELS = {
    "gpt": os.environ.get("OPENAI_MODEL", "gpt-4o"),
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.0-flash",
}

GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {"predicted_label": {"type": "STRING"}},
    "required": ["predicted_label"],
}


def parse_common_args(provider: str, experiment: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run {provider} {experiment} prediction experiment.")
    parser.add_argument("--data_path", default="data/final_dataset.csv")
    parser.add_argument("--output_dir", default=f"results/ablation/{provider}_{experiment}")
    parser.add_argument("--model", default=DEFAULT_MODELS[provider])
    parser.add_argument("--sample_n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_chars", type=int, default=4500)
    parser.add_argument("--max_tokens", type=int, default=180)
    parser.add_argument("--sleep_between_calls", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def clip(value: Any, max_chars: int) -> str:
    return clean_text(value)[:max_chars]


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


def load_sample(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, str | None]]:
    frame = pd.read_csv(args.data_path)
    columns = {
        "label": resolve_column(frame, LABEL_ALIASES),
        "facts": resolve_column(frame, FACTS_ALIASES),
        "plaintiff": resolve_column(frame, PLAINTIFF_ALIASES, required=False),
        "defendant": resolve_column(frame, DEFENDANT_ALIASES, required=False),
        "doc_id": resolve_column(frame, CASE_ID_ALIASES, required=False),
    }
    frame = frame.copy()
    frame["_label"] = frame[columns["label"]].map(clean_text)
    frame = frame[frame["_label"].isin(BINARY_LABELS)].reset_index(drop=True)

    sample_n = min(args.sample_n, len(frame))
    if sample_n == len(frame):
        return frame.sample(frac=1, random_state=args.seed).reset_index(drop=True), columns

    proportions = frame["_label"].value_counts(normalize=True)
    counts = {label: int(round(float(proportions[label]) * sample_n)) for label in proportions.index}
    diff = sample_n - sum(counts.values())
    if diff:
        counts[proportions.idxmax()] += diff

    parts = []
    for label, count in counts.items():
        subset = frame[frame["_label"] == label]
        if count > 0:
            parts.append(subset.sample(n=min(count, len(subset)), random_state=args.seed))
    return pd.concat(parts).sample(frac=1, random_state=args.seed).reset_index(drop=True), columns


def build_prompt(experiment: str, row: pd.Series, columns: dict[str, str | None], max_input_chars: int) -> str:
    return PROMPTS[experiment].format(
        facts=clip(row.get(columns["facts"], ""), max_input_chars),
        plaintiff=clip(row.get(columns["plaintiff"], ""), max_input_chars) if columns["plaintiff"] else "",
        defendant=clip(row.get(columns["defendant"], ""), max_input_chars) if columns["defendant"] else "",
    )


def call_gpt(client: OpenAI, model: str, prompt: str, max_tokens: int, max_retries: int = 4) -> tuple[dict[str, Any] | None, str, str]:
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


def call_deepseek(model: str, prompt: str, max_tokens: int, timeout: int, max_retries: int = 4) -> tuple[dict[str, Any] | None, str, str]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only JSON. No markdown. No explanation."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post("https://api.deepseek.com/chat/completions", headers=headers, data=json.dumps(payload), timeout=timeout)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            raw = response.json()["choices"][0]["message"].get("content", "")
            return extract_json(raw), raw, ""
        except Exception as exc:
            last_error = str(exc)
            time.sleep(float(attempt))
    return None, "", last_error


def call_gemini(model: str, prompt: str, max_tokens: int, timeout: int, max_retries: int = 4) -> tuple[dict[str, Any] | None, str, str]:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_SCHEMA,
        },
    }
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=timeout)
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            candidates = response.json().get("candidates") or []
            parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
            raw = parts[0].get("text", "") if parts else ""
            return extract_json(raw), raw, ""
        except Exception as exc:
            last_error = str(exc)
            time.sleep(float(attempt))
    return None, "", last_error


def metric_payload(predictions: pd.DataFrame) -> dict[str, Any]:
    valid = predictions[predictions["pred"].isin(BINARY_LABELS)].copy()
    payload = {
        "rows": int(len(predictions)),
        "valid_predictions": int(len(valid)),
        "valid_prediction_rate": float(len(valid) / len(predictions)) if len(predictions) else 0.0,
    }
    if len(valid):
        payload.update(
            {
                "accuracy": float(accuracy_score(valid["gold"], valid["pred"])),
                "macro_f1": float(f1_score(valid["gold"], valid["pred"], labels=BINARY_LABELS, average="macro", zero_division=0)),
                "confusion_matrix": confusion_matrix(valid["gold"], valid["pred"], labels=BINARY_LABELS).tolist(),
                "classification_report": classification_report(valid["gold"], valid["pred"], labels=BINARY_LABELS, output_dict=True, zero_division=0),
            }
        )
    return payload


def call_provider(provider: str, args: argparse.Namespace, client: OpenAI | None, prompt: str) -> tuple[dict[str, Any] | None, str, str]:
    if provider == "gpt":
        return call_gpt(client, args.model, prompt, args.max_tokens)  # type: ignore[arg-type]
    if provider == "deepseek":
        return call_deepseek(args.model, prompt, args.max_tokens, args.timeout)
    if provider == "gemini":
        return call_gemini(args.model, prompt, args.max_tokens, args.timeout)
    raise ValueError(f"Unknown provider: {provider}")


def run_experiment(provider: str, experiment: str) -> None:
    args = parse_common_args(provider, experiment)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample, columns = load_sample(args)
    client = OpenAI() if provider == "gpt" else None
    rows: list[dict[str, Any]] = []

    for row_index, row in tqdm(sample.iterrows(), total=len(sample), desc=f"{provider}:{experiment}"):
        prompt = build_prompt(experiment, row, columns, args.max_input_chars)
        parsed, raw, error = call_provider(provider, args, client, prompt)
        raw_pred = (parsed or {}).get("predicted_label", "")
        rows.append(
            {
                "row_index": row_index,
                "doc_id": clean_text(row.get(columns["doc_id"], row_index)) if columns["doc_id"] else str(row_index),
                "provider": provider,
                "model": args.model,
                "experiment": experiment,
                "gold": row["_label"],
                "pred": normalize_label(raw_pred),
                "raw_pred": raw_pred,
                "error": error,
                "raw_response": raw,
            }
        )
        time.sleep(args.sleep_between_calls)

    predictions = pd.DataFrame(rows)
    predictions.to_csv(output_dir / f"{provider}_{experiment}_predictions.csv", index=False)
    (output_dir / f"{provider}_{experiment}_metrics.json").write_text(
        json.dumps(metric_payload(predictions), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
