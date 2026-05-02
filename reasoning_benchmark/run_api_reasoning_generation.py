#!/usr/bin/env python3
"""
Generate court-reasoning summaries with hosted API models.

This is the cleaned, script version of the GPT/DeepSeek/Gemini reasoning
notebook. It runs one or more API models on a fixed doc_id subset and writes
one CSV per model, plus an optional merged table.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CASE_ID_COL = "doc_id"
FACTS_COL = "facts"
PLAINTIFF_COL_ALIASES = ["claim_plaintiff", "plaintiff_claims"]
DEFENDANT_COL_ALIASES = ["claim_defendant", "defendant_claims"]
GOLD_REASONING_COLS = ["judicial_reasoning", "reasoning", "reasoning_summary", "court_reasoning", "rationale"]

MODEL_DEFAULTS = {
    "gpt": "gpt-4o",
    "deepseek": "deepseek-chat",
    "gemini": "gemini-2.0-flash",
}

PROMPT_REASONING_UNIFIED = """You are a legal expert.

Task: Write a concise summary of the COURT'S REASONING only.

Rules:
Use ONLY information explicitly stated in the INPUT fields below.
Do NOT add new facts, parties, dates, or procedural events.
Focus on the court's rationale, not the parties' arguments.

A good legal reasoning summary should:
- Explain WHY the court reached its decision,
- Show how key facts support the conclusion,
- Address competing arguments and indicate why one side was accepted or rejected.

Write 2-4 concrete sentences.
Include 2 short evidence cues copied from the INPUT, each no more than 12 words.

INPUT:
Facts:
{facts}

Plaintiff/Appellant Claims:
{plaintiff}

Defendant/Appellee Claims:
{defendant}

Return ONLY valid JSON:
{{"reasoning_summary":"...","evidence_cues":["...","..."]}}
"""

GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "reasoning_summary": {"type": "STRING"},
        "evidence_cues": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["reasoning_summary", "evidence_cues"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT/DeepSeek/Gemini reasoning generation.")
    parser.add_argument("--data_path", default="data/final_dataset.csv")
    parser.add_argument("--ids_path", default="data/reasoning_doc_ids.csv")
    parser.add_argument("--output_dir", default="results/reasoning_benchmark")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_DEFAULTS), default=["gpt", "deepseek", "gemini"])
    parser.add_argument("--gpt_model", default=MODEL_DEFAULTS["gpt"])
    parser.add_argument("--deepseek_model", default=MODEL_DEFAULTS["deepseek"])
    parser.add_argument("--gemini_model", default=MODEL_DEFAULTS["gemini"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max_input_chars", type=int, default=4500)
    parser.add_argument("--sleep_between_calls", type=float, default=0.45)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--merge_outputs", action="store_true")
    return parser.parse_args()


def resolve_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"Missing required column. Tried: {candidates}")


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def clip(value: Any, max_chars: int) -> str:
    return clean_text(value)[:max_chars]


def clean_json_text(text: str) -> str:
    cleaned = clean_text(text).replace("```json", "").replace("```", "").strip()
    if "{" in cleaned and "}" in cleaned:
        cleaned = cleaned[cleaned.find("{") : cleaned.rfind("}") + 1]
    return cleaned


def parse_json_or_none(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(clean_json_text(text))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def make_session() -> requests.Session:
    session = requests.Session()
    retry_cfg = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.0,
        status_forcelist=[408, 409, 425, 429, 500, 502, 503, 504],
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_cfg, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.headers.update({"Connection": "close"})
    return session


def post_json_with_retries(
    session: requests.Session,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: int,
    max_retries: int = 6,
    base_sleep: float = 1.3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
            data = response.json() if response.text else {}
            if response.status_code == 200:
                return data
            last_error = RuntimeError(f"HTTP {response.status_code}: {str(data)[:500]}")
        except Exception as exc:
            last_error = exc

        if attempt < max_retries:
            time.sleep(base_sleep * attempt + random.uniform(0.0, 0.7))

    raise RuntimeError(f"Request failed after retries: {last_error}")


def build_prompt(row: pd.Series, plaintiff_col: str, defendant_col: str, max_input_chars: int) -> str:
    return PROMPT_REASONING_UNIFIED.format(
        facts=clip(row.get(FACTS_COL, ""), max_input_chars),
        plaintiff=clip(row.get(plaintiff_col, ""), max_input_chars),
        defendant=clip(row.get(defendant_col, ""), max_input_chars),
    )


def call_gpt(session: requests.Session, prompt: str, model: str, timeout: int) -> tuple[dict[str, Any] | None, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    data = post_json_with_retries(
        session,
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return only JSON. No markdown. No explanation."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 280,
        },
        timeout,
    )
    raw_text = data["choices"][0]["message"].get("content", "")
    return parse_json_or_none(raw_text), raw_text


def call_deepseek(session: requests.Session, prompt: str, model: str, timeout: int) -> tuple[dict[str, Any] | None, str]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY")
    data = post_json_with_retries(
        session,
        "https://api.deepseek.com/chat/completions",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return only JSON. No markdown. No explanation."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 280,
        },
        timeout,
    )
    raw_text = data["choices"][0]["message"].get("content", "")
    return parse_json_or_none(raw_text), raw_text


def call_gemini(session: requests.Session, prompt: str, model: str, timeout: int) -> tuple[dict[str, Any] | None, str]:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY")
    data = post_json_with_retries(
        session,
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        {"Content-Type": "application/json"},
        {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 280,
                "responseMimeType": "application/json",
                "responseSchema": GEMINI_SCHEMA,
            },
        },
        timeout,
    )
    candidates = data.get("candidates") or []
    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    raw_text = parts[0].get("text", "") if parts else ""
    return parse_json_or_none(raw_text), raw_text


def load_selected_cases(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    frame = pd.read_csv(args.data_path, dtype={CASE_ID_COL: str})
    if CASE_ID_COL not in frame.columns:
        raise ValueError(f"Dataset must contain {CASE_ID_COL}.")

    plaintiff_col = resolve_column(frame, PLAINTIFF_COL_ALIASES)
    defendant_col = resolve_column(frame, DEFENDANT_COL_ALIASES)

    ids = pd.read_csv(args.ids_path, dtype={CASE_ID_COL: str})
    if CASE_ID_COL not in ids.columns:
        raise ValueError(f"IDs file must contain {CASE_ID_COL}.")
    ids = ids.dropna(subset=[CASE_ID_COL]).drop_duplicates(subset=[CASE_ID_COL]).head(args.limit).copy()
    ids[CASE_ID_COL] = ids[CASE_ID_COL].astype(str).str.strip()

    order = {doc_id: index for index, doc_id in enumerate(ids[CASE_ID_COL].tolist())}
    selected = frame[frame[CASE_ID_COL].isin(order)].drop_duplicates(subset=[CASE_ID_COL]).copy()
    selected["_order"] = selected[CASE_ID_COL].map(order)
    selected = selected.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return selected, ids, plaintiff_col, defendant_col


def run_model(
    args: argparse.Namespace,
    session: requests.Session,
    cases: pd.DataFrame,
    model_key: str,
    plaintiff_col: str,
    defendant_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model_name = getattr(args, f"{model_key}_model")

    for _, row in cases.iterrows():
        prompt = build_prompt(row, plaintiff_col, defendant_col, args.max_input_chars)
        parsed = None
        raw_text = ""
        error = ""
        try:
            if model_key == "gpt":
                parsed, raw_text = call_gpt(session, prompt, model_name, args.timeout)
            elif model_key == "deepseek":
                parsed, raw_text = call_deepseek(session, prompt, model_name, args.timeout)
            elif model_key == "gemini":
                parsed, raw_text = call_gemini(session, prompt, model_name, args.timeout)
            if parsed is None:
                error = "json_parse_failed"
        except Exception as exc:
            error = str(exc)

        rows.append(
            {
                CASE_ID_COL: row[CASE_ID_COL],
                "model_key": model_key,
                "model_name": model_name,
                "reasoning_summary": (parsed or {}).get("reasoning_summary"),
                "evidence_cues": json.dumps((parsed or {}).get("evidence_cues"), ensure_ascii=False) if parsed else "",
                "raw_text": raw_text,
                "error": error,
            }
        )
        time.sleep(args.sleep_between_calls)

    return pd.DataFrame(rows)


def save_minimal_merge(args: argparse.Namespace, ids: pd.DataFrame, frame: pd.DataFrame, outputs: dict[str, pd.DataFrame]) -> None:
    gold_col = next((column for column in GOLD_REASONING_COLS if column in frame.columns), None)
    merged = ids[[CASE_ID_COL]].copy()
    if gold_col:
        merged = merged.merge(
            frame[[CASE_ID_COL, gold_col]].drop_duplicates(subset=[CASE_ID_COL]).rename(columns={gold_col: "gold_reasoning"}),
            on=CASE_ID_COL,
            how="left",
        )
    for model_key, predictions in outputs.items():
        merged = merged.merge(
            predictions[[CASE_ID_COL, "reasoning_summary"]].rename(columns={"reasoning_summary": f"{model_key}_reasoning"}),
            on=CASE_ID_COL,
            how="left",
        )
    output_path = Path(args.output_dir) / "api_reasoning_same_subset_merged.csv"
    merged.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases, ids, plaintiff_col, defendant_col = load_selected_cases(args)
    session = make_session()

    outputs: dict[str, pd.DataFrame] = {}
    for model_key in args.models:
        predictions = run_model(args, session, cases, model_key, plaintiff_col, defendant_col)
        model_name = getattr(args, f"{model_key}_model").replace("/", "__")
        predictions.to_csv(output_dir / f"reasoning_{model_key}_{model_name}.csv", index=False)
        outputs[model_key] = predictions

    if args.merge_outputs:
        full_frame = pd.read_csv(args.data_path, dtype={CASE_ID_COL: str})
        save_minimal_merge(args, ids, full_frame, outputs)


if __name__ == "__main__":
    main()
