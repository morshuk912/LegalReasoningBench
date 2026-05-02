#!/usr/bin/env python3
"""
Create the structured legal-case dataset from US-LegalKit.

This script is the cleaned version of the original exploratory notebook flow.
It loads cases from Hugging Face, asks an OpenAI model to extract the structured
fields used in the thesis, assigns an initial outcome label from the extracted
conclusion, maps the label into the final thesis binary task when applicable,
and appends deduplicated rows to a master CSV.

Required environment:
  OPENAI_API_KEY

Example:
  python extract_structured_cases.py \
    --master_csv uslegalkit_structured_dataset.csv \
    --intermediate_csv intermediate_results.csv \
    --batch_size 600
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


STRUCTURED_COLUMNS = [
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
EXCLUDED_LABEL = "Excluded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured fields from US-LegalKit cases.")
    parser.add_argument("--dataset_name", default="macadeliccc/US-LegalKit")
    parser.add_argument("--split", default="train")
    parser.add_argument("--master_csv", default="uslegalkit_structured_dataset.csv")
    parser.add_argument("--intermediate_csv", default="intermediate_results.csv")
    parser.add_argument("--batch_size", type=int, default=600)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_doc_chars", type=int, default=4000)
    return parser.parse_args()


def clean_reply(reply: str | None) -> str:
    text = (reply or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def stable_doc_id(record: dict[str, Any]) -> str:
    if record.get("id") is not None:
        return str(record["id"])
    document = str(record.get("document", ""))
    return hashlib.sha256(document.encode("utf-8")).hexdigest()


def stable_sort_key(record: dict[str, Any]) -> tuple[str, str]:
    if record.get("id") is not None:
        return ("id", str(record["id"]))
    return ("hash", stable_doc_id(record))


def load_processed_ids(*paths: Path) -> set[str]:
    processed: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if "doc_id" in frame.columns:
            processed.update(frame["doc_id"].dropna().astype(str).tolist())
    return processed


def select_next_batch(records: list[dict[str, Any]], processed_ids: set[str], batch_size: int) -> list[dict[str, Any]]:
    batch: list[dict[str, Any]] = []
    for record in sorted(records, key=stable_sort_key):
        if stable_doc_id(record) in processed_ids:
            continue
        batch.append(record)
        if len(batch) >= batch_size:
            break
    return batch


def has_pair(text: str, first: str, second: str) -> bool:
    return re.search(rf"\b{first}\w*\b(?:\W+\w+){{0,12}}\b{second}\w*\b", text) is not None


def has_remand_cue(text: str) -> bool:
    cues = [
        r"\bremand(ed)?\b",
        r"\bfor\s+further\s+proceedings\b",
        r"\bfor\s+(a\s+)?new\s+trial\b",
        r"\bfor\s+resentencing\b",
        r"\bproceedings\s+consistent\s+with\s+this\s+opinion\b",
        r"\bwith\s+instructions\s+to\b",
        r"\bfor\s+entry\s+of\s+a\s+new\s+judgment\b",
    ]
    return any(re.search(pattern, text) for pattern in cues)


def is_motion_to_dismiss_denied(text: str) -> bool:
    return bool(re.search(r"\bmotion\s+to\s+dismiss\b.*\b(denied|overruled)\b", text))


def is_appeal_dismissed(text: str) -> bool:
    appeal_words = r"(appeal|cross[-\s]?appeal|interlocutory\s+appeal|appeals)"
    petition_words = r"(petition|application)\s+(for\s+)?(review|rehearing|certiorari|leave\s+to\s+appeal|writ)"
    writ_words = r"(writ|certiorari|mandamus|prohibition|habeas)"
    if re.search(rf"\b{appeal_words}\b.*\b(dismiss(ed|al))\b", text):
        return True
    if re.search(rf"\b(dismiss(ed|al))\b.*\b{appeal_words}\b", text):
        return True
    if re.search(rf"\b({petition_words}|{writ_words})\b.*\b(dismiss(ed|al))\b", text):
        return True
    if re.search(rf"\b(dismiss(ed|al))\b.*\b({petition_words}|{writ_words})\b", text):
        return True
    return bool(re.search(r"\b(dismissed\s+as\s+moot|dismissed\s+for\s+lack\s+of\s+jurisdiction|untimely\s+appeal)\b", text))


def is_case_dismissed(text: str) -> bool:
    if is_motion_to_dismiss_denied(text):
        return False
    if re.search(r"\b(appeal|cross[-\s]?appeal|interlocutory\s+appeal|appeals)\b", text):
        return False
    case_objects = r"(complaint|indictment|information|action|case|claim|count|petition)"
    if re.search(rf"\b{case_objects}\b.*\b(dismiss(ed|al))\b", text):
        return True
    if re.search(rf"\b(dismiss(ed|al))\b.*\b{case_objects}\b", text):
        return True
    return bool(re.search(r"\bdismiss(ed|al)\b", text))


def classify_detailed_outcome_label(conclusion_text: Any) -> str:
    if not isinstance(conclusion_text, str):
        return "Unknown"
    text = conclusion_text.lower().strip()
    if not text:
        return "Unknown"

    if "clarif" in text or "opinion clarified" in text or "motion for clarification" in text:
        return "Clarified"
    if has_pair(text, "affirm", "remand") or has_pair(text, "remand", "affirm"):
        return "Affirmed and Remanded"
    if has_pair(text, "reverse", "remand") or has_pair(text, "remand", "reverse"):
        return "Reversed and Remanded"
    if has_pair(text, "vacat", "remand") or has_pair(text, "remand", "vacat"):
        return "Vacated and Remanded"
    if is_appeal_dismissed(text):
        return "Appeal Dismissed"
    if is_case_dismissed(text):
        return "Dismissed"
    if "affirm" in text:
        return "Affirmed"
    if "reverse" in text:
        return "Reversed"
    if "remand" in text or has_remand_cue(text):
        return "Remanded"
    if "vacat" in text:
        return "Vacated"
    if "writ awarded" in text or "granted" in text:
        return "Granted"
    if "certiorari denied" in text or "denied" in text:
        return "Denied"
    if re.search(r"\bmodified\b|\breduced\s+to\b|\bamended\s+to\b", text):
        return "Modified"
    return "Unknown"


def map_to_final_binary_group(detailed_label: str) -> str:
    """Map detailed extraction labels to the final thesis binary task.

    Rows mapped to EXCLUDED_LABEL are kept in the master file for traceability
    but are removed by prepare_final_binary_dataset.py.
    """
    affirmed_labels = {
        "Affirmed",
        "Affirmed and Remanded",
        "Clarified",
        "Modified",
    }
    reversed_vacated_labels = {
        "Reversed",
        "Reversed and Remanded",
        "Remanded",
        "Vacated",
        "Vacated and Remanded",
    }
    if detailed_label in affirmed_labels:
        return "Affirmed"
    if detailed_label in reversed_vacated_labels:
        return "Reversed/Vacated"
    return EXCLUDED_LABEL


def truncate_document(document: str, max_chars: int) -> str:
    document = document.strip()
    if len(document) <= max_chars:
        return document
    half = max_chars // 2
    return document[:half] + "\n...\n" + document[-half:]


def build_extraction_prompt(document: str) -> str:
    return f"""You are a legal document analysis expert.

Your task is to extract structured information from the following U.S. court decision.

Return valid JSON only with these keys:
- facts
- claim_plaintiff
- claim_defendant
- reasoning
- conclusion

Field requirements:
1. facts: A clear chronological summary of the real-world facts and events that led to the dispute. Identify party roles when available. Exclude judicial reasoning.
2. claim_plaintiff: A concise neutral summary of the plaintiff/appellant claims.
3. claim_defendant: A concise neutral summary of the defendant/appellee claims.
4. reasoning: The judge's legal reasoning only.
5. conclusion: A short sentence stating the final court decision.

--- LEGAL DOCUMENT START ---
{document}
--- LEGAL DOCUMENT END ---
"""


def parse_model_json(raw_reply: str) -> dict[str, str]:
    cleaned = clean_reply(raw_reply)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = {}
        match = re.search(r'"?conclusion"?\s*:\s*"([^"]+?)"', cleaned)
        if match:
            parsed["conclusion"] = match.group(1)

    return {
        "facts": str(parsed.get("facts", "") or "").strip(),
        "claim_plaintiff": str(parsed.get("claim_plaintiff", "") or "").strip(),
        "claim_defendant": str(parsed.get("claim_defendant", "") or "").strip(),
        "reasoning": str(parsed.get("reasoning", "") or "").strip(),
        "conclusion": str(parsed.get("conclusion", "") or "").strip(),
    }


def save_checkpoint(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows, columns=STRUCTURED_COLUMNS).to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)


def append_to_master(batch_rows: list[dict[str, Any]], master_csv: Path) -> pd.DataFrame:
    batch_frame = pd.DataFrame(batch_rows, columns=STRUCTURED_COLUMNS)
    if master_csv.exists():
        master_frame = pd.read_csv(master_csv)
    else:
        master_frame = pd.DataFrame(columns=STRUCTURED_COLUMNS)

    combined = pd.concat([master_frame, batch_frame], ignore_index=True)
    combined["doc_id"] = combined["doc_id"].astype(str)
    combined = combined.drop_duplicates(subset=["doc_id"], keep="first")
    combined.to_csv(master_csv, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
    return combined


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in the environment.")

    from datasets import load_dataset
    from openai import OpenAI

    master_csv = Path(args.master_csv)
    intermediate_csv = Path(args.intermediate_csv)
    client = OpenAI(api_key=api_key)

    dataset = load_dataset(args.dataset_name, split=args.split)
    records = list(dataset)
    processed_ids = load_processed_ids(master_csv, intermediate_csv)
    batch = select_next_batch(records, processed_ids, args.batch_size)
    print(f"Selected {len(batch)} cases from {args.dataset_name}:{args.split}.")

    rows: list[dict[str, Any]] = []
    if intermediate_csv.exists():
        existing = pd.read_csv(intermediate_csv)
        rows = existing.to_dict(orient="records")

    existing_ids = {str(row.get("doc_id", "")) for row in rows}
    for index, record in enumerate(batch, start=1):
        doc_id = stable_doc_id(record)
        if doc_id in existing_ids:
            continue

        full_document = str(record.get("document", "") or "")
        prompt = build_extraction_prompt(truncate_document(full_document, args.max_doc_chars))
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
        )
        parsed = parse_model_json(response.choices[0].message.content or "")
        parsed["doc_id"] = doc_id
        parsed["full_document"] = full_document
        parsed["final_outcome_label"] = classify_detailed_outcome_label(parsed["conclusion"])
        parsed["binary_outcome_label"] = map_to_final_binary_group(parsed["final_outcome_label"])
        rows.append(parsed)

        if index % args.checkpoint_every == 0:
            save_checkpoint(rows, intermediate_csv)
            print(f"Checkpoint saved: {len(rows)} rows.")

    save_checkpoint(rows, intermediate_csv)
    combined = append_to_master(rows, master_csv)
    print(f"Master saved: {master_csv} ({len(combined)} rows).")


if __name__ == "__main__":
    main()
