#!/usr/bin/env python3
"""
Controlled RAG experiment for judicial-reasoning generation.

The script reconstructs the same train/validation/test split strategy used by
reasoning_instruction_tuning/gemma_legal_sft.py, retrieves only from train
cases, and generates paired no-RAG/RAG outputs for the same held-out test cases.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sklearn.model_selection import train_test_split
from urllib3.util.retry import Retry


CASE_ID_COL = "doc_id"
FACTS_COL = "facts"
BINARY_LABELS = ["Affirmed", "Reversed/Vacated"]
COLUMN_ALIASES = {
    "facts": ["facts"],
    "plaintiff_claims": ["plaintiff_claims", "claim_plaintiff"],
    "defendant_claims": ["defendant_claims", "claim_defendant"],
    "outcome_label": ["outcome_label", "binary_outcome_label", "final_outcome_group_expert4"],
    "judicial_reasoning": ["judicial_reasoning", "reasoning", "reasoning_summary", "court_reasoning", "rationale"],
    "doc_id": ["doc_id", "document_id", "case_id", "id"],
}

PROMPT_VERSION = "controlled_rag_cot_json_v1"

RAG_PROMPT = """You are a legal expert.

Task: Write the COURT'S REASONING using structured reasoning steps.

Rules:
- Use ONLY information explicitly stated in the CURRENT CASE INPUT as the factual source.
- Do NOT add new facts, parties, dates, outcomes, statutes, or procedural events.
- Retrieved train cases are provided only as examples of reasoning structure and legal reasoning pattern.
- Do NOT copy facts, parties, dates, outcomes, evidence cues, or reasoning content from retrieved cases.
- Focus on the court's rationale, not merely summarizing the parties' arguments.

Follow these steps internally:
Step 1: Identify the central legal conflict in the current case.
Step 2: Use the retrieved examples only to understand how similar legal reasoning is structured.
Step 3: Explain how the current case facts support or contradict each side's claims.
Step 4: Derive the court's reasoning for the current case only.

Final Answer:
Write a concise 2-4 sentence summary of the court's reasoning based only on the current case.

Additional requirements:
- Include exactly 2 short evidence cues copied from the CURRENT CASE INPUT.
- Each evidence cue must be max 12 words.
- Evidence cues must not be copied from retrieved cases.

RETRIEVED TRAIN CASE EXAMPLES FOR STRUCTURE ONLY:
{retrieved_examples}

CURRENT CASE INPUT:
Facts:
{facts}

Plaintiff/Appellant Claims:
{plaintiff}

Defendant/Appellee Claims:
{defendant}

Return ONLY valid JSON:
{{"reasoning_summary":"...","evidence_cues":["...","..."]}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a controlled train-facts RAG reasoning experiment.")
    parser.add_argument("--data_path", default="data/final_dataset.csv")
    parser.add_argument("--output_dir", default="results/reasoning_benchmark/rag_controlled")
    parser.add_argument("--generator_model", default="gpt-4o")
    parser.add_argument("--embedding_model", default="text-embedding-3-small")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--eval_ids_path", default=None, help="Optional doc_id CSV. All IDs must belong to the reconstructed test split.")
    parser.add_argument("--max_input_chars", type=int, default=4500)
    parser.add_argument("--max_retrieved_chars", type=int, default=1400)
    parser.add_argument("--embedding_batch_size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=320)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep_between_calls", type=float, default=0.45)
    return parser.parse_args()


def load_dotenv_if_present() -> None:
    candidates = []
    for base in [Path.cwd(), Path(__file__).resolve()]:
        directory = base if base.is_dir() else base.parent
        candidates.extend(parent / ".env" for parent in [directory, *directory.parents])
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def clip(value: Any, max_chars: int) -> str:
    return clean_text(value)[:max_chars]


def resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in frame.columns:
                resolved[canonical] = alias
                break
        if canonical != "doc_id" and canonical not in resolved:
            raise ValueError(f"Missing required column for {canonical}: aliases={aliases}")
    return resolved


def load_frame(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = pd.read_csv(path, dtype={CASE_ID_COL: str})
    columns = resolve_columns(frame)
    normalized = frame.copy()
    normalized["_facts"] = normalized[columns["facts"]].map(clean_text)
    normalized["_plaintiff_claims"] = normalized[columns["plaintiff_claims"]].map(clean_text)
    normalized["_defendant_claims"] = normalized[columns["defendant_claims"]].map(clean_text)
    normalized["_outcome_label"] = normalized[columns["outcome_label"]].map(clean_text)
    normalized["_judicial_reasoning"] = normalized[columns["judicial_reasoning"]].map(clean_text)
    normalized["_doc_id"] = normalized[columns["doc_id"]].astype(str).str.strip() if "doc_id" in columns else normalized.index.astype(str)
    normalized = normalized[
        normalized["_outcome_label"].isin(BINARY_LABELS)
        & normalized["_judicial_reasoning"].ne("")
        & normalized["_facts"].ne("")
    ].drop_duplicates(subset=["_doc_id"])
    return normalized.reset_index(drop=True), columns


def split_frame(frame: pd.DataFrame, seed: int, test_size: float, val_size: float) -> pd.DataFrame:
    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame["_outcome_label"],
    )
    relative_val = val_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=train_val["_outcome_label"],
    )
    train = train.copy()
    validation = validation.copy()
    test = test.copy()
    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"
    return pd.concat([train, validation, test], ignore_index=True)


def select_eval_cases(split: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    test = split[split["split"] == "test"].copy()
    if args.eval_ids_path:
        ids = pd.read_csv(args.eval_ids_path, dtype={CASE_ID_COL: str})
        if CASE_ID_COL not in ids.columns:
            raise ValueError(f"--eval_ids_path must contain {CASE_ID_COL}.")
        ids = ids.dropna(subset=[CASE_ID_COL]).drop_duplicates(subset=[CASE_ID_COL]).head(args.limit).copy()
        ids[CASE_ID_COL] = ids[CASE_ID_COL].astype(str).str.strip()
        test_ids = set(test["_doc_id"])
        outside_test = sorted(set(ids[CASE_ID_COL]) - test_ids)
        if outside_test:
            raise ValueError(f"{len(outside_test)} eval IDs are not in the reconstructed test split. First IDs: {outside_test[:5]}")
        order = {doc_id: index for index, doc_id in enumerate(ids[CASE_ID_COL].tolist())}
        selected = test[test["_doc_id"].isin(order)].copy()
        selected["_order"] = selected["_doc_id"].map(order)
        return selected.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    sample_n = min(args.limit, len(test))
    if sample_n < args.limit:
        raise ValueError(f"Requested {args.limit} evaluation cases, but test split contains only {len(test)}.")
    label_counts = test["_outcome_label"].value_counts()
    raw_targets = (label_counts / len(test) * sample_n).to_dict()
    sample_counts = {label: int(np.floor(target)) for label, target in raw_targets.items()}
    remaining = sample_n - sum(sample_counts.values())
    remainders = sorted(raw_targets, key=lambda label: raw_targets[label] - sample_counts[label], reverse=True)
    for label in remainders[:remaining]:
        sample_counts[label] += 1

    parts = []
    for label, count in sample_counts.items():
        label_frame = test[test["_outcome_label"] == label]
        parts.append(label_frame.sample(n=min(count, len(label_frame)), random_state=args.seed))
    selected = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if len(selected) != sample_n:
        raise RuntimeError(f"Expected {sample_n} selected test cases, got {len(selected)}.")
    return selected


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


def openai_headers() -> dict[str, str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def embed_texts(session: requests.Session, texts: list[str], model: str, batch_size: int, timeout: int) -> np.ndarray:
    vectors: list[list[float]] = []
    headers = openai_headers()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        data = post_json_with_retries(
            session,
            "https://api.openai.com/v1/embeddings",
            headers,
            {"model": model, "input": batch},
            timeout,
        )
        vectors.extend(item["embedding"] for item in sorted(data["data"], key=lambda item: item["index"]))
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def top_k_retrieval(train: pd.DataFrame, train_embeddings: np.ndarray, query_embedding: np.ndarray, k: int) -> pd.DataFrame:
    scores = train_embeddings @ query_embedding
    top_indices = np.argsort(-scores)[:k]
    retrieved = train.iloc[top_indices].copy()
    retrieved["_similarity_score"] = scores[top_indices]
    return retrieved.reset_index(drop=True)


def retrieved_examples_text(retrieved: pd.DataFrame, max_chars: int) -> str:
    chunks = []
    for index, row in retrieved.iterrows():
        chunks.append(
            f"""Example {index + 1} (train case; structure only)
Case ID: {row["_doc_id"]}
Similarity: {row["_similarity_score"]:.6f}
Facts:
{clip(row["_facts"], max_chars)}
Plaintiff/Appellant Claims:
{clip(row["_plaintiff_claims"], max_chars)}
Defendant/Appellee Claims:
{clip(row["_defendant_claims"], max_chars)}
Judicial Reasoning:
{clip(row["_judicial_reasoning"], max_chars)}
"""
        )
    return "\n---\n".join(chunks)


def build_prompt(row: pd.Series, args: argparse.Namespace, retrieved: pd.DataFrame) -> str:
    values = {
        "facts": clip(row["_facts"], args.max_input_chars),
        "plaintiff": clip(row["_plaintiff_claims"], args.max_input_chars),
        "defendant": clip(row["_defendant_claims"], args.max_input_chars),
    }
    return RAG_PROMPT.format(retrieved_examples=retrieved_examples_text(retrieved, args.max_retrieved_chars), **values)


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


def call_gpt(session: requests.Session, prompt: str, args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    payload: dict[str, Any] = {
        "model": args.generator_model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON. No markdown. No explanation."},
            {"role": "user", "content": prompt},
        ],
        "top_p": args.top_p,
    }
    if args.generator_model.startswith("o"):
        payload["max_completion_tokens"] = args.max_tokens
    else:
        payload["temperature"] = args.temperature
        payload["max_tokens"] = args.max_tokens
    data = post_json_with_retries(
        session,
        "https://api.openai.com/v1/chat/completions",
        openai_headers(),
        payload,
        args.timeout,
    )
    raw_text = data["choices"][0]["message"].get("content", "")
    return parse_json_or_none(raw_text), raw_text


def json_list(values: list[Any]) -> str:
    return json.dumps(values, ensure_ascii=False)


def retrieval_metadata(retrieved: pd.DataFrame) -> dict[str, list[Any]]:
    return {
        "retrieved_case_ids": retrieved["_doc_id"].tolist(),
        "retrieved_similarity_scores": [float(score) for score in retrieved["_similarity_score"].tolist()],
        "retrieved_facts": retrieved["_facts"].tolist(),
        "retrieved_plaintiff_claims": retrieved["_plaintiff_claims"].tolist(),
        "retrieved_defendant_claims": retrieved["_defendant_claims"].tolist(),
        "retrieved_judicial_reasoning": retrieved["_judicial_reasoning"].tolist(),
    }


def validate_no_leakage(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, eval_cases: pd.DataFrame) -> None:
    train_ids = set(train["_doc_id"])
    validation_ids = set(validation["_doc_id"])
    test_ids = set(test["_doc_id"])
    eval_ids = set(eval_cases["_doc_id"])
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("Split leakage detected: doc_id appears in more than one split.")
    if not eval_ids <= test_ids:
        raise ValueError("Evaluation cases must be a subset of the held-out test split.")


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    load_dotenv_if_present()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, resolved_columns = load_frame(Path(args.data_path))
    split = split_frame(frame, args.seed, args.test_size, args.val_size)
    train = split[split["split"] == "train"].reset_index(drop=True)
    validation = split[split["split"] == "validation"].reset_index(drop=True)
    test = split[split["split"] == "test"].reset_index(drop=True)
    eval_cases = select_eval_cases(split, args)
    validate_no_leakage(train, validation, test, eval_cases)

    split[["_doc_id", "split", "_outcome_label"]].rename(
        columns={"_doc_id": CASE_ID_COL, "_outcome_label": "final_outcome_label"}
    ).to_csv(output_dir / "controlled_rag_split_assignments.csv", index=False)
    eval_cases[["_doc_id", "split", "_outcome_label"]].rename(
        columns={"_doc_id": CASE_ID_COL, "_outcome_label": "final_outcome_label"}
    ).to_csv(output_dir / "controlled_rag_eval_case_ids.csv", index=False)

    session = make_session()
    train_embeddings = embed_texts(session, train["_facts"].tolist(), args.embedding_model, args.embedding_batch_size, args.timeout)
    query_embeddings = embed_texts(session, eval_cases["_facts"].tolist(), args.embedding_model, args.embedding_batch_size, args.timeout)

    rows: list[dict[str, Any]] = []
    prompt_records: list[dict[str, Any]] = []

    for query_index, row in eval_cases.reset_index(drop=True).iterrows():
        retrieved = top_k_retrieval(train, train_embeddings, query_embeddings[query_index], args.top_k)
        if row["_doc_id"] in set(retrieved["_doc_id"]):
            raise ValueError(f"Leakage detected: test case {row['_doc_id']} was retrieved from train pool.")
        metadata = retrieval_metadata(retrieved)

        condition = f"rag_facts_top{args.top_k}"
        prompt = build_prompt(row, args, retrieved)
        parsed = None
        raw_text = ""
        error = ""
        try:
            parsed, raw_text = call_gpt(session, prompt, args)
            if parsed is None:
                error = "json_parse_failed"
            else:
                cues = parsed.get("evidence_cues")
                if not isinstance(cues, list) or len(cues) != 2:
                    error = "invalid_evidence_cue_count"
        except Exception as exc:
            error = str(exc)

        evidence_cues = (parsed or {}).get("evidence_cues", [])
        if not isinstance(evidence_cues, list):
            evidence_cues = []

        row_record = {
            "case_id": row["_doc_id"],
            "split": "test",
            "condition": condition,
            "generator_model": args.generator_model,
            "prompt_version": PROMPT_VERSION,
            "facts": row["_facts"],
            "plaintiff_claims": row["_plaintiff_claims"],
            "defendant_claims": row["_defendant_claims"],
            "retrieved_case_ids": json_list(metadata["retrieved_case_ids"]),
            "retrieved_similarity_scores": json_list(metadata["retrieved_similarity_scores"]),
            "retrieved_facts": json_list(metadata["retrieved_facts"]),
            "retrieved_plaintiff_claims": json_list(metadata["retrieved_plaintiff_claims"]),
            "retrieved_defendant_claims": json_list(metadata["retrieved_defendant_claims"]),
            "retrieved_judicial_reasoning": json_list(metadata["retrieved_judicial_reasoning"]),
            "generated_reasoning_summary": clean_text((parsed or {}).get("reasoning_summary", "")),
            "evidence_cues": json_list([clean_text(cue) for cue in evidence_cues]),
            "ground_truth_judicial_reasoning": row["_judicial_reasoning"],
            "final_outcome_label": row["_outcome_label"],
            "raw_model_output": raw_text,
            "error": error,
        }
        rows.append(row_record)
        prompt_records.append(
            {
                "case_id": row["_doc_id"],
                "condition": condition,
                "generator_model": args.generator_model,
                "prompt_version": PROMPT_VERSION,
                "prompt": prompt,
                "raw_model_output": raw_text,
                "error": error,
            }
        )
        time.sleep(args.sleep_between_calls)

    output_path = output_dir / f"controlled_rag_reasoning_{args.generator_model.replace('/', '__')}_top{args.top_k}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    save_jsonl(output_dir / f"controlled_rag_prompts_raw_{args.generator_model.replace('/', '__')}_top{args.top_k}.jsonl", prompt_records)
    (output_dir / "controlled_rag_run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "prompt_version": PROMPT_VERSION,
                "resolved_columns": resolved_columns,
                "split_strategy": "Same as reasoning_instruction_tuning/gemma_legal_sft.py: stratified train_test_split test, then stratified validation from train_val.",
                "conditions": [f"rag_facts_top{args.top_k}"],
                "retrieval_pool": "train split only",
                "held_out_policy": "validation and test excluded from retrieval corpus; evaluation cases selected only from test split.",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
