#!/usr/bin/env python3
"""
Supervised instruction tuning for Gemma-style instruction models on a legal
outcome prediction + concise judicial reasoning dataset.

The script:
- loads a CSV dataset
- validates and normalizes required columns
- performs stratified train/validation/test splits
- converts rows into chat-formatted supervision examples
- fine-tunes with TRL SFTTrainer using full-model fine-tuning
- evaluates by generating JSON outputs on validation and test splits
- saves splits, full model, tokenizer, logs, metrics, malformed generations, and
  predictions

Example:
python reasoning_instruction_tuning/gemma_legal_full_ft.py \
  --model_name google/gemma-2-2b-it \
  --input_csv data/final_dataset.csv \
  --output_dir results/gemma_legal_full_ft
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


SYSTEM_PROMPT = (
    "You are a legal reasoning model. Use only the provided case information. "
    "Predict the outcome and generate concise judicial reasoning. Do not use "
    "external knowledge. Return strict valid JSON only. Output exactly one JSON "
    "object with the keys outcome_label and judicial_reasoning. Do not add markdown, "
    "preambles, explanations, or any text outside the JSON object."
)

USER_TEMPLATE = """Instruction:
{system_prompt}

Facts:
{facts}

Plaintiff/Appellant Claims:
{plaintiff_claims}

Defendant/Appellee Claims:
{defendant_claims}"""

REQUIRED_COLUMN_ALIASES = {
    "facts": ["facts"],
    "plaintiff_claims": ["plaintiff_claims", "claim_plaintiff"],
    "defendant_claims": ["defendant_claims", "claim_defendant"],
    "outcome_label": ["outcome_label", "binary_outcome_label", "final_outcome_group_expert4"],
    "judicial_reasoning": ["judicial_reasoning", "reasoning"],
}

OPTIONAL_COLUMN_ALIASES = {
    "doc_id": ["doc_id", "document_id", "case_id", "id"],
}

BINARY_LABELS = ["Affirmed", "Reversed/Vacated"]

@dataclass
class ParsedPrediction:
    outcome_label: str
    judicial_reasoning: str
    json_valid: bool
    parse_error: str
    raw_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemma legal full-model supervised fine-tuning with TRL SFTTrainer."
    )
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b-it")
    parser.add_argument(
        "--input_csv",
        type=str,
        default="data/final_dataset.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/gemma_legal_full_ft",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--generation_max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--early_stopping_patience", type=int, default=1)
    parser.add_argument(
        "--group_split_column",
        type=str,
        default=None,
        help="Optional group column to prevent the same group appearing across multiple splits.",
    )
    parser.set_defaults(gradient_checkpointing=True)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logging(output_dir: Path) -> logging.Logger:
    ensure_dir(output_dir)
    logger = logging.getLogger("gemma_legal_full_ft")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(output_dir / "run.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for canonical_name, aliases in REQUIRED_COLUMN_ALIASES.items():
        match = next((candidate for candidate in aliases if candidate in df.columns), None)
        if match is None:
            alias_text = ", ".join(aliases)
            raise ValueError(
                f"Missing required column for '{canonical_name}'. "
                f"Accepted names: {alias_text}"
            )
        resolved[canonical_name] = match
    return resolved


def resolve_optional_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    resolved: Dict[str, Optional[str]] = {}
    for canonical_name, aliases in OPTIONAL_COLUMN_ALIASES.items():
        resolved[canonical_name] = next((candidate for candidate in aliases if candidate in df.columns), None)
    return resolved


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_dataframe(
    df: pd.DataFrame,
    logger: logging.Logger,
    group_split_column: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Optional[str]]]:
    columns = resolve_columns(df)
    optional_columns = resolve_optional_columns(df)
    renamed = df.rename(columns={v: k for k, v in columns.items()}).copy()

    if optional_columns.get("doc_id"):
        renamed = renamed.rename(columns={optional_columns["doc_id"]: "doc_id"})
    elif "doc_id" not in renamed.columns:
        renamed["doc_id"] = ""

    for text_col in ["facts", "plaintiff_claims", "defendant_claims", "judicial_reasoning", "doc_id"]:
        renamed[text_col] = renamed[text_col].apply(clean_text)

    renamed["outcome_label"] = renamed["outcome_label"].apply(clean_text)
    renamed["row_id"] = np.arange(len(renamed))

    before = len(renamed)
    renamed = renamed[
        renamed["outcome_label"].ne("") & renamed["judicial_reasoning"].ne("")
    ].reset_index(drop=True)
    dropped = before - len(renamed)
    if dropped:
        logger.info("Dropped %d rows with missing target label/reasoning.", dropped)

    before_binary = len(renamed)
    renamed = renamed[renamed["outcome_label"].isin(BINARY_LABELS)].reset_index(drop=True)
    dropped_binary = before_binary - len(renamed)
    if dropped_binary:
        logger.info(
            "Dropped %d rows outside the binary label set %s.",
            dropped_binary,
            BINARY_LABELS,
        )

    if group_split_column:
        if group_split_column in renamed.columns:
            renamed[group_split_column] = renamed[group_split_column].apply(clean_text)
        else:
            raise ValueError(f"group_split_column '{group_split_column}' does not exist in the dataset.")

    if renamed["outcome_label"].nunique() < 2:
        raise ValueError("Need at least two outcome classes after cleaning.")

    class_counts = renamed["outcome_label"].value_counts()
    if (class_counts < 2).any():
        rare = class_counts[class_counts < 2].to_dict()
        raise ValueError(
            f"Each class needs at least 2 examples for stratified splitting. Found: {rare}"
        )

    return renamed, columns, optional_columns


def build_messages(row: pd.Series) -> List[Dict[str, str]]:
    user_message = USER_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        facts=row["facts"],
        plaintiff_claims=row["plaintiff_claims"],
        defendant_claims=row["defendant_claims"],
    )
    assistant_payload = {
        "outcome_label": row["outcome_label"],
        "judicial_reasoning": row["judicial_reasoning"],
    }
    return [
        {"role": "user", "content": user_message},
        {
            "role": "assistant",
            "content": json.dumps(assistant_payload, ensure_ascii=False),
        },
    ]


def format_chat_example(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def add_training_text(df: pd.DataFrame, tokenizer: AutoTokenizer) -> pd.DataFrame:
    frame = df.copy()
    frame["messages"] = frame.apply(build_messages, axis=1)
    frame["text"] = frame["messages"].apply(
        lambda messages: format_chat_example(tokenizer, messages, add_generation_prompt=False)
    )
    frame["prompt_text"] = frame.apply(
        lambda row: format_chat_example(
            tokenizer,
            [
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        system_prompt=SYSTEM_PROMPT,
                        facts=row["facts"],
                        plaintiff_claims=row["plaintiff_claims"],
                        defendant_claims=row["defendant_claims"],
                    ),
                },
            ],
            add_generation_prompt=True,
        ),
        axis=1,
    )
    frame["assistant_json"] = frame.apply(
        lambda row: json.dumps(
            {
                "outcome_label": row["outcome_label"],
                "judicial_reasoning": row["judicial_reasoning"],
            },
            ensure_ascii=False,
        ),
        axis=1,
    )
    return frame


def choose_group_stratified_holdout(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df[group_col].nunique() < 2:
        raise ValueError(f"group_split_column '{group_col}' must contain at least 2 distinct groups.")

    desired_size = len(df) * test_size
    n_splits = int(round(1.0 / test_size)) if test_size > 0 else 2
    n_splits = max(2, min(n_splits, df[group_col].nunique()))

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    full_distribution = df["outcome_label"].value_counts(normalize=True)

    best_train_idx = None
    best_test_idx = None
    best_score = None

    for train_idx, test_idx in splitter.split(df, y=df["outcome_label"], groups=df[group_col]):
        test_distribution = df.iloc[test_idx]["outcome_label"].value_counts(normalize=True)
        size_gap = abs(len(test_idx) - desired_size) / max(desired_size, 1.0)
        dist_gap = full_distribution.subtract(test_distribution, fill_value=0.0).abs().sum()
        score = (size_gap, dist_gap)
        if best_score is None or score < best_score:
            best_score = score
            best_train_idx = train_idx
            best_test_idx = test_idx

    if best_train_idx is None or best_test_idx is None:
        raise RuntimeError("Failed to construct a group-aware stratified split.")

    return (
        df.iloc[best_train_idx].reset_index(drop=True),
        df.iloc[best_test_idx].reset_index(drop=True),
    )


def stratified_split(
    df: pd.DataFrame,
    seed: int,
    test_size: float,
    val_size: float,
    group_split_column: Optional[str],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < test_size < 1:
        raise ValueError("--test_size must be between 0 and 1.")
    if not 0 < val_size < 1:
        raise ValueError("--val_size must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1.")

    if group_split_column:
        logger.info("Using group-aware split with column: %s", group_split_column)
        train_val_df, test_df = choose_group_stratified_holdout(df, group_split_column, test_size, seed)
        relative_val_size = val_size / (1.0 - test_size)
        train_df, val_df = choose_group_stratified_holdout(
            train_val_df,
            group_split_column,
            relative_val_size,
            seed + 1,
        )
    else:
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["outcome_label"],
            random_state=seed,
        )
        relative_val_size = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            stratify=train_val_df["outcome_label"],
            random_state=seed,
        )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compute_split_distributions(
    split_frames: Dict[str, pd.DataFrame],
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    distribution_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for split_name, split_df in split_frames.items():
        counts = split_df["outcome_label"].value_counts().sort_index()
        total = int(len(split_df))
        distribution_summary[split_name] = {
            label: {
                "count": int(count),
                "percentage": float(count / total) if total else 0.0,
            }
            for label, count in counts.items()
        }
        logger.info("%s class distribution: %s", split_name, distribution_summary[split_name])
    save_json(distribution_summary, output_dir / "split_class_distribution.json")


def save_split_artifacts(df: pd.DataFrame, split_name: str, output_dir: Path) -> None:
    split_dir = output_dir / "processed_splits"
    ensure_dir(split_dir)
    df.to_csv(split_dir / f"{split_name}.csv", index=False)
    df.to_json(split_dir / f"{split_name}.jsonl", orient="records", lines=True, force_ascii=False)


def dataframe_to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[["text"]], preserve_index=False)


def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_tokenizer(args: argparse.Namespace) -> AutoTokenizer:
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model_for_training(args: argparse.Namespace) -> AutoModelForCausalLM:
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=hf_token,
        torch_dtype=get_torch_dtype(),
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def save_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_prediction_label(raw_label: Any, allowed_labels: List[str]) -> str:
    if raw_label is None:
        return ""
    candidate = clean_text(raw_label)
    if candidate in allowed_labels:
        return candidate

    lowered = candidate.lower()
    for label in allowed_labels:
        if lowered == label.lower():
            return label

    if "affirm" in lowered:
        for label in allowed_labels:
            if "affirm" in label.lower():
                return label

    if any(term in lowered for term in ["reverse", "revers", "vacat", "remand", "dismiss"]):
        for label in allowed_labels:
            label_lower = label.lower()
            if any(term in label_lower for term in ["reverse", "revers", "vacat", "remand", "dismiss"]):
                return label
    return candidate


def parse_generated_json(raw_text: str, allowed_labels: List[str]) -> ParsedPrediction:
    cleaned = clean_text(raw_text)
    if not cleaned:
        return ParsedPrediction("", "", False, "empty_generation", raw_text)

    fence_stripped = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    fence_stripped = re.sub(r"\s*```$", "", fence_stripped)

    candidate = fence_stripped
    parsed_obj: Optional[Dict[str, Any]] = None
    parse_error = ""

    try:
        maybe = json.loads(candidate)
        if isinstance(maybe, dict):
            parsed_obj = maybe
        else:
            parse_error = "json_not_object"
    except Exception as exc:
        parse_error = str(exc)

    if parsed_obj is None:
        match = re.search(r"\{[\s\S]*\}", candidate)
        if match:
            try:
                maybe = json.loads(match.group(0))
                if isinstance(maybe, dict):
                    parsed_obj = maybe
                else:
                    parse_error = "json_not_object"
            except Exception as exc:
                parse_error = str(exc)

    if parsed_obj is None:
        return ParsedPrediction("", "", False, parse_error or "json_parse_failed", raw_text)

    outcome_label = normalize_prediction_label(parsed_obj.get("outcome_label"), allowed_labels)
    judicial_reasoning = clean_text(parsed_obj.get("judicial_reasoning", ""))

    if not outcome_label:
        return ParsedPrediction("", judicial_reasoning, False, "missing_or_invalid_outcome_label", raw_text)
    if not judicial_reasoning:
        return ParsedPrediction(outcome_label, "", False, "missing_judicial_reasoning", raw_text)

    return ParsedPrediction(outcome_label, judicial_reasoning, True, "", raw_text)


def create_generation_prompt(row: pd.Series, tokenizer: AutoTokenizer) -> str:
    return format_chat_example(
        tokenizer,
        [
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    system_prompt=SYSTEM_PROMPT,
                    facts=row["facts"],
                    plaintiff_claims=row["plaintiff_claims"],
                    defendant_claims=row["defendant_claims"],
                ),
            },
        ],
        add_generation_prompt=True,
    )


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
        "temperature": max(temperature, 1e-5) if temperature > 0 else None,
        "top_p": top_p,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    with torch.no_grad():
        output = model.generate(**encoded, **generation_kwargs)

    prompt_length = encoded["input_ids"].shape[1]
    generated_ids = output[0][prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def build_prediction_export_row(row: pd.Series, split_name: str, parsed: ParsedPrediction, raw_response: str) -> Dict[str, Any]:
    return {
        "row_id": int(row["row_id"]),
        "doc_id": row.get("doc_id", ""),
        "split": split_name,
        "facts": row["facts"],
        "plaintiff_claims": row["plaintiff_claims"],
        "defendant_claims": row["defendant_claims"],
        "gold_outcome_label": row["outcome_label"],
        "pred_outcome_label": parsed.outcome_label,
        "gold_judicial_reasoning": row["judicial_reasoning"],
        "pred_judicial_reasoning": parsed.judicial_reasoning,
        "json_valid": parsed.json_valid,
        "parse_error": parsed.parse_error,
        "raw_response": raw_response,
    }


def build_train_placeholder_predictions(train_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in train_df.iterrows():
        rows.append(
            {
                "row_id": int(row["row_id"]),
                "doc_id": row.get("doc_id", ""),
                "split": "train",
                "facts": row["facts"],
                "plaintiff_claims": row["plaintiff_claims"],
                "defendant_claims": row["defendant_claims"],
                "gold_outcome_label": row["outcome_label"],
                "pred_outcome_label": "",
                "gold_judicial_reasoning": row["judicial_reasoning"],
                "pred_judicial_reasoning": "",
                "json_valid": "",
                "parse_error": "",
                "raw_response": "",
            }
        )
    return pd.DataFrame(rows)


def choose_incorrect_label(true_label: str, allowed_labels: List[str]) -> str:
    for label in allowed_labels:
        if label != true_label:
            return label
    return true_label


def compute_mode_metrics(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
    }


def save_confusion_matrix(
    split_name: str,
    mode_name: str,
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
    output_dir: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_payload = {
        "split": split_name,
        "mode": mode_name,
        "labels": labels,
        "matrix": matrix.tolist(),
    }
    save_json(matrix_payload, output_dir / f"{split_name}_{mode_name}_confusion_matrix.json")
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / f"{split_name}_{mode_name}_confusion_matrix.csv"
    )


def evaluate_split(
    split_name: str,
    df: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    allowed_labels: List[str],
    output_dir: Path,
    generation_max_new_tokens: int,
    temperature: float,
    top_p: float,
    logger: logging.Logger,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    predictions: List[Dict[str, Any]] = []
    malformed_records: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        prompt = row["prompt_text"] if "prompt_text" in row else create_generation_prompt(row, tokenizer)
        raw_response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=generation_max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        parsed = parse_generated_json(raw_response, allowed_labels)

        if not parsed.json_valid:
            malformed_records.append(
                {
                    "split": split_name,
                    "row_id": int(row["row_id"]),
                    "doc_id": row.get("doc_id", ""),
                    "gold_outcome_label": row["outcome_label"],
                    "parse_error": parsed.parse_error,
                    "raw_response": raw_response,
                }
            )
            logger.warning(
                "Malformed JSON on %s row_id=%s doc_id=%s error=%s",
                split_name,
                row["row_id"],
                row.get("doc_id", ""),
                parsed.parse_error,
            )

        predictions.append(build_prediction_export_row(row, split_name, parsed, raw_response))

    pred_df = pd.DataFrame(predictions)
    # Invalid outputs are counted as prediction errors rather than a separate class,
    # so macro metrics stay on the original label set without artificial distortion.
    pred_df["pred_outcome_label_all_predictions"] = pred_df.apply(
        lambda row: row["pred_outcome_label"]
        if bool(row["json_valid"])
        else choose_incorrect_label(row["gold_outcome_label"], allowed_labels),
        axis=1,
    )
    pred_df.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    malformed_path = output_dir / f"{split_name}_malformed_json.jsonl"
    save_jsonl(malformed_records, malformed_path)

    valid_mask = pred_df["json_valid"].astype(bool)
    json_validity_rate = float(valid_mask.mean()) if len(pred_df) else 0.0

    valid_predictions = pred_df.loc[valid_mask].copy()
    if len(valid_predictions) > 0:
        valid_only_metrics = compute_mode_metrics(
            valid_predictions["gold_outcome_label"].tolist(),
            valid_predictions["pred_outcome_label"].tolist(),
            allowed_labels,
        )
        save_confusion_matrix(
            split_name,
            "valid_only",
            valid_predictions["gold_outcome_label"].tolist(),
            valid_predictions["pred_outcome_label"].tolist(),
            allowed_labels,
            output_dir,
        )
    else:
        valid_only_metrics = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "macro_f1": 0.0,
        }
        save_confusion_matrix(split_name, "valid_only", [], [], allowed_labels, output_dir)

    all_predictions_metrics = compute_mode_metrics(
        pred_df["gold_outcome_label"].tolist(),
        pred_df["pred_outcome_label_all_predictions"].tolist(),
        allowed_labels,
    )
    save_confusion_matrix(
        split_name,
        "all_predictions",
        pred_df["gold_outcome_label"].tolist(),
        pred_df["pred_outcome_label_all_predictions"].tolist(),
        allowed_labels,
        output_dir,
    )

    metrics = {
        "split": split_name,
        "num_examples": int(len(pred_df)),
        "num_valid_json": int(valid_mask.sum()),
        "num_malformed_json": int((~valid_mask).sum()),
        "json_validity_rate": json_validity_rate,
        "valid_only": valid_only_metrics,
        "all_predictions": all_predictions_metrics,
    }
    save_json(metrics, output_dir / f"{split_name}_metrics.json")
    logger.info("%s metrics: %s", split_name, json.dumps(metrics, ensure_ascii=False))
    return metrics, pred_df


def export_training_history(trainer: SFTTrainer, output_dir: Path) -> None:
    state = trainer.state.log_history if trainer.state is not None else []
    save_json({"log_history": state}, output_dir / "training_logs.json")


def prepare_inference_model(
    model_dir: Path,
    trust_remote_code: bool,
    hf_token: Optional[str],
) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        token=hf_token,
        torch_dtype=get_torch_dtype(),
        trust_remote_code=trust_remote_code,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    model.config.use_cache = True
    return model


def predict_single_example(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    facts: str,
    plaintiff_claims: str,
    defendant_claims: str,
    allowed_labels: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    row = pd.Series(
        {
            "facts": clean_text(facts),
            "plaintiff_claims": clean_text(plaintiff_claims),
            "defendant_claims": clean_text(defendant_claims),
        }
    )
    prompt = create_generation_prompt(row, tokenizer)
    raw_response = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    parsed = parse_generated_json(raw_response, allowed_labels)
    return {
        "raw_response": raw_response,
        "parsed_prediction": asdict(parsed),
    }


def save_example_inference(
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    allowed_labels: List[str],
    output_dir: Path,
) -> None:
    if test_df.empty:
        return

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    model = prepare_inference_model(
        model_dir=output_dir / "full_model",
        trust_remote_code=args.trust_remote_code,
        hf_token=hf_token,
    )

    sample = test_df.iloc[0]
    result = predict_single_example(
        model=model,
        tokenizer=tokenizer,
        facts=sample["facts"],
        plaintiff_claims=sample["plaintiff_claims"],
        defendant_claims=sample["defendant_claims"],
        allowed_labels=allowed_labels,
        max_new_tokens=args.generation_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    save_json(result, output_dir / "example_inference.json")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    logger = setup_logging(output_dir)

    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. The script is designed for GPU training and may be slow.")
    if args.temperature != 0.0:
        logger.warning("Non-zero temperature can reduce JSON reliability. Recommended value is 0.0.")

    set_reproducibility(args.seed)
    save_json(vars(args), output_dir / "run_config.json")

    logger.info("Loading dataset from %s", args.input_csv)
    raw_df = pd.read_csv(args.input_csv)
    df, resolved_columns, optional_columns = normalize_dataframe(raw_df, logger, args.group_split_column)
    save_json(resolved_columns, output_dir / "resolved_columns.json")
    save_json(optional_columns, output_dir / "resolved_optional_columns.json")

    logger.info("Rows after normalization: %d", len(df))
    logger.info("Class distribution: %s", df["outcome_label"].value_counts().to_dict())

    tokenizer = load_tokenizer(args)

    train_df, val_df, test_df = stratified_split(
        df=df,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        group_split_column=args.group_split_column,
        logger=logger,
    )
    train_df = add_training_text(train_df, tokenizer)
    val_df = add_training_text(val_df, tokenizer)
    test_df = add_training_text(test_df, tokenizer)

    compute_split_distributions(
        {"train": train_df, "validation": val_df, "test": test_df},
        output_dir,
        logger,
    )

    save_split_artifacts(train_df, "train", output_dir)
    save_split_artifacts(val_df, "validation", output_dir)
    save_split_artifacts(test_df, "test", output_dir)

    train_dataset = dataframe_to_dataset(train_df)
    val_dataset = dataframe_to_dataset(val_df)

    model = load_model_for_training(args)

    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        report_to=[] if args.report_to == "none" else [args.report_to],
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adafactor",
        lr_scheduler_type="cosine",
        packing=False,
        logging_dir=str(output_dir / "logs"),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    logger.info("Starting training.")
    # This version performs full-model fine-tuning, so all model weights are updated.
    # The previous version used parameter-efficient tuning with LoRA adapters only.
    trainer.train()
    trainer.save_model(str(output_dir / "full_model"))
    tokenizer.save_pretrained(output_dir / "tokenizer")
    export_training_history(trainer, output_dir)

    allowed_labels = sorted(df["outcome_label"].unique().tolist())
    save_json({"allowed_labels": allowed_labels}, output_dir / "allowed_labels.json")
    logger.info("Allowed labels for JSON parsing: %s", allowed_labels)

    eval_model = trainer.model
    eval_model.eval()

    # Detailed reasoning quality is evaluated in a separate pipeline.
    # This script focuses on training, outcome prediction, JSON validity, and exporting outputs.
    validation_metrics, validation_pred_df = evaluate_split(
        split_name="validation",
        df=val_df,
        model=eval_model,
        tokenizer=tokenizer,
        allowed_labels=allowed_labels,
        output_dir=output_dir,
        generation_max_new_tokens=args.generation_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        logger=logger,
    )
    test_metrics, test_pred_df = evaluate_split(
        split_name="test",
        df=test_df,
        model=eval_model,
        tokenizer=tokenizer,
        allowed_labels=allowed_labels,
        output_dir=output_dir,
        generation_max_new_tokens=args.generation_max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        logger=logger,
    )

    train_pred_df = build_train_placeholder_predictions(train_df)
    train_pred_df.to_csv(output_dir / "train_predictions.csv", index=False)
    pd.concat([train_pred_df, validation_pred_df, test_pred_df], ignore_index=True).to_csv(
        output_dir / "all_split_predictions.csv",
        index=False,
    )

    save_json(
        {
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        },
        output_dir / "metrics_summary.json",
    )

    save_example_inference(
        test_df=test_df,
        tokenizer=tokenizer,
        args=args,
        allowed_labels=allowed_labels,
        output_dir=output_dir,
    )
    logger.info("Finished successfully.")


if __name__ == "__main__":
    main()
