#!/usr/bin/env python3
"""
Instruction-tune a Gemma-style causal LM to predict legal outcome + reasoning.

The model is supervised to return strict JSON:
  {"outcome_label": "...", "judicial_reasoning": "..."}

Outputs include split prediction CSVs, metrics JSON files, run config, and the
trained PEFT adapter/tokenizer.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer


BINARY_LABELS = ["Affirmed", "Reversed/Vacated"]
SYSTEM_PROMPT = (
    "You are a legal reasoning model. Use only the provided case information. "
    "Return strict valid JSON only with keys outcome_label and judicial_reasoning."
)
USER_TEMPLATE = """Facts:
{facts}

Plaintiff/Appellant Claims:
{plaintiff_claims}

Defendant/Appellee Claims:
{defendant_claims}"""

COLUMN_ALIASES = {
    "facts": ["facts"],
    "plaintiff_claims": ["plaintiff_claims", "claim_plaintiff"],
    "defendant_claims": ["defendant_claims", "claim_defendant"],
    "outcome_label": ["outcome_label", "binary_outcome_label", "final_outcome_group_expert4"],
    "judicial_reasoning": ["judicial_reasoning", "reasoning"],
    "doc_id": ["doc_id", "document_id", "case_id", "id"],
}

DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class ParsedPrediction:
    outcome_label: str
    judicial_reasoning: str
    json_valid: bool
    parse_error: str
    raw_response: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemma legal instruction tuning with PEFT LoRA/QLoRA.")
    parser.add_argument("--model_name", default="google/gemma-2-2b-it")
    parser.add_argument("--input_csv", default="data/final_dataset.csv")
    parser.add_argument("--output_dir", default="results/gemma_legal_sft")
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation_max_new_tokens", type=int, default=160)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--hf_token", default=None)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def resolve_columns(frame: pd.DataFrame) -> dict[str, str]:
    resolved = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in frame.columns:
                resolved[canonical] = alias
                break
        if canonical != "doc_id" and canonical not in resolved:
            raise ValueError(f"Missing required column for {canonical}: aliases={aliases}")
    return resolved


def load_frame(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    frame = pd.read_csv(path)
    columns = resolve_columns(frame)
    frame = frame.copy()
    frame["_facts"] = frame[columns["facts"]].map(clean_text)
    frame["_plaintiff_claims"] = frame[columns["plaintiff_claims"]].map(clean_text)
    frame["_defendant_claims"] = frame[columns["defendant_claims"]].map(clean_text)
    frame["_outcome_label"] = frame[columns["outcome_label"]].map(clean_text)
    frame["_judicial_reasoning"] = frame[columns["judicial_reasoning"]].map(clean_text)
    frame["_doc_id"] = frame[columns["doc_id"]].astype(str) if "doc_id" in columns else frame.index.astype(str)
    frame = frame[frame["_outcome_label"].isin(BINARY_LABELS)].drop_duplicates(subset=["_doc_id"]).reset_index(drop=True)
    return frame, columns


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


def build_user(row: pd.Series) -> str:
    return USER_TEMPLATE.format(
        facts=row["_facts"],
        plaintiff_claims=row["_plaintiff_claims"],
        defendant_claims=row["_defendant_claims"],
    )


def build_answer(row: pd.Series) -> str:
    return json.dumps(
        {
            "outcome_label": row["_outcome_label"],
            "judicial_reasoning": row["_judicial_reasoning"],
        },
        ensure_ascii=False,
    )


def format_example(row: pd.Series, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user(row)},
        {"role": "assistant", "content": build_answer(row)},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return f"{SYSTEM_PROMPT}\n\n{build_user(row)}\n\n{build_answer(row)}"


def format_prompt(row: pd.Series, tokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user(row)},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{build_user(row)}\n\n"


def parse_response(text: str) -> ParsedPrediction:
    raw = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        cleaned = match.group(0)
    try:
        obj = json.loads(cleaned)
        outcome = clean_text(obj.get("outcome_label", ""))
        reasoning = clean_text(obj.get("judicial_reasoning", ""))
        if outcome not in BINARY_LABELS:
            outcome = ""
        return ParsedPrediction(outcome, reasoning, True, "", raw)
    except Exception as exc:
        return ParsedPrediction("", "", False, str(exc), raw)


def metric_dict(gold: list[str], pred: list[str]) -> dict[str, Any]:
    valid_pred = [label if label in BINARY_LABELS else "INVALID" for label in pred]
    all_labels = BINARY_LABELS + ["INVALID"]
    return {
        "rows": len(gold),
        "valid_json_rate": float(np.mean([label != "INVALID" for label in valid_pred])) if valid_pred else 0.0,
        "accuracy_all_predictions": float(accuracy_score(gold, valid_pred)),
        "macro_f1_all_predictions": float(f1_score(gold, valid_pred, labels=all_labels, average="macro", zero_division=0)),
        "binary_macro_f1_valid_only": float(
            f1_score(
                [g for g, p in zip(gold, valid_pred) if p in BINARY_LABELS],
                [p for p in valid_pred if p in BINARY_LABELS],
                labels=BINARY_LABELS,
                average="macro",
                zero_division=0,
            )
        )
        if any(p in BINARY_LABELS for p in valid_pred)
        else 0.0,
        "macro_precision_all_predictions": float(precision_score(gold, valid_pred, labels=all_labels, average="macro", zero_division=0)),
        "macro_recall_all_predictions": float(recall_score(gold, valid_pred, labels=all_labels, average="macro", zero_division=0)),
        "confusion_matrix_all_predictions": confusion_matrix(gold, valid_pred, labels=all_labels).tolist(),
    }


def make_dataset(frame: pd.DataFrame, tokenizer) -> Dataset:
    rows = [{"text": format_example(row, tokenizer)} for _, row in frame.iterrows()]
    return Dataset.from_list(rows)


def load_model_and_tokenizer(args: argparse.Namespace):
    token = args.hf_token
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if args.use_4bit:
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=token,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        quantization_config=quantization_config,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def generate_predictions(model, tokenizer, frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    model.eval()
    for _, row in frame.iterrows():
        prompt = format_prompt(row, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_length).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.generation_max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        response = decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()
        parsed = parse_response(response)
        rows.append(
            {
                "doc_id": row["_doc_id"],
                "split": row["split"],
                "gold_outcome_label": row["_outcome_label"],
                "pred_outcome_label": parsed.outcome_label,
                "gold_judicial_reasoning": row["_judicial_reasoning"],
                "pred_judicial_reasoning": parsed.judicial_reasoning,
                "json_valid": parsed.json_valid,
                "parse_error": parsed.parse_error,
                "raw_response": parsed.raw_response,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, resolved_columns = load_frame(Path(args.input_csv))
    split = split_frame(frame, args.seed, args.test_size, args.val_size)
    model, tokenizer = load_model_and_tokenizer(args)

    train_dataset = make_dataset(split[split["split"] == "train"], tokenizer)
    validation_dataset = make_dataset(split[split["split"] == "validation"], tokenizer)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    sft_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to=[],
        fp16=torch.cuda.is_available(),
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(output_dir / "adapter")

    all_predictions = []
    metrics_summary = {}
    for split_name in ["validation", "test"]:
        split_frame = split[split["split"] == split_name].reset_index(drop=True)
        predictions = generate_predictions(trainer.model, tokenizer, split_frame, args)
        predictions.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)
        metrics = metric_dict(predictions["gold_outcome_label"].tolist(), predictions["pred_outcome_label"].tolist())
        (output_dir / f"{split_name}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        metrics_summary[split_name] = metrics
        all_predictions.append(predictions)

    pd.concat(all_predictions, ignore_index=True).to_csv(output_dir / "all_eval_predictions.csv", index=False)
    (output_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "resolved_columns.json").write_text(json.dumps(resolved_columns, indent=2) + "\n", encoding="utf-8")
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    print(f"Saved instruction-tuning outputs to {output_dir}")


if __name__ == "__main__":
    main()
