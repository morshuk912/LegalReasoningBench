#!/usr/bin/env python3
"""
Prediction benchmark: causal language models fine-tuned with LoRA/QLoRA.

This script runs 5-fold stratified CV for instruction-style causal LMs such as
Llama, SaulLM, LawMA, and Law-LLM. It supports both facts-only and facts+claims
inputs.

Leakage controls:
- The binary label is filtered before splitting.
- Each case appears in exactly one held-out fold.
- The inner validation split is drawn only from the training side of each fold.
- The training loss is masked so prompt tokens do not contribute to the loss;
  only the outcome label tokens are supervised.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


NEG_LABEL = "Affirmed"
POS_LABEL = "Reversed/Vacated"
BINARY_LABELS = [NEG_LABEL, POS_LABEL]
LABEL_COL = "binary_outcome_label"
LEGACY_LABEL_COLS = ["final_outcome_group_expert4"]
LABEL_TO_TEXT = {NEG_LABEL: "AFFIRMED", POS_LABEL: "REVERSED"}
TEXT_TO_LABEL = {"AFFIRMED": NEG_LABEL, "REVERSED": POS_LABEL}
TEXT_VARIANTS = ["facts", "facts_claims"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run causal-LM LoRA 5-fold binary prediction benchmark.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text_variant", choices=TEXT_VARIANTS, required=True)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--train_bs", type=int, default=2)
    parser.add_argument("--eval_bs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--hf_token", default=None)
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
    required = ["facts", "claim_plaintiff", "claim_defendant", "doc_id", LABEL_COL]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df[LABEL_COL] = df[LABEL_COL].map(clean_text)
    df = df[df[LABEL_COL].isin(BINARY_LABELS)].copy()
    df = df.drop_duplicates(subset=["doc_id"], keep="first").reset_index(drop=True)
    return df


def build_text(df: pd.DataFrame, variant: str) -> list[str]:
    facts = df["facts"].fillna("").astype(str)
    plaintiff = df["claim_plaintiff"].fillna("").astype(str)
    defendant = df["claim_defendant"].fillna("").astype(str)
    if variant == "facts":
        return ("[FACTS]\n" + facts).tolist()
    if variant == "facts_claims":
        return (
            "[FACTS]\n" + facts
            + "\n\n[PLAINTIFF CLAIMS]\n" + plaintiff
            + "\n\n[DEFENDANT CLAIMS]\n" + defendant
        ).tolist()
    raise ValueError(f"Unknown text variant: {variant}")


def format_prompt(text: str) -> str:
    return (
        "You are a legal outcome prediction model.\n"
        "Predict the appellate outcome using only the case information.\n"
        "Return exactly one token label: AFFIRMED or REVERSED.\n\n"
        f"[CASE]\n{text}\n\n[OUTCOME]\n"
    )


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


def main() -> None:
    args = parse_args()
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, Trainer, TrainingArguments
    except Exception as exc:
        print(f"Missing dependency: {exc}")
        print("Install requirements and retry: pip install -r requirements.txt")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_binary_dataset(Path(args.data_path))
    y = df[LABEL_COL].to_numpy(dtype=object)
    y_ids = np.array([1 if label == POS_LABEL else 0 for label in y], dtype=int)
    texts = np.array(build_text(df, args.text_variant), dtype=object)
    prompts = np.array([format_prompt(text) for text in texts], dtype=object)

    print(f"Loaded {len(df)} binary rows from {args.data_path}")
    print(df[LABEL_COL].value_counts().to_string())
    print(f"Model: {args.model_name}")
    print(f"Text variant: {args.text_variant}")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def make_supervised_dataset(indices: np.ndarray) -> Dataset:
        rows = []
        for idx in indices:
            prompt = str(prompts[idx])
            answer = LABEL_TO_TEXT[str(y[idx])]
            prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=args.max_length).input_ids
            answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
            input_ids = (prompt_ids + answer_ids + [tokenizer.eos_token_id])[: args.max_length]
            labels = [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id]
            labels = labels[: args.max_length]
            attention_mask = [1] * len(input_ids)
            pad_len = args.max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [tokenizer.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
                labels += [-100] * pad_len
            rows.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
        dataset = Dataset.from_list(rows)
        dataset.set_format(type="torch")
        return dataset

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    all_predictions = np.empty(len(y), dtype=object)
    fold_assignments = np.empty(len(y), dtype=int)
    fold_metrics = []
    safe_model_name = args.model_name.replace("/", "__")

    for fold, (train_val_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y_ids)), y_ids), start=1):
        train_val_idx = np.array(train_val_idx)
        test_idx = np.array(test_idx)
        inner = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed + 1000 + fold)
        train_inner_idx, val_inner_idx = next(inner.split(train_val_idx, y_ids[train_val_idx]))
        train_idx = train_val_idx[train_inner_idx]
        val_idx = train_val_idx[val_inner_idx]

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            token=hf_token,
            torch_dtype=torch.float16 if torch.cuda.is_available() else None,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quantization_config,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        if args.use_4bit:
            model = prepare_model_for_kbit_training(model)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints" / f"{safe_model_name}_{args.text_variant}_fold_{fold}"),
            per_device_train_batch_size=args.train_bs,
            per_device_eval_batch_size=args.eval_bs,
            learning_rate=args.lr,
            num_train_epochs=args.max_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_strategy="epoch",
            seed=args.seed,
            fp16=torch.cuda.is_available(),
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=make_supervised_dataset(train_idx),
            eval_dataset=make_supervised_dataset(val_idx),
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        )
        trainer.train()
        eval_metrics = trainer.evaluate()

        model.eval()

        def score_label(prompt: str, label_text: str) -> float:
            prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_length).input_ids.to(model.device)
            label_ids = tokenizer(label_text, add_special_tokens=False).input_ids
            label_tensor = torch.tensor(label_ids, device=model.device).unsqueeze(0)
            input_ids = torch.cat([prompt_ids, label_tensor], dim=1)
            with torch.no_grad():
                logits = model(input_ids).logits
            log_probs = torch.log_softmax(logits, dim=-1)
            prompt_len = prompt_ids.shape[1]
            scores = []
            for pos_offset, token_id in enumerate(label_ids):
                scores.append(log_probs[0, prompt_len - 1 + pos_offset, token_id])
            return float(torch.sum(torch.stack(scores)).item())

        fold_pred = []
        for idx in test_idx:
            prompt = str(prompts[idx])
            score_affirmed = score_label(prompt, "AFFIRMED")
            score_reversed = score_label(prompt, "REVERSED")
            fold_pred.append(NEG_LABEL if score_affirmed >= score_reversed else POS_LABEL)

        fold_pred_array = np.array(fold_pred, dtype=object)
        all_predictions[test_idx] = fold_pred_array
        fold_assignments[test_idx] = fold
        fold_metrics.append(
            {
                "fold": fold,
                "train_size": int(len(train_idx)),
                "validation_size": int(len(val_idx)),
                "test_size": int(len(test_idx)),
                "validation_loss": float(eval_metrics.get("eval_loss", np.nan)),
                "test_accuracy": float(accuracy_score(y[test_idx], fold_pred_array)),
                "test_macro_f1": float(f1_score(y[test_idx], fold_pred_array, labels=BINARY_LABELS, average="macro", zero_division=0)),
            }
        )

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = metric_dict(y, all_predictions)
    metrics.update(
        {
            "model": args.model_name,
            "text_variant": args.text_variant,
            "n_splits": args.n_splits,
            "seed": args.seed,
            "fold_metrics": fold_metrics,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules,
            "use_4bit": args.use_4bit,
        }
    )

    prediction_path = output_dir / f"lora_{safe_model_name}_{args.text_variant}_predictions.csv"
    metrics_path = output_dir / f"lora_{safe_model_name}_{args.text_variant}_metrics.json"
    pd.DataFrame(
        {
            "doc_id": df["doc_id"].astype(str),
            "fold": fold_assignments,
            "gold": y,
            "pred": all_predictions,
        }
    ).to_csv(prediction_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]}, indent=2))
    print(f"Saved predictions: {prediction_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
