#!/usr/bin/env python3
"""
Prediction benchmark: classical baselines and encoder fine-tuning.

Models covered:
- Prior-weighted random baseline
- Logistic Regression + TF-IDF, facts only
- Logistic Regression + TF-IDF, facts + claims
- Optional Hugging Face encoder models with 5-fold stratified CV

The script uses the final binary thesis label column, binary_outcome_label.
For compatibility with earlier local files, it can also read
final_outcome_group_expert4 and rename it internally.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


NEG_LABEL = "Affirmed"
POS_LABEL = "Reversed/Vacated"
BINARY_LABELS = [NEG_LABEL, POS_LABEL]
LABEL_COL = "binary_outcome_label"
LEGACY_LABEL_COLS = ["final_outcome_group_expert4"]

TEXT_VARIANTS = ["facts", "facts_claims"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean binary prediction benchmarks.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument(
        "--run",
        choices=["classical", "encoders", "all"],
        default="classical",
        help="Which benchmark family to run.",
    )
    parser.add_argument(
        "--encoder_model",
        action="append",
        default=[],
        help="HF encoder model to fine-tune. Can be passed multiple times.",
    )
    parser.add_argument(
        "--encoder_text_variant",
        action="append",
        choices=TEXT_VARIANTS,
        default=[],
        help="Text variant to run for encoder models. Defaults to both facts and facts_claims.",
    )
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--train_bs", type=int, default=8)
    parser.add_argument("--eval_bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--lr_class_weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--lr_max_features", type=int, default=200_000)
    parser.add_argument("--lr_min_df", type=int, default=2)
    parser.add_argument("--lr_solver", default="liblinear")
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
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[LABEL_COL] = df[LABEL_COL].map(clean_text)
    df = df[df[LABEL_COL].isin(BINARY_LABELS)].copy()
    df = df.drop_duplicates(subset=["doc_id"], keep="first").reset_index(drop=True)
    if df[LABEL_COL].nunique() != 2:
        raise ValueError(f"Expected exactly two labels after filtering, got {df[LABEL_COL].unique().tolist()}")
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


def label_to_id(labels: np.ndarray) -> np.ndarray:
    return np.array([1 if label == POS_LABEL else 0 for label in labels], dtype=int)


def id_to_label(ids: np.ndarray) -> np.ndarray:
    return np.array([POS_LABEL if int(idx) == 1 else NEG_LABEL for idx in ids], dtype=object)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = BINARY_LABELS
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def save_outputs(
    output_dir: Path,
    model_key: str,
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    folds: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(
        {
            "doc_id": df["doc_id"].astype(str),
            "fold": folds,
            "gold": y_true,
            "pred": y_pred,
        }
    )
    pred_df.to_csv(output_dir / f"{model_key}_predictions.csv", index=False)
    (output_dir / f"{model_key}_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_cv_predictions(
    y: np.ndarray,
    n_splits: int,
    seed: int,
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    predictions = np.empty(len(y), dtype=object)
    folds = np.empty(len(y), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        predictions[test_idx] = fit_predict(train_idx, test_idx)
        folds[test_idx] = fold
    return predictions, folds


def run_prior_random(df: pd.DataFrame, output_dir: Path, n_splits: int, seed: int) -> dict[str, Any]:
    y = df[LABEL_COL].to_numpy(dtype=object)

    def fit_predict(train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
        # Matches the original benchmark script: each fold draws from the
        # empirical training-label distribution using the same fixed seed.
        rng = np.random.default_rng(seed)
        train_labels, counts = np.unique(y[train_idx], return_counts=True)
        probabilities = counts / counts.sum()
        return rng.choice(train_labels, size=len(test_idx), p=probabilities)

    pred, folds = run_cv_predictions(y, n_splits, seed, fit_predict)
    metrics = metric_dict(y, pred)
    metrics.update({"model": "prior_weighted_random", "n_splits": n_splits, "seed": seed})
    save_outputs(output_dir, "prior_weighted_random", df, y, pred, folds, metrics)
    return metrics


def run_tfidf_lr(
    df: pd.DataFrame,
    text_variant: str,
    output_dir: Path,
    n_splits: int,
    seed: int,
    class_weight: str | None,
    max_features: int,
    min_df: int,
    solver: str,
) -> dict[str, Any]:
    y = df[LABEL_COL].to_numpy(dtype=object)
    texts = np.array(build_text(df, text_variant), dtype=object)

    def fit_predict(train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
        vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=(1, 2), max_features=max_features)
        x_train = vectorizer.fit_transform(texts[train_idx])
        x_test = vectorizer.transform(texts[test_idx])
        classifier = LogisticRegression(
            max_iter=3000,
            class_weight=class_weight,
            solver=solver,
            n_jobs=None if solver == "liblinear" else -1,
            random_state=seed,
        )
        classifier.fit(x_train, y[train_idx])
        return classifier.predict(x_test)

    pred, folds = run_cv_predictions(y, n_splits, seed, fit_predict)
    metrics = metric_dict(y, pred)
    metrics.update(
        {
            "model": "tfidf_logistic_regression",
            "text_variant": text_variant,
            "n_splits": n_splits,
            "seed": seed,
            "class_weight": class_weight or "none",
            "max_features": max_features,
            "min_df": min_df,
            "solver": solver,
        }
    )
    save_outputs(output_dir, f"tfidf_lr_{text_variant}", df, y, pred, folds, metrics)
    return metrics


def run_encoder_cv(
    df: pd.DataFrame,
    model_name: str,
    text_variant: str,
    output_dir: Path,
    n_splits: int,
    seed: int,
    max_epochs: int,
    patience: int,
    val_ratio: float,
    max_length: int,
    train_bs: int,
    eval_bs: int,
    lr: float,
    use_class_weights: bool,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from sklearn.utils.class_weight import compute_class_weight
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments

    y = df[LABEL_COL].to_numpy(dtype=object)
    y_ids = label_to_id(y)
    texts = np.array(build_text(df, text_variant), dtype=object)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def make_dataset(indices: np.ndarray) -> Dataset:
        dataset = Dataset.from_dict({"text": texts[indices].tolist(), "labels": y_ids[indices].tolist()})

        def tokenize(batch):
            return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

        dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
        dataset.set_format(type="torch")
        return dataset

    class WeightedTrainer(Trainer):
        def __init__(self, class_weights=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs["labels"]
            outputs = model(**{key: value for key, value in inputs.items() if key != "labels"})
            logits = outputs.logits
            if self.class_weights is None:
                loss_fn = torch.nn.CrossEntropyLoss()
            else:
                loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred_ids = np.argmax(logits, axis=1)
        return {
            "accuracy": float(accuracy_score(labels, pred_ids)),
            "macro_f1": float(f1_score(labels, pred_ids, average="macro", zero_division=0)),
        }

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_pred_ids = np.empty(len(y_ids), dtype=int)
    folds = np.empty(len(y_ids), dtype=int)
    fold_metrics: list[dict[str, Any]] = []
    safe_model_name = model_name.replace("/", "__")

    for fold, (train_val_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y_ids)), y_ids), start=1):
        train_val_idx = np.array(train_val_idx)
        test_idx = np.array(test_idx)
        inner = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed + 1000 + fold)
        train_inner_idx, val_inner_idx = next(inner.split(train_val_idx, y_ids[train_val_idx]))
        train_idx = train_val_idx[train_inner_idx]
        val_idx = train_val_idx[val_inner_idx]

        class_weights = None
        if use_class_weights:
            weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_ids[train_idx])
            class_weights = torch.tensor(weights, dtype=torch.float32)

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            id2label={0: NEG_LABEL, 1: POS_LABEL},
            label2id={NEG_LABEL: 0, POS_LABEL: 1},
        ).to(device)

        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints" / f"{safe_model_name}_{text_variant}_fold_{fold}"),
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=eval_bs,
            learning_rate=lr,
            num_train_epochs=max_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            logging_strategy="epoch",
            seed=seed,
            fp16=torch.cuda.is_available(),
            report_to=[],
        )

        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=make_dataset(train_idx),
            eval_dataset=make_dataset(val_idx),
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        )
        trainer.train()
        eval_metrics = trainer.evaluate()
        predictions = trainer.predict(make_dataset(test_idx)).predictions
        pred_ids = np.argmax(predictions, axis=1)
        all_pred_ids[test_idx] = pred_ids
        folds[test_idx] = fold
        fold_metrics.append(
            {
                "fold": fold,
                "train_size": int(len(train_idx)),
                "validation_size": int(len(val_idx)),
                "test_size": int(len(test_idx)),
                "validation_accuracy": float(eval_metrics.get("eval_accuracy", np.nan)),
                "validation_macro_f1": float(eval_metrics.get("eval_macro_f1", np.nan)),
                "test_accuracy": float(accuracy_score(y_ids[test_idx], pred_ids)),
                "test_macro_f1": float(f1_score(y_ids[test_idx], pred_ids, average="macro", zero_division=0)),
            }
        )

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    y_pred = id_to_label(all_pred_ids)
    metrics = metric_dict(y, y_pred)
    metrics.update(
        {
            "model": model_name,
            "text_variant": text_variant,
            "n_splits": n_splits,
            "seed": seed,
            "fold_metrics": fold_metrics,
            "use_class_weights": use_class_weights,
        }
    )
    model_key = f"encoder_{safe_model_name}_{text_variant}"
    save_outputs(output_dir, model_key, df, y, y_pred, folds, metrics)
    return metrics


def save_summary(output_dir: Path, metrics_list: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for metrics in metrics_list:
        summary_rows.append(
            {
                "model": metrics.get("model"),
                "text_variant": metrics.get("text_variant", ""),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "metrics_summary.csv", index=False)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    df = load_binary_dataset(Path(args.data_path))
    print(f"Loaded {len(df)} binary rows from {args.data_path}")
    print(df[LABEL_COL].value_counts().to_string())

    all_metrics: list[dict[str, Any]] = []
    if args.run in {"classical", "all"}:
        all_metrics.append(run_prior_random(df, output_dir, args.n_splits, args.seed))
        lr_class_weight = None if args.lr_class_weight == "none" else args.lr_class_weight
        for variant in TEXT_VARIANTS:
            all_metrics.append(
                run_tfidf_lr(
                    df,
                    variant,
                    output_dir,
                    args.n_splits,
                    args.seed,
                    lr_class_weight,
                    args.lr_max_features,
                    args.lr_min_df,
                    args.lr_solver,
                )
            )

    if args.run in {"encoders", "all"}:
        encoder_models = args.encoder_model or ["distilbert-base-uncased", "nlpaueb/legal-bert-base-uncased"]
        encoder_variants = args.encoder_text_variant or TEXT_VARIANTS
        for model_name in encoder_models:
            for variant in encoder_variants:
                all_metrics.append(
                    run_encoder_cv(
                        df=df,
                        model_name=model_name,
                        text_variant=variant,
                        output_dir=output_dir,
                        n_splits=args.n_splits,
                        seed=args.seed,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        val_ratio=args.val_ratio,
                        max_length=args.max_length,
                        train_bs=args.train_bs,
                        eval_bs=args.eval_bs,
                        lr=args.lr,
                        use_class_weights=not args.no_class_weights,
                    )
                )

    save_summary(output_dir, all_metrics)


if __name__ == "__main__":
    main()
