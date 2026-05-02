#!/usr/bin/env python3
"""
Generate concise judicial-reasoning summaries for a fixed comparison subset.

This is the cleaned repo version of the domain-LLM reasoning comparison flow.
It runs one model at a time, writes one CSV per model, and keeps raw generations
for auditability.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


CASE_ID_COL = "doc_id"
FACTS_COL = "facts"
PLAINTIFF_COL = "claim_plaintiff"
DEFENDANT_COL = "claim_defendant"
LABEL_COL = "binary_outcome_label"
LEGACY_LABEL_COLS = ["final_outcome_group_expert4"]
GOLD_REASONING_COLS = ["judicial_reasoning", "reasoning"]

PROMPT_TEMPLATE = """You are a legal expert.

Task: Write a concise summary of the COURT'S REASONING only.

Rules:
Use ONLY information explicitly stated in the input fields below.
Do not add facts, parties, dates, statutes, or procedural events that are not in the input.
Focus on why the court reached its decision, not on repeating the parties' claims.
Write 2-4 concrete sentences.

Facts:
{facts}

Plaintiff/Appellant Claims:
{plaintiff}

Defendant/Appellee Claims:
{defendant}

Return only the reasoning summary as plain text.
"""

LAWMA_PROMPT_TEMPLATE = """Write exactly 2 plain-text sentences summarizing the court's reasoning.
Use only the information below.
Do not repeat the prompt.
Do not list numbers or bullets.
Do not output JSON.

Facts:
{facts}

Plaintiff/Appellant Claims:
{plaintiff}

Defendant/Appellee Claims:
{defendant}

Court reasoning summary:
"""

MODEL_CONFIGS = {
    "saullm": {
        "model_name": "Equall/Saul-7B-Instruct-v1",
        "use_4bit": False,
        "trust_remote_code": True,
        "prompt_style": "default",
        "max_new_tokens": 280,
    },
    "lawma": {
        "model_name": "ricdomolm/lawma-8b",
        "use_4bit": False,
        "trust_remote_code": False,
        "prompt_style": "lawma_simple",
        "max_new_tokens": 96,
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 6,
    },
    "lawllm": {
        "model_name": "AdaptLLM/law-LLM-13B",
        "use_4bit": True,
        "trust_remote_code": True,
        "prompt_style": "default",
        "max_new_tokens": 280,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reasoning generation for one legal LLM.")
    parser.add_argument("--data_path", default="data/reasoning_minimal_same50.csv")
    parser.add_argument("--output_dir", default="results/reasoning_benchmark")
    parser.add_argument("--model_key", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--max_input_chars", type=int, default=4500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--sleep_between_cases", type=float, default=0.05)
    parser.add_argument("--hf_token", default=None)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def clip(value: Any, max_chars: int) -> str:
    return clean_text(value)[:max_chars]


def resolve_column(frame: pd.DataFrame, preferred: str, aliases: list[str] | None = None) -> str:
    for column in [preferred, *(aliases or [])]:
        if column in frame.columns:
            return column
    raise ValueError(f"Missing required column: {preferred}")


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if LABEL_COL not in frame.columns:
        for legacy_col in LEGACY_LABEL_COLS:
            if legacy_col in frame.columns:
                frame = frame.rename(columns={legacy_col: LABEL_COL})
                break
    for column in [CASE_ID_COL, FACTS_COL, PLAINTIFF_COL, DEFENDANT_COL, LABEL_COL]:
        if column not in frame.columns:
            raise ValueError(f"Missing required column in {path}: {column}")
    return frame.copy()


def gold_reasoning(row: pd.Series) -> str:
    for column in GOLD_REASONING_COLS:
        if column in row.index:
            value = clean_text(row.get(column))
            if value:
                return value
    return ""


def build_prompt(row: pd.Series, max_chars: int, prompt_style: str) -> str:
    values = {
        "facts": clip(row.get(FACTS_COL, ""), max_chars),
        "plaintiff": clip(row.get(PLAINTIFF_COL, ""), max_chars),
        "defendant": clip(row.get(DEFENDANT_COL, ""), max_chars),
    }
    template = LAWMA_PROMPT_TEMPLATE if prompt_style == "lawma_simple" else PROMPT_TEMPLATE
    return template.format(**values)


def cleanup_reasoning(text: str, prompt_style: str) -> str:
    cleaned = clean_text(text)
    cleaned = re.sub(r"^Output:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\[/INST\]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if prompt_style == "lawma_simple":
        marker = "court reasoning summary:"
        lower = cleaned.lower()
        if marker in lower:
            cleaned = cleaned[lower.find(marker) + len(marker) :].strip()
        if cleaned.lower().startswith("summarize the court"):
            return ""
        if re.search(r"\b(?:[A-T]\s+){8,}[A-T]\b", cleaned):
            return ""
        if re.fullmatch(r"[\d\s.,;:-]+", cleaned):
            return ""
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", cleaned) if sentence.strip()]
        if sentences:
            cleaned = " ".join(sentences[:4])
    return cleaned


def load_model(cfg: dict[str, Any], adapter_dir: str | None, hf_token: str | None):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        trust_remote_code=cfg.get("trust_remote_code", True),
        use_fast=cfg.get("prompt_style") != "lawma_simple",
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if cfg.get("use_4bit"):
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        trust_remote_code=cfg.get("trust_remote_code", True),
        token=hf_token,
    )
    if adapter_dir:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed but --adapter_dir was provided.")
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tokenizer, model


def generate(prompt: str, tokenizer, model, cfg: dict[str, Any], temperature: float, top_p: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    gen_kwargs = {
        "max_new_tokens": cfg.get("max_new_tokens", 280),
        "do_sample": temperature > 0,
        "top_p": top_p,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": cfg.get("repetition_penalty", 1.0),
    }
    if cfg.get("no_repeat_ngram_size"):
        gen_kwargs["no_repeat_ngram_size"] = cfg["no_repeat_ngram_size"]
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt) :].strip() if decoded.startswith(prompt) else decoded.strip()


def main() -> None:
    args = parse_args()
    cfg = MODEL_CONFIGS[args.model_key]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(Path(args.data_path))
    tokenizer, model = load_model(cfg, args.adapter_dir, args.hf_token)

    rows = []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f"Reasoning {args.model_key}"):
        prompt = build_prompt(row, args.max_input_chars, cfg.get("prompt_style", "default"))
        raw_text = ""
        reasoning_summary = ""
        error = ""
        try:
            raw_text = generate(prompt, tokenizer, model, cfg, args.temperature, args.top_p)
            reasoning_summary = cleanup_reasoning(raw_text, cfg.get("prompt_style", "default"))
            if not reasoning_summary:
                error = "empty_after_cleanup"
        except Exception as exc:
            error = str(exc)

        rows.append(
            {
                "doc_id": row.get(CASE_ID_COL, ""),
                "gold_outcome_label": row.get(LABEL_COL, ""),
                "gold_reasoning": gold_reasoning(row),
                "reasoning_summary": reasoning_summary,
                "raw_text": raw_text,
                "error": error,
                "model_key": args.model_key,
                "model_name": cfg["model_name"],
            }
        )
        time.sleep(args.sleep_between_cases)

    output_path = output_dir / f"reasoning_{args.model_key}_same50.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    run_config = {**cfg, "data_path": args.data_path, "adapter_dir": args.adapter_dir}
    (output_dir / f"reasoning_{args.model_key}_run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
