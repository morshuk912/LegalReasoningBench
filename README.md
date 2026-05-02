# LegalReasoningBench

This repository contains the code used for the thesis experiments, organized by pipeline stage.

## Structure

- `dataset_creation/` - scripts for extracting, validating, and preparing the final binary dataset.
- `prediction_benchmark/` - classical, encoder, and packaged LoRA prediction benchmarks.
- `reasoning_benchmark/` - reasoning generation and merge utilities for model comparison.
- `reasoning_instruction_tuning/` - Gemma supervised fine-tuning scripts for outcome prediction and judicial reasoning.
- `ablation/` - ablation-study code, including CoT/story-facts prompt variants.
- `scripts/` - SLURM job scripts for running the main experiments.
- `configs/` - optional run configuration files.
- `data/` - local input CSVs. Data files are ignored by git.
- `results/` - local experiment outputs. Result files are ignored by git.

## Expected Data Files

Place local datasets under `data/` or pass explicit paths with the relevant CLI flags.

- `data/final_dataset.csv`
- `data/reasoning_minimal_same50.csv`
- `data/reasoning_doc_ids.csv`

The scripts accept alternative paths via arguments such as `--data_path`, `--input_csv`, `--base_csv`, and `--output_dir`.

## Running Examples

Prediction benchmark:

```bash
python prediction_benchmark/classical_and_encoder_benchmarks.py \
  --data_path data/final_dataset.csv \
  --output_dir results/prediction_benchmark
```

LLM prediction benchmark, Prompt 2, 200 cases, facts only and facts+claims:

```bash
python prediction_benchmark/run_gpt_prediction_200.py \
  --data_path data/final_dataset.csv

python prediction_benchmark/run_deepseek_prediction_200.py \
  --data_path data/final_dataset.csv

python prediction_benchmark/run_gemini_prediction_200.py \
  --data_path data/final_dataset.csv
```

Reasoning generation:

```bash
python reasoning_benchmark/run_reasoning_generation.py \
  --data_path data/reasoning_minimal_same50.csv \
  --output_dir results/reasoning_benchmark \
  --model_key saullm
```

API reasoning generation with GPT, DeepSeek, and Gemini:

```bash
python reasoning_benchmark/run_api_reasoning_generation.py \
  --data_path data/final_dataset.csv \
  --ids_path data/reasoning_doc_ids.csv \
  --output_dir results/reasoning_benchmark \
  --models gpt deepseek gemini \
  --merge_outputs
```

Gemma instruction tuning:

```bash
python reasoning_instruction_tuning/gemma_legal_sft.py \
  --input_csv data/final_dataset.csv \
  --output_dir results/gemma_legal_sft
```

Additional prediction experiments:

```bash
python ablation/run_gpt_cot.py \
  --data_path data/final_dataset.csv \
  --output_dir results/ablation/gpt_cot

python ablation/run_deepseek_story_facts.py \
  --data_path data/final_dataset.csv \
  --output_dir results/ablation/deepseek_story_facts

python ablation/run_gemini_story_facts.py \
  --data_path data/final_dataset.csv \
  --output_dir results/ablation/gemini_story_facts
```

## Notes

Large datasets, model checkpoints, generated predictions, and final experiment outputs are intentionally excluded from git. The SLURM scripts can be configured with `REPO_DIR`, `DATA_PATH`, and `OUTPUT_DIR` environment variables.
