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

## Dataset Construction Statistics

The reviewed source file contains 4,782 cases before binary filtering. Outcome labels were first annotated with 15 unique fine-grained labels, then grouped into the binary benchmark labels used in the main prediction task.

| Statistic | Value |
| --- | ---: |
| Source cases before binary filtering | 4,782 |
| Final binary benchmark cases | 4,042 |
| Affirmed cases | 2,747 |
| Reversed/Vacated cases | 1,295 |
| Unique outcome labels before binary grouping | 15 |
| Avg. full opinion words/tokens, non-empty rows | 2,401.1 |
| Median full opinion words/tokens, non-empty rows | 2,031.0 |
| Avg. extracted facts words/tokens | 135.7 |
| Median extracted facts words/tokens | 135.0 |
| Avg. judicial reasoning words/tokens | 83.3 |
| Median judicial reasoning words/tokens | 83.0 |
| Avg. plaintiff/appellant claim words/tokens | 43.3 |
| Avg. defendant/appellee claim words/tokens | 39.9 |

Lengths are regex word-token counts computed by `dataset_creation/prepare_final_binary_dataset.py`. The full-opinion average excludes 207 binary rows with empty `full_document`; extracted facts are present for all 4,042 binary rows, and judicial reasoning is present for 4,041 rows.

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

Controlled RAG reasoning generation with GPT-4o:

```bash
python reasoning_benchmark/run_controlled_rag_reasoning.py \
  --data_path data/final_dataset.csv \
  --eval_ids_path data/reasoning_doc_ids.csv \
  --output_dir results/reasoning_benchmark/rag_controlled \
  --generator_model gpt-4o \
  --top_k 3 \
  --limit 50
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
