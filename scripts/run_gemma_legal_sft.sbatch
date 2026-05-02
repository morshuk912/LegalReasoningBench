#!/bin/bash
# Gemma LoRA/QLoRA instruction-tuning run for outcome + judicial reasoning.

#SBATCH --job-name=gemma_legal_sft
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=main
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-12:00:00

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate my_env

export TOKENIZERS_PARALLELISM=false

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_PATH="${DATA_PATH:-$REPO_DIR/data/final_dataset.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/results/gemma_legal_sft}"

python -m pip install --upgrade pip
python -m pip install -r "$REPO_DIR/reasoning_instruction_tuning/requirements.txt"

python "$REPO_DIR/reasoning_instruction_tuning/gemma_legal_sft.py" \
  --model_name google/gemma-2-2b-it \
  --input_csv "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length 2048 \
  --batch_size 1 \
  --learning_rate 2e-4 \
  --num_train_epochs 3 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --seed 42 \
  --gradient_accumulation_steps 8 \
  --generation_max_new_tokens 160 \
  --gradient_checkpointing \
  --use_4bit \
  ${HF_TOKEN:+--hf_token "$HF_TOKEN"}
