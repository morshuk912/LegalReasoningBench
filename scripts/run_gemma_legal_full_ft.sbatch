#!/bin/bash
# Gemma full-model instruction tuning for outcome + judicial reasoning.
# This updates the full base model weights; it does not use LoRA/QLoRA.

#SBATCH --job-name=gemma_legal_full_ft
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=main
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate my_env

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_PATH="${DATA_PATH:-$REPO_DIR/data/final_dataset.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/results/gemma_legal_full_ft}"

python -m pip install --upgrade pip
python -m pip install -r "$REPO_DIR/reasoning_instruction_tuning/requirements.txt"

python "$REPO_DIR/reasoning_instruction_tuning/gemma_legal_full_ft.py" \
  --model_name google/gemma-2-2b-it \
  --input_csv "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_seq_length 1024 \
  --batch_size 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --seed 42 \
  --gradient_accumulation_steps 8 \
  --generation_max_new_tokens 160 \
  --gradient_checkpointing \
  ${HF_TOKEN:+--hf_token "$HF_TOKEN"}
