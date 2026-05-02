#!/bin/bash
# Rerun only the missing LegalBERT facts+claims encoder benchmark.

#SBATCH --job-name=legalbert_fc_cv
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=main
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate my_env

export TOKENIZERS_PARALLELISM=false

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_PATH="${DATA_PATH:-$REPO_DIR/data/final_dataset.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/results/encoder_benchmark_rerun_gpu}"

python -m pip install --upgrade pip
python -m pip install -r "$REPO_DIR/prediction_benchmark/requirements.txt"

python "$REPO_DIR/prediction_benchmark/classical_and_encoder_benchmarks.py" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --run encoders \
  --encoder_model nlpaueb/legal-bert-base-uncased \
  --encoder_text_variant facts_claims \
  --n_splits 5 \
  --seed 42 \
  --max_epochs 10 \
  --patience 2 \
  --val_ratio 0.1 \
  --max_length 384 \
  --train_bs 8 \
  --eval_bs 16 \
  --lr 2e-5
