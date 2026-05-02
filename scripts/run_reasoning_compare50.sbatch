#!/bin/bash
# Submit one model by passing MODEL_KEY=saullm|lawma|lawllm.
# Example:
#   MODEL_KEY=saullm sbatch scripts/run_reasoning_compare50.sbatch

#SBATCH --job-name=reasoning_compare50
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=main
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate my_env

export TOKENIZERS_PARALLELISM=false

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_PATH="${DATA_PATH:-$REPO_DIR/data/reasoning_minimal_same50.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/results/reasoning_benchmark}"

python -m pip install --upgrade pip
python -m pip install -r "$REPO_DIR/reasoning_benchmark/requirements.txt"

python "$REPO_DIR/reasoning_benchmark/run_reasoning_generation.py" \
  --data_path "$DATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --model_key "${MODEL_KEY:?Set MODEL_KEY=saullm, lawma, or lawllm}" \
  ${HF_TOKEN:+--hf_token "$HF_TOKEN"}
