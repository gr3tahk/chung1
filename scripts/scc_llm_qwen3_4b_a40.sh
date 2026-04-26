#!/bin/bash -l

# Set SCC project
#$ -P cs585

# Name the job and output log
#$ -N overcooked_qwen3_4b
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.out

# Request wall time, 4 CPUs, and 1 A40 GPU
#$ -l gpus=1
#$ -l gpu_type=A40

set -e

# Use the SCC academic machine learning environment
module load miniconda
module load academic-ml/spring-2026
conda activate spring-2026-pyt

cd "$SGE_O_WORKDIR"

export HF_HOME="$SGE_O_WORKDIR/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" logs results

echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "PYTHON=$(which python)"
python --version

echo "== GPU =="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "== install project =="
python -m pip install -e .

echo "== local LLM experiment =="
python benchmark.py \
  --pair llm-llm \
  --backend local \
  --local-model Qwen/Qwen3-4B-Instruct-2507 \
  --dtype bfloat16 \
  --device-map auto \
  --max-new-tokens 160 \
  --trials 3 \
  --max-ticks 60 \
  --experiment-output results/llm_qwen3_4b_cramped.json \
  --suite-trace-dir results/qwen3_4b_traces

echo "== summary =="
python -m overcooked_benchmark.summarize results/llm_qwen3_4b_cramped.json

echo "DONE"
