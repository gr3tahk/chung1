#!/bin/bash -l

# Set SCC project
#$ -P cs585

# Name the job and output log
#$ -N overcooked_gpt52_vlm_pilot
#$ -j y
#$ -o logs/$JOB_NAME.$JOB_ID.out

# Request wall time and 1 CPU. OpenAI runs do not need a GPU.
#$ -l h_rt=03:00:00
#$ -pe omp 1

set -e

# Use the SCC academic machine learning environment
module load miniconda
module load academic-ml/spring-2026
conda activate spring-2026-pyt

cd "$SGE_O_WORKDIR"
mkdir -p logs results

if [ -z "$OPENAI_API_KEY" ]; then
  echo "OPENAI_API_KEY is not set. Submit with: qsub -v OPENAI_API_KEY scripts/scc_openai_gpt52_vlm_pilot.sh"
  exit 1
fi

echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "PYTHON=$(which python)"
python --version

echo "== install project =="
python -m pip install -e .

echo "== GPT-5.2 VLM pilot =="
python benchmark.py \
  --pair vlm-vlm \
  --vision-model gpt-5.2 \
  --trials 3 \
  --max-ticks 60 \
  --experiment-output results/gpt52_vlm_pilot.json \
  --suite-trace-dir results/gpt52_vlm_pilot_traces

echo "== summary =="
python -m overcooked_benchmark.summarize results/gpt52_vlm_pilot.json

echo "DONE"
