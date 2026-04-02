# Overcooked VLM vs LLM Benchmark

Comparing VLM and LLM agents in the Overcooked cooperative cooking environment.

## Setup
```bash
conda create -n overcooked_vlm python=3.10
conda activate overcooked_vlm
pip install overcooked-ai openai pillow numpy
```

## Running LLM Baseline
```bash
export OPENAI_API_KEY=your_key_here
python benchmark.py
```

## Running With A Local Model
```bash
pip install torch transformers accelerate sentencepiece
python benchmark.py --backend local
```

Optional: override the default local model.

```bash
python benchmark.py --backend local --local-model Qwen/Qwen2.5-3B-Instruct
```

## Running TES/ITES Task Benchmark
```bash
python benchmark.py --mode tasks --backend local
```

Run a single task and write to a custom results file:

```bash
python benchmark.py --mode tasks --task-id cramped_room_single_delivery --output task_results.json
```

Task specs live in `benchmark_tasks.json`. Task-mode results include reward, success, condensed symbolic trajectory, per-tick action logs, and TES/ITES scores against the best-matching reference trajectory.

## Project Structure
- `benchmark.py` — LLM agent and game loop
- `metrics.py` — TES/ITES scoring helpers
- `benchmark_tasks.json` — task specs and reference trajectories
- `vlm_agent.py` — VLM agent (Qwen-VL) [Josh]
- `test_llm.py` — quick test scripts
