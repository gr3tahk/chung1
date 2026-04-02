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

## Project Structure
- `benchmark.py` — LLM agent and game loop
- `vlm_agent.py` — VLM agent (Qwen-VL) [Josh]
- `test_llm.py` — quick test scripts
