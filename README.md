# Overcooked VLM vs LLM Benchmark

Comparing VLM and LLM agents in the Overcooked cooperative cooking environment.

## Setup
```bash
uv venv --python 3.11
uv sync
```

## Paired LLM/VLM Benchmark

The benchmark harness supports low-level paired agents with the same action space:

- `llm-llm`: both agents receive symbolic text state.
- `vlm-vlm`: both agents receive a rendered board image plus the action list.
- `scripted-scripted`: deterministic local smoke-test agents, no API key needed.

Run a local smoke trace:

```bash
uv run python benchmark.py --pair scripted-scripted --layout cramped_room --max-ticks 8 --collect-trajectory
```

Run one LLM or VLM trace:

```bash
export OPENAI_API_KEY=your_key_here
uv run python benchmark.py --pair llm-llm --layout cramped_room --max-ticks 80 --collect-trajectory
uv run python benchmark.py --pair vlm-vlm --layout cramped_room --max-ticks 80 --collect-trajectory --vision-model gpt-4o
```

Run local text models, for example on SCC:

```bash
uv run python benchmark.py --pair llm-llm --backend local --local-model Qwen/Qwen2.5-7B-Instruct --trials 3 --max-ticks 60 --experiment-output results/llm_qwen25_7b.json
```

Run a small suite:

```bash
uv run python benchmark.py --pair llm-llm --openai-model gpt-5.4 --trials 3 --max-ticks 60 --experiment-output results/llm_llm.json
uv run python benchmark.py --pair vlm-vlm --vision-model gpt-5.4 --trials 3 --max-ticks 60 --experiment-output results/vlm_vlm.json
uv run python -m overcooked_benchmark.summarize results/llm_llm.json results/vlm_vlm.json
```

Suite runs also print the same summary table automatically.

## Collab-Overcooked Metrics

This repo implements the trajectory metrics from `colab.pdf`:

- `TES`: ordered trajectory efficiency against a reference action trajectory.
- `ITES`: incremental TES contribution of a new action or action batch.
- `PC`: per-agent progress completeness, averaged across agents.
- `IC` / `RC`: helper scoring for annotated collaboration initiations and responses once the LLM/VLM agents emit request/response events.

Run a local trajectory with metrics:

```bash
uv run python benchmark.py --pair scripted-scripted --layout cramped_room --collect-trajectory
```

The replay JSON includes `symbolic_actions`, `agent_histories`, and `metrics`, so LLM-LLM and VLM-VLM runs can be compared using the same output schema.

The paired runner records score, success, ticks, TES, PC, invalid action count, symbolic action history, agent messages, and prompt/response logs.

## Project Structure
- `benchmark.py` — thin CLI shim
- `overcooked_benchmark/agents/` — common agent interface plus OpenAI text/vision and scripted agents
- `overcooked_benchmark/runners/` — paired-agent and suite runners
- `overcooked_benchmark/tasks.py` — task definitions and reference trajectories
- `overcooked_benchmark/metrics.py` — TES, ITES, PC, IC, and RC metric helpers
- `overcooked_benchmark/symbolic.py` — transition-to-symbolic-action extraction
- `overcooked_benchmark/summarize.py` — result table summarizer
- `overcooked_benchmark/traces.py` — replay JSON serialization
- `overcooked_benchmark/rendering.py` — board renderer used for VLM observations
- `tests/` — unit and smoke tests
