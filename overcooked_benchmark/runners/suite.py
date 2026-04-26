from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from overcooked_benchmark.config import DEFAULT_OPENAI_MODEL, DEFAULT_VISION_MODEL
from overcooked_benchmark.runners.paired import run_agent_pair
from overcooked_benchmark.tasks import get_task_by_id


def run_experiment_suite(
    pair: str,
    task_ids: list[str],
    trials: int,
    max_ticks: int,
    backend: str = "openai",
    text_model: str = DEFAULT_OPENAI_MODEL,
    vision_model: str = DEFAULT_VISION_MODEL,
    local_model: str = "Qwen/Qwen3-8B",
    dtype: str = "auto",
    device_map: str = "auto",
    max_new_tokens: int = 160,
    output_path: str | Path = "experiment_results.json",
    trace_dir: str | Path | None = None,
) -> dict[str, Any]:
    results = []
    model = local_model if backend == "local" else text_model if pair == "llm-llm" else vision_model if pair == "vlm-vlm" else "scripted"
    trace_root = Path(trace_dir) if trace_dir else None
    if trace_root:
        trace_root.mkdir(parents=True, exist_ok=True)
    for task_id in task_ids:
        task = get_task_by_id(task_id)
        for trial in range(trials):
            run_result = run_agent_pair(
                pair=pair,
                layout_name=task["layout"],
                task_id=task_id,
                max_ticks=max_ticks,
                backend=backend,
                text_model=text_model,
                vision_model=vision_model,
                local_model=local_model,
                dtype=dtype,
                device_map=device_map,
                max_new_tokens=max_new_tokens,
                collect_trajectory=trace_root is not None,
            )
            if trace_root:
                summary, trajectory = run_result
                trace_path = trace_root / f"{pair}_{backend}_{task_id}_trial{trial}.json"
                with trace_path.open("w") as handle:
                    json.dump(trajectory, handle, indent=2)
                summary["trace_path"] = str(trace_path)
            else:
                summary = run_result
            summary["trial"] = trial
            summary["model"] = model
            summary["max_ticks"] = max_ticks
            summary["flat_metrics"] = {
                "success": summary["success"],
                "score": summary["score"],
                "tes": summary["metrics"].get("tes", 0.0),
                "pc": summary["metrics"].get("progress_completeness", {}).get("pc", 0.0),
                "ticks": summary["metrics"].get("ticks", max_ticks),
                "invalid_action_count": summary["metrics"].get("invalid_action_count", 0),
            }
            results.append(summary)

    aggregate = {
        "pair": pair,
        "backend": backend,
        "model": model,
        "trials": trials,
        "max_ticks": max_ticks,
        "task_ids": task_ids,
        "results": results,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(aggregate, handle, indent=2)
    return aggregate
