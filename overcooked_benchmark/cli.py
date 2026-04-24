from __future__ import annotations

import argparse

from overcooked_benchmark.config import (
    DEFAULT_MAX_TICKS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TRACE_OUTPUT,
    DEFAULT_TRIALS,
    DEFAULT_VISION_MODEL,
    LAYOUTS,
)
from overcooked_benchmark.runners.paired import run_agent_pair
from overcooked_benchmark.runners.suite import run_experiment_suite
from overcooked_benchmark.summarize import flatten_run, format_summary_table, summarize_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Run paired LLM/VLM Overcooked benchmarks.")
    parser.add_argument(
        "--pair",
        choices=["llm-llm", "vlm-vlm", "scripted-scripted"],
        default="scripted-scripted",
        help="Agent pair to benchmark.",
    )
    parser.add_argument("--layout", choices=LAYOUTS, help="Run a single layout and task trace.")
    parser.add_argument("--task-id", default="cramped_room_single_delivery", help="Task metadata to use for trajectory metrics.")
    parser.add_argument("--backend", choices=["openai", "local"], default="openai", help="Inference backend for llm-llm agents.")
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL, help="OpenAI text model for --pair llm-llm.")
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL, help="OpenAI vision model for --pair vlm-vlm.")
    parser.add_argument("--local-model", default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model for --backend local.")
    parser.add_argument("--dtype", default="auto", help="Local model dtype: auto, float16, bfloat16, or float32.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map for local models.")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Local model generation token budget.")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="Trials for suite runs.")
    parser.add_argument("--max-ticks", type=int, default=DEFAULT_MAX_TICKS, help="Number of game ticks to simulate.")
    parser.add_argument("--collect-trajectory", action="store_true", help="Collect a detailed replay trajectory for the UI.")
    parser.add_argument("--trace-output", default=str(DEFAULT_TRACE_OUTPUT), help="Where to save replay JSON.")
    parser.add_argument("--experiment-output", default="experiment_results.json", help="Output JSON for suite runs.")
    return parser.parse_args()


def print_summary(summary: dict):
    metrics = summary["metrics"]
    pc = metrics.get("progress_completeness", {}).get("pc")
    print(f"Pair: {summary['pair']}")
    print(f"Score: {summary['score']}")
    print(f"Success: {summary['success']}")
    print(f"TES: {metrics.get('tes', 0):.3f}")
    if pc is not None:
        print(f"PC: {pc:.3f}")
    print(f"Invalid actions: {metrics.get('invalid_action_count', 0)}")


def main():
    args = parse_args()
    if args.layout:
        result = run_agent_pair(
            pair=args.pair,
            layout_name=args.layout,
            task_id=args.task_id,
            max_ticks=args.max_ticks,
            backend=args.backend,
            text_model=args.openai_model,
            vision_model=args.vision_model,
            local_model=args.local_model,
            dtype=args.dtype,
            device_map=args.device_map,
            max_new_tokens=args.max_new_tokens,
            collect_trajectory=args.collect_trajectory,
            trace_output_path=args.trace_output if args.collect_trajectory else None,
        )
        if args.collect_trajectory:
            summary, _ = result
            print(f"\nSaved replay trace to {args.trace_output}")
        else:
            summary = result
        print_summary(summary)
        return

    aggregate = run_experiment_suite(
        pair=args.pair,
        task_ids=[args.task_id],
        trials=args.trials,
        max_ticks=args.max_ticks,
        backend=args.backend,
        text_model=args.openai_model,
        vision_model=args.vision_model,
        local_model=args.local_model,
        dtype=args.dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        output_path=args.experiment_output,
    )
    print(f"Saved experiment results to {args.experiment_output}")
    print(f"Runs: {len(aggregate['results'])}")
    print(format_summary_table(summarize_rows([flatten_run(run, default_model=aggregate["model"]) for run in aggregate["results"]])))


if __name__ == "__main__":
    main()
