from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


SUMMARY_COLUMNS = [
    "pair",
    "model",
    "task_id",
    "trials",
    "success",
    "avg_score",
    "avg_tes",
    "avg_pc",
    "avg_ticks",
    "avg_invalid",
]


def _pc(metrics: dict[str, Any]) -> float:
    return float(metrics.get("progress_completeness", {}).get("pc", 0.0))


def flatten_run(run: dict[str, Any], default_model: str = "") -> dict[str, Any]:
    metrics = run.get("metrics", {})
    return {
        "pair": run.get("pair", ""),
        "model": run.get("model", default_model),
        "task_id": run.get("task_id", ""),
        "trial": run.get("trial"),
        "success": bool(run.get("success", metrics.get("success", False))),
        "score": float(run.get("score", 0)),
        "tes": float(metrics.get("tes", run.get("tes", 0.0))),
        "pc": _pc(metrics) if "progress_completeness" in metrics else float(run.get("pc", 0.0)),
        "ticks": float(metrics.get("ticks", run.get("ticks", 0))),
        "invalid": float(metrics.get("invalid_action_count", run.get("invalid_action_count", 0))),
    }


def load_result_rows(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        default_model = payload.get("model", "")
        if "results" in payload:
            rows.extend(flatten_run(run, default_model=default_model) for run in payload["results"])
        else:
            rows.append(flatten_run(payload, default_model=default_model))
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["pair"], row["model"], row["task_id"])].append(row)

    summaries = []
    for (pair, model, task_id), group in sorted(groups.items()):
        successes = sum(1 for row in group if row["success"])
        trials = len(group)
        summaries.append(
            {
                "pair": pair,
                "model": model,
                "task_id": task_id,
                "trials": trials,
                "success": f"{successes}/{trials}",
                "avg_score": mean(row["score"] for row in group),
                "avg_tes": mean(row["tes"] for row in group),
                "avg_pc": mean(row["pc"] for row in group),
                "avg_ticks": mean(row["ticks"] for row in group),
                "avg_invalid": mean(row["invalid"] for row in group),
            }
        )
    return summaries


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def format_summary_table(summaries: list[dict[str, Any]]) -> str:
    if not summaries:
        return "No runs found."

    rendered_rows = [{column: _format_value(row[column]) for column in SUMMARY_COLUMNS} for row in summaries]
    widths = {
        column: max(len(column), *(len(row[column]) for row in rendered_rows))
        for column in SUMMARY_COLUMNS
    }
    header = "  ".join(column.ljust(widths[column]) for column in SUMMARY_COLUMNS)
    divider = "  ".join("-" * widths[column] for column in SUMMARY_COLUMNS)
    body = [
        "  ".join(row[column].ljust(widths[column]) for column in SUMMARY_COLUMNS)
        for row in rendered_rows
    ]
    return "\n".join([header, divider, *body])


def summarize_files(paths: list[str | Path]) -> str:
    rows = load_result_rows(paths)
    return format_summary_table(summarize_rows(rows))


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize Overcooked benchmark result JSON files.")
    parser.add_argument("paths", nargs="+", help="Result JSON files to summarize.")
    return parser.parse_args()


def main():
    args = parse_args()
    print(summarize_files(args.paths))


if __name__ == "__main__":
    main()
