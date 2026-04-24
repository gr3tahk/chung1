from __future__ import annotations

from pathlib import Path
from typing import Any

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from overcooked_benchmark.agents import AgentObservation, LocalTextAgent, OpenAITextAgent, OpenAIVisionAgent, ScriptedAgent
from overcooked_benchmark.config import DEFAULT_OPENAI_MODEL, DEFAULT_VISION_MODEL, PLAYER_NAMES
from overcooked_benchmark.evaluation import evaluate_task_trajectory
from overcooked_benchmark.openai_client import get_openai_client
from overcooked_benchmark.local_client import get_local_text_client
from overcooked_benchmark.phase import task_phase_hint
from overcooked_benchmark.symbolic import classify_player_action
from overcooked_benchmark.tasks import get_task_by_id
from overcooked_benchmark.traces import (
    build_event_entries,
    build_frame,
    build_headline,
    build_layout_snapshot,
    save_trajectory,
    serialize_action,
)


def make_agent_pair(
    pair: str,
    backend: str,
    text_model: str,
    vision_model: str,
    local_model: str,
    dtype: str,
    device_map: str,
    max_new_tokens: int,
):
    if pair == "llm-llm":
        if backend == "local":
            client = get_local_text_client(
                model_name=local_model,
                dtype=dtype,
                device_map=device_map,
                max_new_tokens=max_new_tokens,
            )
            return [
                LocalTextAgent(0, PLAYER_NAMES[0], client),
                LocalTextAgent(1, PLAYER_NAMES[1], client),
            ]
        if backend != "openai":
            raise ValueError(f"Unsupported backend for llm-llm: {backend}")
        client = get_openai_client()
        return [
            OpenAITextAgent(0, PLAYER_NAMES[0], client, text_model),
            OpenAITextAgent(1, PLAYER_NAMES[1], client, text_model),
        ]
    if pair == "vlm-vlm":
        client = get_openai_client()
        return [
            OpenAIVisionAgent(0, PLAYER_NAMES[0], client, vision_model),
            OpenAIVisionAgent(1, PLAYER_NAMES[1], client, vision_model),
        ]
    if pair == "scripted-scripted":
        return [
            ScriptedAgent(0, PLAYER_NAMES[0], ["up", "left", "interact", "right", "interact", "stay"]),
            ScriptedAgent(1, PLAYER_NAMES[1], ["up", "right", "interact", "left", "interact", "stay"]),
        ]
    raise ValueError(f"Unsupported pair: {pair}")


def _decision_trace(agent) -> dict[str, Any]:
    if agent.last_decision is None:
        return {
            "playerId": agent.player_id,
            "playerName": agent.player_name,
            "goal": "low_level_action",
            "goalLabel": "stay",
            "previousGoal": "low_level_action",
            "previousGoalLabel": "stay",
            "changed": False,
            "action": "stay",
            "message": "",
            "plan": "",
            "rawResponse": "",
            "valid": False,
            "invalidReason": "agent did not produce a decision",
        }
    return agent.last_decision.to_trace()


def _held_name(player: dict[str, Any]) -> str:
    held = player.get("held_object")
    return held.get("name", "nothing") if held else "nothing"


def _player_snapshot(state, player_id: int) -> dict[str, Any]:
    player = state.to_dict()["players"][player_id]
    return {
        "position": tuple(player["position"]),
        "orientation": tuple(player["orientation"]),
        "held": _held_name(player),
    }


def _event_messages_for_player(events: list[dict[str, Any]], player_id: int) -> list[str]:
    return [event["message"] for event in events if event["playerId"] == player_id]


def _build_action_feedback(
    before_state,
    after_state,
    player_id: int,
    decision: dict[str, Any],
    events: list[dict[str, Any]],
    symbolic_action: str | None,
) -> str:
    before = _player_snapshot(before_state, player_id)
    after = _player_snapshot(after_state, player_id)
    changes = []
    for key in ["position", "orientation", "held"]:
        if before[key] != after[key]:
            changes.append(f"{key}: {before[key]} -> {after[key]}")

    player_events = _event_messages_for_player(events, player_id)
    parts = [f"previous action `{decision['action']}`"]
    if changes:
        parts.append("state changed (" + "; ".join(changes) + ")")
    if player_events:
        parts.append("events: " + ", ".join(player_events))
    if symbolic_action:
        parts.append(f"task progress: {symbolic_action}")
    if not changes and not player_events and not symbolic_action:
        parts.append("NO-OP: nothing changed and no station interaction succeeded")
    return "; ".join(parts)


def _is_no_op_feedback(feedback: str) -> bool:
    return "NO-OP" in feedback


def _build_no_op_warning(action: str | None, count: int) -> str:
    if not action or count < 2:
        return ""
    return (
        f"You have tried `{action}` {count} times in a row with no effect. "
        "Do not repeat it now; move or turn to face a useful station first."
    )


def run_agent_pair(
    pair: str,
    layout_name: str,
    task_id: str,
    max_ticks: int,
    backend: str = "openai",
    text_model: str = DEFAULT_OPENAI_MODEL,
    vision_model: str = DEFAULT_VISION_MODEL,
    local_model: str = "Qwen/Qwen2.5-7B-Instruct",
    dtype: str = "auto",
    device_map: str = "auto",
    max_new_tokens: int = 160,
    collect_trajectory: bool = False,
    trace_output_path: str | Path | None = None,
):
    task = get_task_by_id(task_id)
    if task.get("layout") and task["layout"] != layout_name:
        raise ValueError(f"Task {task_id} is for layout {task['layout']}, not {layout_name}")

    mdp = OvercookedGridworld.from_layout_name(layout_name)
    state = mdp.get_standard_start_state()
    agents = make_agent_pair(
        pair,
        backend=backend,
        text_model=text_model,
        vision_model=vision_model,
        local_model=local_model,
        dtype=dtype,
        device_map=device_map,
        max_new_tokens=max_new_tokens,
    )
    score = 0
    messages = ["", ""]
    executed_actions: list[str] = []
    agent_histories: dict[str, list[str]] = {str(player_id): [] for player_id in range(len(agents))}
    invalid_action_count = 0
    prompt_logs: list[dict[str, Any]] = []
    ticks_run = 0
    action_feedbacks = ["No previous action yet." for _ in agents]
    plans = ["" for _ in agents]
    repeated_no_op_actions: list[str | None] = [None for _ in agents]
    repeated_no_op_counts = [0 for _ in agents]

    trajectory = None
    if collect_trajectory:
        trajectory = {
            "meta": {
                "layout": layout_name,
                "pair": pair,
                "backend": backend if pair != "scripted-scripted" else "scripted",
                "openai_model": text_model,
                "vision_model": vision_model,
                "local_model": local_model if backend == "local" else "",
                "max_ticks": max_ticks,
                "task_id": task_id,
            },
            "layout": build_layout_snapshot(mdp),
            "frames": [build_frame(state, mdp, agents, tick=0, score=0)],
            "tick_events": [],
            "symbolic_actions": [],
            "agent_histories": agent_histories,
            "messages": [],
            "prompt_logs": prompt_logs,
        }

    for tick in range(max_ticks):
        ticks_run = tick + 1
        before_state = state
        feedbacks_seen = list(action_feedbacks)
        plans_seen = list(plans)
        phase_hint = task_phase_hint(state, mdp)
        no_op_warnings_seen = [
            _build_no_op_warning(repeated_no_op_actions[player_id], repeated_no_op_counts[player_id])
            for player_id in range(len(agents))
        ]
        observations = [
            AgentObservation(
                state=state,
                mdp=mdp,
                tick=tick,
                score=score,
                player_id=player_id,
                task=task,
                teammate_message=messages[1 - player_id],
                current_plan=plans_seen[player_id],
                phase_hint=phase_hint,
                action_feedback=feedbacks_seen[player_id],
                no_op_warning=no_op_warnings_seen[player_id],
            )
            for player_id in range(len(agents))
        ]
        actions = [agent.act(observations[index]) for index, agent in enumerate(agents)]
        decisions = [_decision_trace(agent) for agent in agents]
        invalid_action_count += sum(1 for decision in decisions if not decision.get("valid", True))
        messages = [decision.get("message", "") for decision in decisions]
        plans = [
            decision.get("plan", "").strip() or plans[player_id]
            for player_id, decision in enumerate(decisions)
        ]

        state, info = mdp.get_state_transition(state, actions)
        score_delta = int(sum(info["sparse_reward_by_agent"]))
        score += score_delta

        events = build_event_entries(info["event_infos"])
        symbolic_actions = []
        symbolic_actions_by_player: dict[int, str] = {}
        for player_id in range(len(agents)):
            symbolic_action = classify_player_action(
                before_state,
                state,
                player_id,
                info.get("event_infos", {}),
                actions[player_id],
            )
            if symbolic_action:
                symbolic_actions_by_player[player_id] = symbolic_action
                agent_key = str(player_id)
                agent_histories[agent_key].append(symbolic_action)
                executed_actions.append(symbolic_action)
                symbolic_actions.append(
                    {
                        "playerId": player_id,
                        "playerName": PLAYER_NAMES[player_id],
                        "action": symbolic_action,
                    }
                )

        action_feedbacks = [
            _build_action_feedback(
                before_state,
                state,
                player_id,
                decisions[player_id],
                events,
                symbolic_actions_by_player.get(player_id),
            )
            for player_id in range(len(agents))
        ]
        for player_id, feedback in enumerate(action_feedbacks):
            action = decisions[player_id]["action"]
            if _is_no_op_feedback(feedback):
                if repeated_no_op_actions[player_id] == action:
                    repeated_no_op_counts[player_id] += 1
                else:
                    repeated_no_op_actions[player_id] = action
                    repeated_no_op_counts[player_id] = 1
            else:
                repeated_no_op_actions[player_id] = None
                repeated_no_op_counts[player_id] = 0
        if collect_trajectory:
            for decision in decisions:
                player_id = decision["playerId"]
                prompt_logs.append(
                    {
                        "tick": tick,
                        "playerId": player_id,
                        "playerName": decision["playerName"],
                        "action": decision["action"],
                        "message": decision.get("message", ""),
                        "planSeen": plans_seen[player_id],
                        "planAfter": plans[player_id],
                        "phaseHint": phase_hint,
                        "feedbackSeen": feedbacks_seen[player_id],
                        "feedbackAfter": action_feedbacks[player_id],
                        "noOpWarningSeen": no_op_warnings_seen[player_id],
                        "prompt": decision.get("prompt", ""),
                        "rawResponse": decision.get("rawResponse", ""),
                        "valid": decision.get("valid", True),
                        "invalidReason": decision.get("invalidReason"),
                    }
                )
                if decision.get("message"):
                    trajectory["messages"].append(
                        {
                            "tick": tick,
                            "playerId": decision["playerId"],
                            "playerName": decision["playerName"],
                            "message": decision["message"],
                        }
                    )

            trajectory["tick_events"].append(
                {
                    "tick": tick + 1,
                    "actions": [serialize_action(action) for action in actions],
                    "decisions": decisions,
                    "goal_changes": [],
                    "events": events,
                    "symbolic_actions": symbolic_actions,
                    "scoreDelta": score_delta,
                    "scoreAfter": score,
                    "headline": build_headline(events, score_delta),
                }
            )
            trajectory["frames"].append(build_frame(state, mdp, agents, tick + 1, score))
            trajectory["symbolic_actions"].extend(
                {**symbolic_action, "tick": tick + 1} for symbolic_action in symbolic_actions
            )

        if score_delta > 0:
            break

    metrics = evaluate_task_trajectory(task, executed_actions, agent_histories=agent_histories)
    metrics["invalid_action_count"] = invalid_action_count
    metrics["success"] = score > 0
    metrics["ticks"] = ticks_run

    summary = {
        "pair": pair,
        "layout": layout_name,
        "task_id": task_id,
        "score": score,
        "soups_delivered": score // 20,
        "success": score > 0,
        "metrics": metrics,
        "agent_histories": agent_histories,
        "symbolic_actions": executed_actions,
    }

    if collect_trajectory:
        trajectory["score"] = score
        trajectory["soups_delivered"] = score // 20
        trajectory["metrics"] = metrics
        trajectory["summary"] = summary
        if trace_output_path:
            save_trajectory(trajectory, Path(trace_output_path))
        return summary, trajectory
    return summary
