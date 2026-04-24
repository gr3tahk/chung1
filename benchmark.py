from __future__ import annotations

from collections import deque
import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

LAYOUTS = [
    "cramped_room",
    "asymmetric_advantages",
    "coordination_ring",
    "forced_coordination",
    "counter_circuit_o_1order",
]
NUM_TICKS = 400
NUM_TRIALS = 3
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_LOCAL_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TRACE_OUTPUT = Path("replay-ui/public/traces/latest.json")

BACKEND = "openai"
OPENAI_MODEL = DEFAULT_OPENAI_MODEL
LOCAL_MODEL = DEFAULT_LOCAL_MODEL
client = None
tokenizer = None
model = None

PLAYER_NAMES = ["Alice", "Bob"]
GOAL_LABELS = {
    "get_onion": "Grab onion",
    "place_onion": "Fill pot",
    "start_cooking": "Start pot",
    "get_plate": "Fetch plate",
    "load_soup": "Plate soup",
    "deliver_soup": "Serve dish",
    "wait": "Wait",
}
ACTION_LABELS = {
    (0, -1): "up",
    (0, 1): "down",
    (1, 0): "right",
    (-1, 0): "left",
    Action.STAY: "stay",
    Action.INTERACT: "interact",
}
EVENT_MESSAGES = {
    "onion_pickup": "picked up an onion",
    "onion_drop": "set down an onion",
    "potting_onion": "added an onion to the pot",
    "dish_pickup": "grabbed a plate",
    "dish_drop": "set down a plate",
    "soup_pickup": "picked up soup",
    "soup_delivery": "served a soup",
    "soup_drop": "set down soup",
    "tomato_pickup": "picked up a tomato",
    "tomato_drop": "set down a tomato",
    "potting_tomato": "added a tomato to the pot",
}
TERRAIN_THEME = {
    " ": "path",
    "X": "counter",
    "O": "onion",
    "D": "dish",
    "P": "pot",
    "S": "serve",
}


def configure_llm_backend(
    backend: str,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    local_model: str = DEFAULT_LOCAL_MODEL,
):
    global BACKEND, OPENAI_MODEL, LOCAL_MODEL
    BACKEND = backend
    OPENAI_MODEL = openai_model
    LOCAL_MODEL = local_model


def get_openai_client():
    global client
    if client is None:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return client


def get_local_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading local model: {LOCAL_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL,
            torch_dtype="auto",
            device_map="auto",
        )
    return tokenizer, model


def query_openai(prompt: str) -> str:
    response = get_openai_client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def query_local_model(prompt: str) -> str:
    tokenizer, model = get_local_model()
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def is_adjacent_to(pos_a, pos_b):
    return abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1]) == 1


def direction_toward(my_pos, target_pos):
    dx = target_pos[0] - my_pos[0]
    dy = target_pos[1] - my_pos[1]
    if abs(dx) >= abs(dy):
        return (1 if dx > 0 else -1, 0)
    return (0, 1 if dy > 0 else -1)


def bfs_next_action(mdp, state, player_id, goal_locs):
    state_dict = state.to_dict()
    my_pos = state_dict["players"][player_id]["position"]
    other_id = 1 - player_id
    other_pos = state_dict["players"][other_id]["position"]
    my_orient = tuple(state_dict["players"][player_id]["orientation"])

    for gloc in goal_locs:
        if is_adjacent_to(my_pos, gloc):
            needed_dir = direction_toward(my_pos, gloc)
            if my_orient == needed_dir:
                return Action.INTERACT
            return needed_dir

    queue = deque([(my_pos, None)])
    visited = {my_pos}
    dirs = [(0, -1), (0, 1), (1, 0), (-1, 0)]
    while queue:
        pos, first_step = queue.popleft()
        for dc, dr in dirs:
            npos = (pos[0] + dc, pos[1] + dr)
            cx, cy = npos
            if cx < 0 or cy < 0 or cx >= mdp.width or cy >= mdp.height:
                continue
            if npos in visited:
                continue
            step = first_step if first_step is not None else (dc, dr)
            if npos in goal_locs:
                return step
            terrain = mdp.get_terrain_type_at_pos(npos)
            if terrain == " " and npos != other_pos:
                visited.add(npos)
                queue.append((npos, step))

    queue = deque([(my_pos, None)])
    visited = {my_pos}
    while queue:
        pos, first_step = queue.popleft()
        for dc, dr in dirs:
            npos = (pos[0] + dc, pos[1] + dr)
            cx, cy = npos
            if cx < 0 or cy < 0 or cx >= mdp.width or cy >= mdp.height:
                continue
            if npos in visited:
                continue
            step = first_step if first_step is not None else (dc, dr)
            if npos in goal_locs:
                return step
            terrain = mdp.get_terrain_type_at_pos(npos)
            if terrain == " ":
                visited.add(npos)
                queue.append((npos, step))
    return Action.STAY


def get_pot_info(state, mdp):
    pot_locs = mdp.get_pot_locations()
    any_ready = False
    needs_onions = []
    needs_cooking = []
    ready_locs = []
    for pot_pos in pot_locs:
        if state.has_object(pot_pos):
            soup = state.get_object(pot_pos)
            ingredient_count = len(soup.ingredients)
            if soup.is_ready:
                any_ready = True
                ready_locs.append(pot_pos)
            elif soup.is_cooking:
                pass
            elif ingredient_count >= 3:
                needs_cooking.append(pot_pos)
            else:
                needs_onions.append(pot_pos)
        else:
            needs_onions.append(pot_pos)
    return any_ready, needs_onions, needs_cooking, ready_locs


def get_assigned_onion(mdp, player_id):
    onion_locs = mdp.get_onion_dispenser_locations()
    if len(onion_locs) == 1:
        return onion_locs
    if player_id == 0:
        return [min(onion_locs, key=lambda pos: pos[0])]
    return [max(onion_locs, key=lambda pos: pos[0])]


def compute_goal(state, mdp, player_id, llm_call_fn):
    state_dict = state.to_dict()
    held = state_dict["players"][player_id]["held_object"]
    held_name = held["name"] if held else "nothing"
    any_ready, needs_onions, needs_cooking, _ = get_pot_info(state, mdp)

    if held_name == "onion":
        return "place_onion" if needs_onions else "wait"
    if held_name == "dish":
        return "load_soup" if any_ready else "wait"
    if held_name in ["soup", "soup in plate"]:
        return "deliver_soup"

    if needs_cooking:
        return "start_cooking"
    if any_ready:
        return llm_call_fn()
    if needs_onions:
        return "get_onion"
    return "wait"


def get_llm_goal(state, mdp, player_id):
    state_dict = state.to_dict()
    other_id = 1 - player_id
    other_held = state_dict["players"][other_id]["held_object"]
    other_held_str = other_held["name"] if other_held else "nothing"
    player_name = PLAYER_NAMES[player_id]
    other_name = PLAYER_NAMES[other_id]

    prompt = f"""You are {player_name} in Overcooked. You are holding nothing.
{other_name} is holding: {other_held_str}
A pot of soup is READY TO PLATE.

Should you get a plate to collect the soup, or get another onion to prepare a new soup?
Avoid doing the same thing as {other_name}.

Reply with one of: get_plate, get_onion"""

    try:
        if BACKEND == "local":
            reply = query_local_model(prompt).lower()
        else:
            reply = query_openai(prompt).lower()
        return "get_plate" if "plate" in reply else "get_onion"
    except Exception as exc:
        print(f"  LLM error ({BACKEND}): {exc}")
        return "get_plate"


def goal_to_action(goal, state, mdp, player_id):
    plate_locs = mdp.get_dish_dispenser_locations()
    delivery_locs = mdp.get_serving_locations()
    pot_locs = list(mdp.get_pot_locations())
    _, needs_onions, needs_cooking, ready_locs = get_pot_info(state, mdp)

    if goal == "get_onion":
        return bfs_next_action(mdp, state, player_id, get_assigned_onion(mdp, player_id))
    if goal == "place_onion":
        targets = needs_onions if needs_onions else pot_locs
        return bfs_next_action(mdp, state, player_id, targets)
    if goal == "start_cooking":
        targets = needs_cooking if needs_cooking else pot_locs
        return bfs_next_action(mdp, state, player_id, targets)
    if goal == "get_plate":
        return bfs_next_action(mdp, state, player_id, plate_locs)
    if goal == "load_soup":
        targets = ready_locs if ready_locs else pot_locs
        return bfs_next_action(mdp, state, player_id, targets)
    if goal == "deliver_soup":
        return bfs_next_action(mdp, state, player_id, delivery_locs)
    return Action.STAY


def serialize_position(pos) -> list[int]:
    return [int(pos[0]), int(pos[1])]


def serialize_object(obj: dict[str, Any] | None):
    if not obj:
        return None
    payload = {"name": obj["name"], "position": serialize_position(obj["position"])}
    if "ingredients" in obj:
        payload["ingredients"] = list(obj["ingredients"])
    if "is_cooking" in obj:
        payload["is_cooking"] = bool(obj["is_cooking"])
    if "is_ready" in obj:
        payload["is_ready"] = bool(obj["is_ready"])
    return payload


def serialize_action(action) -> str:
    return ACTION_LABELS.get(action, str(action))


def build_layout_snapshot(mdp):
    terrain = []
    for y in range(mdp.height):
        row = []
        for x in range(mdp.width):
            tile = mdp.get_terrain_type_at_pos((x, y))
            row.append(
                {
                    "terrain": tile,
                    "theme": TERRAIN_THEME.get(tile, "counter"),
                    "position": [x, y],
                }
            )
        terrain.append(row)

    stations = []
    for station_type, positions in [
        ("pot", mdp.get_pot_locations()),
        ("onion", mdp.get_onion_dispenser_locations()),
        ("dish", mdp.get_dish_dispenser_locations()),
        ("serve", mdp.get_serving_locations()),
    ]:
        for pos in positions:
            stations.append({"type": station_type, "position": serialize_position(pos)})

    return {"width": mdp.width, "height": mdp.height, "terrain": terrain, "stations": stations}


def build_pot_snapshots(state, mdp):
    pots = []
    for pot_pos in mdp.get_pot_locations():
        pot = {"position": serialize_position(pot_pos), "stage": "empty", "ingredients": [], "ingredient_count": 0}
        if state.has_object(pot_pos):
            soup = state.get_object(pot_pos)
            ingredient_names = [getattr(ingredient, "name", ingredient) for ingredient in soup.ingredients]
            pot["ingredients"] = ingredient_names
            pot["ingredient_count"] = len(ingredient_names)
            if soup.is_ready:
                pot["stage"] = "ready"
            elif soup.is_cooking:
                pot["stage"] = "cooking"
            elif len(ingredient_names) >= 3:
                pot["stage"] = "full"
            else:
                pot["stage"] = "filling"
        pots.append(pot)
    return pots


def build_counter_objects(state):
    return [serialize_object(obj) for obj in state.to_dict()["objects"]]


def build_frame(state, mdp, agents, tick: int, score: int):
    state_dict = state.to_dict()
    players = []
    for player_id, player in enumerate(state_dict["players"]):
        agent = agents[player_id]
        players.append(
            {
                "id": player_id,
                "name": PLAYER_NAMES[player_id],
                "position": serialize_position(player["position"]),
                "orientation": serialize_position(player["orientation"]),
                "held_object": serialize_object(player["held_object"]),
                "goal": agent.current_goal,
                "goal_label": GOAL_LABELS.get(agent.current_goal, agent.current_goal),
            }
        )

    return {
        "tick": tick,
        "score": score,
        "players": players,
        "pots": build_pot_snapshots(state, mdp),
        "counter_objects": build_counter_objects(state),
    }


def build_event_entries(event_infos):
    events = []
    for event_name, flags in event_infos.items():
        for player_id, happened in enumerate(flags):
            if happened:
                events.append(
                    {
                        "type": event_name,
                        "playerId": player_id,
                        "playerName": PLAYER_NAMES[player_id],
                        "message": EVENT_MESSAGES.get(event_name, event_name.replace("_", " ")),
                    }
                )
    return events


def build_headline(goal_changes, events, score_delta):
    if score_delta > 0:
        return f"Soup served! Team score +{score_delta}"
    if events:
        first = events[0]
        return f"{first['playerName']} {first['message']}"
    if goal_changes:
        first = goal_changes[0]
        return f"{first['playerName']} switched to {GOAL_LABELS.get(first['goal'], first['goal'])}"
    return "The team keeps moving through the kitchen."


class LLMAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.current_goal = "get_onion"
        self.last_decision = None

    def act(self, state, mdp):
        llm_fn = lambda: get_llm_goal(state, mdp, self.player_id)
        previous_goal = self.current_goal
        goal = compute_goal(state, mdp, self.player_id, llm_fn)
        changed = goal != previous_goal
        if changed:
            print(f"    P{self.player_id}: {previous_goal} -> {goal}")
            self.current_goal = goal
        action = goal_to_action(self.current_goal, state, mdp, self.player_id)
        self.last_decision = {
            "playerId": self.player_id,
            "playerName": PLAYER_NAMES[self.player_id],
            "goal": self.current_goal,
            "goalLabel": GOAL_LABELS.get(self.current_goal, self.current_goal),
            "previousGoal": previous_goal,
            "previousGoalLabel": GOAL_LABELS.get(previous_goal, previous_goal),
            "changed": changed,
            "action": serialize_action(action),
        }
        return action


def save_trajectory(trajectory, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(trajectory, handle, indent=2)


def run_game(
    layout_name,
    max_ticks: int = NUM_TICKS,
    collect_trajectory: bool = False,
    trace_output_path: str | Path | None = None,
):
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    state = mdp.get_standard_start_state()
    agents = [LLMAgent(0), LLMAgent(1)]
    score = 0
    trajectory = None

    if collect_trajectory:
        trajectory = {
            "meta": {
                "layout": layout_name,
                "backend": BACKEND,
                "openai_model": OPENAI_MODEL,
                "local_model": LOCAL_MODEL,
                "max_ticks": max_ticks,
            },
            "layout": build_layout_snapshot(mdp),
            "frames": [build_frame(state, mdp, agents, tick=0, score=0)],
            "tick_events": [],
        }

    for tick in range(max_ticks):
        actions = [agents[0].act(state, mdp), agents[1].act(state, mdp)]
        state, info = mdp.get_state_transition(state, actions)
        score_delta = int(sum(info["sparse_reward_by_agent"]))
        score += score_delta

        if collect_trajectory:
            decisions = [agent.last_decision for agent in agents]
            goal_changes = [decision for decision in decisions if decision["changed"]]
            events = build_event_entries(info["event_infos"])
            trajectory["tick_events"].append(
                {
                    "tick": tick + 1,
                    "actions": [serialize_action(action) for action in actions],
                    "decisions": decisions,
                    "goal_changes": goal_changes,
                    "events": events,
                    "scoreDelta": score_delta,
                    "scoreAfter": score,
                    "headline": build_headline(goal_changes, events, score_delta),
                }
            )
            trajectory["frames"].append(build_frame(state, mdp, agents, tick + 1, score))

        if tick % 50 == 0:
            print(f"    tick {tick:03d} | score {score}")

    if collect_trajectory:
        trajectory["score"] = score
        trajectory["soups_delivered"] = score // 20
        if trace_output_path:
            save_trajectory(trajectory, Path(trace_output_path))
        return score, trajectory
    return score


def run_benchmark(layout_name):
    scores = []
    for trial in range(NUM_TRIALS):
        print(f"  trial {trial + 1}/{NUM_TRIALS}")
        score = run_game(layout_name)
        scores.append(score)
        print(f"  -> score: {score}")
    mean = float(np.mean(scores))
    stderr = float(np.std(scores) / np.sqrt(NUM_TRIALS))
    return mean, stderr, scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "local"], default="openai", help="LLM backend to use")
    parser.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL, help="OpenAI model name for --backend openai")
    parser.add_argument("--local-model", default=DEFAULT_LOCAL_MODEL, help="Transformers model name for --backend local")
    parser.add_argument("--layout", choices=LAYOUTS, help="Run a single layout instead of the full benchmark suite")
    parser.add_argument("--max-ticks", type=int, default=NUM_TICKS, help="Number of game ticks to simulate")
    parser.add_argument("--collect-trajectory", action="store_true", help="Collect a detailed replay trajectory for the UI")
    parser.add_argument("--trace-output", default=str(DEFAULT_TRACE_OUTPUT), help="Where to save replay JSON when --collect-trajectory is enabled")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_llm_backend(args.backend, args.openai_model, args.local_model)

    if args.layout:
        result = run_game(
            args.layout,
            max_ticks=args.max_ticks,
            collect_trajectory=args.collect_trajectory,
            trace_output_path=args.trace_output if args.collect_trajectory else None,
        )
        if args.collect_trajectory:
            score, _ = result
            print(f"\nSaved replay trace to {args.trace_output}")
        else:
            score = result
        print(f"Final score for {args.layout}: {score}")
    else:
        all_results = {}
        for layout in LAYOUTS:
            print(f"\n=== {layout} ===")
            mean, stderr, scores = run_benchmark(layout)
            all_results[layout] = {"mean": mean, "stderr": stderr, "scores": scores}
            print(f"  RESULT: {mean:.1f} +/- {stderr:.1f}")
        with open("llm_results.json", "w") as handle:
            json.dump(all_results, handle, indent=2)
        print("\nSaved to llm_results.json")
