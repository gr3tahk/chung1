from collections import Counter, deque
import argparse
import json
import os
from pathlib import Path

import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

from metrics import score_against_references

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
DEFAULT_TASKS_FILE = Path(__file__).with_name("benchmark_tasks.json")
DEFAULT_LAYOUT_RESULTS_FILE = "llm_results.json"
DEFAULT_TASK_RESULTS_FILE = "task_results.json"

BACKEND = "openai"
OPENAI_MODEL = DEFAULT_OPENAI_MODEL
LOCAL_MODEL = DEFAULT_LOCAL_MODEL
client = None
tokenizer = None
model = None


def configure_llm_backend(backend, openai_model=DEFAULT_OPENAI_MODEL, local_model=DEFAULT_LOCAL_MODEL):
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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading local model: {LOCAL_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_MODEL,
            torch_dtype="auto",
            device_map="auto",
        )
    return tokenizer, model


def query_openai(prompt):
    response = get_openai_client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def query_local_model(prompt):
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
    generated = outputs[0][inputs["input_ids"].shape[1]:]
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
            n = len(soup.ingredients)
            if soup.is_ready:
                any_ready = True
                ready_locs.append(pot_pos)
            elif soup.is_cooking:
                pass
            elif n >= 3:
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
        return [min(onion_locs, key=lambda x: x[0])]
    return [max(onion_locs, key=lambda x: x[0])]


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
    player_name = "Alice" if player_id == 0 else "Bob"
    other_name = "Bob" if player_id == 0 else "Alice"

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
    except Exception as e:
        print(f"  LLM error ({BACKEND}): {e}")
        return "get_plate"


def goal_to_action(goal, state, mdp, player_id):
    plate_locs = mdp.get_dish_dispenser_locations()
    delivery_locs = mdp.get_serving_locations()
    pot_locs = list(mdp.get_pot_locations())
    _, needs_onions, needs_cooking, ready_locs = get_pot_info(state, mdp)

    if goal == "get_onion":
        assigned = get_assigned_onion(mdp, player_id)
        return bfs_next_action(mdp, state, player_id, assigned)
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


class LLMAgent:
    def __init__(self, player_id):
        self.player_id = player_id
        self.current_goal = "get_onion"

    def act(self, state, mdp):
        llm_fn = lambda: get_llm_goal(state, mdp, self.player_id)
        goal = compute_goal(state, mdp, self.player_id, llm_fn)
        if goal != self.current_goal:
            print(f"    P{self.player_id}: {self.current_goal} -> {goal}")
            self.current_goal = goal
        return goal_to_action(self.current_goal, state, mdp, self.player_id)


def normalize_object_name(name):
    if name is None:
        return "nothing"
    if name == "soup in plate":
        return "soup"
    return name


def object_name(obj):
    if obj is None:
        return "nothing"
    return normalize_object_name(getattr(obj, "name", None))


def get_counter_locations(mdp):
    getter = getattr(mdp, "get_counter_locations", None)
    if getter is None:
        return []
    return list(getter())


def snapshot_counter_objects(state, mdp):
    snapshot = {}
    for pos in get_counter_locations(mdp):
        if state.has_object(pos):
            snapshot[pos] = object_name(state.get_object(pos))
    return snapshot


def snapshot_pots(state, mdp):
    snapshot = {}
    for pos in mdp.get_pot_locations():
        if state.has_object(pos):
            soup = state.get_object(pos)
            snapshot[pos] = {
                "ingredients": len(soup.ingredients),
                "is_cooking": bool(soup.is_cooking),
                "is_ready": bool(soup.is_ready),
            }
        else:
            snapshot[pos] = {
                "ingredients": 0,
                "is_cooking": False,
                "is_ready": False,
            }
    return snapshot


def make_state_snapshot(state, mdp):
    state_dict = state.to_dict()
    players = []
    for player in state_dict["players"]:
        held = player["held_object"]
        players.append(
            {
                "position": tuple(player["position"]),
                "orientation": tuple(player["orientation"]),
                "held_object": normalize_object_name(held["name"]) if held else "nothing",
            }
        )
    return {
        "players": players,
        "counters": snapshot_counter_objects(state, mdp),
        "pots": snapshot_pots(state, mdp),
    }


def expand_counter_delta(before_counters, after_counters):
    before_counts = Counter(before_counters.values())
    after_counts = Counter(after_counters.values())
    removed = before_counts - after_counts
    added = after_counts - before_counts
    return removed, added


def started_cooking_pots(before_snapshot, after_snapshot):
    started = []
    for pos, after in after_snapshot["pots"].items():
        before = before_snapshot["pots"].get(
            pos,
            {"ingredients": 0, "is_cooking": False, "is_ready": False},
        )
        if before["ingredients"] >= 3 and not before["is_cooking"] and after["is_cooking"]:
            started.append(pos)
    return started


def onion_placed(before_snapshot, after_snapshot):
    for pos, after in after_snapshot["pots"].items():
        before = before_snapshot["pots"].get(
            pos,
            {"ingredients": 0, "is_cooking": False, "is_ready": False},
        )
        if after["ingredients"] == before["ingredients"] + 1:
            return True
    return False


def encode_symbolic_action(player_id, action_name):
    return f"p{player_id}:{action_name}"


def classify_player_action(
    player_id,
    before_snapshot,
    after_snapshot,
    low_level_action,
    score_delta,
    removed_counters,
    added_counters,
):
    before_player = before_snapshot["players"][player_id]
    after_player = after_snapshot["players"][player_id]
    before_held = before_player["held_object"]
    after_held = after_player["held_object"]

    if before_held == "dish" and after_held == "soup":
        return "pickup_soup"
    if before_held == "nothing" and after_held == "onion":
        if removed_counters["onion"] > 0:
            return "pickup_from_counter"
        return "pickup_onion"
    if before_held == "nothing" and after_held == "dish":
        if removed_counters["dish"] > 0:
            return "pickup_from_counter"
        return "pickup_dish"
    if before_held == "nothing" and after_held == "soup":
        if removed_counters["soup"] > 0:
            return "pickup_from_counter"
        return "pickup_soup"
    if before_held == "onion" and after_held == "nothing":
        if onion_placed(before_snapshot, after_snapshot):
            return "place_onion_in_pot"
        if added_counters["onion"] > 0:
            return "place_on_counter"
    if before_held in {"dish", "soup"} and after_held == "nothing":
        if before_held == "soup" and score_delta > 0:
            return "deliver_soup"
        if added_counters[before_held] > 0:
            return "place_on_counter"

    if low_level_action == Action.INTERACT:
        for pot_pos in started_cooking_pots(before_snapshot, after_snapshot):
            if is_adjacent_to(before_player["position"], pot_pos):
                return "start_cooking"

    if low_level_action == Action.STAY or before_held == after_held:
        return "wait"
    return "wait"


class TrajectoryLogger:
    def __init__(self):
        self.tick_events = []
        self.executed_actions = []

    def record(self, tick, before_snapshot, after_snapshot, actions, goals, score_delta):
        removed_counters, added_counters = expand_counter_delta(
            before_snapshot["counters"], after_snapshot["counters"]
        )
        tick_event = {"tick": tick, "players": []}
        for player_id, low_level_action in enumerate(actions):
            action_name = classify_player_action(
                player_id,
                before_snapshot,
                after_snapshot,
                low_level_action,
                score_delta,
                removed_counters,
                added_counters,
            )
            event = {
                "player_id": player_id,
                "goal": goals[player_id],
                "symbolic_action": action_name,
                "held_before": before_snapshot["players"][player_id]["held_object"],
                "held_after": after_snapshot["players"][player_id]["held_object"],
                "position_before": list(before_snapshot["players"][player_id]["position"]),
                "position_after": list(after_snapshot["players"][player_id]["position"]),
            }
            tick_event["players"].append(event)
            if action_name != "wait":
                self.executed_actions.append(encode_symbolic_action(player_id, action_name))
        self.tick_events.append(tick_event)


def load_tasks(tasks_file=DEFAULT_TASKS_FILE):
    with open(tasks_file) as f:
        payload = json.load(f)
    return payload["tasks"]


def get_task_by_id(task_id, tasks_file=DEFAULT_TASKS_FILE):
    for task in load_tasks(tasks_file):
        if task["task_id"] == task_id:
            return task
    raise ValueError(f"Unknown task_id: {task_id}")


def create_initial_state(task, mdp):
    initial_state = task.get("initial_state", "standard")
    if initial_state != "standard":
        raise ValueError(f"Unsupported initial_state: {initial_state}")
    return mdp.get_standard_start_state()


def run_game(layout_name, max_ticks=NUM_TICKS, stop_on_score=None, collect_trajectory=False):
    mdp = OvercookedGridworld.from_layout_name(layout_name)
    state = mdp.get_standard_start_state()
    agents = [LLMAgent(0), LLMAgent(1)]
    score = 0
    logger = TrajectoryLogger() if collect_trajectory else None

    for tick in range(max_ticks):
        before_snapshot = make_state_snapshot(state, mdp) if logger else None
        a0 = agents[0].act(state, mdp)
        a1 = agents[1].act(state, mdp)
        goals = [agents[0].current_goal, agents[1].current_goal]
        state, info = mdp.get_state_transition(state, [a0, a1])
        reward_delta = sum(info["sparse_reward_by_agent"])
        score += reward_delta
        if logger:
            after_snapshot = make_state_snapshot(state, mdp)
            logger.record(tick, before_snapshot, after_snapshot, [a0, a1], goals, reward_delta)
        if tick % 50 == 0:
            print(f"    tick {tick:03d} | score {score}")
        if stop_on_score is not None and score >= stop_on_score:
            break

    result = {"score": score}
    if logger:
        result["trajectory"] = logger.executed_actions
        result["tick_events"] = logger.tick_events
    return result


def run_benchmark(layout_name):
    scores = []
    for trial in range(NUM_TRIALS):
        print(f"  trial {trial + 1}/{NUM_TRIALS}")
        result = run_game(layout_name)
        score = result["score"]
        scores.append(score)
        print(f"  -> score: {score}")
    mean = float(np.mean(scores))
    stderr = float(np.std(scores) / np.sqrt(NUM_TRIALS))
    return mean, stderr, scores


def evaluate_task_trajectory(task, executed_actions):
    return score_against_references(executed_actions, task["reference_trajectories"])


def run_task(task):
    mdp = OvercookedGridworld.from_layout_name(task["layout_name"])
    create_initial_state(task, mdp)
    run_result = run_game(
        task["layout_name"],
        max_ticks=task.get("max_ticks", NUM_TICKS),
        stop_on_score=task.get("target_score"),
        collect_trajectory=True,
    )
    score_result = evaluate_task_trajectory(task, run_result["trajectory"])
    success = run_result["score"] >= task.get("target_score", 0)
    return {
        "task_id": task["task_id"],
        "layout_name": task["layout_name"],
        "score": run_result["score"],
        "success": success,
        "target_score": task.get("target_score", 0),
        "executed_trajectory": run_result["trajectory"],
        "tick_events": run_result["tick_events"],
        "best_reference_id": score_result["best_reference_id"],
        "tes": score_result["tes"],
        "ites": score_result["ites"],
    }


def run_task_benchmark(tasks):
    all_results = {}
    for task in tasks:
        print(f"\n=== {task['task_id']} ===")
        trials = []
        for trial in range(NUM_TRIALS):
            print(f"  trial {trial + 1}/{NUM_TRIALS}")
            result = run_task(task)
            trials.append(result)
            print(
                f"  -> score: {result['score']} | success: {result['success']} | "
                f"TES: {result['tes']:.3f} | ITES: {result['ites']:.3f}"
            )
        all_results[task["task_id"]] = {
            "layout_name": task["layout_name"],
            "mean_score": float(np.mean([trial["score"] for trial in trials])),
            "mean_tes": float(np.mean([trial["tes"] for trial in trials])),
            "mean_ites": float(np.mean([trial["ites"] for trial in trials])),
            "success_rate": float(np.mean([1.0 if trial["success"] else 0.0 for trial in trials])),
            "trials": trials,
        }
    return all_results


def default_output_path(mode):
    if mode == "tasks":
        return DEFAULT_TASK_RESULTS_FILE
    return DEFAULT_LAYOUT_RESULTS_FILE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["openai", "local"],
        default="openai",
        help="LLM backend to use",
    )
    parser.add_argument(
        "--mode",
        choices=["layouts", "tasks"],
        default="layouts",
        help="Benchmark mode to run",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help="OpenAI model name for --backend openai",
    )
    parser.add_argument(
        "--local-model",
        default=DEFAULT_LOCAL_MODEL,
        help="Transformers model name for --backend local",
    )
    parser.add_argument(
        "--tasks-file",
        default=str(DEFAULT_TASKS_FILE),
        help="Task spec JSON file for --mode tasks",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Optional single task_id to run in --mode tasks",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path",
    )
    args = parser.parse_args()

    configure_llm_backend(args.backend, args.openai_model, args.local_model)
    output_path = args.output or default_output_path(args.mode)

    if args.mode == "tasks":
        tasks = load_tasks(args.tasks_file)
        if args.task_id:
            tasks = [task for task in tasks if task["task_id"] == args.task_id]
            if not tasks:
                raise ValueError(f"task_id not found in {args.tasks_file}: {args.task_id}")
        results = run_task_benchmark(tasks)
    else:
        results = {}
        for layout in LAYOUTS:
            print(f"\n=== {layout} ===")
            mean, stderr, scores = run_benchmark(layout)
            results[layout] = {"mean": mean, "stderr": stderr, "scores": scores}
            print(f"  RESULT: {mean:.1f} +/- {stderr:.1f}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
