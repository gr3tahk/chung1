from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action
from openai import OpenAI
from collections import deque
import numpy as np
import os, json

LAYOUTS = [
    'cramped_room',
    'asymmetric_advantages',
    'coordination_ring',
    'forced_coordination',
    'counter_circuit_o_1order'
]
NUM_TICKS = 400
NUM_TRIALS = 3
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def is_adjacent_to(pos_a, pos_b):
    return abs(pos_a[0]-pos_b[0]) + abs(pos_a[1]-pos_b[1]) == 1

def direction_toward(my_pos, target_pos):
    dx = target_pos[0] - my_pos[0]
    dy = target_pos[1] - my_pos[1]
    if abs(dx) >= abs(dy):
        return (1 if dx > 0 else -1, 0)
    else:
        return (0, 1 if dy > 0 else -1)

def bfs_next_action(mdp, state, player_id, goal_locs):
    state_dict = state.to_dict()
    my_pos    = state_dict['players'][player_id]['position']
    other_id  = 1 - player_id
    other_pos = state_dict['players'][other_id]['position']
    my_orient = tuple(state_dict['players'][player_id]['orientation'])

    for gloc in goal_locs:
        if is_adjacent_to(my_pos, gloc):
            needed_dir = direction_toward(my_pos, gloc)
            if my_orient == needed_dir:
                return Action.INTERACT
            else:
                return needed_dir

    queue = deque([(my_pos, None)])
    visited = {my_pos}
    dirs = [(0,-1),(0,1),(1,0),(-1,0)]
    while queue:
        pos, first_step = queue.popleft()
        for dc, dr in dirs:
            npos = (pos[0]+dc, pos[1]+dr)
            cx, cy = npos
            if cx < 0 or cy < 0 or cx >= mdp.width or cy >= mdp.height:
                continue
            if npos in visited:
                continue
            step = first_step if first_step is not None else (dc, dr)
            if npos in goal_locs:
                return step
            terrain = mdp.get_terrain_type_at_pos(npos)
            if terrain == ' ' and npos != other_pos:
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

def compute_goal(state, mdp, player_id, current_goal, llm_call_fn):
    """
    Determine goal with inventory-based persistence:
    - If holding onion, must place it (don't replan)
    - If holding dish and soup ready, must load it
    - If holding soup, must deliver it
    - Otherwise replan
    """
    state_dict = state.to_dict()
    held = state_dict['players'][player_id]['held_object']
    held_name = held['name'] if held else 'nothing'
    any_ready, needs_onions, needs_cooking, ready_locs = get_pot_info(state, mdp)

    # Hard inventory-based rules — never deviate from these
    if held_name == 'onion':
        return 'place_onion' if needs_onions else 'wait'
    if held_name == 'dish':
        return 'load_soup' if any_ready else 'wait'
    if held_name in ['soup', 'soup in plate']:
        return 'deliver_soup'

    # Holding nothing — decide what to do next
    if needs_cooking:
        return 'start_cooking'
    if any_ready:
        return llm_call_fn()  # Only LLM decision point
    if needs_onions:
        return 'get_onion'
    return 'wait'

def get_llm_goal(state, mdp, player_id):
    state_dict = state.to_dict()
    other_id = 1 - player_id
    other_held = state_dict['players'][other_id]['held_object']
    other_held_str = other_held['name'] if other_held else 'nothing'
    player_name = "Alice" if player_id == 0 else "Bob"
    other_name  = "Bob"   if player_id == 0 else "Alice"

    prompt = f"""You are {player_name} in Overcooked. You are holding nothing.
{other_name} is holding: {other_held_str}
A pot of soup is READY TO PLATE.

Should you get a plate to collect the soup, or get another onion to prepare a new soup?
Avoid doing the same thing as {other_name}.

Reply with one of: get_plate, get_onion"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0
        )
        reply = response.choices[0].message.content.strip().lower()
        return 'get_plate' if 'plate' in reply else 'get_onion'
    except Exception as e:
        print(f"  LLM error: {e}")
        return 'get_plate'

def goal_to_action(goal, state, mdp, player_id):
    onion_locs    = mdp.get_onion_dispenser_locations()
    plate_locs    = mdp.get_dish_dispenser_locations()
    delivery_locs = mdp.get_serving_locations()
    pot_locs      = list(mdp.get_pot_locations())
    _, needs_onions, needs_cooking, ready_locs = get_pot_info(state, mdp)

    if goal == 'get_onion':
        return bfs_next_action(mdp, state, player_id, onion_locs)
    elif goal == 'place_onion':
        targets = needs_onions if needs_onions else pot_locs
        return bfs_next_action(mdp, state, player_id, targets)
    elif goal == 'start_cooking':
        targets = needs_cooking if needs_cooking else pot_locs
        return bfs_next_action(mdp, state, player_id, targets)
    elif goal == 'get_plate':
        return bfs_next_action(mdp, state, player_id, plate_locs)
    elif goal == 'load_soup':
        targets = ready_locs if ready_locs else pot_locs
        return bfs_next_action(mdp, state, player_id, targets)
    elif goal == 'deliver_soup':
        return bfs_next_action(mdp, state, player_id, delivery_locs)
    return Action.STAY

class LLMAgent:
    def __init__(self, player_id):
        self.player_id    = player_id
        self.current_goal = 'get_onion'

    def act(self, state, mdp):
        llm_fn = lambda: get_llm_goal(state, mdp, self.player_id)
        goal = compute_goal(state, mdp, self.player_id, self.current_goal, llm_fn)
        if goal != self.current_goal:
            print(f"    P{self.player_id}: {self.current_goal} -> {goal}")
            self.current_goal = goal
        return goal_to_action(self.current_goal, state, mdp, self.player_id)

def run_game(layout_name):
    mdp    = OvercookedGridworld.from_layout_name(layout_name)
    state  = mdp.get_standard_start_state()
    agents = [LLMAgent(0), LLMAgent(1)]
    score  = 0
    for tick in range(NUM_TICKS):
        a0 = agents[0].act(state, mdp)
        a1 = agents[1].act(state, mdp)
        state, info = mdp.get_state_transition(state, [a0, a1])
        score += sum(info['sparse_reward_by_agent'])
        if tick % 50 == 0:
            print(f"    tick {tick:03d} | score {score}")
    return score

def run_benchmark(layout_name):
    scores = []
    for trial in range(NUM_TRIALS):
        print(f"  trial {trial+1}/{NUM_TRIALS}")
        s = run_game(layout_name)
        scores.append(s)
        print(f"  -> score: {s}")
    mean   = float(np.mean(scores))
    stderr = float(np.std(scores) / np.sqrt(NUM_TRIALS))
    return mean, stderr, scores

if __name__ == '__main__':
    all_results = {}
    for layout in LAYOUTS:
        print(f"\n=== {layout} ===")
        mean, stderr, scores = run_benchmark(layout)
        all_results[layout] = {'mean': mean, 'stderr': stderr, 'scores': scores}
        print(f"  RESULT: {mean:.1f} +/- {stderr:.1f}")
    with open('llm_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to llm_results.json")
