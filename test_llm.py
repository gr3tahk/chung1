from benchmark import LLMAgent, run_game
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

print("Running full 400 tick game on cramped_room...")
score = run_game('cramped_room')
print(f"\nFinal score: {score}")
print(f"Soups delivered: {score // 20}")
