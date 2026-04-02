import unittest
from unittest.mock import patch

import benchmark


class TaskBenchmarkTest(unittest.TestCase):
    def test_task_loading(self):
        tasks = benchmark.load_tasks()
        self.assertGreaterEqual(len(tasks), 3)
        self.assertIn("task_id", tasks[0])
        self.assertIn("reference_trajectories", tasks[0])

    def test_symbolic_extraction_detects_key_actions(self):
        before = {
            "players": [
                {"position": (1, 1), "orientation": (1, 0), "held_object": "nothing"},
                {"position": (3, 3), "orientation": (-1, 0), "held_object": "nothing"},
            ],
            "counters": {},
            "pots": {(2, 1): {"ingredients": 0, "is_cooking": False, "is_ready": False}},
        }
        after_pickup = {
            "players": [
                {"position": (1, 1), "orientation": (1, 0), "held_object": "onion"},
                {"position": (3, 3), "orientation": (-1, 0), "held_object": "nothing"},
            ],
            "counters": {},
            "pots": {(2, 1): {"ingredients": 0, "is_cooking": False, "is_ready": False}},
        }
        action = benchmark.classify_player_action(
            0,
            before,
            after_pickup,
            (0, 1),
            0,
            benchmark.Counter(),
            benchmark.Counter(),
        )
        self.assertEqual(action, "pickup_onion")

        before_place = after_pickup
        after_place = {
            "players": [
                {"position": (1, 1), "orientation": (1, 0), "held_object": "nothing"},
                {"position": (3, 3), "orientation": (-1, 0), "held_object": "nothing"},
            ],
            "counters": {},
            "pots": {(2, 1): {"ingredients": 1, "is_cooking": False, "is_ready": False}},
        }
        action = benchmark.classify_player_action(
            0,
            before_place,
            after_place,
            benchmark.Action.INTERACT,
            0,
            benchmark.Counter(),
            benchmark.Counter(),
        )
        self.assertEqual(action, "place_onion_in_pot")

    def test_exact_reference_scores_best(self):
        task = benchmark.get_task_by_id("cramped_room_single_delivery")
        executed = list(task["reference_trajectories"][0]["actions"])
        result = benchmark.evaluate_task_trajectory(task, executed)
        self.assertEqual(result["best_reference_id"], task["reference_trajectories"][0]["id"])
        self.assertEqual(result["tes"], 1.0)

    def test_reward_benchmark_path_still_runs(self):
        with patch("benchmark.get_llm_goal", return_value="get_plate"):
            result = benchmark.run_game("cramped_room", max_ticks=20, collect_trajectory=True)
        self.assertIn("score", result)
        self.assertIn("trajectory", result)
        self.assertIn("tick_events", result)


if __name__ == "__main__":
    unittest.main()
