import unittest

from overcooked_benchmark.agents.base import parse_agent_response
from overcooked_benchmark.agents.local_text import LocalTextAgent
from overcooked_benchmark.local_client import get_local_text_client
from overcooked_benchmark.runners.paired import _build_no_op_warning
from overcooked_benchmark.runners.paired import run_agent_pair


class ExperimentTest(unittest.TestCase):
    def test_response_parser_accepts_json(self):
        action, message, plan, valid, reason = parse_agent_response(
            '{"action":"left","message":"taking onion","plan":"get onion then pot"}'
        )
        self.assertEqual(action, "left")
        self.assertEqual(message, "taking onion")
        self.assertEqual(plan, "get onion then pot")
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_response_parser_falls_back_to_stay(self):
        action, _, _, valid, reason = parse_agent_response("launch rocket")
        self.assertEqual(action, "stay")
        self.assertFalse(valid)
        self.assertIsNotNone(reason)

    def test_repeated_no_op_warning(self):
        self.assertEqual(_build_no_op_warning("interact", 1), "")
        self.assertIn("interact", _build_no_op_warning("interact", 2))

    def test_local_text_agent_uses_shared_client(self):
        class FakeClient:
            def __init__(self):
                self.calls = 0

            def generate(self, prompt):
                self.calls += 1
                return '{"action":"stay","message":"ok","plan":"wait"}'

        fake = FakeClient()
        agent = LocalTextAgent(0, "Alice", fake)
        action = agent.act(
            type(
                "Observation",
                (),
                {
                    "task": {"description": "test"},
                    "player_id": 0,
                    "state": type(
                        "State",
                        (),
                        {
                            "to_dict": lambda self: {
                                "players": [
                                    {"position": (1, 1), "orientation": (0, -1), "held_object": None},
                                    {"position": (2, 1), "orientation": (0, -1), "held_object": None},
                                ],
                                "objects": [],
                            },
                            "has_object": lambda self, pos: False,
                        },
                    )(),
                    "mdp": type(
                        "Mdp",
                        (),
                        {
                            "height": 1,
                            "width": 1,
                            "get_terrain_type_at_pos": lambda self, pos: " ",
                            "get_pot_locations": lambda self: [],
                            "get_onion_dispenser_locations": lambda self: [],
                            "get_dish_dispenser_locations": lambda self: [],
                            "get_serving_locations": lambda self: [],
                        },
                    )(),
                    "teammate_message": "",
                    "current_plan": "",
                    "action_feedback": "",
                    "no_op_warning": "",
                    "phase_hint": "",
                },
            )()
        )
        self.assertEqual(action, (0, 0))
        self.assertEqual(fake.calls, 1)
        self.assertEqual(agent.last_decision.plan, "wait")

    def test_local_text_client_cache_reuses_instance(self):
        first = get_local_text_client("fake/model", max_new_tokens=7)
        second = get_local_text_client("fake/model", max_new_tokens=7)
        third = get_local_text_client("fake/model", max_new_tokens=8)
        self.assertIs(first, second)
        self.assertIsNot(first, third)

    def test_scripted_pair_smoke_run(self):
        summary, trajectory = run_agent_pair(
            pair="scripted-scripted",
            layout_name="cramped_room",
            task_id="cramped_room_single_delivery",
            max_ticks=4,
            collect_trajectory=True,
        )
        self.assertIn("metrics", summary)
        self.assertIn("tick_events", trajectory)
        self.assertIn("feedbackAfter", trajectory["prompt_logs"][0])
        self.assertIn("planAfter", trajectory["prompt_logs"][0])
        self.assertIn("phaseHint", trajectory["prompt_logs"][0])
        self.assertIn("plan", trajectory["tick_events"][0]["decisions"][0])
        self.assertEqual(summary["metrics"]["invalid_action_count"], 0)


if __name__ == "__main__":
    unittest.main()
