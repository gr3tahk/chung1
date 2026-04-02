import unittest

from metrics import compute_ites, compute_tes, score_against_references


class MetricsTest(unittest.TestCase):
    def test_tes_exact_match(self):
        executed = ["p0:pickup_onion", "p0:place_onion_in_pot"]
        reference = ["p0:pickup_onion", "p0:place_onion_in_pot"]
        self.assertEqual(compute_tes(executed, reference), 1.0)

    def test_tes_penalizes_redundancy(self):
        executed = ["p0:pickup_onion", "p0:pickup_onion", "p0:place_onion_in_pot"]
        reference = ["p0:pickup_onion", "p0:place_onion_in_pot"]
        self.assertLess(compute_tes(executed, reference), 1.0)

    def test_tes_penalizes_wrong_order(self):
        executed = ["p0:place_onion_in_pot", "p0:pickup_onion"]
        reference = ["p0:pickup_onion", "p0:place_onion_in_pot"]
        self.assertLess(compute_tes(executed, reference), 1.0)

    def test_tes_penalizes_partial_completion(self):
        executed = ["p0:pickup_onion"]
        reference = ["p0:pickup_onion", "p0:place_onion_in_pot"]
        self.assertLess(compute_tes(executed, reference), 1.0)

    def test_ites_rewards_progress_and_penalizes_stalls(self):
        reference = ["p0:pickup_onion", "p0:place_onion_in_pot"]
        improving = ["p0:pickup_onion", "p0:place_onion_in_pot"]
        stalled = ["p0:pickup_onion", "p1:wait", "p1:wait", "p1:wait"]
        self.assertGreater(compute_ites(improving, reference), compute_ites(stalled, reference))

    def test_best_reference_selected(self):
        executed = ["p0:pickup_onion", "p0:place_onion_in_pot"]
        references = [
            {"id": "wrong", "actions": ["p1:pickup_onion", "p1:place_onion_in_pot"]},
            {"id": "exact", "actions": ["p0:pickup_onion", "p0:place_onion_in_pot"]},
        ]
        result = score_against_references(executed, references)
        self.assertEqual(result["best_reference_id"], "exact")


if __name__ == "__main__":
    unittest.main()
