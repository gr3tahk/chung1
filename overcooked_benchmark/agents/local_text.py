from __future__ import annotations

from overcooked_benchmark.agents.base import ACTION_TO_OVERCOOKED, AgentDecision, AgentObservation, BenchmarkAgent, parse_agent_response
from overcooked_benchmark.agents.prompts import build_action_prompt


class LocalTextAgent(BenchmarkAgent):
    def __init__(self, player_id: int, player_name: str, client):
        super().__init__(player_id, player_name)
        self.client = client

    def act(self, observation: AgentObservation):
        prompt = build_action_prompt(observation, include_text_state=True)
        raw = self.client.generate(prompt)
        action, message, plan, valid, invalid_reason = parse_agent_response(raw)
        self.last_decision = AgentDecision(
            player_id=self.player_id,
            player_name=self.player_name,
            action=action,
            message=message,
            plan=plan,
            raw_response=raw,
            prompt=prompt,
            valid=valid,
            invalid_reason=invalid_reason,
        )
        return ACTION_TO_OVERCOOKED[action]
