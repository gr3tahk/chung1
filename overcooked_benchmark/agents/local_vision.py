from __future__ import annotations

from overcooked_benchmark.agents.base import ACTION_TO_OVERCOOKED, AgentDecision, AgentObservation, BenchmarkAgent, parse_agent_response
from overcooked_benchmark.agents.prompts import build_action_prompt
from overcooked_benchmark.rendering import render_state_image


class LocalVisionAgent(BenchmarkAgent):
    def __init__(self, player_id: int, player_name: str, client):
        super().__init__(player_id, player_name)
        self.client = client

    def act(self, observation: AgentObservation):
        prompt = build_action_prompt(observation, include_text_state=False)
        image = render_state_image(observation.state, observation.mdp, tick=observation.tick, score=observation.score)
        raw = self.client.generate(prompt, image)
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
