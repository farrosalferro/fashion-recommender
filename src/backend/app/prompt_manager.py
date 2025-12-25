from langsmith import Client
from typing import Any


class PromptManager:

    def __init__(self, auto_load: bool = True):
        self.client = Client()
        self._prompts: dict[str, Any] = {}
        if auto_load:
            self.load_all()

    def load_all(self) -> None:
        try:
            self._prompts["agent"] = self.client.pull_prompt("agent-prompt:v1-0-1")
            self._prompts["descriptor"] = self.client.pull_prompt("descriptor-prompt:v1-0-0")
            self._prompts["recommender"] = self.client.pull_prompt("recommender-prompt:v1-0-0")
            self._prompts["vton"] = self.client.pull_prompt("vton-prompt:v1-0-0")
        except Exception as e:
            raise ValueError(f"Failed to load prompts: {e}")

    def get_prompt(self, agent: str) -> str:
        prompt = self._prompts.get(agent)
        if prompt is None:
            raise ValueError(f"Prompt for {agent} not found")

        try:
            return prompt.messages[0].prompt.template
        except Exception as e:
            raise ValueError(f"Failed to get prompt for {agent}: {e}")

    def refresh(self, name: str = None) -> None:
        if name:
            self._prompts[name] = self.client.pull_prompt(f"{name}-prompt:latest")
        else:
            self.load_all()
