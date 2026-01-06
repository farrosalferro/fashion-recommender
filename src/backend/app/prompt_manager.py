from langsmith import Client
from typing import Any

url_template = "farrosalferro/{agent}-prompt:latest"


class PromptManager:

    def __init__(self, auto_load: bool = True):
        self.client = Client()
        self._prompts: dict[str, Any] = {}
        if auto_load:
            self.load_all()

    def load_all(self) -> None:
        try:
            self._prompts["agent"] = self.client.pull_prompt(url_template.format(agent="agent"))
            self._prompts["descriptor"] = self.client.pull_prompt(url_template.format(agent="descriptor"))
            self._prompts["recommender"] = self.client.pull_prompt(url_template.format(agent="recommender"))
            self._prompts["vton"] = self.client.pull_prompt(url_template.format(agent="vton"))
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
            self._prompts[name] = self.client.pull_prompt(url_template.format(agent=name))
        else:
            self.load_all()
