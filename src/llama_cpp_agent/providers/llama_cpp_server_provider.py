import json

import requests

from dataclasses import dataclass, field
from typing import List, Union
from collections import defaultdict


@dataclass
class LlamaCppServerGenerationSettings:
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    n_predict: int = -1
    n_keep: int = 0
    stream: bool = False
    stop: List[str] = field(default_factory=list)
    tfs_z: float = 1.0
    typical_p: float = 1.0
    repeat_penalty: float = 1.1
    repeat_last_n: int = 64
    penalize_nl: bool = True
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    penalty_prompt: Union[None, str, List[int]] = None
    mirostat: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    seed: int = -1
    ignore_eos: bool = False

    def save(self, file_path: str):
        with open(file_path, 'w', encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load_from_file(file_path: str) -> "LlamaCppServerGenerationSettings":
        with open(file_path, 'r', encoding="utf-8") as file:
            loaded_settings = json.load(file)
            return LlamaCppServerGenerationSettings(**loaded_settings)

    @staticmethod
    def load_from_dict(settings: dict) -> "LlamaCppServerGenerationSettings":
        return LlamaCppServerGenerationSettings(**settings)

    def as_dict(self) -> dict:
        return self.__dict__


@dataclass
class LlamaCppServerLLMSettings:
    server_url: str

    def save(self, file_path: str):
        with open(file_path, 'w', encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load_from_file(file_path: str) -> "LlamaCppServerLLMSettings":
        with open(file_path, 'r', encoding="utf-8") as file:
            loaded_settings = json.load(file)
            return LlamaCppServerLLMSettings(**loaded_settings)

    @staticmethod
    def load_from_dict(settings: dict) -> "LlamaCppServerLLMSettings":
        return LlamaCppServerLLMSettings(**settings)

    def as_dict(self) -> dict:
        return self.__dict__

    def get_response(self, prompt, grammar, generation_settings: LlamaCppServerGenerationSettings):
        headers = {"Content-Type": "application/json"}

        data = generation_settings.as_dict()
        data["prompt"] = prompt
        data["grammar"] = grammar

        response = requests.post(self.server_url, headers=headers, json=data)
        return response.json()["content"]
