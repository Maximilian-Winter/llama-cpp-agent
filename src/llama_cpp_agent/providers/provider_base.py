import json
from abc import ABC
from copy import copy
from dataclasses import dataclass
from typing import Any, Dict, List

import requests
from pydantic import BaseModel

from llama_cpp_agent.messages import ChatMessage


@dataclass
class LLMGenerationSetting:
    name: str
    value: Any


class LLMGenerationSettings(BaseModel):
    settings: Dict[str, LLMGenerationSetting]

    def get_settings_dict(self) -> Dict[str, Any]:
        return {name: setting.value for name, setting in self.settings.items()}

    def set_setting(self, name: str, value: Any):
        if name in self.settings:
            self.settings[name].value = value
        else:
            self.settings[name] = LLMGenerationSetting(name=name, value=value)

    def get_setting(self, name: str) -> Any:
        if name in self.settings:
            return self.settings[name].value
        return None

    def __getattr__(self, name: str) -> Any:
        if name in self.settings:
            return self.settings[name].value
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class LLMProviderBase(ABC):

    def get_default_settings(self) -> LLMGenerationSettings:
        pass

    def create_completion(self, prompt: str, grammar: str, settings: LLMGenerationSettings):
        pass

    def create_chat_completion(self, messages: List[Dict[str, str]], grammar: str, settings: LLMGenerationSettings):
        pass

    def tokenize(self, prompt: str):
        pass


class LlamaCppServerProvider(LLMProviderBase):
    def __init__(self, server_address: str, api_key: str = "NO-API-KEY"):
        self.server_address = server_address
        self.api_key = api_key

    def create_completion(self, prompt: str, grammar: str, settings: LLMGenerationSettings):
        if settings.stream:
            return self.get_response_stream(prompt, grammar, settings)

        headers = {"Content-Type": "application/json"}

        data = copy(settings.as_dict())
        data["prompt"] = prompt
        data["grammar"] = grammar
        data["mirostat"] = data["mirostat_mode"]
        data["stop"] = data["stop_sequences"]
        del data["mirostat_mode"]
        del data["stop_sequences"]
        if "samplers" not in data or data["samplers"] is None:
            data["samplers"] = [
                "top_k",
                "tfs_z",
                "typical_p",
                "top_p",
                "min_p",
                "temperature",
            ]
        response = requests.post(
            self.server_address + "/completion", headers=headers, json=data
        )
        data = response.json()

        returned_data = {"choices": [{"text": data["content"]}]}
        return returned_data

    def get_response_stream(self, prompt, grammar, generation_settings):
        headers = {"Content-Type": "application/json"}

        data = copy(generation_settings.as_dict())
        data["prompt"] = prompt
        data["grammar"] = grammar
        data["mirostat"] = data["mirostat_mode"]
        data["stop"] = data["stop_sequences"]
        del data["mirostat_mode"]
        del data["stop_sequences"]
        if "samplers" not in data or data["samplers"] is None:
            data["samplers"] = [
                "top_k",
                "tfs_z",
                "typical_p",
                "top_p",
                "min_p",
                "temperature",
            ]
        response = requests.post(
            self.completions_endpoint_url,
            headers=headers,
            json=data,
            stream=generation_settings.stream,
        )

        # Check if the request was successful
        response.raise_for_status()

        # Define a generator function to yield text chunks
        def generate_text_chunks():
            try:
                decoded_chunk = ""
                for chunk in response.iter_lines():
                    if chunk:
                        decoded_chunk += chunk.decode("utf-8")
                        new_data = json.loads(decoded_chunk.replace("data: ", ""))
                        returned_data = {"choices": [{"text": new_data["content"]}]}
                        yield returned_data
                        decoded_chunk = ""

            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")

        return generate_text_chunks()