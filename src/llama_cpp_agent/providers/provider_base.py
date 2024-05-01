import json
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from typing import Any, Dict, List

import requests
from pydantic import BaseModel


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

    @abstractmethod
    def get_default_generation_settings(self) -> LLMGenerationSettings:
        pass

    @abstractmethod
    def create_completion(self, prompt: str, grammar: str, settings: LLMGenerationSettings):
        pass

    @abstractmethod
    def create_chat_completion(self, messages: List[Dict[str, str]], grammar: str, settings: LLMGenerationSettings):
        pass

    @abstractmethod
    def tokenize(self, prompt: str):
        pass


class LlamaCppServerProvider(LLMProviderBase):
    def __init__(self, server_address: str, api_key: str = None):
        self.server_address = server_address
        self.server_completion_endpoint = self.server_address + "/completion"
        self.server_chat_completion_endpoint = self.server_address + "/v1/chat/completions"
        self.server_tokenize_endpoint = self.server_address + "/tokenize"
        self.api_key = api_key

    def get_default_generation_settings(self) -> LLMGenerationSettings:
        default_settings = LLMGenerationSettings()
        dic = {
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.95,
            "min_p": 0.05,
            "n_predict": -1,
            "n_keep": 0,
            "stream": False,
            "stop_sequences": [],
            "tfs_z": 1.0,
            "typical_p": 1.0,
            "repeat_penalty": 1.1,
            "repeat_last_n": -1,
            "penalize_nl": False,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "penalty_prompt": None,
            "mirostat_mode": 0,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1,
            "seed": -1,
            "ignore_eos": False
        }

        # Populate the LLMGenerationSettings instance
        for key, value in dic.items():
            default_settings.set_setting(key, value)

        return default_settings

    def create_completion(self, prompt: str, grammar: str, settings: LLMGenerationSettings):
        if self.api_key is not None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"  # Add API key as bearer token
            }
        else:
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
        if settings.stream:
            return self.get_response_stream(headers, data, self.server_completion_endpoint)

        response = requests.post(
            self.server_completion_endpoint, headers=headers, json=data
        )
        data = response.json()

        returned_data = {"choices": [{"text": data["content"]}]}
        return returned_data

    def create_chat_completion(self, messages: List[Dict[str, str]], grammar: str, settings: LLMGenerationSettings):
        if self.api_key is not None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"  # Add API key as bearer token
            }
        else:
            headers = {"Content-Type": "application/json"}

        data = copy(settings.as_dict())
        data["messages"] = messages
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
        if settings.stream:
            return self.get_response_stream(headers, data, self.server_chat_completion_endpoint)

        response = requests.post(
            self.server_chat_completion_endpoint, headers=headers, json=data
        )
        data = response.json()

        returned_data = {"choices": [{"text": data["content"]}]}
        return returned_data

    def tokenize(self, prompt: str) -> list[int]:
        if self.api_key is not None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"  # Add API key as bearer token
            }
        else:
            headers = {"Content-Type": "application/json"}
        response = requests.post(self.server_tokenize_endpoint, headers=headers, json={"content": prompt})
        if response.status_code == 200:
            tokens = response.json()["tokens"]
            return tokens
        else:
            raise Exception(
                f"Tokenization request failed. Status code: {response.status_code}\nResponse: {response.text}")

    @staticmethod
    def get_response_stream(headers, data, endpoint_address):

        response = requests.post(
            endpoint_address,
            headers=headers,
            json=data,
            stream=True,
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


class OpenAIProvider(LLMProviderBase):
    def get_default_settings(self) -> LLMGenerationSettings:
        default_settings = LLMGenerationSettings()
        dic = {
            "max_tokens": 0,
            "temperature": 0.8,
            "top_p": 0.95,
            "min_p": 0.05,
            "echo": False,
            "stop_sequences": None,
            "stream": False,
            "logprobs": None,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logit_bias": None,
            "seed": None,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "logit_bias_type": None,
            "mirostat_mode": 0,
            "mirostat_tau": 5.0,
            "mirostat_eta": 0.1,
            "grammar": None
        }

        # Populate the LLMGenerationSettings instance
        for key, value in dic.items():
            default_settings.set_setting(key, value)

        return default_settings
