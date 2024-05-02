import json
from copy import copy
from typing import Union, List, Optional, Dict, Literal
from dataclasses import dataclass

import aiohttp
import requests


@dataclass
class OpenAIGenerationSettings:
    max_tokens: Optional[int] = 0
    temperature: float = 0.8
    top_p: float = 0.95
    min_p: float = 0.05
    echo: bool = False
    stop_sequences: Optional[Union[str, List[str]]] = None
    stream: bool = False
    logprobs: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    seed: Optional[int] = None
    top_k: int = 40
    repeat_penalty: float = 1.1
    logit_bias_type: Optional[Literal["input_ids", "tokens"]] = None
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1

    def save(self, file_path: str):
        """
        Save the settings to a file.

        Args:
            file_path (str): The path to the file.
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load_from_file(file_path: str) -> "OpenAIGenerationSettings":
        """
        Load the settings from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            CreateCompletionRequest: The loaded settings.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_settings = json.load(file)
            return OpenAIGenerationSettings(**loaded_settings)

    @staticmethod
    def load_from_dict(settings: dict) -> "OpenAIGenerationSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            CreateCompletionRequest: The loaded settings.
        """
        return OpenAIGenerationSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        return self.__dict__


@dataclass
class OpenAIEndpointSettings:
    """
    Settings for interacting with an OpenAI endpoint that support GBNF grammars like the llama-cpp-python server.

    Args:
        completions_endpoint_url (str): The URL for the completions endpoint.

    Attributes:
        completions_endpoint_url (str): The URL for the completions endpoint.

    Methods:
        save(file_path: str): Save the settings to a file.
        load_from_file(file_path: str) -> OpenAIEndpointSettings: Load the settings from a file.
        load_from_dict(settings: dict) -> OpenAIEndpointSettings: Load the settings from a dictionary.
        as_dict() -> dict: Convert the settings to a dictionary.
        create_completion(prompt, grammar, generation_settings: CompletionRequestSettings): Create a completion using the server.

    """

    completions_endpoint_url: str

    def save(self, file_path: str):
        """
        Save the settings to a file.

        Args:
            file_path (str): The path to the file.
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load_from_file(file_path: str) -> "OpenAIEndpointSettings":
        """
        Load the settings from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            OpenAIEndpointSettings: The loaded settings.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_settings = json.load(file)
            return OpenAIEndpointSettings(**loaded_settings)

    @staticmethod
    def load_from_dict(settings: dict) -> "OpenAIEndpointSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            OpenAIEndpointSettings: The loaded settings.
        """
        return OpenAIEndpointSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        return self.__dict__

    def create_completion(
        self, prompt, grammar, generation_settings: OpenAIGenerationSettings
    ):
        """
        Create a completion using the Llama.cpp server.

        Args:
            prompt: The input prompt.
            grammar: The grammar for completion.
            generation_settings (OpenAIGenerationSettings): The generation settings.

        Returns:
            dict or generator: The completion response.
        """
        if generation_settings.stream:
            return self.get_response_stream(prompt, grammar, generation_settings)

        headers = {"Content-Type": "application/json"}

        data = copy(generation_settings.as_dict())
        data["prompt"] = prompt
        data["grammar"] = grammar
        data["stop"] = data["stop_sequences"]
        del data["stop_sequences"]

        response = requests.post(
            self.completions_endpoint_url, headers=headers, json=data
        )
        data = response.json()

        returned_data = {"choices": [{"text": data["content"]}]}
        return returned_data

    def get_response_stream(self, prompt, grammar, generation_settings):
        headers = {"Content-Type": "application/json"}

        data = copy(generation_settings.as_dict())
        data["prompt"] = prompt
        data["grammar"] = grammar
        data["stop"] = data["stop_sequences"]
        del data["stop_sequences"]
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
