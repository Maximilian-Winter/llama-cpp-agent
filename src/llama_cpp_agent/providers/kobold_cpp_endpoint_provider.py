import json
from copy import copy
from dataclasses import dataclass, field
from typing import List, Optional, Union

import requests


@dataclass
class KoboldCppGenerationSettings:
    max_context_length: int
    max_length: int
    prompt: str
    rep_pen: float
    rep_pen_range: int
    sampler_order: List[int]
    sampler_seed: int
    stop_sequence: List[str]
    temperature: float
    tfs: float
    top_a: float
    top_k: int
    top_p: float
    min_p: float
    typical: float
    stream: bool = True
    use_default_badwordsids: bool = False
    dynatemp_range: float = 0
    mirostat: Optional[int] = None
    mirostat_tau: float = 0
    mirostat_eta: float = 0
    genkey: Optional[str] = None
    grammar_retain_state: bool = False
    memory: Optional[str] = None
    trim_stop: bool = False
    logit_bias: Optional[dict] = field(default_factory=dict)

    def save(self, file_path: str):
        """
        Save the settings to a file.

        Args:
            file_path (str): The path to the file.
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load_from_file(file_path: str) -> "KoboldCppGenerationSettings":
        """
        Load the settings from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            KoboldCppGenerationSettings: The loaded settings.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_settings = json.load(file)
            return KoboldCppGenerationSettings(**loaded_settings)

    @staticmethod
    def load_from_dict(settings: dict) -> "KoboldCppGenerationSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            KoboldCppGenerationSettings: The loaded settings.
        """
        return KoboldCppGenerationSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        return self.__dict__


@dataclass
class KoboldCppEndpointSettings:
    """
    Settings for interacting with the kobold.cpp server.

    Args:
        completions_endpoint_url (str): The URL for the completions endpoint.

    Attributes:
        completions_endpoint_url (str): The URL for the completions endpoint.

    Methods:
        save(file_path: str): Save the settings to a file.
        load_from_file(file_path: str) -> LlamaCppServerLLMSettings: Load the settings from a file.
        load_from_dict(settings: dict) -> LlamaCppServerLLMSettings: Load the settings from a dictionary.
        as_dict() -> dict: Convert the settings to a dictionary.
        create_completion(prompt, grammar, generation_settings: LlamaCppServerGenerationSettings): Create a completion using the server.

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
    def load_from_file(file_path: str) -> "KoboldCppEndpointSettings":
        """
        Load the settings from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            KoboldCppEndpointSettings: The loaded settings.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_settings = json.load(file)
            return KoboldCppEndpointSettings(**loaded_settings)

    @staticmethod
    def load_from_dict(settings: dict) -> "KoboldCppEndpointSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            KoboldCppEndpointSettings: The loaded settings.
        """
        return KoboldCppEndpointSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        return self.__dict__

    def create_completion(
        self, prompt, grammar, generation_settings: KoboldCppGenerationSettings
    ):
        """
        Create a completion using the Llama.cpp server.

        Args:
            prompt: The input prompt.
            grammar: The grammar for completion.
            generation_settings (LlamaCppGenerationSettings): The generation settings.

        Returns:
            dict or generator: The completion response.
        """
        if generation_settings.stream:
            return self.get_response_stream(prompt, grammar, generation_settings)

        headers = {"Content-Type": "application/json"}

        data = generation_settings.as_dict()
        data["prompt"] = prompt
        data["grammar"] = grammar
        data["mirostat"] = data["mirostat_mode"]
        data["stop"] = data["stop_sequences"]
        del data["mirostat_mode"]
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
        data["mirostat"] = data["mirostat_mode"]
        data["stop_sequence"] = data["stop_sequences"]
        del data["mirostat_mode"]
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
