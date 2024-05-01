import json
from copy import copy

import requests
import aiohttp
import json
from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class LlamaCppGenerationSettings:
    """
    Settings for generating completions using the Llama.cpp server.

    Args:
        temperature (float): Controls the randomness of the generated completions. Higher values make the output more random.
        top_k (int): Controls the diversity of the top-k sampling. Higher values result in more diverse completions.
        top_p (float): Controls the diversity of the nucleus sampling. Higher values result in more diverse completions.
        min_p (float): Minimum probability for nucleus sampling. Lower values result in more focused completions.
        n_predict (int): Number of completions to predict. Set to -1 to use the default value.
        n_keep (int): Number of completions to keep. Set to 0 for all predictions.
        stream (bool): Enable streaming for long completions.
        stop_sequences (List[str]): List of stop sequences to finish completion generation.
        tfs_z (float): Controls the temperature for top frequent sampling.
        typical_p (float): Typical probability for top frequent sampling.
        repeat_penalty (float): Penalty for repeating tokens in completions.
        repeat_last_n (int): Number of tokens to consider for repeat penalty.
        penalize_nl (bool): Enable penalizing newlines in completions.
        presence_penalty (float): Penalty for presence of certain tokens.
        frequency_penalty (float): Penalty based on token frequency.
        penalty_prompt (Union[None, str, List[int]]): Prompts to apply penalty for certain tokens.
        mirostat_mode (int): Mirostat level.
        mirostat_tau (float): Mirostat temperature.
        mirostat_eta (float): Mirostat eta parameter.
        seed (int): Seed for randomness. Set to -1 for no seed.
        ignore_eos (bool): Ignore end-of-sequence token.

    Attributes:
        temperature (float): Controls the randomness of the generated completions. Higher values make the output more random.
        top_k (int): Controls the diversity of the top-k sampling. Higher values result in more diverse completions.
        top_p (float): Controls the diversity of the nucleus sampling. Higher values result in more diverse completions.
        min_p (float): Minimum probability for nucleus sampling. Lower values result in more focused completions.
        n_predict (int): Number of completions to predict. Set to -1 to use the default value.
        n_keep (int): Number of completions to keep. Set to 0 for all predictions.
        stream (bool): Enable streaming for long completions.
        stop_sequences (List[str]): List of stop sequences to finish completion generation.
        tfs_z (float): Controls the temperature for top frequent sampling.
        typical_p (float): Typical probability for top frequent sampling.
        repeat_penalty (float): Penalty for repeating tokens in completions.
        repeat_last_n (int): Number of tokens to consider for repeat penalty.
        penalize_nl (bool): Enable penalizing newlines in completions.
        presence_penalty (float): Penalty for presence of certain tokens.
        frequency_penalty (float): Penalty based on token frequency.
        penalty_prompt (Union[None, str, List[int]]): Prompts to apply penalty for certain tokens.
        mirostat_mode (int): Mirostat level.
        mirostat_tau (float): Mirostat temperature.
        mirostat_eta (float): Mirostat eta parameter.
        seed (int): Seed for randomness. Set to -1 for no seed.
        ignore_eos (bool): Ignore end-of-sequence token.
    Methods:
        save(file_path: str): Save the settings to a file.
        load_from_file(file_path: str) -> LlamaCppServerGenerationSettings: Load the settings from a file.
        load_from_dict(settings: dict) -> LlamaCppServerGenerationSettings: Load the settings from a dictionary.
        as_dict() -> dict: Convert the settings to a dictionary.

    """

    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    n_predict: int = -1
    n_keep: int = 0
    stream: bool = False
    stop_sequences: List[str] = None
    tfs_z: float = 1.0
    typical_p: float = 1.0
    repeat_penalty: float = 1.1
    repeat_last_n: int = -1
    penalize_nl: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    penalty_prompt: Union[None, str, List[int]] = None
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    cache_prompt: bool = True
    seed: int = -1
    ignore_eos: bool = False
    samplers: List[str] = None

    def save(self, file_path: str):
        """
        Save the settings to a file.

        Args:
            file_path (str): The path to the file.
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load_from_file(file_path: str) -> "LlamaCppGenerationSettings":
        """
        Load the settings from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            LlamaCppGenerationSettings: The loaded settings.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_settings = json.load(file)
            return LlamaCppGenerationSettings(**loaded_settings)

    @staticmethod
    def load_from_dict(settings: dict) -> "LlamaCppGenerationSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            LlamaCppGenerationSettings: The loaded settings.
        """
        return LlamaCppGenerationSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        return self.__dict__


@dataclass
class LlamaCppEndpointSettings:
    """
    Settings for interacting with the Llama.cpp server.

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
    def load_from_file(file_path: str) -> "LlamaCppEndpointSettings":
        """
        Load the settings from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            LlamaCppEndpointSettings: The loaded settings.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_settings = json.load(file)
            return LlamaCppEndpointSettings(**loaded_settings)

    @staticmethod
    def load_from_dict(settings: dict) -> "LlamaCppEndpointSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            LlamaCppEndpointSettings: The loaded settings.
        """
        return LlamaCppEndpointSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        return self.__dict__

    def create_completion(
        self, prompt, grammar, generation_settings: LlamaCppGenerationSettings
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
