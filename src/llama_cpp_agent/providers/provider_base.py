import json
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Union

import requests

from llama_cpp_agent.structured_output.settings import StructuredOutputSettings


class LlmProviderId(Enum):
    llama_cpp = "llama_cpp"
    tgi = "text_generation_inference"
    vllm = "vllm"


class LlmGenerationSettings(ABC):
    @abstractmethod
    def get_provider_identifier(self) -> LlmProviderId:
        """
        Returns a internally used provider identifier.

        Returns:
            provider_identifier(str): The provider identifier.
        """
        pass

    @abstractmethod
    def save(self, file_path: str):
        """
        Save the settings to a file.

        Args:
            file_path (str): The path to the file.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_from_file(file_path: str) -> "LlmGenerationSettings":
        """
        Load the settings from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            LlmGenerationSettings: The loaded settings.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_from_dict(settings: dict) -> "LlmGenerationSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            LlmGenerationSettings: The loaded settings.
        """
        pass

    @abstractmethod
    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        pass


class LlmProviderBase(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @abstractmethod
    def get_provider_identifier(self) -> str:
        pass

    @abstractmethod
    def create_completion(
        self,
        prompt: str,
        structured_output_settings: StructuredOutputSettings,
        settings: LlmGenerationSettings,
    ):
        """Create a completion request with the LLM provider and returns the result."""
        pass

    @abstractmethod
    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        structured_output_settings: StructuredOutputSettings,
        settings: LlmGenerationSettings,
    ):
        """Create a chat completion request with the LLM provider and returns the result."""
        pass

    @abstractmethod
    def tokenize(self, prompt: str):
        """Tokenize the given prompt."""
        pass


@dataclass
class LlamaCppGenerationSettings(LlmGenerationSettings):
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

    def get_provider_identifier(self) -> str:
        return """llama_cpp"""

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


class LlamaCppServerProvider(LlmProviderBase):
    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.llama_cpp

    def __init__(
        self,
        server_address: str,
        api_key: str = None,
        llama_cpp_python_server: bool = False,
    ):
        self.server_address = server_address
        if llama_cpp_python_server:
            self.server_completion_endpoint = (
                self.server_address + "/v1/engines/copilot-codex/completions"
            )
        else:
            self.server_completion_endpoint = self.server_address + "/completion"

        self.server_chat_completion_endpoint = (
            self.server_address + "/v1/chat/completions"
        )
        if llama_cpp_python_server:
            self.server_tokenize_endpoint = self.server_address + "/extras/tokenize"
        else:
            self.server_tokenize_endpoint = self.server_address + "/tokenize"
        self.api_key = api_key
        self.llama_cpp_python_server = llama_cpp_python_server

    def create_completion(
        self,
        prompt: str,
        structured_output_settings: StructuredOutputSettings,
        settings: LlamaCppGenerationSettings,
    ):
        if self.api_key is not None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",  # Add API key as bearer token
            }
        else:
            headers = {"Content-Type": "application/json"}

        data = copy(settings.as_dict())
        data["prompt"] = prompt
        data["grammar"] = structured_output_settings.get_gbnf_grammar()
        if not self.llama_cpp_python_server:
            data["mirostat"] = data.pop("mirostat_mode")
        if self.llama_cpp_python_server:
            dic = {
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.95,
                "min_p": 0.05,
                "max_tokens": -1,
                "stream": False,
                "stop_sequences": [],
                "repeat_penalty": 1.1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "mirostat_mode": 0,
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1,
                "seed": -1,
            }
        data["stop"] = data.pop("stop_sequences")
        if not self.llama_cpp_python_server:
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
            return self.get_response_stream(
                headers, data, self.server_completion_endpoint
            )

        response = requests.post(
            self.server_completion_endpoint, headers=headers, json=data
        )
        data = response.json()

        returned_data = {"choices": [{"text": data["content"]}]}
        return returned_data

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        structured_output_settings: StructuredOutputSettings,
        settings: LlamaCppGenerationSettings,
    ):
        if self.api_key is not None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",  # Add API key as bearer token
            }
        else:
            headers = {"Content-Type": "application/json"}

        data = copy(settings.as_dict())
        data["messages"] = messages
        data["grammar"] = structured_output_settings.get_gbnf_grammar()
        if not self.llama_cpp_python_server:
            data["mirostat"] = data.pop("mirostat_mode")
        if self.llama_cpp_python_server:
            dic = {
                "temperature": 0.8,
                "top_k": 40,
                "top_p": 0.95,
                "min_p": 0.05,
                "max_tokens": -1,
                "stream": False,
                "stop_sequences": [],
                "repeat_penalty": 1.1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "mirostat_mode": 0,
                "mirostat_tau": 5.0,
                "mirostat_eta": 0.1,
                "seed": -1,
            }
        data["stop"] = data.pop("stop_sequences")
        if not self.llama_cpp_python_server:
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
            return self.get_response_stream(
                headers, data, self.server_chat_completion_endpoint
            )

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
                "Authorization": f"Bearer {self.api_key}",  # Add API key as bearer token
            }
        else:
            headers = {"Content-Type": "application/json"}
        if self.llama_cpp_python_server:
            response = requests.post(
                self.server_tokenize_endpoint, headers=headers, json={"input": prompt}
            )
        else:
            response = requests.post(
                self.server_tokenize_endpoint, headers=headers, json={"content": prompt}
            )
        if response.status_code == 200:
            tokens = response.json()["tokens"]
            return tokens
        else:
            raise Exception(
                f"Tokenization request failed. Status code: {response.status_code}\nResponse: {response.text}"
            )

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
