import datetime
import json
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Union

import requests
from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel, Field

from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType


class LlmProviderId(Enum):
    llama_cpp_server = "llama_cpp_server"
    llama_cpp_python = "llama_cpp_python"
    tgi_server = "text_generation_inference"
    vllm_server = "vllm"


class LlmSamplingSettings(ABC):
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
    def load_from_file(file_path: str) -> "LlmSamplingSettings":
        """
        Load the settings from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            LlmSamplingSettings: The loaded settings.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_from_dict(settings: dict) -> "LlmSamplingSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            LlmSamplingSettings: The loaded settings.
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


class LlmProvider(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @abstractmethod
    def get_provider_identifier(self) -> str:
        pass

    @abstractmethod
    def get_provider_default_settings(self) -> LlmSamplingSettings:
        """Returns the default generation settings of the provider."""
        pass

    @abstractmethod
    def create_completion(
            self,
            prompt: str,
            structured_output_settings: LlmStructuredOutputSettings,
            settings: LlmSamplingSettings,
    ):
        """Create a completion request with the LLM provider and returns the result."""
        pass

    @abstractmethod
    def create_chat_completion(
            self,
            messages: List[Dict[str, str]],
            structured_output_settings: LlmStructuredOutputSettings,
            settings: LlmSamplingSettings,
    ):
        """Create a chat completion request with the LLM provider and returns the result."""
        pass

    @abstractmethod
    def tokenize(self, prompt: str):
        """Tokenize the given prompt."""
        pass


@dataclass
class LlamaCppSamplingSettings(LlmSamplingSettings):
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

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.llama_cpp_server

    def save(self, file_path: str):
        """
        Save the settings to a file.

        Args:
            file_path (str): The path to the file.
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load_from_file(file_path: str) -> "LlamaCppSamplingSettings":
        """
        Load the settings from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            LlamaCppSamplingSettings: The loaded settings.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_settings = json.load(file)
            return LlamaCppSamplingSettings(**loaded_settings)

    @staticmethod
    def load_from_dict(settings: dict) -> "LlamaCppSamplingSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            LlamaCppSamplingSettings: The loaded settings.
        """
        return LlamaCppSamplingSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        return self.__dict__


class LlamaCppServerProvider(LlmProvider):

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

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.llama_cpp_server

    def get_provider_default_settings(self) -> LlamaCppSamplingSettings:
        return LlamaCppSamplingSettings()

    def create_completion(
            self,
            prompt: str,
            structured_output_settings: LlmStructuredOutputSettings,
            settings: LlamaCppSamplingSettings,
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
        data = self.prepare_generation_settings(data, structured_output_settings)

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
            structured_output_settings: LlmStructuredOutputSettings,
            settings: LlamaCppSamplingSettings,
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
        data = self.prepare_generation_settings(data, structured_output_settings)
        if settings.stream:
            return self.get_response_stream(
                headers, data, self.server_chat_completion_endpoint
            )
        response = requests.post(
            self.server_chat_completion_endpoint, headers=headers, json=data
        )
        data = response.json()

        returned_data = data
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

    def get_response_stream(self, headers, data, endpoint_address):
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
                        if decoded_chunk.strip().startswith("error:"):
                            raise RuntimeError(decoded_chunk)
                        new_data = json.loads(decoded_chunk.replace("data:", ""))
                        if self.llama_cpp_python_server:
                            returned_data = new_data
                        else:
                            if "choices" not in new_data:
                                returned_data = {"choices": [{"text": new_data["content"]}]}
                            else:
                                returned_data = new_data
                        yield returned_data
                        decoded_chunk = ""

            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")

        return generate_text_chunks()

    def prepare_generation_settings(self, settings_dictionary: dict,
                                    structured_output_settings: LlmStructuredOutputSettings) -> dict:
        if structured_output_settings.output_type != LlmStructuredOutputType.no_structured_output:
            settings_dictionary["grammar"] = structured_output_settings.get_gbnf_grammar()
        if not self.llama_cpp_python_server:
            settings_dictionary["mirostat"] = settings_dictionary.pop("mirostat_mode")
        if self.llama_cpp_python_server:
            settings_dictionary["max_tokens"] = settings_dictionary.pop("n_predict")

        settings_dictionary["stop"] = settings_dictionary.pop("stop_sequences")
        if not self.llama_cpp_python_server:
            if "samplers" not in settings_dictionary or settings_dictionary["samplers"] is None:
                settings_dictionary["samplers"] = [
                    "top_k",
                    "tfs_z",
                    "typical_p",
                    "top_p",
                    "min_p",
                    "temperature",
                ]
        else:
            if "samplers" in settings_dictionary:
                del settings_dictionary["samplers"]

        return settings_dictionary


class LlamaCppPythonProvider(LlmProvider):

    def __init__(
            self,
            llama_model: Llama
    ):
        self.llama_model = llama_model
        self.grammar_cache = {}

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.llama_cpp_python

    def get_provider_default_settings(self) -> LlamaCppSamplingSettings:
        return LlamaCppSamplingSettings()

    def create_completion(
            self,
            prompt: str,
            structured_output_settings: LlmStructuredOutputSettings,
            settings: LlamaCppSamplingSettings,
    ):
        grammar = None
        if structured_output_settings.output_type != LlmStructuredOutputType.no_structured_output:
            grammar = structured_output_settings.get_gbnf_grammar()
            if grammar in self.grammar_cache:
                grammar = self.grammar_cache[grammar]
            else:
                self.grammar_cache[grammar] = LlamaGrammar.from_string(grammar)
                grammar = self.grammar_cache[grammar]

        settings_dictionary = copy(settings.as_dict())
        settings_dictionary["max_tokens"] = settings_dictionary.pop("n_predict")
        settings_dictionary["stop"] = settings_dictionary.pop("stop_sequences")

        settings_dictionary.pop("n_keep")
        settings_dictionary.pop("repeat_last_n")
        settings_dictionary.pop("penalize_nl")
        settings_dictionary.pop("penalty_prompt")
        settings_dictionary.pop("samplers")
        settings_dictionary.pop("cache_prompt")
        settings_dictionary.pop("ignore_eos")

        return self.llama_model.create_completion(prompt, grammar=grammar, **settings_dictionary)

    def create_chat_completion(
            self,
            messages: List[Dict[str, str]],
            structured_output_settings: LlmStructuredOutputSettings,
            settings: LlamaCppSamplingSettings,
    ):
        grammar = None
        if structured_output_settings.output_type != LlmStructuredOutputType.no_structured_output:
            grammar = structured_output_settings.get_gbnf_grammar()
            if grammar in self.grammar_cache:
                grammar = self.grammar_cache[grammar]
            else:
                self.grammar_cache[grammar] = LlamaGrammar.from_string(grammar)
                grammar = self.grammar_cache[grammar]
        settings_dictionary = copy(settings.as_dict())
        settings_dictionary["max_tokens"] = settings_dictionary.pop("n_predict")
        settings_dictionary["stop"] = settings_dictionary.pop("stop_sequences")

        settings_dictionary.pop("n_keep")
        settings_dictionary.pop("repeat_last_n")
        settings_dictionary.pop("penalize_nl")
        settings_dictionary.pop("penalty_prompt")
        settings_dictionary.pop("samplers")
        settings_dictionary.pop("cache_prompt")
        settings_dictionary.pop("ignore_eos")

        return self.llama_model.create_chat_completion(messages, grammar=grammar, **settings_dictionary)

    def tokenize(self, prompt: str) -> list[int]:
        return self.llama_model.tokenize(prompt.encode("utf-8"))


from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TGIServerSamplingSettings(LlmSamplingSettings):
    """
    TGIServerSamplingSettings dataclass
    """

    best_of: Optional[int] = field(default=None, metadata={'minimum': 0})
    decoder_input_details: bool = False
    details: bool = True
    do_sample: bool = False
    frequency_penalty: Optional[float] = field(default=None, metadata={'exclusiveMinimum': -2})
    grammar: Optional[dict] = None
    max_new_tokens: Optional[int] = field(default=None, metadata={'minimum': 0})
    repetition_penalty: Optional[float] = field(default=None, metadata={'exclusiveMinimum': 0})
    return_full_text: Optional[bool] = field(default=None)
    seed: Optional[int] = field(default=None, metadata={'minimum': 0})
    stop: Optional[List[str]] = field(default_factory=list)
    temperature: Optional[float] = field(default=None, metadata={'exclusiveMinimum': 0})
    top_k: Optional[int] = field(default=None, metadata={'exclusiveMinimum': 0})
    top_n_tokens: Optional[int] = field(default=None, metadata={'minimum': 0, 'exclusiveMinimum': 0})
    top_p: Optional[float] = field(default=None, metadata={'maximum': 1, 'exclusiveMinimum': 0})
    truncate: Optional[int] = field(default=None, metadata={'minimum': 0})
    typical_p: Optional[float] = field(default=None, metadata={'maximum': 1, 'exclusiveMinimum': 0})
    watermark: bool = False
    stream: bool = False

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.tgi_server

    def save(self, file_path: str):
        """
        Save the settings to a file.

        Args:
            file_path (str): The path to the file.
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load_from_file(file_path: str) -> "TGIServerSamplingSettings":
        """
        Load the settings from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            LlamaCppSamplingSettings: The loaded settings.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_settings = json.load(file)
            return TGIServerSamplingSettings(**loaded_settings)

    @staticmethod
    def load_from_dict(settings: dict) -> "TGIServerSamplingSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            LlamaCppSamplingSettings: The loaded settings.
        """
        return TGIServerSamplingSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        return self.__dict__


class TGIServerProvider(LlmProvider):

    def __init__(
            self,
            server_address: str,
            api_key: str = None
    ):
        self.server_address = server_address
        self.server_completion_endpoint = self.server_address + "/generate"
        self.server_streaming_completion_endpoint = self.server_address + "/generate_stream"
        self.server_chat_completion_endpoint = (
                self.server_address + "/v1/chat/completions"
        )
        self.server_tokenize_endpoint = self.server_address + "/tokenize"
        self.api_key = api_key

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.tgi_server

    def get_provider_default_settings(self) -> TGIServerSamplingSettings:
        return TGIServerSamplingSettings()

    def create_completion(
            self,
            prompt: str,
            structured_output_settings: LlmStructuredOutputSettings,
            settings: TGIServerSamplingSettings,
    ):
        if self.api_key is not None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",  # Add API key as bearer token
            }
        else:
            headers = {"Content-Type": "application/json"}

        settings_dict = copy(settings.as_dict())

        data = {"parameters": settings_dict, "inputs": prompt}
        grammar = None
        if structured_output_settings.output_type != LlmStructuredOutputType.no_structured_output:
            grammar = structured_output_settings.get_json_schema()

        if grammar is not None:
            data["parameters"]["grammar"] = {"type": "json", "value": grammar}
        if settings.stream:
            return self.get_response_stream(
                headers, data, self.server_streaming_completion_endpoint
            )

        response = requests.post(
            self.server_completion_endpoint, headers=headers, json=data
        )
        data = response.json()

        returned_data = {"choices": [{"text": data["generated_text"]}]}
        return returned_data

    def create_chat_completion(
            self,
            messages: List[Dict[str, str]],
            structured_output_settings: LlmStructuredOutputSettings,
            settings: TGIServerSamplingSettings,
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
        data["model"] = "tgi"
        grammar = None
        if structured_output_settings.output_type != LlmStructuredOutputType.no_structured_output:
            grammar = structured_output_settings.get_json_schema()

        if grammar is not None:
            data["parameters"]["grammar"] = {"type": "json", "value": grammar}
        if settings.stream:
            return self.get_response_stream(
                headers, data, self.server_chat_completion_endpoint
            )
        response = requests.post(
            self.server_chat_completion_endpoint, headers=headers, json=data
        )
        data = response.json()

        returned_data = data
        return returned_data

    def tokenize(self, prompt: str) -> list[int]:
        if self.api_key is not None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",  # Add API key as bearer token
            }
        else:
            headers = {"Content-Type": "application/json"}
        response = requests.post(
            self.server_tokenize_endpoint, headers=headers, json={"inputs": prompt}
        )
        if response.status_code == 200:
            tokens = response.json()
            return [tok["id"] for tok in tokens]
        else:
            raise Exception(
                f"Tokenization request failed. Status code: {response.status_code}\nResponse: {response.text}"
            )

    def get_response_stream(self, headers, data, endpoint_address):
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
                        new_data = json.loads(decoded_chunk.replace("data:", ""))
                        if "generated_text" in new_data and (new_data["generated_text"] is not None):
                            returned_data = {"choices": [{"text": new_data["generated_text"]}]}
                            yield returned_data
                        decoded_chunk = ""

            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")

        return generate_text_chunks()


@dataclass
class VLLMServerSamplingSettings(LlmSamplingSettings):
    """
    TGIServerSamplingSettings dataclass
    """

    best_of: Optional[int] = None
    use_beam_search = False
    top_k: float = -1
    top_p: float = 1
    min_p: float = 0.0
    temperature: float = 0.7
    max_tokens: int = 16
    repetition_penalty: Optional[float] = 1.0
    length_penalty: Optional[float] = 1.0
    early_stopping: Optional[bool] = False
    ignore_eos: Optional[bool] = False
    min_tokens: Optional[int] = 0
    stop_token_ids: Optional[List[int]] = field(default_factory=list)
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    stream: bool = False

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.tgi_server

    def save(self, file_path: str):
        """
        Save the settings to a file.

        Args:
            file_path (str): The path to the file.
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.as_dict(), file, indent=4)

    @staticmethod
    def load_from_file(file_path: str) -> "VLLMServerSamplingSettings":
        """
        Load the settings from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            LlamaCppSamplingSettings: The loaded settings.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_settings = json.load(file)
            return VLLMServerSamplingSettings(**loaded_settings)

    @staticmethod
    def load_from_dict(settings: dict) -> "VLLMServerSamplingSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            LlamaCppSamplingSettings: The loaded settings.
        """
        return VLLMServerSamplingSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        return self.__dict__


class VLLMServerProvider(LlmProvider):

    def __init__(
            self,
            base_url: str,
            model: str,
            api_key: str = None
    ):
        from openai import OpenAI
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.tgi_server

    def get_provider_default_settings(self) -> VLLMServerSamplingSettings:
        return VLLMServerSamplingSettings()

    def create_completion(
            self,
            prompt: str,
            structured_output_settings: LlmStructuredOutputSettings,
            settings: VLLMServerSamplingSettings,
    ):
        grammar = None
        if structured_output_settings.output_type != LlmStructuredOutputType.no_structured_output:
            grammar = structured_output_settings.get_json_schema()
        top_p = settings.top_p
        stream = settings.stream
        temperature = settings.temperature
        max_tokens = settings.max_tokens

        settings_dict = copy(settings.as_dict())
        settings_dict.pop("top_p")
        settings_dict.pop("stream")
        settings_dict.pop("temperature")
        settings_dict.pop("max_tokens")
        if grammar is not None:
            settings_dict["guided_json"] = grammar

        if settings.stream:
            result = self.client.completions.create(prompt=prompt, model=self.model, extra_body=settings_dict,
                                                    top_p=top_p, stream=stream, temperature=temperature,
                                                    max_tokens=max_tokens)

            def generate_chunks():
                for chunk in result:
                    if chunk.choices[0].text is not None:
                        yield {"choices": [{"text": chunk.choices[0].text}]}

            return generate_chunks()
        else:

            result = self.client.completions.create(prompt=prompt, model=self.model, extra_body=settings_dict,
                                                    top_p=top_p, stream=stream, temperature=temperature,
                                                    max_tokens=max_tokens)
            return {"choices": [{"text": result.choices[0].text}]}

    def create_chat_completion(
            self,
            messages: List[Dict[str, str]],
            structured_output_settings: LlmStructuredOutputSettings,
            settings: VLLMServerSamplingSettings,
    ):
        grammar = None
        if structured_output_settings.output_type != LlmStructuredOutputType.no_structured_output:
            grammar = structured_output_settings.get_json_schema()

        top_p = settings.top_p
        stream = settings.stream
        temperature = settings.temperature
        max_tokens = settings.max_tokens

        settings_dict = copy(settings.as_dict())
        settings_dict.pop("top_p")
        settings_dict.pop("stream")
        settings_dict.pop("temperature")
        settings_dict.pop("max_tokens")
        if grammar is not None:
            settings_dict["guided_json"] = grammar

        if settings.stream:
            result = self.client.chat.completions.create(messages=messages, model=self.model, extra_body=settings_dict,
                                                         top_p=top_p, stream=stream, temperature=temperature,
                                                         max_tokens=max_tokens)

            def generate_chunks():
                for chunk in result:
                    if chunk.choices[0].delta.content is not None:
                        yield {"choices": [{"text": chunk.choices[0].delta.content}]}

            return generate_chunks()
        else:

            result = self.client.chat.completions.create(messages=messages, model=self.model, extra_body=settings_dict,
                                                         top_p=top_p, stream=stream, temperature=temperature,
                                                         max_tokens=max_tokens)
            return {"choices": [{"text": result.choices[0].message.content}]}

    def tokenize(self, prompt: str) -> list[int]:
        result = self.tokenize(prompt=prompt)
        return result

