import json
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Union

import requests

from llama_cpp_agent.llm_output_settings import (
    LlmStructuredOutputType,
    LlmStructuredOutputSettings,
)
from llama_cpp_agent.providers.provider_base import (
    LlmProviderId,
    LlmProvider,
    LlmSamplingSettings,
)


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
        additional_stop_sequences (List[str]): List of stop sequences to finish completion generation. The official stop sequences of the model get added automatically.
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
        additional_stop_sequences (List[str]): List of stop sequences to finish completion generation. The official stop sequences of the model get added automatically.
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
    stream: bool = True
    additional_stop_sequences: List[str] = None
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

    def get_additional_stop_sequences(self) -> List[str]:
        if self.additional_stop_sequences is None:
            self.additional_stop_sequences = []
        return self.additional_stop_sequences

    def add_additional_stop_sequences(self, sequences: List[str]):
        if self.additional_stop_sequences is None:
            self.additional_stop_sequences = []
        self.additional_stop_sequences.extend(sequences)

    def is_streaming(self):
        return self.stream

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

    def is_using_json_schema_constraints(self):
        return False

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.llama_cpp_server

    def get_provider_default_settings(self) -> LlamaCppSamplingSettings:
        return LlamaCppSamplingSettings()

    def create_completion(
        self,
        prompt: str,
        structured_output_settings: LlmStructuredOutputSettings,
        settings: LlamaCppSamplingSettings,
        bos_token: str,
    ):
        if self.api_key is not None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",  # Add API key as bearer token
            }
        else:
            headers = {"Content-Type": "application/json"}

        data = deepcopy(settings.as_dict())
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
        settings: LlamaCppSamplingSettings
    ):
        if self.api_key is not None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",  # Add API key as bearer token
            }
        else:
            headers = {"Content-Type": "application/json"}

        data = deepcopy(settings.as_dict())
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
                                returned_data = {
                                    "choices": [{"text": new_data["content"]}]
                                }
                            else:
                                returned_data = new_data
                        yield returned_data
                        decoded_chunk = ""

            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")

        return generate_text_chunks()

    def prepare_generation_settings(
        self,
        settings_dictionary: dict,
        structured_output_settings: LlmStructuredOutputSettings,
    ) -> dict:
        if (
            structured_output_settings.output_type
            != LlmStructuredOutputType.no_structured_output
        ):
            settings_dictionary[
                "grammar"
            ] = structured_output_settings.get_gbnf_grammar()
        if not self.llama_cpp_python_server:
            settings_dictionary["mirostat"] = settings_dictionary.pop("mirostat_mode")
        if self.llama_cpp_python_server:
            settings_dictionary["max_tokens"] = settings_dictionary.pop("n_predict")

        settings_dictionary["stop"] = settings_dictionary.pop(
            "additional_stop_sequences"
        )
        if not self.llama_cpp_python_server:
            if (
                "samplers" not in settings_dictionary
                or settings_dictionary["samplers"] is None
            ):
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
