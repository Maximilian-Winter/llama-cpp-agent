import json
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Union

from llama_cpp import Llama, LlamaGrammar

from llama_cpp_agent.llm_output_settings import (
    LlmStructuredOutputSettings,
    LlmStructuredOutputType,
)
from llama_cpp_agent.providers.provider_base import (
    LlmProvider,
    LlmProviderId,
    LlmSamplingSettings,
)


@dataclass
class LlamaCppPythonSamplingSettings(LlmSamplingSettings):
    """
    Settings for generating completions using the Llama.cpp server.

    Args:
        temperature (float): Controls the randomness of the generated completions. Higher values make the output more random.
        top_k (int): Controls the diversity of the top-k sampling. Higher values result in more diverse completions.
        top_p (float): Controls the diversity of the nucleus sampling. Higher values result in more diverse completions.
        min_p (float): Minimum probability for nucleus sampling. Lower values result in more focused completions.
        max_tokens (int): Number of max tokens to generate.
        stream (bool): Enable streaming for long completions.
        additional_stop_sequences (List[str]): List of stop sequences to finish completion generation. The official stop sequences of the model get added automatically.
        tfs_z (float): Controls the temperature for top frequent sampling.
        typical_p (float): Typical probability for top frequent sampling.
        repeat_penalty (float): Penalty for repeating tokens in completions.
        presence_penalty (float): Penalty for presence of certain tokens.
        frequency_penalty (float): Penalty based on token frequency.
        mirostat_mode (int): Mirostat level.
        mirostat_tau (float): Mirostat temperature.
        mirostat_eta (float): Mirostat eta parameter.
        seed (int): Seed for randomness. Set to -1 for no seed.


    Attributes:
        temperature (float): Controls the randomness of the generated completions. Higher values make the output more random.
        top_k (int): Controls the diversity of the top-k sampling. Higher values result in more diverse completions.
        top_p (float): Controls the diversity of the nucleus sampling. Higher values result in more diverse completions.
        min_p (float): Minimum probability for nucleus sampling. Lower values result in more focused completions.
        max_tokens (int): Number of max tokens to generate.
        stream (bool): Enable streaming for long completions.
        additional_stop_sequences (List[str]): List of stop sequences to finish completion generation. The official stop sequences of the model get added automatically.
        tfs_z (float): Controls the temperature for top frequent sampling.
        typical_p (float): Typical probability for top frequent sampling.
        repeat_penalty (float): Penalty for repeating tokens in completions.
        presence_penalty (float): Penalty for presence of certain tokens.
        frequency_penalty (float): Penalty based on token frequency.
        mirostat_mode (int): Mirostat level.
        mirostat_tau (float): Mirostat temperature.
        mirostat_eta (float): Mirostat eta parameter.
        seed (int): Seed for randomness. Set to -1 for no seed.
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
    max_tokens: int = -1
    stream: bool = False
    additional_stop_sequences: List[str] = None
    tfs_z: float = 1.0
    typical_p: float = 1.0
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    mirostat_mode: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    seed: int = -1

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
    def load_from_dict(settings: dict) -> "LlamaCppPythonSamplingSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            LlamaCppPythonSamplingSettings: The loaded settings.
        """
        return LlamaCppPythonSamplingSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        return self.__dict__


class LlamaCppPythonProvider(LlmProvider):
    def __init__(self, llama_model: Llama):
        self.llama_model = llama_model
        self.grammar_cache = {}

    def is_using_json_schema_constraints(self):
        return False

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.llama_cpp_python

    def get_provider_default_settings(self) -> LlamaCppPythonSamplingSettings:
        return LlamaCppPythonSamplingSettings()

    def create_completion(
        self,
        prompt: str,
        structured_output_settings: LlmStructuredOutputSettings,
        settings: LlamaCppPythonSamplingSettings,
        bos_token: str,
    ):
        grammar = None
        if (
            structured_output_settings.output_type
            != LlmStructuredOutputType.no_structured_output
        ):
            grammar = structured_output_settings.get_gbnf_grammar()
            if grammar in self.grammar_cache:
                grammar = self.grammar_cache[grammar]
            else:
                self.grammar_cache[grammar] = LlamaGrammar.from_string(grammar)
                grammar = self.grammar_cache[grammar]

        settings_dictionary = deepcopy(settings.as_dict())

        settings_dictionary["stop"] = settings_dictionary.pop(
            "additional_stop_sequences"
        )

        return self.llama_model.create_completion(
            prompt, grammar=grammar, **settings_dictionary
        )

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        structured_output_settings: LlmStructuredOutputSettings,
        settings: LlamaCppPythonSamplingSettings
    ):
        grammar = None
        if (
            structured_output_settings.output_type
            != LlmStructuredOutputType.no_structured_output
        ):
            grammar = structured_output_settings.get_gbnf_grammar()
            if grammar in self.grammar_cache:
                grammar = self.grammar_cache[grammar]
            else:
                self.grammar_cache[grammar] = LlamaGrammar.from_string(grammar)
                grammar = self.grammar_cache[grammar]
        settings_dictionary = deepcopy(settings.as_dict())
        settings_dictionary["max_tokens"] = settings_dictionary.pop("n_predict")
        settings_dictionary["stop"] = settings_dictionary.pop("stop_sequences")

        return self.llama_model.create_chat_completion(
            messages, grammar=grammar, **settings_dictionary
        )

    def tokenize(self, prompt: str) -> list[int]:
        return self.llama_model.tokenize(prompt.encode("utf-8"))
