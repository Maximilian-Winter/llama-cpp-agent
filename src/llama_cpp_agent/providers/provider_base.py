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
    def get_additional_stop_sequences(self) -> List[str]:
        pass

    @abstractmethod
    def add_additional_stop_sequences(self, sequences: List[str]):
        pass

    @abstractmethod
    def is_streaming(self):
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
