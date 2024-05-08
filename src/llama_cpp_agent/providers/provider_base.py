import json
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Union, Optional, Any, Callable, Tuple

import requests
from pydantic import BaseModel, Field

from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import \
    generate_gbnf_grammar_from_pydantic_models
from llama_cpp_agent.json_schema_generator.schema_generator import generate_json_schemas
from llama_cpp_agent.pydantic_model_documentation.documentation_context import generate_text_documentation


class LlmOutputType(Enum):
    no_structured_output = "no_structured_output"
    object_instance = "object_instance"
    list_of_objects = "list_of_objects"
    function_call = "function_call"
    parallel_function_call = "parallel_function_call"


class StructuredOutputSettings(BaseModel):
    """
    Llm settings for structured output, like function calling and object creation.
    """
    output_type: Optional[LlmOutputType] = Field(LlmOutputType.no_structured_output,
                                                 description="The output type of the llm")
    function_tools: Optional[List[LlamaCppFunctionTool]] = Field(None,
                                                                 description="List of functions tools for function calling")
    pydantic_models: Optional[List[BaseModel]] = Field(None,
                                                       description="List of pydantic models for structured output")

    @staticmethod
    def from_llama_cpp_function_tools(llama_cpp_function_tools: List[LlamaCppFunctionTool], output_type: LlmOutputType):
        return StructuredOutputSettings(output_type=output_type, function_tools=llama_cpp_function_tools)

    @staticmethod
    def from_pydantic_models(models: List[BaseModel], output_type: LlmOutputType):
        if output_type is LlmOutputType.no_structured_output:
            raise NotImplementedError("LlmOutputType: no_structured_output not supported for structured output and function calling!")
        elif output_type is LlmOutputType.object_instance:
            return StructuredOutputSettings(output_type=LlmOutputType.object_instance, pydantic_models=models)
        elif output_type is LlmOutputType.list_of_objects:
            return StructuredOutputSettings(output_type=LlmOutputType.list_of_objects, pydantic_models=models)
        elif output_type is LlmOutputType.function_call:
            return StructuredOutputSettings(output_type=LlmOutputType.function_call,
                                            function_tools=[LlamaCppFunctionTool(model) for model in models])
        elif output_type is LlmOutputType.parallel_function_call:
            return StructuredOutputSettings(output_type=LlmOutputType.parallel_function_call,
                                            function_tools=[LlamaCppFunctionTool(model) for model in models])

    @staticmethod
    def from_open_ai_tools(tools: List[Tuple[Dict[str, Any], Callable]], output_type: LlmOutputType):
        if output_type is LlmOutputType.no_structured_output:
            raise NotImplementedError("LlmOutputType: no_structured_output not supported for structured output and function calling!")
        elif output_type is LlmOutputType.function_call:
            return StructuredOutputSettings(output_type=LlmOutputType.function_call,
                                            function_tools=[LlamaCppFunctionTool(model) for model in tools])
        elif output_type is LlmOutputType.parallel_function_call:
            return StructuredOutputSettings(output_type=LlmOutputType.parallel_function_call,
                                            function_tools=[LlamaCppFunctionTool(model) for model in tools])
        else:
            raise NotImplementedError(f"LlmOutputType: {output_type.value} not supported for tools!")

    @staticmethod
    def from_functions(tools: List[Callable], output_type: LlmOutputType):
        if output_type is LlmOutputType.no_structured_output:
            raise NotImplementedError("LlmOutputType: no_structured_output not supported for structured output and function calling!")
        elif output_type is LlmOutputType.function_call:
            return StructuredOutputSettings(output_type=LlmOutputType.function_call,
                                            function_tools=[LlamaCppFunctionTool(model) for model in tools])
        elif output_type is LlmOutputType.parallel_function_call:
            return StructuredOutputSettings(output_type=LlmOutputType.parallel_function_call,
                                            function_tools=[LlamaCppFunctionTool(model) for model in tools])
        else:
            raise NotImplementedError(f"LlmOutputType: {output_type.value} not supported for tools!")

    @staticmethod
    def from_llama_index_tools(tools: list, output_type: LlmOutputType):
        if output_type is LlmOutputType.no_structured_output:
            raise NotImplementedError("LlmOutputType: no_structured_output not supported for structured output and function calling!")
        elif output_type is LlmOutputType.function_call:
            return StructuredOutputSettings(output_type=LlmOutputType.function_call,
                                            function_tools=[LlamaCppFunctionTool.from_llama_index_tool(model) for model
                                                            in tools])
        elif output_type is LlmOutputType.parallel_function_call:
            return StructuredOutputSettings(output_type=LlmOutputType.parallel_function_call,
                                            function_tools=[LlamaCppFunctionTool.from_llama_index_tool(model) for model
                                                            in tools])
        else:
            raise NotImplementedError(f"LlmOutputType: {output_type.value} not supported for tools!")

    def add_llama_cpp_function_tool(self, tool: LlamaCppFunctionTool):
        self.function_tools.append(tool)

    def add_pydantic_model(self, model: BaseModel):
        if self.output_type is LlmOutputType.no_structured_output:
            raise NotImplementedError("LlmOutputType: no_structured_output not supported for structured output and function calling!")
        elif self.output_type is LlmOutputType.object_instance:
            self.pydantic_models.append(model)
        elif self.output_type is LlmOutputType.list_of_objects:
            self.pydantic_models.append(model)
        elif self.output_type is LlmOutputType.function_call:
            self.function_tools.append(LlamaCppFunctionTool(model))
        elif self.output_type is LlmOutputType.parallel_function_call:
            self.function_tools.append(LlamaCppFunctionTool(model))

    def add_open_ai_tool(self, open_ai_schema_and_function: Tuple[Dict[str, Any], Callable]):
        if self.output_type is LlmOutputType.no_structured_output:
            raise NotImplementedError("LlmOutputType: no_structured_output not supported for structured output and function calling!")
        elif self.output_type is LlmOutputType.function_call:
            self.function_tools.append(LlamaCppFunctionTool(open_ai_schema_and_function))
        elif self.output_type is LlmOutputType.parallel_function_call:
            self.function_tools.append(LlamaCppFunctionTool(open_ai_schema_and_function))
        else:
            raise NotImplementedError(f"LlmOutputType: {self.output_type.value} not supported for tools!")

    def add_function_tool(self, function: Callable):
        if self.output_type is LlmOutputType.no_structured_output:
            raise NotImplementedError("LlmOutputType: no_structured_output not supported for structured output and function calling!")
        elif self.output_type is LlmOutputType.function_call:
            self.function_tools.append(LlamaCppFunctionTool(function))
        elif self.output_type is LlmOutputType.parallel_function_call:
            self.function_tools.append(LlamaCppFunctionTool(function))
        else:
            raise NotImplementedError(f"LlmOutputType: {self.output_type.value} not supported for tools!")

    def add_llama_index_tool(self, tool):
        if self.output_type is LlmOutputType.no_structured_output:
            raise NotImplementedError("LlmOutputType: no_structured_output not supported for structured output and function calling!")
        elif self.output_type is LlmOutputType.function_call:
            self.function_tools.append(LlamaCppFunctionTool.from_llama_index_tool(tool))
        elif self.output_type is LlmOutputType.parallel_function_call:
            self.function_tools.append(LlamaCppFunctionTool.from_llama_index_tool(tool))
        else:
            raise NotImplementedError(f"LlmOutputType: {self.output_type.value} not supported for tools!")

    def get_llm_documentation(self):
        if self.output_type == LlmOutputType.no_structured_output:
            raise NotImplementedError("LlmOutputType: no_structured_output not supported for structured output and function calling!")
        elif self.output_type == LlmOutputType.object_instance:
            return generate_text_documentation(self.pydantic_models)
        elif self.output_type == LlmOutputType.list_of_objects:
            return generate_text_documentation(self.pydantic_models)
        elif self.output_type == LlmOutputType.function_call:
            return generate_text_documentation([tool.model for tool in self.function_tools], model_prefix="Function",
                                               fields_prefix="Parameters")
        elif self.output_type == LlmOutputType.parallel_function_call:
            return generate_text_documentation([tool.model for tool in self.function_tools], model_prefix="Function",
                                               fields_prefix="Parameters")

    def get_gbnf_grammar(self, add_inner_thoughts: bool = False,
                         allow_only_inner_thoughts: bool = False, add_request_heartbeat: bool = False):
        if self.output_type == LlmOutputType.no_structured_output:
            raise NotImplementedError("LlmOutputType: no_structured_output not supported for structured output and function calling!")
        elif self.output_type == LlmOutputType.object_instance:
            return generate_gbnf_grammar_from_pydantic_models(self.pydantic_models, list_of_outputs=False,
                                                              add_inner_thoughts=add_inner_thoughts,
                                                              allow_only_inner_thoughts=allow_only_inner_thoughts,
                                                              add_request_heartbeat=add_request_heartbeat)
        elif self.output_type == LlmOutputType.list_of_objects:
            return generate_gbnf_grammar_from_pydantic_models(self.pydantic_models, list_of_outputs=True,
                                                              add_inner_thoughts=add_inner_thoughts,
                                                              allow_only_inner_thoughts=allow_only_inner_thoughts,
                                                              add_request_heartbeat=add_request_heartbeat)
        elif self.output_type == LlmOutputType.function_call:
            return generate_gbnf_grammar_from_pydantic_models([tool.model for tool in self.function_tools],
                                                              list_of_outputs=False,
                                                              add_inner_thoughts=add_inner_thoughts,
                                                              allow_only_inner_thoughts=allow_only_inner_thoughts,
                                                              add_request_heartbeat=add_request_heartbeat)
        elif self.output_type == LlmOutputType.parallel_function_call:
            return generate_gbnf_grammar_from_pydantic_models([tool.model for tool in self.function_tools],
                                                              list_of_outputs=True,
                                                              add_inner_thoughts=add_inner_thoughts,
                                                              allow_only_inner_thoughts=allow_only_inner_thoughts,
                                                              add_request_heartbeat=add_request_heartbeat)

    def get_json_schema(self):
        if self.output_type == LlmOutputType.no_structured_output:
            raise NotImplementedError("LlmOutputType: no_structured_output not supported for structured output and function calling!")
        elif self.output_type == LlmOutputType.object_instance:
            return generate_json_schemas(self.pydantic_models, allow_list=False, outer_object_name="model",
                                         outer_object_properties_name="fields")
        elif self.output_type == LlmOutputType.list_of_objects:
            return generate_json_schemas(self.pydantic_models, allow_list=True, outer_object_name="model",
                                         outer_object_properties_name="fields")
        elif self.output_type == LlmOutputType.function_call:
            return generate_json_schemas([tool.model for tool in self.function_tools],
                                         allow_list=False, outer_object_name="function",
                                         outer_object_properties_name="arguments")
        elif self.output_type == LlmOutputType.parallel_function_call:
            return generate_json_schemas([tool.model for tool in self.function_tools],
                                         allow_list=True, outer_object_name="function",
                                         outer_object_properties_name="arguments")


class LLMGenerationSettings(ABC):

    @abstractmethod
    def get_provider_identifier(self) -> str:
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
    def load_from_file(file_path: str) -> "LLMGenerationSettings":
        """
        Load the settings from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            LLMGenerationSettings: The loaded settings.
        """
        pass

    @staticmethod
    @abstractmethod
    def load_from_dict(settings: dict) -> "LLMGenerationSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            LLMGenerationSettings: The loaded settings.
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


class LLMProviderBase(ABC):
    """
    Abstract base class for all LLM providers.
    """

    @abstractmethod
    def get_provider_identifier(self) -> str:
        pass

    @abstractmethod
    def create_completion(
            self, prompt: str, structured_output_settings: StructuredOutputSettings, settings: LLMGenerationSettings
    ):
        """Create a completion request with the LLM provider and returns the result."""
        pass

    @abstractmethod
    def create_chat_completion(
            self,
            messages: List[Dict[str, str]],
            structured_output_settings: StructuredOutputSettings,
            settings: LLMGenerationSettings,
    ):
        """Create a chat completion request with the LLM provider and returns the result."""
        pass

    @abstractmethod
    def tokenize(self, prompt: str):
        """Tokenize the given prompt."""
        pass


@dataclass
class LlamaCppGenerationSettings(LLMGenerationSettings):
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


class LlamaCppServerProvider(LLMProviderBase):
    def get_provider_identifier(self) -> str:
        return """llama_cpp"""

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

    def get_default_generation_settings(self) -> LLMGenerationSettings:
        default_settings = LLMGenerationSettings()
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
        else:
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
                "ignore_eos": False,
            }
        # Populate the LLMGenerationSettings instance
        for key, value in dic.items():
            default_settings.set_setting(key, value)

        return default_settings

    def create_completion(
            self, prompt: str, structured_output_settings: StructuredOutputSettings,
            settings: LlamaCppGenerationSettings
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
