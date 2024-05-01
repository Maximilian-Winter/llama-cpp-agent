import json
from copy import copy
from typing import Type, Callable, Union

from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel

from .llm_agent import LlamaCppAgent, StreamingResponse
from .llm_prompt_template import PromptTemplate
from .llm_settings import LlamaLLMGenerationSettings, LlamaLLMSettings
from .output_parser import extract_object_from_response
from .messages_formatter import MessagesFormatterType, MessagesFormatter
from .gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import (
    generate_gbnf_grammar_and_documentation,
)
from .providers.llama_cpp_endpoint_provider import (
    LlamaCppEndpointSettings,
    LlamaCppGenerationSettings,
)
from .providers.openai_endpoint_provider import (
    OpenAIGenerationSettings,
    OpenAIEndpointSettings,
)


class StructuredOutputAgent:
    """
    An agent that creates structured output based on pydantic models from unstructured text.

    Args:
        llama_llm (Union[Llama, LlamaLLMSettings, LlamaCppEndpointSettings, OpenAIEndpointSettings]): An instance of Llama, LlamaLLMSettings, LlamaCppServerLLMSettings, OpenAIEndpointSettings as LLM.
        llama_generation_settings (Union[LlamaLLMGenerationSettings, LlamaCppGenerationSettings, OpenAIGenerationSettings]): Generation settings for Llama or LlamaCppServer.
        messages_formatter_type (MessagesFormatterType): Type of messages formatter.
        custom_messages_formatter (MessagesFormatter): Custom messages formatter.
        streaming_callback (Callable[[StreamingResponse], None]): Callback function for streaming responses.
        debug_output (bool): Enable debug output.

    Attributes:
        llama_generation_settings (Union[LlamaLLMGenerationSettings, LlamaCppServerGenerationSettings]): Generation settings for Llama or LlamaCppServer.
        grammar_cache (dict): Cache for generated grammars.
        system_prompt_template (PromptTemplate): Template for the system prompt.
        creation_prompt_template (PromptTemplate): Template for the creation prompt.
        llama_cpp_agent (LlamaCppAgent): LlamaCppAgent instance for interaction.
        streaming_callback (Callable[[StreamingResponse], None]): Callback function for streaming responses.

    Methods:
        save(file_path: str): Save the agent's state to a file.
        load_from_file(file_path: str, llama_llm, streaming_callback) -> StructuredOutputAgent: Load the agent's state from a file.
        load_from_dict(agent_dict: dict) -> StructuredOutputAgent: Load the agent's state from a dictionary.
        as_dict() -> dict: Convert the agent's state to a dictionary.
        create_object(model: Type[BaseModel], data: str = "") -> object: Create an object of the given model from the given data.

    """

    def __init__(
        self,
        llama_llm: Union[
            Llama, LlamaLLMSettings, LlamaCppEndpointSettings, OpenAIEndpointSettings
        ],
        llama_generation_settings: Union[
            LlamaLLMGenerationSettings,
            LlamaCppGenerationSettings,
            OpenAIGenerationSettings,
        ] = None,
        messages_formatter_type: MessagesFormatterType = MessagesFormatterType.CHATML,
        custom_messages_formatter: MessagesFormatter = None,
        streaming_callback: Callable[[StreamingResponse], None] = None,
        debug_output: bool = False,
    ):
        """
        Initialize the StructuredOutputAgent.

        Args:
            llama_llm (Union[Llama, LlamaLLMSettings, LlamaCppEndpointSettings, OpenAIEndpointSettings]): An instance of Llama, LlamaLLMSettings, or LlamaCppServerLLMSettings as LLM.
            llama_generation_settings (Union[LlamaLLMGenerationSettings, LlamaCppGenerationSettings, OpenAIGenerationSettings]): Generation settings for Llama or LlamaCppServer or OpenAIEndpoint.
            messages_formatter_type (MessagesFormatterType): Type of messages formatter.
            custom_messages_formatter (MessagesFormatter): Custom messages formatter.
            streaming_callback (Callable[[StreamingResponse], None]): Callback function for streaming responses.
            debug_output (bool): Enable debug output.
        """
        if llama_generation_settings is None:
            if isinstance(llama_llm, Llama) or isinstance(llama_llm, LlamaLLMSettings):
                llama_generation_settings = LlamaLLMGenerationSettings()
            elif isinstance(llama_llm, OpenAIEndpointSettings):
                llama_generation_settings = OpenAIGenerationSettings()
            else:
                llama_generation_settings = LlamaCppGenerationSettings()

        if isinstance(
            llama_generation_settings, LlamaLLMGenerationSettings
        ) and isinstance(llama_llm, LlamaCppEndpointSettings):
            raise Exception(
                "Wrong generation settings for llama.cpp server endpoint, use LlamaCppServerGenerationSettings under llama_cpp_agent.providers.llama_cpp_server_provider!"
            )
        if (
            isinstance(llama_llm, Llama) or isinstance(llama_llm, LlamaLLMSettings)
        ) and isinstance(llama_generation_settings, LlamaCppGenerationSettings):
            raise Exception(
                "Wrong generation settings for llama-cpp-python, use LlamaLLMGenerationSettings under llama_cpp_agent.llm_settings!"
            )

        if isinstance(llama_llm, OpenAIEndpointSettings) and not isinstance(
            llama_generation_settings, OpenAIGenerationSettings
        ):
            raise Exception(
                "Wrong generation settings for OpenAI endpoint, use CompletionRequestSettings under llama_cpp_agent.providers.openai_endpoint_provider!"
            )

        self.llama_generation_settings = llama_generation_settings
        self.grammar_cache = {}
        self.system_prompt_template = PromptTemplate.from_string(
            "You are an advanced AI agent. You are tasked to assist the user by creating structured output in JSON format.\n\n{documentation}"
        )
        self.creation_prompt_template = PromptTemplate.from_string(
            "Create an JSON response based on the following input.\n\nInput:\n\n{user_input}"
        )

        self.llama_cpp_agent = LlamaCppAgent(
            llama_llm,
            debug_output=debug_output,
            system_prompt="",
            predefined_messages_formatter_type=messages_formatter_type,
            custom_messages_formatter=custom_messages_formatter,
        )
        self.streaming_callback = streaming_callback

    def save(self, file_path: str):
        """
        Save the agent's state to a file.

        Args:
            file_path (str): The path to the file.
        """
        with open(file_path, "w", encoding="utf-8") as file:
            dic = copy(self.as_dict())
            del dic["llama_cpp_agent"]
            del dic["grammar_cache"]
            del dic["system_prompt_template"]
            del dic["creation_prompt_template"]
            del dic["streaming_callback"]
            dic["debug_output"] = self.llama_cpp_agent.debug_output
            dic["llama_generation_settings"] = self.llama_generation_settings.as_dict()
            dic[
                "custom_messages_formatter"
            ] = self.llama_cpp_agent.messages_formatter.as_dict()
            json.dump(dic, file, indent=4)

    @staticmethod
    def load_from_file(
        file_path: str,
        llama_llm: Union[Llama, LlamaLLMSettings],
        streaming_callback: Callable[[StreamingResponse], None] = None,
    ) -> "StructuredOutputAgent":
        """
        Load the agent's state from a file.

        Args:
            file_path (str): The path to the file.
            llama_llm (Union[Llama, LlamaLLMSettings, LlamaCppEndpointSettings]): An instance of Llama, LlamaLLMSettings, or LlamaCppServerLLMSettings as LLM.
            streaming_callback (Callable[[StreamingResponse], None]): Callback function for streaming responses.

        Returns:
            StructuredOutputAgent: The loaded StructuredOutputAgent instance.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_agent = json.load(file)
            loaded_agent["llama_llm"] = llama_llm
            loaded_agent["streaming_callback"] = streaming_callback
            loaded_agent[
                "llama_generation_settings"
            ] = LlamaLLMGenerationSettings.load_from_dict(
                loaded_agent["llama_generation_settings"]
            )
            loaded_agent[
                "custom_messages_formatter"
            ] = MessagesFormatter.load_from_dict(
                loaded_agent["custom_messages_formatter"]
            )
            return StructuredOutputAgent(**loaded_agent)

    @staticmethod
    def load_from_dict(agent_dict: dict) -> "StructuredOutputAgent":
        """
        Load the agent's state from a dictionary.

        Args:
            agent_dict (dict): The dictionary containing the agent's state.

        Returns:
            StructuredOutputAgent: The loaded StructuredOutputAgent instance.
        """
        return StructuredOutputAgent(**agent_dict)

    def as_dict(self) -> dict:
        """
        Convert the agent's state to a dictionary.

        Returns:
            dict: The dictionary representation of the agent's state.
        """
        return self.__dict__

    def create_object(self, model: Type[BaseModel], data: str = "") -> object:
        """
        Creates an object of the given model from the given data.

        Args:
            model (Type[BaseModel]): The model to create the object from.
            data (str): The data to create the object from.

        Returns:
            object: The created object.
        """
        if model not in self.grammar_cache:
            grammar, documentation = generate_gbnf_grammar_and_documentation(
                [model],
                model_prefix="Response Model",
                fields_prefix="Response Model Field",
            )

            self.grammar_cache[model] = grammar, documentation
        else:
            grammar, documentation = self.grammar_cache[model]

        system_prompt = self.system_prompt_template.generate_prompt(
            {"documentation": documentation}
        )
        if data == "":
            prompt = "Create a random JSON response based on the response model."
        else:
            prompt = self.creation_prompt_template.generate_prompt({"user_input": data})
        response = self.llama_cpp_agent.get_chat_response(
            prompt,
            system_prompt=system_prompt,
            grammar=grammar,
            add_response_to_chat_history=False,
            add_message_to_chat_history=False,
            streaming_callback=self.streaming_callback,
            **self.llama_generation_settings.as_dict(),
        )
        return extract_object_from_response(response, model)
