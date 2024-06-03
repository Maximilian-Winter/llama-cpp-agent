import json
from copy import copy
from typing import Type, Callable, Union

from llama_cpp import Llama
from pydantic import BaseModel

from .llm_agent import LlamaCppAgent, StreamingResponse
from .llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType
from .llm_prompt_template import PromptTemplate
from .output_parser import extract_object_from_response
from .messages_formatter import MessagesFormatterType, MessagesFormatter


from .providers.provider_base import LlmProvider, LlmSamplingSettings


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
        llama_llm: LlmProvider,
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

    def as_dict(self) -> dict:
        """
        Convert the agent's state to a dictionary.

        Returns:
            dict: The dictionary representation of the agent's state.
        """
        return self.__dict__

    def create_object(
        self,
        model: Type[BaseModel],
        data: str = "",
        llm_sampling_settings: LlmSamplingSettings = None,
        returns_streaming_generator: bool = False,
    ) -> object:
        """
        Creates an object of the given model from the given data.

        Args:
            model (Type[BaseModel]): The model to create the object from.
            data (str): The data to create the object from.

        Returns:
            object: The created object.
        """
        output_settings = LlmStructuredOutputSettings.from_pydantic_models(
            [model], output_type=LlmStructuredOutputType.object_instance
        )

        system_prompt = self.system_prompt_template.generate_prompt(
            {
                "documentation": output_settings.get_llm_documentation(
                    self.llama_cpp_agent.provider
                ).strip()
            }
        )
        if data == "":
            prompt = "Create a random JSON response based on the response model."
        else:
            prompt = self.creation_prompt_template.generate_prompt({"user_input": data})
        response = self.llama_cpp_agent.get_chat_response(
            prompt,
            system_prompt=system_prompt,
            returns_streaming_generator=returns_streaming_generator,
            add_response_to_chat_history=False,
            add_message_to_chat_history=False,
            streaming_callback=self.streaming_callback,
            structured_output_settings=output_settings,
            llm_sampling_settings=llm_sampling_settings,
        )
        return response
