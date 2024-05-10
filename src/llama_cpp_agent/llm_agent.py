import json
import re
from copy import copy
from dataclasses import dataclass
from typing import List, Dict, Literal, Callable, Union, Generator

from pydantic import BaseModel

from .chat_history.basic_chat_history import BasicChatHistory
from .chat_history.chat_history_base import ChatHistory

from .llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType

from .messages_formatter import (
    MessagesFormatterType,
    get_predefined_messages_formatter,
    MessagesFormatter,
)
from .function_calling import LlamaCppFunctionTool, LlamaCppFunctionToolRegistry
from .output_parser import parse_json_response

from .providers.provider_base import LlmProvider, LlmSamplingSettings


@dataclass
class StreamingResponse:
    """
    Represents a streaming response with text and an indicator for the last response.
    """

    text: str
    is_last_response: bool

    def __init__(self, text: str, is_last_response: bool):
        """
        Initializes a new StreamingResponse object.

        Args:
            text (str): The text content of the streaming response.
            is_last_response (bool): Indicates whether this is the last response in the stream.
        """
        self.text = text
        self.is_last_response = is_last_response


class LlamaCppAgent:
    """
    A base agent that can be used for chat, structured output and function calling.
     Is used as part of all other agents.
    """

    def __init__(
            self,
            provider: LlmProvider,
            name: str = "llamacpp_agent",
            system_prompt: str = "You are a helpful assistant.",
            predefined_messages_formatter_type: MessagesFormatterType = MessagesFormatterType.CHATML,
            custom_messages_formatter: MessagesFormatter = None,
            chat_history: ChatHistory = None,
            debug_output: bool = False,
    ):
        """
        Initializes a new LlamaCppAgent object.

        Args:
           provider (LlmProvider):The underlying llm provider (LlamaCppServerProvider, LlamaCppPythonProvider, TGIServerProvider or VLLMServerProvider).
           name (str): The name of the agent.
           system_prompt (str): The system prompt used in chat interactions.
           predefined_messages_formatter_type (MessagesFormatterType): The type of predefined messages formatter.
           custom_messages_formatter (MessagesFormatter): Custom message's formatter.
           chat_history (ChatHistory): This will handle the chat history.
           debug_output (bool): Indicates whether debug output should be enabled.
        """
        self.provider = provider
        self.name = name
        self.system_prompt = system_prompt
        self.debug_output = debug_output
        self.messages = []
        self.grammar_cache = {}
        if custom_messages_formatter is not None:
            self.messages_formatter = custom_messages_formatter
        else:
            self.messages_formatter = get_predefined_messages_formatter(
                predefined_messages_formatter_type
            )
        self.last_response = ""
        if chat_history is None:
            self.chat_history = BasicChatHistory()
        else:
            self.chat_history = chat_history

    def add_message(
            self,
            message: str,
            role: (
                    Literal["system"]
                    | Literal["user"]
                    | Literal["assistant"]
                    | Literal["function"]
            ) = "user",
    ):
        """
        Adds a message to the chat history.

        Args:
            message (str): The content of the message.
            role (Literal["system"] | Literal["user"] | Literal["assistant"]): The role of the message sender.
        """
        self.messages.append(
            {
                "role": role,
                "content": message,
            },
        )

    def get_text_response(
            self,
            prompt: str | list[int] = None,
            structured_output_settings: LlmStructuredOutputSettings = None,
            llm_samplings_settings: LlmSamplingSettings = None,
            streaming_callback: Callable[[StreamingResponse], None] = None,
            print_output: bool = False,
    ) -> Union[str, List[dict], BaseModel]:
        """
        Get a text response from the LLM provider.

        Args:
            prompt (str | list[int]): The prompt or tokenized prompt for the LLM.
            structured_output_settings (LlmStructuredOutputSettings): Settings for structured output.
            llm_samplings_settings (LlmSamplingSettings): Sampling settings for the LLM.
            streaming_callback (Callable[[StreamingResponse], None]): Callback for streaming responses.
            print_output (bool): Whether to print the output.

        Returns:
            Union[str, List[dict], BaseModel]: The generated text response.
        """

        if self.debug_output:
            if type(prompt) is str:
                print(prompt, end="")

        if structured_output_settings is None:
            structured_output_settings = LlmStructuredOutputSettings(
                output_type=LlmStructuredOutputType.no_structured_output)
        if llm_samplings_settings is None:
            llm_samplings_settings = self.provider.get_provider_default_settings()
        else:
            llm_samplings_settings = copy(llm_samplings_settings)

        if llm_samplings_settings.get_additional_stop_sequences() is not None:
            llm_samplings_settings.add_additional_stop_sequences(self.messages_formatter.DEFAULT_STOP_SEQUENCES)

        if self.provider:
            completion = self.get_text_completion(
                prompt=prompt, structured_output_settings=structured_output_settings,
                llm_samplings_settings=llm_samplings_settings
            )
            if llm_samplings_settings.is_streaming():
                full_response = ""
                for out in completion:
                    text = out["choices"][0]["text"]
                    full_response += text
                    if streaming_callback is not None:
                        streaming_callback(
                            StreamingResponse(text=text, is_last_response=False)
                        )
                    if print_output:
                        print(text, end="")
                if streaming_callback is not None:
                    streaming_callback(
                        StreamingResponse(text="", is_last_response=True)
                    )
                if print_output:
                    print("")
                self.last_response = full_response
                return self.handle_structured_output(full_response, structured_output_settings, llm_samplings_settings)
            else:
                full_response = ""
                text = completion["choices"][0]["text"]
                full_response += text
                if print_output:
                    print(full_response)
                self.last_response = full_response
                return self.handle_structured_output(full_response, structured_output_settings, llm_samplings_settings)
        return "Error: No model loaded!"

    def get_chat_response(
            self,
            message: str = None,
            role: Literal["system", "user", "assistant", "function"] = "user",
            response_role: Literal["user", "assistant"] | None = None,
            system_prompt: str = None,
            prompt_suffix: str = None,
            add_message_to_chat_history: bool = True,
            add_response_to_chat_history: bool = True,
            structured_output_settings: LlmStructuredOutputSettings = None,
            llm_samplings_settings: LlmSamplingSettings = None,
            streaming_callback: Callable[[StreamingResponse], None] = None,
            print_output: bool = False,
            k_last_messages: int = 0,

    ) -> Union[str, List[dict], BaseModel]:
        """
        Get a chat response based on the input message and context.

        Args:
            message (str): The input message.
            role (Literal["system", "user", "assistant", "function"]): The role of the message sender.
            response_role (Literal["user", "assistant"]): The role of the message response.
            system_prompt (str): The system prompt used in chat interactions.
            prompt_suffix (str): Suffix to append after the prompt.
            add_message_to_chat_history (bool): Whether to add the input message to the chat history.
            add_response_to_chat_history (bool): Whether to add the generated response to the chat history.
            structured_output_settings (LlmStructuredOutputSettings): Settings for structured output.
            llm_samplings_settings (LlmSamplingSettings): Sampling settings for the LLM.
            streaming_callback (Callable[[StreamingResponse], None]): Callback for streaming responses.
            print_output (bool): Whether to print the generated response.
            k_last_messages (int): Number of last messages to consider from the chat history.

        Returns:
            Union[str, List[dict], BaseModel]: The generated chat response.
        """
        if structured_output_settings is None:
            structured_output_settings = LlmStructuredOutputSettings(
                output_type=LlmStructuredOutputType.no_structured_output)
        if llm_samplings_settings is None:
            llm_samplings_settings = self.provider.get_provider_default_settings()
        else:
            llm_samplings_settings = copy(llm_samplings_settings)

        completion, response_role = self.get_response_role_and_completion(
            system_prompt=system_prompt,
            message=message,
            add_message_to_chat_history=add_message_to_chat_history,
            role=role,
            response_role=response_role,
            prompt_suffix=prompt_suffix,
            k_last_messages=k_last_messages,
            structured_output_settings=structured_output_settings,
            llm_samplings_settings=llm_samplings_settings
        )
        if self.provider:
            if llm_samplings_settings.is_streaming():
                full_response = ""
                for out in completion:
                    text = out["choices"][0]["text"]
                    full_response += text
                    if streaming_callback is not None:
                        streaming_callback(
                            StreamingResponse(text=text, is_last_response=False)
                        )
                    if print_output:
                        print(text, end="")
                if streaming_callback is not None:
                    streaming_callback(
                        StreamingResponse(text="", is_last_response=True)
                    )
                if print_output:
                    print("")

                self.last_response = full_response
                if add_response_to_chat_history:
                    self.messages.append(
                        {
                            "role": response_role,
                            "content": full_response,
                        },
                    )
                return self.handle_structured_output(full_response, structured_output_settings, llm_samplings_settings)
            else:
                text = completion["choices"][0]["text"]
                if print_output:
                    print(text)
                self.last_response = text
                if add_response_to_chat_history:
                    self.messages.append(
                        {
                            "role": response_role,
                            "content": text,
                        },
                    )
                return self.handle_structured_output(text, structured_output_settings, llm_samplings_settings)
        return "Error: No model loaded!"

    def get_text_completion(
            self,
            prompt: str | list[int] = None,
            structured_output_settings: LlmStructuredOutputSettings = None,
            llm_samplings_settings: LlmSamplingSettings = None
    ):
        if structured_output_settings is None:
            structured_output_settings = LlmStructuredOutputSettings(
                output_type=LlmStructuredOutputType.no_structured_output)
        if llm_samplings_settings is None:
            llm_samplings_settings = self.provider.get_provider_default_settings()
        else:
            llm_samplings_settings = copy(llm_samplings_settings)
        return self.provider.create_completion(prompt, structured_output_settings, llm_samplings_settings, self.messages_formatter.BOS_TOKEN)

    def get_response_role_and_completion(
            self,
            system_prompt: str = None,
            message: str = None,
            add_message_to_chat_history: bool = True,
            role: Literal["system", "user", "assistant", "function"] = "user",
            response_role: Literal["user", "assistant"] | None = None,
            prompt_suffix: str = None,
            llm_samplings_settings: LlmSamplingSettings = None,
            structured_output_settings: LlmStructuredOutputSettings = None,
            k_last_messages: int = 0,
    ):

        if system_prompt is None:
            system_prompt = self.system_prompt
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        if message is not None and add_message_to_chat_history:
            self.messages.append(
                {
                    "role": role,
                    "content": message,
                },
            )
        if not add_message_to_chat_history and message is not None:
            messages.append(
                {
                    "role": role,
                    "content": message,
                },
            )
        if k_last_messages > 0:
            messages.extend(self.messages[-k_last_messages:])
        else:
            messages.extend(self.messages)

        prompt, response_role = self.messages_formatter.format_messages(
            messages, response_role
        )

        if prompt_suffix:
            prompt += prompt_suffix

        if self.debug_output:
            print(prompt, end="")

        if structured_output_settings is None:
            structured_output_settings = LlmStructuredOutputSettings(
                output_type=LlmStructuredOutputType.no_structured_output)
        if llm_samplings_settings is None:
            llm_samplings_settings = self.provider.get_provider_default_settings()
        else:
            llm_samplings_settings = copy(llm_samplings_settings)

        if llm_samplings_settings.get_additional_stop_sequences() is not None:
            llm_samplings_settings.add_additional_stop_sequences(self.messages_formatter.DEFAULT_STOP_SEQUENCES)

        return self.provider.create_completion(prompt, structured_output_settings,
                                               llm_samplings_settings, self.messages_formatter.BOS_TOKEN), response_role

    def handle_structured_output(self, llm_output: str, structured_output_settings: LlmStructuredOutputSettings,
                                 llm_sampling_settings: LlmSamplingSettings):
        if structured_output_settings.output_type is LlmStructuredOutputType.function_calling or structured_output_settings.output_type is LlmStructuredOutputType.parallel_function_calling:

            function_tool_registry = self.get_function_tool_registry(structured_output_settings.function_tools,
                                                                     inner_thoughts_field_name=structured_output_settings.thoughts_and_reasoning_field_name,
                                                                     allow_parallel_function_calling=True if structured_output_settings.output_type is LlmStructuredOutputType.parallel_function_calling else False,
                                                                     tool_root=structured_output_settings.function_calling_name_field_name,
                                                                     tool_rule_content=structured_output_settings.function_calling_content,
                                                                     add_inner_thoughts=structured_output_settings.add_thoughts_and_reasoning_field)
            output = parse_json_response(llm_output)
            output = self.clean_keys(output)

            return function_tool_registry.handle_function_call(
                output
            )
        elif structured_output_settings.output_type == LlmStructuredOutputType.object_instance:
            output = json.loads(llm_output)
            output = self.clean_keys(output)
            model_name = output[structured_output_settings.output_model_name_field_name]
            model_attributes = output[structured_output_settings.output_model_attributes_field_name]
            for model in structured_output_settings.pydantic_models:
                if model_name == model.__name__:
                    return model(**model_attributes)

        elif structured_output_settings.output_type == LlmStructuredOutputType.list_of_objects:
            output = json.loads(llm_output)
            output = self.clean_keys(output)
            models = []
            for out in output:
                for model in structured_output_settings.pydantic_models:
                    model_name = out[structured_output_settings.output_model_name_field_name]
                    model_attributes = out[structured_output_settings.output_model_attributes_field_name]
                    if model_name == model.__name__:
                        models.append(model(**model_attributes))
            return models
        return llm_output

    def remove_last_k_chat_messages(self, k: int):
        """
        Removes the last k messages from the chat history.

        Args:
            k (int): Number of last messages to remove.
        """
        # Ensure k is not greater than the length of the messages list
        k = min(k, len(self.messages))

        # Remove the last k elements
        self.messages = self.messages[:-k] if k > 0 else self.messages

    def remove_first_k_chat_messages(self, k: int):
        """
        Removes the first k messages from the chat history.

        Args:
            k (int): Number of first messages to remove.
        """
        # Ensure k is not greater than the length of the messages list
        k = min(k, len(self.messages))

        # Remove the first k elements
        self.messages = self.messages[k:] if k > 0 else self.messages

    def save_messages(self, file_path: str):
        """
        Saves the chat history messages to a file.

        Args:
           file_path (str): The file path to save the messages.
        """
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.messages, file, indent=4)

    def load_messages(self, file_path: str):
        """
        Loads chat history messages from a file.

        Args:
            file_path (str): The file path to load the messages from.
        """
        with open(file_path, "r", encoding="utf-8") as file:
            loaded_messages = json.load(file)
            self.messages.extend(loaded_messages)

    def clean_keys(self, data):
        if isinstance(data, dict):
            # Create a new dictionary with modified keys
            new_dict = {}
            for key, value in data.items():
                # Remove the leading 'XXX_' from keys
                new_key = re.sub(r'^\d{3}_', '', key)
                # Recursively clean nested dictionaries and lists
                new_dict[new_key] = self.clean_keys(value)
            return new_dict
        elif isinstance(data, list):
            # Process each item in the list
            return [self.clean_keys(item) for item in data]
        else:
            # Return the item as is if it's not a dict or list
            return data

    @staticmethod
    def get_function_tool_registry(
            function_tool_list: List[LlamaCppFunctionTool],
            allow_parallel_function_calling=False,
            add_inner_thoughts=False,
            allow_inner_thoughts_only=False,
            add_request_heartbeat=False,
            tool_root="function",
            tool_rule_content="parameters",
            model_prefix="function",
            fields_prefix="parameters",
            inner_thoughts_field_name="thoughts_and_reasoning",
            request_heartbeat_field_name="request_heartbeat",
    ):
        """
        Creates and returns a function tool registry from a list of LlamaCppFunctionTool instances.

        Args:
            function_tool_list (List[LlamaCppFunctionTool]): List of function tools to register.
            allow_parallel_function_calling: Allow parallel function calling (Default=False)
            add_inner_thoughts: Add inner thoughts field (Default=False)
            allow_inner_thoughts_only: Allow inner thoughts only (Default=False)
            add_request_heartbeat: Add request heartbeat field (Default=False)
            tool_root: The root name of the tool (Default="function")
            tool_rule_content: The content of the tool rule (Default="parameters")
            model_prefix: The prefix for the model in the documentation (Default="function")
            fields_prefix: The prefix for the fields in the documentation (Default="parameters")
            inner_thoughts_field_name: The name of the inner thoughts field (Default="thoughts_and_reasoning")
            request_heartbeat_field_name: The name of the request heartbeat field (Default="request_heartbeat")
        Returns:
            LlamaCppFunctionToolRegistry: The created function tool registry.
        """
        function_tool_registry = LlamaCppFunctionToolRegistry(
            allow_parallel_function_calling,
            add_inner_thoughts,
            allow_inner_thoughts_only,
            add_request_heartbeat,
            tool_root,
            tool_rule_content,
            model_prefix,
            fields_prefix,
            inner_thoughts_field_name,
            request_heartbeat_field_name,
        )

        for function_tool in function_tool_list:
            function_tool_registry.register_function_tool(function_tool)
        function_tool_registry.finalize()
        return function_tool_registry