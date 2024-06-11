from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Callable, Union, Generator, Any

from pydantic import BaseModel

from .chat_history.basic_chat_history import BasicChatHistory
from .chat_history.chat_history_base import ChatHistory
from .chat_history.messages import Roles

from .llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType

from .messages_formatter import (
    MessagesFormatterType,
    get_predefined_messages_formatter,
    MessagesFormatter,
)
from .prompt_templates import function_calling_thoughts_and_reasoning_templater, \
    function_calling_system_prompt_templater, function_calling_heart_beats_templater, \
    function_calling_function_list_templater, structured_output_templater, \
    structured_output_thoughts_and_reasoning_templater

from .providers.provider_base import LlmProvider, LlmSamplingSettings

class SystemPromptModulePosition(Enum):
    after_system_instructions = 1
    at_end = 2


class SystemPromptModule:

    def __init__(self, section_name: str, prefix: str = "", suffix: str = "", position: SystemPromptModulePosition = SystemPromptModulePosition.at_end):
        self.section_name = section_name
        self.prefix = prefix
        self.suffix = suffix
        self.content = ""
        self.position = position

    def set_content(self, content: str):
        self.content = content

    def get_formatted_content(self):
        return f"{self.prefix}\n<{self.section_name}>\n{self.content}\n</{self.section_name}>\n{self.suffix}\n"



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
    """

    def __init__(
            self,
            provider: LlmProvider,
            name: str = "llamacpp_agent",
            system_prompt: str = "You are a helpful assistant.",
            predefined_messages_formatter_type: MessagesFormatterType = MessagesFormatterType.CHATML,
            custom_messages_formatter: MessagesFormatter = None,
            chat_history: ChatHistory = None,
            add_tools_and_structures_documentation_to_system_prompt: bool = True,
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
           add_tools_and_structures_documentation_to_system_prompt (bool): Will suffix system prompt dynamically with documentation for function calling or structured output.
           debug_output (bool): Indicates whether debug output should be enabled.
        """
        self.provider = provider
        self.name = name
        self.debug_output = debug_output
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

        self.add_message(role=Roles.system, message=system_prompt)
        self.system_prompt = system_prompt
        self.add_tools_and_structures_documentation_to_system_prompt = add_tools_and_structures_documentation_to_system_prompt

    def add_message(
            self,
            message: str,
            role: Roles,
    ):
        """
        Adds a message to the chat history.

        Args:
            message (str): The content of the message.
            role (Literal["system"] | Literal["user"] | Literal["assistant"] | Literal["tool"]): The role of the message sender.
        """
        self.chat_history.add_message(
            {
                "role": role,
                "content": message,
            }
        )

    def get_text_response(
            self,
            prompt: str = None,
            structured_output_settings: LlmStructuredOutputSettings = None,
            llm_sampling_settings: LlmSamplingSettings = None,
            streaming_callback: Callable[[StreamingResponse], None] = None,
            returns_streaming_generator: bool = False,
            print_output: bool = False,
    ) -> Union[
        str,
        List[dict],
        BaseModel,
        Generator[Any, Any, str | BaseModel | list[BaseModel]],
    ]:
        """
        Get a text response from the LLM provider.

        Args:
            prompt (str | list[int]): The prompt for the LLM.
            structured_output_settings (LlmStructuredOutputSettings): Settings for structured output.
            llm_sampling_settings (LlmSamplingSettings): Sampling settings for the LLM.
            streaming_callback (Callable[[StreamingResponse], None]): Callback for streaming responses.
            returns_streaming_generator (bool): Whether to return a generator streaming the results.
            print_output (bool): Whether to print the output.

        Returns:
            Union[str, List[dict], BaseModel, Generator[Any, Any, str | BaseModel | list[BaseModel]]: The generated response. A string message, a list of function calls, an object from structured output or a generator for the response
        """

        if self.debug_output:
            if type(prompt) is str:
                print(prompt, end="")

        if structured_output_settings is None:
            structured_output_settings = LlmStructuredOutputSettings(
                output_type=LlmStructuredOutputType.no_structured_output
            )
        if llm_sampling_settings is None:
            llm_sampling_settings = self.provider.get_provider_default_settings()
        else:
            llm_sampling_settings = deepcopy(llm_sampling_settings)

        if llm_sampling_settings.get_additional_stop_sequences() is not None:
            llm_sampling_settings.add_additional_stop_sequences(
                self.messages_formatter.default_stop_sequences
            )

        if self.provider:
            completion = self.get_text_completion(
                prompt=prompt,
                structured_output_settings=structured_output_settings,
                llm_samplings_settings=llm_sampling_settings,
            )

            def stream_results():
                full_response_stream = ""
                for out_stream in completion:
                    out_text = out_stream["choices"][0]["text"]
                    full_response_stream += out_text
                    yield out_text

                return structured_output_settings.handle_structured_output(
                    full_response_stream
                )

            if llm_sampling_settings.is_streaming():
                full_response = ""
                if returns_streaming_generator:
                    return stream_results()
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
                if print_output or self.debug_output:
                    print("")
                self.last_response = full_response
                return structured_output_settings.handle_structured_output(
                    full_response
                )
            else:
                full_response = ""
                text = completion["choices"][0]["text"]
                full_response += text
                if print_output or self.debug_output:
                    print(full_response)
                self.last_response = full_response
                return structured_output_settings.handle_structured_output(
                    full_response
                )
        return "Error: No model loaded!"

    def get_chat_response(
            self,
            message: str = None,
            role: Roles = Roles.user,
            prompt_suffix: str = None,
            chat_history: ChatHistory = None,
            system_prompt: str = None,
            system_prompt_modules: list[SystemPromptModule] = None,
            add_message_to_chat_history: bool = True,
            add_response_to_chat_history: bool = True,
            structured_output_settings: LlmStructuredOutputSettings = None,
            llm_sampling_settings: LlmSamplingSettings = None,
            streaming_callback: Callable[[StreamingResponse], None] = None,
            returns_streaming_generator: bool = False,
            print_output: bool = False,
    ) -> Union[
        str,
        List[dict],
        BaseModel,
        Generator[Any, Any, str | BaseModel | list[BaseModel]],
    ]:
        """
        Get a chat response based on the input message and context.

        Args:
            message (str): The input message.
            role (Literal["system", "user", "assistant", "tool"]): The role of the message sender.
            prompt_suffix (str): Suffix to append after the prompt.
            chat_history (ChatHistory): Overwrite internal ChatHistory of the agent.
            system_prompt (str): Overwrites the system prompt set on the agent initialization.
            system_prompt_modules (SystemPromptModules): Additional sections added to the system prompt.
            add_message_to_chat_history (bool): Whether to add the input message to the chat history.
            add_response_to_chat_history (bool): Whether to add the generated response to the chat history.
            structured_output_settings (LlmStructuredOutputSettings): Settings for structured output.
            llm_sampling_settings (LlmSamplingSettings): Sampling settings for the LLM.
            streaming_callback (Callable[[StreamingResponse], None]): Callback for streaming responses.
            returns_streaming_generator (bool): Whether to return a generator streaming the results.
            print_output (bool): Whether to print the generated response.

        Returns:
            Union[str, List[dict], BaseModel, Generator[Any, Any, str | BaseModel | list[BaseModel]]: The generated chat response. A string message, a list of function calls, an object from structured output or a generator for the response
        """
        if chat_history is None:
            chat_history = self.chat_history

        if structured_output_settings is None:
            structured_output_settings = LlmStructuredOutputSettings(
                output_type=LlmStructuredOutputType.no_structured_output
            )
        if llm_sampling_settings is None:
            llm_sampling_settings = self.provider.get_provider_default_settings()
        else:
            llm_sampling_settings = deepcopy(llm_sampling_settings)

        if llm_sampling_settings.get_additional_stop_sequences() is not None:
            llm_sampling_settings.add_additional_stop_sequences(
                self.messages_formatter.default_stop_sequences
            )

        completion, response_role = self.get_response_role_and_completion(
            message=message,
            chat_history=chat_history,
            system_prompt=system_prompt,
            system_prompt_modules=system_prompt_modules,
            add_message_to_chat_history=add_message_to_chat_history,
            role=role,
            prompt_suffix=prompt_suffix,
            structured_output_settings=structured_output_settings,
            llm_sampling_settings=llm_sampling_settings,
        )

        def stream_results():
            full_response_stream = ""
            for out_stream in completion:
                out_text = out_stream["choices"][0]["text"]
                if out_text != self.messages_formatter.eos_token:
                    full_response_stream += out_text
                    yield out_text
            if prompt_suffix:
                full_response_stream = prompt_suffix + full_response_stream
            self.last_response = full_response_stream
            if add_response_to_chat_history:
                chat_history.add_message(
                    {
                        "role": response_role,
                        "content": full_response_stream,
                    }
                )
            return structured_output_settings.handle_structured_output(
                full_response_stream, prompt_suffix=prompt_suffix
            )

        if self.provider:
            if returns_streaming_generator:
                return stream_results()
            if llm_sampling_settings.is_streaming():
                full_response = ""
                for out in completion:
                    text = out["choices"][0]["text"]
                    if text != self.messages_formatter.eos_token:
                        full_response += text
                        if streaming_callback is not None:
                            streaming_callback(
                                StreamingResponse(text=text, is_last_response=False)
                            )
                        if print_output or self.debug_output:
                            print(text, end="")
                if streaming_callback is not None:
                    streaming_callback(
                        StreamingResponse(text="", is_last_response=True)
                    )
                if print_output or self.debug_output:
                    print("")
                if prompt_suffix:
                    full_response = prompt_suffix + full_response
                self.last_response = full_response
                if add_response_to_chat_history:
                    chat_history.add_message(
                        {
                            "role": response_role,
                            "content": full_response,
                        }
                    )

                return structured_output_settings.handle_structured_output(
                    full_response, prompt_suffix=prompt_suffix
                )
            else:
                text = completion["choices"][0]["text"]
                if text.strip().endswith(self.messages_formatter.eos_token):
                    text = text.replace(self.messages_formatter.eos_token, "")
                if print_output or self.debug_output:
                    print(text)
                if prompt_suffix:
                    text = prompt_suffix + text
                self.last_response = text
                if add_response_to_chat_history:
                    chat_history.add_message(
                        {
                            "role": response_role,
                            "content": text,
                        }
                    )

                return structured_output_settings.handle_structured_output(text, prompt_suffix=prompt_suffix)
        return "Error: No model loaded!"

    def get_text_completion(
            self,
            prompt: str | list[int] = None,
            structured_output_settings: LlmStructuredOutputSettings = None,
            llm_samplings_settings: LlmSamplingSettings = None,
    ):
        return self.provider.create_completion(
            prompt,
            structured_output_settings,
            llm_samplings_settings,
            self.messages_formatter.bos_token,
        )

    def get_response_role_and_completion(
            self,
            message: str = None,
            chat_history: ChatHistory = None,
            system_prompt: str = None,
            system_prompt_modules: list[SystemPromptModule] = None,
            add_message_to_chat_history: bool = True,
            role: Roles = Roles.user,
            prompt_suffix: str = None,
            llm_sampling_settings: LlmSamplingSettings = None,
            structured_output_settings: LlmStructuredOutputSettings = None,
    ):
        if len(chat_history.get_chat_messages()) == 0:
            if system_prompt:
                chat_history.add_message({"role": Roles.system, "content": system_prompt})
            else:
                chat_history.add_message({"role": Roles.system, "content": self.system_prompt})

        if message is not None and add_message_to_chat_history:
            chat_history.add_message(
                {
                    "role": role,
                    "content": message,
                }
            )

        messages = chat_history.get_chat_messages()
        if message is not None and not add_message_to_chat_history:
            messages.append(
                {
                    "role": role,
                    "content": message,
                },
            )

        if system_prompt:
            if messages[0]["role"] != Roles.system and (messages[0]["role"] != Roles.system.value):
                messages.insert(0, {"role": Roles.system, "content": system_prompt})
            else:
                messages[0]["content"] = system_prompt
        else:
            if messages[0]["role"] != Roles.system and (messages[0]["role"] != Roles.system.value):
                messages.insert(0, {"role": Roles.system, "content": self.system_prompt})
            else:
                messages[0]["content"] = self.system_prompt

        additional_suffix = ""
        if self.add_tools_and_structures_documentation_to_system_prompt:
            after_system_instructions_list = []
            after_system_instructions = ""
            if system_prompt_modules is not None:
                for module in system_prompt_modules:
                    if module.position == SystemPromptModulePosition.after_system_instructions:
                        after_system_instructions_list.append(module.get_formatted_content())
                if len(after_system_instructions_list) > 0:
                    after_system_instructions = "\n\n".join(after_system_instructions_list)
                else:
                    after_system_instructions = ""
            if structured_output_settings.output_type != LlmStructuredOutputType.no_structured_output:
                # additional_suffix = "\n"
                thoughts_and_reasoning = ""

                if structured_output_settings.output_type == LlmStructuredOutputType.function_calling or structured_output_settings.output_type == LlmStructuredOutputType.parallel_function_calling:
                    if structured_output_settings.add_thoughts_and_reasoning_field and self.provider.is_using_json_schema_constraints():

                        thoughts_and_reasoning = function_calling_thoughts_and_reasoning_templater
                        thoughts_and_reasoning = thoughts_and_reasoning.generate_prompt({
                            "thoughts_and_reasoning_field_name": "001_" + structured_output_settings.thoughts_and_reasoning_field_name})
                        function_field_name = "002_" + structured_output_settings.function_calling_name_field_name
                        arguments_field_name = "003_" + structured_output_settings.function_calling_content
                        heartbeat_beats = ""
                        if structured_output_settings.add_heartbeat_field:
                            heartbeat_field_name = "004_" + structured_output_settings.heartbeat_field_name
                            heartbeat_beats = function_calling_heart_beats_templater
                            heartbeat_beats = heartbeat_beats.generate_prompt(
                                {"heartbeat_field_name": heartbeat_field_name})
                        function_list = structured_output_settings.get_llm_documentation(
                            provider=self.provider)
                        system_prompt = function_calling_system_prompt_templater
                        system_prompt = system_prompt.generate_prompt({"system_instructions": messages[0]["content"],
                                                                       "after_system_instructions": after_system_instructions,
                                                                       "thoughts_and_reasoning": thoughts_and_reasoning,
                                                                       "function_field_name": function_field_name,
                                                                       "arguments_field_name": arguments_field_name,
                                                                       "heart_beats": heartbeat_beats,
                                                                       "function_list": function_calling_function_list_templater.generate_prompt(
                                                                           {"function_list": function_list})})
                        messages[0]["content"] = system_prompt
                    elif not structured_output_settings.add_thoughts_and_reasoning_field and self.provider.is_using_json_schema_constraints():

                        function_field_name = "001_" + structured_output_settings.function_calling_name_field_name
                        arguments_field_name = "002_" + structured_output_settings.function_calling_content
                        heartbeat_beats = ""
                        if structured_output_settings.add_heartbeat_field:
                            heartbeat_field_name = "003_" + structured_output_settings.heartbeat_field_name
                            heartbeat_beats = function_calling_heart_beats_templater
                            heartbeat_beats = heartbeat_beats.generate_prompt(
                                {"heartbeat_field_name": heartbeat_field_name})
                        function_list = structured_output_settings.get_llm_documentation(
                            provider=self.provider)
                        system_prompt = function_calling_system_prompt_templater
                        system_prompt = system_prompt.generate_prompt({"system_instructions": messages[0]["content"],
                                                                       "after_system_instructions": after_system_instructions,
                                                                       "thoughts_and_reasoning": thoughts_and_reasoning,
                                                                       "function_field_name": function_field_name,
                                                                       "arguments_field_name": arguments_field_name,
                                                                       "heart_beats": heartbeat_beats,
                                                                       "function_list": function_calling_function_list_templater.generate_prompt(
                                                                           {"function_list": function_list})})
                        messages[0]["content"] = system_prompt
                    elif structured_output_settings.add_thoughts_and_reasoning_field and not self.provider.is_using_json_schema_constraints():

                        thoughts_and_reasoning = function_calling_thoughts_and_reasoning_templater
                        thoughts_and_reasoning = thoughts_and_reasoning.generate_prompt({
                            "thoughts_and_reasoning_field_name": structured_output_settings.thoughts_and_reasoning_field_name})
                        function_field_name = structured_output_settings.function_calling_name_field_name
                        arguments_field_name = structured_output_settings.function_calling_content
                        heartbeat_beats = ""
                        if structured_output_settings.add_heartbeat_field:
                            heartbeat_field_name = structured_output_settings.heartbeat_field_name
                            heartbeat_beats = function_calling_heart_beats_templater
                            heartbeat_beats = heartbeat_beats.generate_prompt(
                                {"heartbeat_field_name": heartbeat_field_name})
                        function_list = structured_output_settings.get_llm_documentation(
                            provider=self.provider)
                        system_prompt = function_calling_system_prompt_templater
                        system_prompt = system_prompt.generate_prompt({"system_instructions": messages[0]["content"],
                                                                       "after_system_instructions": after_system_instructions,
                                                                       "thoughts_and_reasoning": thoughts_and_reasoning,
                                                                       "function_field_name": function_field_name,
                                                                       "arguments_field_name": arguments_field_name,
                                                                       "heart_beats": heartbeat_beats,
                                                                       "function_list": function_calling_function_list_templater.generate_prompt(
                                                                           {"function_list": function_list})})
                        messages[0]["content"] = system_prompt
                    elif not structured_output_settings.add_thoughts_and_reasoning_field and not self.provider.is_using_json_schema_constraints():

                        thoughts_and_reasoning = ""
                        function_field_name = structured_output_settings.function_calling_name_field_name
                        arguments_field_name = structured_output_settings.function_calling_content
                        heartbeat_beats = ""
                        if structured_output_settings.add_heartbeat_field:
                            heartbeat_field_name = structured_output_settings.heartbeat_field_name
                            heartbeat_beats = function_calling_heart_beats_templater
                            heartbeat_beats = heartbeat_beats.generate_prompt(
                                {"heartbeat_field_name": heartbeat_field_name})
                        function_list = structured_output_settings.get_llm_documentation(
                            provider=self.provider)
                        system_prompt = function_calling_system_prompt_templater
                        system_prompt = system_prompt.generate_prompt({"system_instructions": messages[0]["content"],
                                                                       "after_system_instructions": after_system_instructions,
                                                                       "thoughts_and_reasoning": thoughts_and_reasoning,
                                                                       "function_field_name": function_field_name,
                                                                       "arguments_field_name": arguments_field_name,
                                                                       "heart_beats": heartbeat_beats,
                                                                       "function_list": function_calling_function_list_templater.generate_prompt(
                                                                           {"function_list": function_list})})
                        messages[0]["content"] = system_prompt
                elif structured_output_settings.output_type == LlmStructuredOutputType.object_instance or structured_output_settings.output_type == LlmStructuredOutputType.list_of_objects:
                    if structured_output_settings.add_thoughts_and_reasoning_field and self.provider.is_using_json_schema_constraints():

                        thoughts_and_reasoning = structured_output_thoughts_and_reasoning_templater
                        thoughts_and_reasoning = thoughts_and_reasoning.generate_prompt({
                            "thoughts_and_reasoning_field_name": "001_" + structured_output_settings.thoughts_and_reasoning_field_name})
                        model_field_name = "002_" + structured_output_settings.output_model_name_field_name
                        fields_field_name = "003_" + structured_output_settings.output_model_attributes_field_name

                        output_models = structured_output_settings.get_llm_documentation(
                            provider=self.provider)
                        system_prompt = structured_output_templater
                        system_prompt = system_prompt.generate_prompt({"system_instructions": messages[0]["content"],
                                                                       "after_system_instructions": after_system_instructions,
                                                                       "thoughts_and_reasoning": thoughts_and_reasoning,
                                                                       "model_field_name": model_field_name,
                                                                       "fields_field_name": fields_field_name,
                                                                       "output_models": output_models})
                        messages[0]["content"] = system_prompt
                    elif not structured_output_settings.add_thoughts_and_reasoning_field and self.provider.is_using_json_schema_constraints():

                        thoughts_and_reasoning = ""
                        model_field_name = "001_" + structured_output_settings.output_model_name_field_name
                        fields_field_name = "002_" + structured_output_settings.output_model_attributes_field_name

                        output_models = structured_output_settings.get_llm_documentation(
                            provider=self.provider)
                        system_prompt = structured_output_templater
                        system_prompt = system_prompt.generate_prompt({"system_instructions": messages[0]["content"],
                                                                       "after_system_instructions": after_system_instructions,
                                                                       "thoughts_and_reasoning": thoughts_and_reasoning,
                                                                       "model_field_name": model_field_name,
                                                                       "fields_field_name": fields_field_name,
                                                                       "output_models": output_models})
                        messages[0]["content"] = system_prompt
                    elif structured_output_settings.add_thoughts_and_reasoning_field and not self.provider.is_using_json_schema_constraints():

                        thoughts_and_reasoning = structured_output_thoughts_and_reasoning_templater
                        thoughts_and_reasoning = thoughts_and_reasoning.generate_prompt({
                            "thoughts_and_reasoning_field_name": structured_output_settings.thoughts_and_reasoning_field_name})
                        model_field_name = structured_output_settings.output_model_name_field_name
                        fields_field_name = structured_output_settings.output_model_attributes_field_name

                        output_models = structured_output_settings.get_llm_documentation(
                            provider=self.provider)
                        system_prompt = structured_output_templater
                        system_prompt = system_prompt.generate_prompt({"system_instructions": messages[0]["content"],
                                                                       "after_system_instructions": after_system_instructions,
                                                                       "thoughts_and_reasoning": thoughts_and_reasoning,
                                                                       "model_field_name": model_field_name,
                                                                       "fields_field_name": fields_field_name,
                                                                       "output_models": output_models})
                        messages[0]["content"] = system_prompt
                    elif not structured_output_settings.add_thoughts_and_reasoning_field and not self.provider.is_using_json_schema_constraints():

                        model_field_name = structured_output_settings.output_model_name_field_name
                        fields_field_name = structured_output_settings.output_model_attributes_field_name

                        output_models = structured_output_settings.get_llm_documentation(
                            provider=self.provider)
                        system_prompt = structured_output_templater
                        system_prompt = system_prompt.generate_prompt({"system_instructions": messages[0]["content"],
                                                                       "after_system_instructions": after_system_instructions,
                                                                       "thoughts_and_reasoning": thoughts_and_reasoning,
                                                                       "model_field_name": model_field_name,
                                                                       "fields_field_name": fields_field_name,
                                                                       "output_models": output_models})
                        messages[0]["content"] = system_prompt

            if structured_output_settings.output_type == LlmStructuredOutputType.no_structured_output or structured_output_settings is None:
                messages[0]["content"] += "\n" + after_system_instructions
        at_end_list = []
        if system_prompt_modules is not None:
            for module in system_prompt_modules:
                if module.position == SystemPromptModulePosition.at_end:
                    at_end_list.append(module.get_formatted_content())
            if len(at_end_list) > 0:
                at_end_list = "\n\n".join(at_end_list)
            else:
                at_end_list = ""

            messages[0]["content"] += at_end_list
        prompt, response_role = self.messages_formatter.format_conversation(
            messages, Roles.assistant
        )

        if prompt_suffix:
            prompt += prompt_suffix

        if self.debug_output:
            print(prompt, end="")

        return (
            self.provider.create_completion(
                prompt,
                structured_output_settings,
                llm_sampling_settings,
                self.messages_formatter.bos_token,
            ),
            response_role,
        )

    @staticmethod
    def remove_any(text, list_of_strings):
        for item in list_of_strings:
            text = text.replace(item, "")
        return text
