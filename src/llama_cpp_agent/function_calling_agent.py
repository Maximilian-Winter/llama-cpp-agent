import datetime
import json
import os
from copy import copy
from typing import Type, List, Callable, Union, Literal

from llama_cpp import Llama
from pydantic import BaseModel, Field

from .chat_history.messages import Roles
from .llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType

from .llm_agent import LlamaCppAgent, StreamingResponse
from .messages_formatter import MessagesFormatterType, MessagesFormatter
from .function_calling import LlamaCppFunctionTool

from .providers.provider_base import LlmProvider, LlmSamplingSettings


class activate_message_mode(BaseModel):
    """
    Activates message mode.
    """

    def run(self, agent: "FunctionCallingAgent"):
        agent.without_grammar_mode = True
        agent.prompt_suffix = "\nWrite message in plain text format:"
        agent.without_grammar_mode_function.append(agent.send_message_to_user)
        return True


class send_message(BaseModel):
    """
    Sends a message to the user.
    """

    content: str = Field(..., description="Content of the message to be sent.")

    def run(self, agent: "FunctionCallingAgent"):
        agent.send_message_to_user(self.content)
        return "Message sent."


class write_text_file(BaseModel):
    """
    Writes content to a file.
    """

    file_path: str = Field(..., description="The path to the file.")
    content: str = Field(..., description="The content to write to the file.")

    def run(self, agent: "FunctionCallingAgent"):
        self.write_file(self.content)
        return True

    def write_file(self, content: str):
        """
        Write content to a file.

        Args:
            content (str): The content to write to the file.
        """
        with open(self.file_path, "w", encoding="utf-8") as file:
            file.write(content)
        return None


class read_text_file(BaseModel):
    """
    Reads the content of a file.
    """

    file_path: str = Field(..., description="The path to the file.")

    def run(self):
        return self.read_file()

    def read_file(self):
        """
        Reads the content of a file.
        """
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as file:
                return file.read()
        else:
            return f"File not found."


class FunctionCallingAgent:
    """
    An agent that uses function calling to interact with its environment and the user.

    Args:
        llama_llm (Llama | LlamaLLMSettings | LlamaCppEndpointSettings | OpenAIEndpointSettings): An instance of Llama, LlamaLLMSettings, LlamaCppEndpointSettings or LlamaCppServerLLMSettings as LLM.
        llama_generation_settings (LlamaLLMGenerationSettings | LlamaCppGenerationSettings | OpenAIGenerationSettings): Generation settings for Llama.
        messages_formatter_type (MessagesFormatterType): Type of messages formatter.
        custom_messages_formatter (MessagesFormatter): Optional Custom messages formatter.
        streaming_callback (Callable[[StreamingResponse], None]): Callback function for streaming responses.
        k_last_messages_from_chat_history (int): Number of last messages to consider from chat history.
        system_prompt (str): System prompt for interaction.
        llama_cpp_function_tools(List[LlamaCppFunctionTool]): List of LlamaCppFunctionTool instances.
        allow_parallel_function_calling (bool): Allow parallel function calling (Default=False)
        add_send_message_to_user_function (bool): Flag to add send_message_to_user function.
        send_message_to_user_callback (Callable[[str], None]): Callback for sending a message to the user.
        debug_output (bool): Enable debug output.

    Attributes:
        send_message_to_user_callback (Callable[[str], None]): Callback for sending a message to the user.
        llama_cpp_tools (List[LlamaCppFunctionTool]): List of LlamaCppFunctionTool instances.
        tool_registry (LlamaCppFunctionToolRegistry): Function tool registry.
        llama_generation_settings (LlamaLLMGenerationSettings): Generation settings for Llama.
        system_prompt (str): System prompt for interaction.
        llama_cpp_agent (LlamaCppAgent): LlamaCppAgent instance for interaction.
        k_last_messages_from_chat_history (int): Number of last messages to consider from chat history.
        streaming_callback (Callable[[StreamingResponse], None]): Callback function for streaming responses.

    Methods:
        save(file_path: str): Save the agent's state to a file.
        load_from_file(file_path: str, llama_llm, python_functions, pydantic_functions, send_message_to_user_callback, streaming_callback) -> FunctionCallingAgent:
            Load the agent's state from a file.
        load_from_dict(agent_dict: dict) -> FunctionCallingAgent: Load the agent's state from a dictionary.
        as_dict() -> dict: Convert the agent's state to a dictionary.
        generate_response(message: str): Generate a response based on the input message.
        send_message_to_user(message: str): Send a message to the user.

    """

    def __init__(
        self,
        llama_llm: LlmProvider,
        messages_formatter_type: MessagesFormatterType = MessagesFormatterType.CHATML,
        custom_messages_formatter: MessagesFormatter = None,
        streaming_callback: Callable[[StreamingResponse], None] = None,
        k_last_messages_from_chat_history: int = 0,
        system_prompt: str = None,
        llama_cpp_function_tools: [LlamaCppFunctionTool] = None,
        basic_file_tools: bool = False,
        allow_parallel_function_calling=False,
        add_send_message_to_user_function: bool = True,
        send_message_to_user_callback: Callable[[str], None] = None,
        debug_output: bool = False,
    ):
        """
        Initialize the FunctionCallingAgent.

        Args:
            llama_llm (LlmProvider): The LLM Provider.
            messages_formatter_type (MessagesFormatterType): Type of messages formatter.
            custom_messages_formatter (MessagesFormatter): Optional Custom messages formatter.
            streaming_callback (Callable[[StreamingResponse], None]): Callback function for streaming responses.
            k_last_messages_from_chat_history (int): Number of last messages to consider from chat history.
            system_prompt (str): System prompt for interaction.
            llama_cpp_function_tools(List[LlamaCppFunctionTool]): List of LlamaCppFunctionTool instances.
            allow_parallel_function_calling (bool): Allow parallel function calling (Default=False)
            add_send_message_to_user_function (bool): Flag to add send_message_to_user function.
            send_message_to_user_callback (Callable[[str], None]): Callback for sending a message to the user.
            debug_output (bool): Enable debug output.
        """
        self.llama_cpp_tools = []
        if llama_cpp_function_tools:
            self.llama_cpp_tools = llama_cpp_function_tools

        self.send_message_to_user_callback = send_message_to_user_callback
        if add_send_message_to_user_function:
            self.llama_cpp_tools.append(LlamaCppFunctionTool(send_message, agent=self))

        if basic_file_tools:
            self.llama_cpp_tools.append(LlamaCppFunctionTool(read_text_file))
            self.llama_cpp_tools.append(
                LlamaCppFunctionTool(write_text_file, agent=self)
            )

        self.allow_parallel_function_calling = allow_parallel_function_calling

        self.structured_output_settings = (
            LlmStructuredOutputSettings.from_llama_cpp_function_tools(
                self.llama_cpp_tools, self.allow_parallel_function_calling
            )
        )

        self.without_grammar_mode = False
        self.without_grammar_mode_function = []
        self.prompt_suffix = ""
        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = """You are Funky, an AI assistant that calls functions to perform tasks."""
        self.llama_cpp_agent = LlamaCppAgent(
            llama_llm,
            debug_output=debug_output,
            system_prompt=self.system_prompt,
            predefined_messages_formatter_type=messages_formatter_type,
            custom_messages_formatter=custom_messages_formatter,
        )

        self.k_last_messages_from_chat_history = k_last_messages_from_chat_history
        self.streaming_callback = streaming_callback


    @staticmethod
    def load_from_dict(agent_dict: dict) -> "FunctionCallingAgent":
        """
        Load the agent's state from a dictionary.

        Args:
            agent_dict (dict): The dictionary containing the agent's state.

        Returns:
            FunctionCallingAgent: The loaded FunctionCallingAgent instance.
        """
        return FunctionCallingAgent(**agent_dict)

    def as_dict(self) -> dict:
        """
        Convert the agent's state to a dictionary.

        Returns:
           dict: The dictionary representation of the agent's state.
        """
        return self.__dict__

    def generate_response(
        self,
        message: str,
        llm_sampling_settings: LlmSamplingSettings = None,
        structured_output_settings: LlmStructuredOutputSettings = None,
    ):
        self.llama_cpp_agent.add_message(role=Roles.user, message=message)

        result = self.intern_get_response(llm_sampling_settings=llm_sampling_settings, structured_output_settings=structured_output_settings)

        while True:
            if isinstance(result, str):
                if len(self.without_grammar_mode_function) > 0:
                    func_list = []
                    for func in self.without_grammar_mode_function:
                        if func.__name__ not in func_list:
                            func(result.strip())
                            func_list.append(func.__name__)
                break
            function_message = f"""Function Calling Results:\n\n"""
            count = 0
            if result is not None:
                agent_sent_message = False
                for res in result:
                    count += 1
                    if res["function"] == "send_message":
                        agent_sent_message = True
                    if not isinstance(res, str):
                        if "params" in res:
                            function_message += f"""{count}. Function: "{res["function"]}"\nArguments: "{res["params"]}"\nReturn Value: {res["return_value"]}\n\n"""
                        else:
                            function_message += f"""{count}. Function: "{res["function"]}"\nReturn Value: {res["return_value"]}\n\n"""
                    else:
                        function_message += f"{count}. " + res + "\n\n"
                self.llama_cpp_agent.add_message(
                    role=Roles.tool, message=function_message.strip()
                )
                if agent_sent_message:
                    break
            result = self.intern_get_response(
                llm_sampling_settings=llm_sampling_settings, structured_output_settings=structured_output_settings
            )
        return result

    def intern_get_response(
        self,
        llm_sampling_settings: List[str] = None,
        structured_output_settings: LlmStructuredOutputSettings = None,
    ):
        without_grammar_mode = False
        if self.without_grammar_mode:
            without_grammar_mode = True
            self.without_grammar_mode = False
        result = self.llama_cpp_agent.get_chat_response(
            streaming_callback=self.streaming_callback,
            structured_output_settings=self.structured_output_settings
            if structured_output_settings is None
            else structured_output_settings,
            llm_sampling_settings=llm_sampling_settings,
        )
        if without_grammar_mode:
            self.prompt_suffix = ""
        return result

    def send_message_to_user(self, message: str):
        """
        Send a message to the user.

        Args:
            message: The message send to the user.
        """
        if self.send_message_to_user_callback:
            self.send_message_to_user_callback(message)
        else:
            print(message)
