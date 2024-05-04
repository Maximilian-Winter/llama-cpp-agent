import datetime
import json
import os
from copy import copy
from typing import Type, List, Callable, Union, Literal

from llama_cpp import Llama
from pydantic import BaseModel, Field

from .llm_settings import LlamaLLMGenerationSettings, LlamaLLMSettings
from .llm_agent import LlamaCppAgent, StreamingResponse
from .messages_formatter import MessagesFormatterType, MessagesFormatter
from .function_calling import LlamaCppFunctionTool
from .gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import (
    create_dynamic_model_from_function,
    create_dynamic_models_from_dictionaries,
    add_run_method_to_dynamic_model,
)
from .providers.llama_cpp_endpoint_provider import (
    LlamaCppGenerationSettings,
    LlamaCppEndpointSettings,
)
from .providers.openai_endpoint_provider import (
    OpenAIGenerationSettings,
    OpenAIEndpointSettings,
)


class activate_message_mode(BaseModel):
    """
    Activates message mode.
    """

    def run(self, agent: "FunctionCallingAgent"):
        agent.without_grammar_mode = True
        agent.prompt_suffix = "\nWrite message in plain text format:"
        agent.without_grammar_mode_function.append(agent.send_message_to_user)
        return True


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
        """
        self.llama_cpp_tools = []
        if llama_cpp_function_tools:
            self.llama_cpp_tools = llama_cpp_function_tools

        self.send_message_to_user_callback = send_message_to_user_callback
        if add_send_message_to_user_function:
            self.llama_cpp_tools.append(
                LlamaCppFunctionTool(activate_message_mode, agent=self)
            )

        if basic_file_tools:
            self.llama_cpp_tools.append(LlamaCppFunctionTool(read_text_file))
            self.llama_cpp_tools.append(
                LlamaCppFunctionTool(write_text_file, agent=self)
            )

        self.tool_registry = LlamaCppAgent.get_function_tool_registry(
            self.llama_cpp_tools,
            add_inner_thoughts=True,
            allow_inner_thoughts_only=False,
            allow_parallel_function_calling=allow_parallel_function_calling,
        )

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
                "Wrong generation settings for OpenAI endpoint, use OpenAIGenerationSettings under llama_cpp_agent.providers.openai_endpoint_provider!"
            )

        self.llama_generation_settings = llama_generation_settings

        self.without_grammar_mode = False
        self.without_grammar_mode_function = []
        self.prompt_suffix = ""
        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            # You can also request to return control back to you after a function call is executed by setting the 'return_control' flag in a function call object.
            self.system_prompt = (
                """You are Funky, an AI assistant that calls functions to perform tasks.

To call functions, you respond with a JSON object containing three fields:
"thoughts_and_reasoning": Your thoughts and reasoning behind the function call.
"function": The name of the function you want to call.
"parameters": The arguments required for the function.

After performing a function call, you will receive a response containing the return values of the function calls.

For direct communication with the user, employ the 'activate_message_mode' function. After you have finished your call to 'activate_message_mode', you can freely write a response to the user without any JSON constraints. This enables you to converse naturally.

### Functions:
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

"""
                + self.tool_registry.get_documentation().strip()
            )
        self.llama_cpp_agent = LlamaCppAgent(
            llama_llm,
            debug_output=debug_output,
            system_prompt="",
            predefined_messages_formatter_type=messages_formatter_type,
            custom_messages_formatter=custom_messages_formatter,
        )

        self.k_last_messages_from_chat_history = k_last_messages_from_chat_history
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
            del dic["streaming_callback"]
            del dic["tool_registry"]
            del dic["llama_cpp_tools"]
            del dic["send_message_to_user_callback"]
            del dic["without_grammar_mode_function"]
            dic["debug_output"] = self.llama_cpp_agent.debug_output
            dic["messages"] = self.llama_cpp_agent.messages
            dic["llama_generation_settings"] = self.llama_generation_settings.as_dict()
            dic[
                "custom_messages_formatter"
            ] = self.llama_cpp_agent.messages_formatter.as_dict()
            json.dump(dic, file, indent=4)

    @staticmethod
    def load_from_file(
        file_path: str,
        llama_llm: Union[Llama, LlamaLLMSettings],
        open_ai_functions: (List[dict], List[Callable]) = None,
        python_functions: List[Callable] = None,
        pydantic_functions: List[Type[BaseModel]] = None,
        send_message_to_user_callback: Callable[[str], None] = None,
        streaming_callback: Callable[[StreamingResponse], None] = None,
    ) -> "FunctionCallingAgent":
        """
        Load the agent's state from a file.

        Args:
            file_path (str): The path to the file.
            llama_llm: LLM to use
            open_ai_functions (Tuple[List[dict], List[Callable]]): OpenAI function definitions, and a list of the actual functions as tuple.
            python_functions (List[Callable]): Python functions for interaction.
            pydantic_functions (List[Type[BaseModel]]): Pydantic models representing functions.
            send_message_to_user_callback (Callable[[str], None]): Callback for sending a message to the user.
            streaming_callback (Callable[[StreamingResponse], None]): Callback function for streaming responses.

        Returns:
            FunctionCallingAgent: The loaded FunctionCallingAgent instance.
        """

        with open(file_path, "r", encoding="utf-8") as file:
            loaded_agent = json.load(file)
            loaded_agent["llama_llm"] = llama_llm
            loaded_agent["streaming_callback"] = streaming_callback
            loaded_agent["python_functions"] = python_functions
            loaded_agent["pydantic_functions"] = pydantic_functions
            loaded_agent["open_ai_functions"] = open_ai_functions
            messages = copy(loaded_agent["messages"])
            del loaded_agent["messages"]
            loaded_agent[
                "send_message_to_user_callback"
            ] = send_message_to_user_callback
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
            agent = FunctionCallingAgent(**loaded_agent)

            agent.llama_cpp_agent.messages = messages
            return agent

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
        self, message: str, additional_stop_sequences: List[str] = None
    ):
        self.llama_cpp_agent.add_message(role="user", message=message)

        result = self.intern_get_response(
            additional_stop_sequences=additional_stop_sequences
        )

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
                for res in result:
                    count += 1
                    if not isinstance(res, str):
                        if "params" in res:
                            function_message += f"""{count}. Function: "{res["function"]}"\nArguments: "{res["params"]}"\nReturn Value: {res["return_value"]}\n\n"""
                        else:
                            function_message += f"""{count}. Function: "{res["function"]}"\nReturn Value: {res["return_value"]}\n\n"""
                    else:
                        function_message += f"{count}. " + res + "\n\n"
                self.llama_cpp_agent.add_message(
                    role="function", message=function_message.strip()
                )
            result = self.intern_get_response(
                additional_stop_sequences=additional_stop_sequences
            )
        return result

    def intern_get_response(self, additional_stop_sequences: List[str] = None):
        without_grammar_mode = False
        if self.without_grammar_mode:
            without_grammar_mode = True
            self.without_grammar_mode = False
        if additional_stop_sequences is None:
            additional_stop_sequences = []
        additional_stop_sequences.append("(End of message)")
        result = self.llama_cpp_agent.get_chat_response(
            system_prompt=self.system_prompt,
            streaming_callback=self.streaming_callback,
            function_tool_registry=self.tool_registry
            if not without_grammar_mode
            else None,
            additional_stop_sequences=additional_stop_sequences,
            **self.llama_generation_settings.as_dict(),
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
