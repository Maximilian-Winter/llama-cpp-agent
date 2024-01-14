import json
from copy import copy
from typing import Type, List, Callable, Union, Literal

from llama_cpp import Llama
from pydantic import BaseModel

from .llm_settings import LlamaLLMGenerationSettings, LlamaLLMSettings
from .llm_agent import LlamaCppAgent, StreamingResponse
from .messages_formatter import MessagesFormatterType, MessagesFormatter
from .function_calling import LlamaCppFunctionTool
from .gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import create_dynamic_model_from_function, \
    create_dynamic_models_from_dictionaries, add_run_method_to_dynamic_model
from .providers.llama_cpp_endpoint_provider import LlamaCppGenerationSettings, LlamaCppEndpointSettings
from .providers.openai_endpoint_provider import OpenAIGenerationSettings, OpenAIEndpointSettings


class FunctionCallingAgent:
    """
    An agent that uses function calling to interact with its environment and the user.

    Args:
        llama_llm (Union[Llama, LlamaLLMSettings, LlamaCppEndpointSettings, OpenAIEndpointSettings]): An instance of Llama, LlamaLLMSettings, LlamaCppServerLLMSettings as LLM.
        llama_generation_settings (Union[LlamaLLMGenerationSettings, LlamaCppGenerationSettings, OpenAIGenerationSettings]): Generation settings for Llama.
        messages_formatter_type (MessagesFormatterType): Type of messages formatter.
        custom_messages_formatter (MessagesFormatter): Custom messages formatter.
        streaming_callback (Callable[[StreamingResponse], None]): Callback function for streaming responses.
        k_last_messages_from_chat_history (int): Number of last messages to consider from chat history.
        system_prompt (str): System prompt for interaction.
        open_ai_functions (Tuple[List[dict], List[Callable]]): OpenAI function definitions and a list of the actual functions as tuple.
        python_functions (List[Callable]): Python functions for interaction.
        pydantic_functions (List[Type[BaseModel]]): Pydantic models representing functions.
        add_send_message_to_user_function (bool): Flag to add send_message_to_user function.
        send_message_to_user_callback (Callable[[str], None]): Callback for sending a message to the user.
        debug_output (bool): Enable debug output.

    Attributes:
        pydantic_functions (List[Type[BaseModel]]): List of Pydantic models representing functions.
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

    def __init__(self, llama_llm: Union[Llama, LlamaLLMSettings, LlamaCppEndpointSettings, OpenAIEndpointSettings],
                 llama_generation_settings: Union[
                     LlamaLLMGenerationSettings, LlamaCppGenerationSettings, OpenAIGenerationSettings] = None,
                 messages_formatter_type: MessagesFormatterType = MessagesFormatterType.CHATML,
                 custom_messages_formatter: MessagesFormatter = None,
                 streaming_callback: Callable[[StreamingResponse], None] = None,
                 k_last_messages_from_chat_history: int = 0,
                 system_prompt: str = None,
                 open_ai_functions: (List[dict], List[Callable]) = None,
                 python_functions: List[Callable] = None,
                 pydantic_functions: List[Type[BaseModel]] = None,
                 allow_parallel_function_calling=False,
                 add_send_message_to_user_function: bool = True,
                 send_message_to_user_callback: Callable[[str], None] = None,
                 debug_output: bool = False):
        """
        Initialize the FunctionCallingAgent.

        Args:
            llama_llm (Union[Llama, LlamaLLMSettings, OpenAIEndpointSettings]): An instance of Llama, LlamaLLMSettings or LlamaCppServerLLMSettings as LLM.
            llama_generation_settings (Union[LlamaLLMGenerationSettings, LlamaCppGenerationSettings, OpenAIGenerationSettings]): Generation settings for Llama.
            messages_formatter_type (MessagesFormatterType): Type of messages formatter.
            custom_messages_formatter (MessagesFormatter): Optional Custom messages formatter.
            streaming_callback (Callable[[StreamingResponse], None]): Callback function for streaming responses.
            k_last_messages_from_chat_history (int): Number of last messages to consider from chat history.
            system_prompt (str): System prompt for interaction.
            open_ai_functions (Tuple[List[dict], List[Callable]]): OpenAI function definitions and a list of the actual functions as tuple.
            python_functions (List[Callable]): Python functions for interaction.
            pydantic_functions (List[Type[BaseModel]]): Pydantic models representing functions.
            allow_parallel_function_calling (bool): Allow parallel function calling (Default=False)
            add_send_message_to_user_function (bool): Flag to add send_message_to_user function.
            send_message_to_user_callback (Callable[[str], None]): Callback for sending a message to the user.
            debug_output (bool): Enable debug output.
        """
        if pydantic_functions is None:
            self.pydantic_functions = []
        else:
            self.pydantic_functions = pydantic_functions

        if python_functions is not None:
            for tool in python_functions:
                self.pydantic_functions.append(create_dynamic_model_from_function(tool))

        if open_ai_functions is not None:
            open_ai_models = create_dynamic_models_from_dictionaries(open_ai_functions[0])
            count = 0
            for func in open_ai_functions[1]:
                model = open_ai_models[count]
                self.pydantic_functions.append(add_run_method_to_dynamic_model(model, func))
                count += 1

        self.send_message_to_user_callback = send_message_to_user_callback
        if add_send_message_to_user_function:
            self.llama_cpp_tools = [
                LlamaCppFunctionTool(create_dynamic_model_from_function(self.send_message_to_user))]

        for tool in self.pydantic_functions:
            self.llama_cpp_tools.append(LlamaCppFunctionTool(tool))

        self.tool_registry = LlamaCppAgent.get_function_tool_registry(self.llama_cpp_tools,
                                                                      allow_parallel_function_calling)

        if llama_generation_settings is None:
            if isinstance(llama_llm, Llama) or isinstance(llama_llm, LlamaLLMSettings):
                llama_generation_settings = LlamaLLMGenerationSettings()
            else:
                llama_generation_settings = LlamaCppGenerationSettings()

        if isinstance(llama_generation_settings, LlamaLLMGenerationSettings) and isinstance(llama_llm,
                                                                                            LlamaCppEndpointSettings):
            raise Exception(
                "Wrong generation settings for llama.cpp server endpoint, use LlamaCppServerGenerationSettings under llama_cpp_agent.providers.llama_cpp_server_provider!")
        if isinstance(llama_llm, Llama) or isinstance(llama_llm, LlamaLLMSettings) and isinstance(
                llama_generation_settings, LlamaCppGenerationSettings):
            raise Exception(
                "Wrong generation settings for llama-cpp-python, use LlamaLLMGenerationSettings under llama_cpp_agent.llm_settings!")

        if isinstance(llama_llm, OpenAIEndpointSettings) and not isinstance(
                llama_generation_settings, OpenAIGenerationSettings):
            raise Exception(
                "Wrong generation settings for OpenAI endpoint, use CompletionRequestSettings under llama_cpp_agent.providers.openai_endpoint_provider!")

        self.llama_generation_settings = llama_generation_settings

        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            # You can also request to return control back to you after a function call is executed by setting the 'return_control' flag in a function call object.
            self.system_prompt = "You are 'FunkGPT', an advanced AI assistant. You are interacting with the user and with your environment by calling functions. You call functions by writing JSON objects, which represent specific function calls. You can call functions in parallel by providing a list of function call objects. Always plan the execution of functions step by step!\nBelow is a list of your available function calls:\n\n" + self.tool_registry.get_documentation()
        self.llama_cpp_agent = LlamaCppAgent(llama_llm, debug_output=debug_output,
                                             system_prompt="",
                                             predefined_messages_formatter_type=messages_formatter_type,
                                             custom_messages_formatter=custom_messages_formatter)

        self.k_last_messages_from_chat_history = k_last_messages_from_chat_history
        self.streaming_callback = streaming_callback

    def save(self, file_path: str):
        """
        Save the agent's state to a file.

        Args:
            file_path (str): The path to the file.
        """
        with open(file_path, 'w', encoding="utf-8") as file:
            dic = copy(self.as_dict())
            del dic["llama_cpp_agent"]
            del dic["streaming_callback"]
            del dic["tool_registry"]
            del dic["llama_cpp_tools"]
            del dic["pydantic_functions"]
            del dic["send_message_to_user_callback"]
            dic["debug_output"] = self.llama_cpp_agent.debug_output
            dic["messages"] = self.llama_cpp_agent.messages
            dic["llama_generation_settings"] = self.llama_generation_settings.as_dict()
            dic["custom_messages_formatter"] = self.llama_cpp_agent.messages_formatter.as_dict()
            json.dump(dic, file, indent=4)

    @staticmethod
    def load_from_file(file_path: str, llama_llm: Union[Llama, LlamaLLMSettings],
                       open_ai_functions: (List[dict], List[Callable]) = None,
                       python_functions: List[Callable] = None,
                       pydantic_functions: List[Type[BaseModel]] = None,
                       send_message_to_user_callback: Callable[[str], None] = None,
                       streaming_callback: Callable[[StreamingResponse], None] = None) -> "FunctionCallingAgent":
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

        with open(file_path, 'r', encoding="utf-8") as file:
            loaded_agent = json.load(file)
            loaded_agent["llama_llm"] = llama_llm
            loaded_agent["streaming_callback"] = streaming_callback
            loaded_agent["python_functions"] = python_functions
            loaded_agent["pydantic_functions"] = pydantic_functions
            loaded_agent["open_ai_functions"] = open_ai_functions
            messages = copy(loaded_agent["messages"])
            del loaded_agent["messages"]
            loaded_agent["send_message_to_user_callback"] = send_message_to_user_callback
            loaded_agent["llama_generation_settings"] = LlamaLLMGenerationSettings.load_from_dict(
                loaded_agent["llama_generation_settings"])
            loaded_agent["custom_messages_formatter"] = MessagesFormatter.load_from_dict(
                loaded_agent["custom_messages_formatter"])
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

    def generate_response(self, message: str):
        """
        Generate a response based on the input message.

        Args:
            message (str): The input message.
        """
        count = 0
        msg = copy(message)
        while msg and not ("None" in '\n'.join([str(m) for m in msg])):
            if count > 0:

                msg = "FUNCTION_CALLING_RESULTS:\n" + '\n\n'.join([str(m) for m in msg])
                msg = self.llama_cpp_agent.get_chat_response(msg, role="function", system_prompt=self.system_prompt,
                                                             function_tool_registry=self.tool_registry,
                                                             streaming_callback=self.streaming_callback,
                                                             k_last_messages=self.k_last_messages_from_chat_history,
                                                             **self.llama_generation_settings.as_dict())

            else:
                msg = self.llama_cpp_agent.get_chat_response(msg, role="user", system_prompt=self.system_prompt,
                                                             function_tool_registry=self.tool_registry,
                                                             streaming_callback=self.streaming_callback,
                                                             k_last_messages=self.k_last_messages_from_chat_history,
                                                             **self.llama_generation_settings.as_dict())
            count += 1

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
