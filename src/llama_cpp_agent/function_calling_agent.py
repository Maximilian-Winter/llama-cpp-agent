import json
from copy import copy
from typing import Type, List, Callable, Union, Literal

from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel

from .llm_settings import LlamaLLMGenerationSettings, LlamaLLMSettings
from .llm_agent import LlamaCppAgent, StreamingResponse
from .messages_formatter import MessagesFormatterType, MessagesFormatter
from .function_calling import LlamaCppFunctionTool
from .gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import create_dynamic_model_from_function, \
    create_dynamic_models_from_dictionaries, add_run_method_to_dynamic_model


class FunctionCallingAgent:
    """
    An agent that uses function calling to interact with its environment and the user.
    """

    def __init__(self, llama_llm: Union[Llama, LlamaLLMSettings],
                 llama_generation_settings: LlamaLLMGenerationSettings = LlamaLLMGenerationSettings(),
                 messages_formatter_type: MessagesFormatterType = MessagesFormatterType.CHATML,
                 custom_messages_formatter: MessagesFormatter = None,
                 streaming_callback: Callable[[StreamingResponse], None] = None,
                 k_last_messages_from_chat_history: int = 0,
                 system_prompt: str = None,
                 open_ai_functions: (List[dict], List[Callable]) = None,
                 python_functions: List[Callable] = None,
                 pydantic_functions: List[Type[BaseModel]] = None,
                 add_send_message_to_user_function: bool = True,
                 send_message_to_user_callback: Callable[[str], None] = None,
                 debug_output: bool = False, ):

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

        self.tool_registry = LlamaCppAgent.get_function_tool_registry(self.llama_cpp_tools)
        self.llama_generation_settings = llama_generation_settings

        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = "You are an advanced AI assistant. You are interacting with your environment and the user by calling functions. You call functions by writing JSON objects, which represent specific function calls. Below is a list of your available JSON functions:\n\n" + self.tool_registry.get_documentation()
        self.llama_cpp_agent = LlamaCppAgent(llama_llm, debug_output=debug_output,
                                             system_prompt="",
                                             predefined_messages_formatter_type=messages_formatter_type,
                                             custom_messages_formatter=custom_messages_formatter)

        self.k_last_messages_from_chat_history = k_last_messages_from_chat_history
        self.streaming_callback = streaming_callback

    def save(self, file_path: str):
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
                       python_functions: List[Callable] = None, pydantic_functions: List[Type[BaseModel]] = None,
                       send_message_to_user_callback: Callable[[str], None] = None,
                       streaming_callback: Callable[[StreamingResponse], None] = None) -> "FunctionCallingAgent":
        with open(file_path, 'r', encoding="utf-8") as file:
            loaded_agent = json.load(file)
            loaded_agent["llama_llm"] = llama_llm
            loaded_agent["streaming_callback"] = streaming_callback
            loaded_agent["python_functions"] = python_functions
            loaded_agent["pydantic_functions"] = pydantic_functions
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
        return FunctionCallingAgent(**agent_dict)

    def as_dict(self) -> dict:
        return self.__dict__

    def generate_response(self, message: str):
        count = 0
        while message:
            if count > 0:
                message = f"Function Call Result: {message}"
                message = self.llama_cpp_agent.get_chat_response(message, role="function",
                                                                 system_prompt=self.system_prompt,
                                                                 function_tool_registry=self.tool_registry,
                                                                 streaming_callback=self.streaming_callback,
                                                                 k_last_messages=self.k_last_messages_from_chat_history,
                                                                 **self.llama_generation_settings.as_dict())
            else:
                message = self.llama_cpp_agent.get_chat_response(message, role="user", system_prompt=self.system_prompt,
                                                                 function_tool_registry=self.tool_registry,
                                                                 streaming_callback=self.streaming_callback,
                                                                 k_last_messages=self.k_last_messages_from_chat_history,
                                                                 **self.llama_generation_settings.as_dict())
            count += 1

    def send_message_to_user(self, message: str):
        """
        Sends message to user.
        """
        if self.send_message_to_user_callback:
            self.send_message_to_user_callback(message)
        else:
            print(message)
