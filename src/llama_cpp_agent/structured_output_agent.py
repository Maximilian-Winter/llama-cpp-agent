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
from .gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import generate_gbnf_grammar_and_documentation


class StructuredOutputAgent:
    """
    An agent that creates structured output based on pydantic models from an unstructured text.
    """
    def __init__(self, llama_llm: Union[Llama, LlamaLLMSettings],
                 llama_generation_settings: LlamaLLMGenerationSettings = LlamaLLMGenerationSettings(),
                 messages_formatter_type: MessagesFormatterType = MessagesFormatterType.CHATML,
                 custom_messages_formatter: MessagesFormatter = None,
                 streaming_callback: Callable[[StreamingResponse], None] = None,
                 debug_output: bool = False):

        self.llama_generation_settings = llama_generation_settings
        self.grammar_cache = {}
        self.system_prompt_template = PromptTemplate.from_string(
            "You are an advanced AI agent. You are tasked to assist the user by creating structured output in JSON format.\n\n{documentation}")
        self.creation_prompt_template = PromptTemplate.from_string(
            "Create an JSON response based on the following input.\n\nInput:\n\n{user_input}")

        self.llama_cpp_agent = LlamaCppAgent(llama_llm, debug_output=debug_output,
                                             system_prompt="",
                                             predefined_messages_formatter_type=messages_formatter_type, custom_messages_formatter=custom_messages_formatter)
        self.streaming_callback = streaming_callback

    def save(self, file_path: str):
        with open(file_path, 'w', encoding="utf-8") as file:
            dic = copy(self.as_dict())
            del dic["llama_cpp_agent"]
            del dic["grammar_cache"]
            del dic["system_prompt_template"]
            del dic["creation_prompt_template"]
            del dic["streaming_callback"]
            dic["debug_output"] = self.llama_cpp_agent.debug_output
            dic["llama_generation_settings"] = self.llama_generation_settings.as_dict()
            dic["custom_messages_formatter"] = self.llama_cpp_agent.messages_formatter.as_dict()
            json.dump(dic, file, indent=4)

    @staticmethod
    def load_from_file(file_path: str, llama_llm: Union[Llama, LlamaLLMSettings],
                       streaming_callback: Callable[[StreamingResponse], None] = None) -> "StructuredOutputAgent":
        with open(file_path, 'r', encoding="utf-8") as file:
            loaded_agent = json.load(file)
            loaded_agent["llama_llm"] = llama_llm
            loaded_agent["streaming_callback"] = streaming_callback
            loaded_agent["llama_generation_settings"] = LlamaLLMGenerationSettings.load_from_dict(loaded_agent["llama_generation_settings"])
            loaded_agent["custom_messages_formatter"] = MessagesFormatter.load_from_dict(loaded_agent["custom_messages_formatter"])
            return StructuredOutputAgent(**loaded_agent)

    @staticmethod
    def load_from_dict(agent_dict: dict) -> "StructuredOutputAgent":
        return StructuredOutputAgent(**agent_dict)

    def as_dict(self) -> dict:
        return self.__dict__

    def create_object(self, model: Type[BaseModel], data: str = "") -> BaseModel:
        """
        Creates an object of the given model from the given data.
        :param model: The model to create the object from.
        :param data: The data to create the object from.
        :return: The created object.
        """
        if model not in self.grammar_cache:
            grammar, documentation = generate_gbnf_grammar_and_documentation([model],
                                                                             False,
                                                                             model_prefix="Response Model",
                                                                             fields_prefix="Response Model Field")
            llama_grammar = LlamaGrammar.from_string(grammar, verbose=False)
            self.grammar_cache[model] = grammar, documentation, llama_grammar
        else:
            grammar, documentation, llama_grammar = self.grammar_cache[model]

        system_prompt = self.system_prompt_template.generate_prompt({"documentation": documentation})
        if data == "":
            prompt = "Create a random JSON response based on the response model."
        else:
            prompt = self.creation_prompt_template.generate_prompt({"user_input": data})

        response = self.llama_cpp_agent.get_chat_response(prompt, system_prompt=system_prompt, grammar=llama_grammar, add_response_to_chat_history=False, add_message_to_chat_history=False, streaming_callback=self.streaming_callback, **self.llama_generation_settings.as_dict())
        return extract_object_from_response(response, model)
