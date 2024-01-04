from typing import Type

from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel

from .llm_agent import LlamaCppAgent
from .llm_prompt_template import Prompter

from .output_parser import extract_object_from_response
from .messages_formatter import MessagesFormatterType
from .gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import generate_gbnf_grammar_and_documentation


class StructuredOutputAgent:

    def __init__(self, llama_llm: Llama, messages_formatter_type: MessagesFormatterType = MessagesFormatterType.CHATML,
                 debug_output: bool = False):
        self.grammar_cache = {}
        self.system_prompter = Prompter.from_string(
            "You are an advanced AI agent. You are tasked to assist the user by creating structured output in JSON format.\n\n{documentation}")
        self.creation_prompter = Prompter.from_string(
            "Create an JSON response based on the following input.\n\nInput:\n\n{user_input}")
        self.llama_cpp_agent = LlamaCppAgent(llama_llm, debug_output=debug_output,
                                             system_prompt="",
                                             predefined_messages_formatter_type=messages_formatter_type)

    def create_object(self, model: Type[BaseModel], data: str = "") -> BaseModel:
        """
        Creates an object of the given model from the given data.
        :param model: The model to create the object from.
        :param data: The data to create the object from.
        :return: The created object.
        """
        if model not in self.grammar_cache:
            grammar, documentation = generate_gbnf_grammar_and_documentation([model], False,
                                                                             model_prefix="Response Model",
                                                                             fields_prefix="Response Model Field")
            llama_grammar = LlamaGrammar.from_string(grammar, verbose=False)
            self.grammar_cache[model] = grammar, documentation, llama_grammar
        else:
            grammar, documentation, llama_grammar = self.grammar_cache[model]

        system_prompt = self.system_prompter.generate_prompt({"documentation": documentation})
        if data == "":
            prompt = "Create a random JSON response based on the response model."
        else:
            prompt = self.creation_prompter.generate_prompt({"user_input": data})

        response = self.llama_cpp_agent.get_chat_response(prompt, temperature=0.25, system_prompt=system_prompt, grammar=llama_grammar, add_response_to_chat_history=False, add_message_to_chat_history=False)
        return extract_object_from_response(response, model)
