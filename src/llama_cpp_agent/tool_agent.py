from typing import Type

from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel

from .llm_agent import LlamaCppAgent
from .llm_prompt_template import Prompter

from .output_parser import extract_object_from_response
from .messages_formatter import MessagesFormatterType
from .gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import generate_gbnf_grammar_and_documentation


class ToolAgent:
    pass
