# Based on an example of the Instructor library for OpenAI


import json
from typing import List

from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import \
    generate_gbnf_grammar_and_documentation

main_model = Llama(
    "../gguf-models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=13,
    f16_kv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    offload_kqv=True,
    last_n_tokens_size=1024,
    verbose=True,
    seed=42,
)

from typing import List
from pydantic import Field


class File(BaseModel):
    """
    Correctly named file with contents.
    """

    file_name: str = Field(
        ..., description="The name of the file including the extension"
    )
    file_string: str = Field(..., description="Correct contents of a file")

    def save(self):
        with open(self.file_name, "w") as f:
            f.write(self.file_string)


class Program(BaseModel):
    """
    Set of files that represent a complete and correct program
    """

    files: List[File] = Field(..., description="List of files")


gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([Program], True)

print(gbnf_grammar)
grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=True)

wrapped_model = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt="You are a world class programming AI capable of writing correct python scripts and modules. You will name files correct, include __init__.py files and write correct python code with correct imports.\n\nYou are responding in JSON format.\n\nAvailable JSON response models:\n\n" + documentation.strip() + "\n\nAlways provide full implementation to the user!!!!",
                              predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL)


def develop(data: str) -> Program:
    prompt = data
    response = wrapped_model.get_chat_response(message=prompt, temperature=0.65, mirostat_mode=2, mirostat_tau=4.0,
                                               mirostat_eta=0.1, grammar=grammar)
    ai_program = json.loads(response)
    cls = Program
    ai_program = cls(**ai_program)
    return ai_program


program = develop(
    """
    Implement system for a swarm of large language model agents using huggingface transformers. The system should be based on natural behavior of ants and bees."""
)

for file in program.files:
    print(file.file_name)
    print("-")
    print(file.body)
    print("\n\n\n")
