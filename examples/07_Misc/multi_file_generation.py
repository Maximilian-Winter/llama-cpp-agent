# Based on an example of the Instructor library for OpenAI


import json
from typing import List

from pydantic import BaseModel
from pydantic import Field

from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import (
    generate_gbnf_grammar_and_documentation,
)
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import (
    LlamaCppEndpointSettings,
)

model = LlamaCppEndpointSettings(completions_endpoint_url="http://127.0.0.1:8080/completion")


class File(BaseModel):
    """
    Correctly named file with contents.
    """

    file_name: str = Field(..., description="The name of the file including the extension")

    # markdown_code_block is a special field used to give the LLM relatively free output to generate the file contents
    file_content: str = Field(..., description="Correct contents of a file")

    def save(self):
        with open(self.file_name, "w") as f:
            f.write(self.file_content)


class Program(BaseModel):
    """
    Set of files that represent a complete and correct program
    """

    files: List[File] = Field(..., description="List of files")


gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(pydantic_model_list=[Program])

llama_cpp_agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt="You are a world class programming AI capable of writing correct python scripts and modules. You will name files correct, include __init__.py files and write correct python code with correct imports.\n\nYou are responding in JSON format.\n\nAvailable JSON response models:\n\n"
    + documentation.strip()
    + "\n\nAlways provide full implementation to the user!!!!",
    predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL,
)


def develop(data: str) -> Program:
    prompt = data
    response = llama_cpp_agent.get_chat_response(
        message=prompt,
        temperature=0.45,
        mirostat_mode=0,
        mirostat_tau=4.0,
        mirostat_eta=0.1,
        grammar=gbnf_grammar,
    )
    json_obj = json.loads(response)
    cls = Program
    ai_program = cls(**json_obj)
    return ai_program


program = develop(
    """
Implement system for a swarm of large language model agents using huggingface transformers. The system should be based on natural behavior of ants and bees.""".strip()
)

for file in program.files:
    print(file.file_name)
    print("-")
    print(file.file_content)
    print("\n\n\n")
