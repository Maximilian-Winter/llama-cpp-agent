# llama-cpp-agent Framework

## Introduction
The llama-cpp-agent framework is a tool designed for easy interaction with Large Language Models (LLMs). It provides a simple yet robust interface using llama-cpp-python, allowing users to chat with LLM models, execute structured function calls and get structured output.

## Key Features
- **Simple Chat Interface**: Engage in seamless conversations with LLMs.
- **Structured Output**: Get structured output from LLMs.
- **Function Calling**: Execute structured outputs from LLMs, enhancing the interaction capabilities.
- **Flexibility**: Suited for various applications from casual chatting to specific function executions.

## Installation
To get started with the llama-cpp-agent LLM framework, follow these steps:
1. Ensure you have Python installed on your system.
2. Clone the repository from [GitHub link](https://github.com/Maximilian-Winter/llama-cpp-agent).
3. Install the necessary dependencies as listed in the `requirements.txt` file.

## Usage Examples

### Simple Chat Example
This example demonstrates how to initiate a chat with an LLM model.
```python
import json
from llama_cpp import Llama
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
main_model = Llama(
    "../gguf-models/dpopenhermes-7b-v2.Q8_0.gguf",
    n_gpu_layers=35,
    f16_kv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=False,
    seed=42,
)
wrapped_model = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt="You are an advanced AI assistant.", predefined_messages_formatter_type=MessagesFormatterType.CHATML)

wrapped_model.get_chat_response('Write a long poem about the USA.', temperature=0.7)

```


### Structured Output
This example shows how to get structured JSON output.
```python

from enum import Enum

from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import generate_gbnf_grammar_and_documentation

main_model = Llama(
    "../gguf-models/dpopenhermes-7b-v2.Q8_0.gguf",
    n_gpu_layers=35,
    f16_kv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=False,
    seed=-1,
)


text = """The Feynman Lectures on Physics is a physics textbook based on some lectures by Richard Feynman, a Nobel laureate who has sometimes been called "The Great Explainer". The lectures were presented before undergraduate students at the California Institute of Technology (Caltech), during 1961–1963. The book's co-authors are Feynman, Robert B. Leighton, and Matthew Sands."""


class Category(Enum):
    Fiction = "Fiction"
    NonFiction = "Non-Fiction"


class Book(BaseModel):
    """
    Represents an entry about a book.
    """
    title: str = Field(..., description="Title of the book.")
    author: str = Field(..., description="Author of the book.")
    published_year: int = Field(..., description="Publishing year of the book.")
    keywords: list[str] = Field(..., description="A list of keywords.")
    category: Category = Field(..., description="Category of the book.")
    summary: str = Field(..., description="Summary of the book.")


gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([Book])
grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)


wrapped_model = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt="You are an advanced AI, tasked to create JSON database entries for books.\n\n\n" + documentation)


wrapped_model.get_chat_response(text, temperature=0.15, grammar=grammar)
```


### Function Calling Example
This example shows how to do function calling.
```python
import json

from llama_cpp import Llama, LlamaGrammar

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import generate_gbnf_grammar_and_documentation

from example_function_call_models import SendMessageToUser, GetFileList, ReadTextFile, WriteTextFileSection
from llama_cpp_agent.messages_formatter import MessagesFormatterType

gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(
    [SendMessageToUser, GetFileList, ReadTextFile, WriteTextFileSection], "function", "function_params", "Function",
    "Function Parameter")
grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)

main_model = Llama(
    "../gguf-models/dpopenhermes-7b-v2.Q8_0.gguf",
    n_gpu_layers=35,
    f16_kv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=False,
    seed=42,
)
wrapped_model = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt="You are an advanced AI, tasked to assist the user by calling functions in JSON format.\n\n\n" + documentation,
                              predefined_messages_formatter_type=MessagesFormatterType.CHATML)

response = wrapped_model.get_chat_response('Write a long poem about the USA in the "HelloUSA.txt" file.',
                                           temperature=0.15, grammar=grammar)

function_call = json.loads(response)

if function_call["function"] == "write-text-file-section":
    call_parameters = function_call["function_params"]
    call = WriteTextFileSection(**call_parameters)
    call.run()

```

### Auto coding agent
Auto coding agent example
```python
import json

from llama_cpp import Llama, LlamaGrammar

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import \
    generate_gbnf_grammar_and_documentation, sanitize_json_string, map_grammar_names_to_pydantic_model_class

from example_agent_models import SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile
from llama_cpp_agent.messages_formatter import MessagesFormatterType

pydantic_function_models = [SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile]

gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(
    pydantic_function_models, "function", "function_params", "Function",
    "Function Parameter")
grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)

output_to_pydantic_model = map_grammar_names_to_pydantic_model_class(pydantic_function_models)

main_model = Llama(
    "../gguf-models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=12,
    f16_kv=True,
    offload_kqv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=True,
    seed=42,
)

system_prompt = f'''Act as AutoCoder, your primary task is to autonomously plan, outline and implement complete software projects based on user specifications. This includes in-depth holistic project planning, writing complex code, and effective file management. You have to use JSON objects to perform functions.

# INSTRUCTIONS:
1. Begin by thoroughly understanding the project scope. Engage with the user to gather context and clarify specifications, asking pertinent questions as needed.
2. Utilize the "workspace" folder effectively for storing notes, development tasks, and research findings. Maintain a high level of organization and clarity in file naming and structuring.
3. Once specifications are confirmed, take a step back and create step by step a detailed and holistic development plan within the "workspace/development_plan" folder. For each task, prepare a separate file in markdown format encompassing:
   - All classes
   - All interfaces for these classes
   - All methods of these classes
   - Brief descriptions for each class, interface and method
4. Document your conceptual framework and key information in the "workspace/project_notes" folder, ensuring clarity and coherence.
5. Implement your plan and write the actual code.

Here are you available functions:
{documentation}'''.strip()

wrapped_model = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt=system_prompt,
                              predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL)

user_input = 'Implement a chat bot frontend in HTML, CSS and Javascript under "./workspace".'
while True:

    if user_input is None:
        user_input = "Proceed."

    response = wrapped_model.get_chat_response(
        user_input,
        temperature=0.45, mirostat_mode=2, mirostat_tau=4.0, mirostat_eta=0.1, grammar=grammar)

    sanitized = sanitize_json_string(response)
    function_call = json.loads(sanitized)
    cls = output_to_pydantic_model[function_call["function"]]
    call_parameters = function_call["function_params"]
    call = cls(**call_parameters)
    user_input = call.run()
```

## Additional Information
- **Dependencies**: pydantic for grammars based generation and of course llama-cpp-python.
