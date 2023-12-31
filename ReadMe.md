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


gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([Book] ,False)

print(gbnf_grammar)
grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=True)


wrapped_model = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt="You are an advanced AI, tasked to create JSON database entries for books.\n\n\n" + documentation)


wrapped_model.get_chat_response(text, temperature=0.15, grammar=grammar)


```


### Function Calling Example
This example shows how to do function calling.
```python
from llama_cpp import Llama

from llama_cpp_agent.llm_agent import LlamaCppAgent


from example_function_call_models import SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_call_tools import LlamaCppFunctionTool

function_tools = [LlamaCppFunctionTool(SendMessageToUser), LlamaCppFunctionTool(GetFileList), LlamaCppFunctionTool(ReadTextFile),
                  LlamaCppFunctionTool(WriteTextFile)]

function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

main_model = Llama(
    "../gguf-models/dolphin-2.6-mistral-7b-Q8_0.gguf",
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
                              system_prompt="You are an advanced AI, tasked to assist the user by calling functions in JSON format.\n\n\n" + function_tool_registry.get_documentation(),
                              predefined_messages_formatter_type=MessagesFormatterType.CHATML)

response = wrapped_model.get_chat_response('Write a long poem about the USA in the "HelloUSA.txt" file under "./".',
                                           temperature=0.75, function_tool_registry=function_tool_registry)


```

### Auto coding agent
Auto coding agent example
```python
import datetime

from llama_cpp import Llama

from llama_cpp_agent.llm_agent import LlamaCppAgent

from example_agent_models_auto_coder import SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile,
    agent_dev_folder_setup
from llama_cpp_agent.messages_formatter import MessagesFormatterType

from llama_cpp_agent.function_call_tools import LlamaCppFunctionTool

function_tools = [LlamaCppFunctionTool(SendMessageToUser), LlamaCppFunctionTool(GetFileList), LlamaCppFunctionTool(ReadTextFile),
                  LlamaCppFunctionTool(WriteTextFile, has_field_string=True)]

function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

main_model = Llama(
    "../gguf-models/neuralhermes-2.5-mistral-7b.Q8_0.gguf",
    n_gpu_layers=46,
    f16_kv=True,
    offload_kqv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=False,
    seed=42,
)

system_prompt_planner = f'''You are an advanced AI agent called AutoPlanner. As AutoPlanner your primary task is to autonomously plan complete software projects based on user specifications. You will create a complete development plans.  Your output is constrained to write JSON function call objects. The content of files is constrained to markdown code blocks with different content types like HTML, CSS, Javascript, Python or  Markdown. Here are your available functions:

{function_tool_registry.get_documentation()}'''.strip()
system_prompt_coder = f'''You are an advanced AI agent called AutoCoder. As AutoCoder your primary task is to autonomously implement complete software projects based on a development plan. Your output is constrained to write JSON function call objects. The content of files is constrained to markdown code blocks with different content types like HTML, CSS, Javascript, Python or  Markdown. Here are your available functions:

{function_tool_registry.get_documentation()}'''.strip()

task = 'Create a complete development plan for a chat bot frontend in HTML, CSS and Javascript with a dark UI.'
task_implement = 'Implement the existing development plan in the "./" folder, for a chat bot frontend in HTML, CSS and Javascript with a dark UI.'
timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")

# agent_dev_folder_setup(f"dev_{timestamp}")
agent_dev_folder_setup("agent_auto_coder_auto_planner_output")
planner_agent = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt=system_prompt_planner,
                              predefined_messages_formatter_type=MessagesFormatterType.CHATML)

coder_agent = LlamaCppAgent(main_model, debug_output=True,
                            system_prompt=system_prompt_coder,
                            predefined_messages_formatter_type=MessagesFormatterType.CHATML)
user_input = task_implement
while True:

    if user_input is None:
        user_input = "Proceed with next step."

    user_input = coder_agent.get_chat_response(
        user_input,
        temperature=0.25, top_p=1.0, top_k=0, tfs_z=0.95, repeat_penalty=1.12, function_tool_registry=function_tool_registry)



```

## Additional Information
- **Dependencies**: pydantic for grammars based generation and of course llama-cpp-python.
