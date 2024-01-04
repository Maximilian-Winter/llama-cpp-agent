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

## Usage
The llama-cpp-agent framework is designed to be easy to use. The following sections will guide you through the process of using the framework.

### Chat usage
To chat with an LLM model, you need to create an instance of the `LlamaCppAgent` class. The constructor takes the following parameters:
- `main_model`: The LLM model to use for the chat. This is an instance of the `Llama` class from the llama-cpp-python library.
- `name`: The name of the agent. Defaults to `llamacpp_agent`.
- `system_prompt`: The system prompt to use for the chat. Defaults to `You are a helpful assistant.`.
- `predefined_messages_formatter_type`: The type of predefined messages formatter to use. Defaults to `MessagesFormatterType.CHATML`.
- `debug_output`: Whether to print debug output to the console. Defaults to `False`.

#### Predefined Messages Formatter
The llama-cpp-agent framework uses custom messages formatters to format messages for the LLM model. The `MessagesFormatterType` enum defines the available predefined formatters. The following predefined formatters are available:
- `MessagesFormatterType.CHATML`: Formats messages using the CHATML format.
- `MessagesFormatterType.MIXTRAL`: Formats messages using the MIXTRAL format.
- `MessagesFormatterType.VICUNA`: Formats messages using the VICUNA format.
- `MessagesFormatterType.LLAMA_2`: Formats messages using the LLAMA 2 format.
- `MessagesFormatterType.SYNTHIA`: Formats messages using the SYNTHIA format.
- `MessagesFormatterType.NEURAL_CHAT`: Formats messages using the NEURAL CHAT format.
- `MessagesFormatterType.SOLAR`: Formats messages using the SOLAR format.

You can also define your own custom messages formatter by creating an instance of the `MessagesFormatter` class.
The `MessagesFormatter` class takes the following parameters:
- `PRE_PROMPT`: The pre-prompt to use for the messages. 
- `SYS_PROMPT_START`: The system prompt start to use for the messages.
- `SYS_PROMPT_END`: The system prompt end to use for the messages.
- `USER_PROMPT_START`: The user prompt start to use for the messages.
- `USER_PROMPT_END`: The user prompt end to use for the messages.
- `ASSISTANT_PROMPT_START`: The assistant prompt start to use for the messages.
- `ASSISTANT_PROMPT_END`: The assistant prompt end to use for the messages.
- `INCLUDE_SYS_PROMPT_IN_MESSAGE`: Whether to include the system prompt in the message.
- `DEFAULT_STOP_SEQUENCES`: The default stop sequences to use for the messages.

After creating an instance of the `MessagesFormatter` class, you can use it by setting the `messages_formatter` of the `LlamaCppAgent` instance to the instance of the `MessagesFormatter` class.

#### Chatting
To chat with the LLM model, you can use the `get_chat_response` method of the `LlamaCppAgent` class. The `get_chat_response` method takes the following parameters:
- `message`: The message to send to the LLM model. Defaults to `None`.
- `role`: The role of the message. Defaults to `user`.
- `system_prompt`: A override for the system prompt. Defaults to `None` and uses the agent system prompt passed at creation.
- `grammar`: The grammar to use for constraining the LLM response. Defaults to `None`.
- `function_tool_registry`: The function tool registry to use for the chat. Defaults to `None`.
- `max_tokens`: The maximum number of tokens to use for the chat. Defaults to `0`.
- `temperature`: The temperature to use for the chat. Defaults to `0.4`.
- `top_k`: The top k to use for the chat. Defaults to `0`.
- `top_p`: The top p to use for the chat. Defaults to `1.0`.
- `min_p`: The min p to use for the chat. Defaults to `0.05`.
- `typical_p`: The typical p to use for the chat. Defaults to `1.0`.              
- `repeat_penalty`: The repeat penalty to use for the chat. Defaults to `1.0`.
- `mirostat_mode`: The mirostat mode to use for the chat. Defaults to `0`.
- `mirostat_tau`: The mirostat tau to use for the chat. Defaults to `5.0`.
- `mirostat_eta`: The mirostat eta to use for the chat. Defaults to `0.1`.
- `tfs_z`: The tfs z to use for the chat. Defaults to `1.0`.
- `stop_sequences`: The stop sequences to use for the chat. Defaults to `None`.
- `stream`: Whether to stream the chat. Defaults to `True`.
- `k_last_messages`: The k last messages to use for the chat. Defaults to `-1` which takes all messages in the chat history.
- `add_response_to_chat_history`: Whether to add the response to the chat history. Defaults to `True`.
- `add_message_to_chat_history`: Whether to add the message to the chat history. Defaults to `True`.
- `print_output`: Whether to print the output. Defaults to `True`.

## Structured Output Usage
To get structured output from an LLM model, you can use an instance of the `StructuredOutputAgent` class. The constructor takes the following parameters:
- `main_model`: The LLM model to use for the structured output. This is an instance of the `Llama` class from the llama-cpp-python library.
- `messages_formatter_type`: The type of messages formatter to use. Defaults to `MessagesFormatterType.CHATML`.
- `debug_output`: Whether to print debug output to the console. Defaults to `False`.

To set a custom messages formatter, you can use the `llama_cpp_agent.messages_formatter` property of the `StructuredOutputAgent` class.

#### Structured Output
To create structured output from the LLM model, you can use the `create_object` method of the `StructuredOutputAgent` class. The `create_object` method takes the following parameters:
- `cls`: The pydantic class used for creating the structured output.
- `data`: The data to use for the structured output. Defaults to `None` which creates a random object of cls class.

This will return an instance of the pydantic class.

## Function Calling Usage
To utilize function calling with an LLM model, you can use the get_chat_response method of a `LlamaCppAgent` with a `function_tool_registry`. The `function_tool_registry` is an instance of the `LlamaCppFunctionToolRegistry` class. You can create a `LlamaCppFunctionToolRegistry` instance by passing a list of `LlamaCppFunctionTool` instances to the static `get_function_tool_registry` method of the `LlamaCppAgent` class. The `LlamaCppFunctionTool` class takes the following parameters:
- `model`: The pydantic class defining the function call, it must have a `run` to actually execute the function call.
- `has_field_string`: Whether the model has a `field_string` field. Defaults to `False`. A `field_string` field is a special field used to allow the LLM to write relatively unconstrained output by letting it write the `field_string` as a Markdown code block. Which is useful for file writing.

After passing the list of `LlamaCppFunctionTool` instances to the `get_function_tool_registry` method, you can use the returned `LlamaCppFunctionToolRegistry` instance as the `function_tool_registry` parameter of the `get_chat_response` method of the `LlamaCppAgent` class.

## Usage Examples

### Simple Chat Example
This example demonstrates how to initiate a chat with an LLM model.
```python
from llama_cpp import Llama
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
main_model = Llama(
    "../gguf-models/dolphin-2.7-mixtral-8x7b.Q4_K_M.gguf",
    n_gpu_layers=13,
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
llama_cpp_agent = LlamaCppAgent(main_model, debug_output=False,
                              system_prompt="You are Dolphin, an uncensored and unbiased AI assistant.", predefined_messages_formatter_type=MessagesFormatterType.CHATML)


while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    user_input = llama_cpp_agent.get_chat_response(user_input, temperature=0.7)
    print("AI: " + user_input)

```
### Structured Output
This example shows how to get structured JSON output using the StructureOutputAgent class.
```python
# Example agent that uses the StructuredOutputAgent class to create a dataset entry of a book out of unstructured data.

from enum import Enum

from llama_cpp import Llama
from pydantic import BaseModel, Field

from llama_cpp_agent.structured_output_agent import StructuredOutputAgent


# Example enum for our output model
class Category(Enum):
    Fiction = "Fiction"
    NonFiction = "Non-Fiction"


# Example output model
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


main_model = Llama(
    "../gguf-models/nous-hermes-2-solar-10.7b.Q6_K.gguf",
    n_gpu_layers=49,
    offload_kqv=True,
    f16_kv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=4096,
    last_n_tokens_size=1024,
    verbose=False,
    seed=42,
)

structured_output_agent = StructuredOutputAgent(main_model, debug_output=True)

text = """The Feynman Lectures on Physics is a physics textbook based on some lectures by Richard Feynman, a Nobel laureate who has sometimes been called "The Great Explainer". The lectures were presented before undergraduate students at the California Institute of Technology (Caltech), during 1961–1963. The book's co-authors are Feynman, Robert B. Leighton, and Matthew Sands."""
print(structured_output_agent.create_object(Book, text))
```
Example output
```text
 { "title": "The Feynman Lectures on Physics"  ,  "author": "Richard Feynman, Robert B. Leighton, Matthew Sands"  ,  "published_year": 1963 ,  "keywords": [ "physics" , "textbook" , "Nobel laureate" , "The Great Explainer" , "California Institute of Technology" , "undergraduate" , "lectures"  ] ,  "category": "Non-Fiction" ,  "summary": "The Feynman Lectures on Physics is a physics textbook based on lectures by Nobel laureate Richard Feynman, known as 'The Great Explainer'. The lectures were presented to undergraduate students at Caltech between 1961 and 1963. Co-authors of the book are Feynman, Robert B. Leighton, and Matthew Sands."  }


title='The Feynman Lectures on Physics' author='Richard Feynman, Robert B. Leighton, Matthew Sands' published_year=1963 keywords=['physics', 'textbook', 'Nobel laureate', 'The Great Explainer', 'California Institute of Technology', 'undergraduate', 'lectures'] category=<Category.NonFiction: 'Non-Fiction'> summary="The Feynman Lectures on Physics is a physics textbook based on lectures by Nobel laureate Richard Feynman, known as 'The Great Explainer'. The lectures were presented to undergraduate students at Caltech between 1961 and 1963. Co-authors of the book are Feynman, Robert B. Leighton, and Matthew Sands."

```

### Function Calling Example
This example shows how to do function calling.
```python
from enum import Enum

from llama_cpp import Llama
from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent

from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_call_tools import LlamaCppFunctionTool


# Simple calculator tool for the agent that can add, subtract, multiply, and divide.
class MathOperation(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


class Calculator(BaseModel):
    """
    Perform a math operation on two numbers.
    """
    number_one: float = Field(..., description="First number.", max_precision=5, min_precision=2)
    operation: MathOperation = Field(..., description="Math operation to perform.")
    number_two: float = Field(..., description="Second number.", max_precision=5, min_precision=2)

    def run(self):
        if self.operation == MathOperation.ADD:
            return self.number_one + self.number_two
        elif self.operation == MathOperation.SUBTRACT:
            return self.number_one - self.number_two
        elif self.operation == MathOperation.MULTIPLY:
            return self.number_one * self.number_two
        elif self.operation == MathOperation.DIVIDE:
            return self.number_one / self.number_two
        else:
            raise ValueError("Unknown operation.")


function_tools = [LlamaCppFunctionTool(Calculator)]

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
llama_cpp_agent = LlamaCppAgent(main_model, debug_output=False,
                                system_prompt="You are an advanced AI, tasked to assist the user by calling functions in JSON format.\n\n\n" + function_tool_registry.get_documentation(),
                                predefined_messages_formatter_type=MessagesFormatterType.CHATML)
user_input = 'What is 42 * 42?'
print(llama_cpp_agent.get_chat_response(user_input, temperature=0.45, function_tool_registry=function_tool_registry))

```
Example output
```text
{ "function": "calculator","function_parameters": { "number_one": 42.00000 ,  "operation": "multiply" ,  "number_two": 42.00000 }}
1764.0
```
### Knowledge Graph Creation Example
This example, based on an example of the Instructor library for OpenAI,
demonstrates how to create a knowledge graph using the llama-cpp-agent framework.
```python
import json
from typing import List

from enum import Enum

from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import generate_gbnf_grammar_and_documentation

main_model = Llama(
    "../gguf-models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=13,
    f16_kv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=True,
    seed=42,
)

class Node(BaseModel):
    id: int
    label: str
    color: str


class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"


class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(..., default_factory=list)
    edges: List[Edge] = Field(..., default_factory=list)




gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([KnowledgeGraph],False)

print(gbnf_grammar)
grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=True)


llama_cpp_agent = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt="You are an advanced AI assistant responding in JSON format.\n\nAvailable JSON response models:\n\n" + documentation)


from graphviz import Digraph


def visualize_knowledge_graph(kg: KnowledgeGraph):
    dot = Digraph(comment="Knowledge Graph")

    # Add nodes
    for node in kg.nodes:
        dot.node(str(node.id), node.label, color=node.color)

    # Add edges
    for edge in kg.edges:
        dot.edge(str(edge.source), str(edge.target), label=edge.label, color=edge.color)

    # Render the graph
    dot.render("knowledge_graph.gv", view=True)


def generate_graph(user_input: str) -> KnowledgeGraph:
    prompt = f'''Help me understand the following by describing it as a detailed knowledge graph: {user_input}'''.strip()
    response = llama_cpp_agent.get_chat_response(message=prompt, temperature=0.65, mirostat_mode=0, mirostat_tau=3.0,
                                               mirostat_eta=0.1, grammar=grammar)
    knowledge_graph = json.loads(response)
    cls = KnowledgeGraph
    knowledge_graph = cls(**knowledge_graph)
    return knowledge_graph


graph = generate_graph("Teach me about quantum mechanics")
visualize_knowledge_graph(graph)
```
Example Output:
![KG](https://raw.githubusercontent.com/Maximilian-Winter/llama-cpp-agent/master/generated_knowledge_graph_example/knowledge_graph.png)
### Auto coding agent
Auto coding agent example with `field_string`
```python
from llama_cpp import Llama

from llama_cpp_agent.llm_agent import LlamaCppAgent

from llama_cpp_agent.messages_formatter import MessagesFormatterType

from llama_cpp_agent.function_call_tools import LlamaCppFunctionTool


import datetime
import os
from enum import Enum
from pathlib import Path

from pydantic import Field, BaseModel

base_folder = "dev"



def agent_dev_folder_setup(custom_base_folder=None):
    global base_folder
    base_folder = custom_base_folder
    os.makedirs(base_folder, exist_ok=True)


class WriteOperation(Enum):
    CREATE_FILE = "create-file"
    APPEND_FILE = "append-file"
    OVERWRITE_FILE = "overwrite-file"


class WriteTextFile(BaseModel):
    """
    Open file for writing and modification.
    """

    directory: str = Field(
        ...,
        description="Path to the directory where the file is located or will be created. Without filename !!!!"
    )

    filename_without_extension: str = Field(
        ...,
        description="Name of the target file without the file extension."
    )

    filename_extension: str = Field(
        ...,
        description="File extension indicating the file type, such as '.txt', '.py', '.md', etc."
    )

    write_operation: WriteOperation = Field(...,
                                            description="Write operation performed, 'create-file', 'append-file' or 'overwrite-file'")

    # Allow free output for the File Content to Enhance LLM Output

    file_string: str = Field(...,
                             description="Special markdown code block for unconstrained output.")

    def run(self):

        if self.directory == "":
            self.directory = "./"
        if self.filename_extension == "":
            self.filename_extension = ".txt"
        if self.filename_extension[0] != ".":
            self.filename_extension = "." + self.filename_extension
        if self.directory[0] == "." and len(self.directory) == 1:
            self.directory = "./"

        if self.directory[0] == "." and len(self.directory) > 1 and self.directory[1] != "/":
            self.directory = "./" + self.directory[1:]

        if self.directory[0] == "/":
            self.directory = self.directory[1:]

        if self.directory.endswith(f"{self.filename_without_extension}{self.filename_extension}"):
            self.directory = self.directory.replace(f"{self.filename_without_extension}{self.filename_extension}", "")
        file_path = os.path.join(self.directory, f"{self.filename_without_extension}{self.filename_extension}")
        file_path = os.path.join(base_folder, file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Determine the write mode based on the write_operation attribute
        if self.write_operation == WriteOperation.CREATE_FILE:
            write_mode = 'w'  # Create a new file, error if file exists
        elif self.write_operation == WriteOperation.APPEND_FILE:
            write_mode = 'a'  # Append if file exists, create if not
        elif self.write_operation == WriteOperation.OVERWRITE_FILE:
            write_mode = 'w'  # Overwrite file if it exists, create if not
        else:
            raise ValueError(f"Invalid write operation: {self.write_operation}")

        # Write back to file
        with open(file_path, write_mode, encoding="utf-8") as file:
            file.writelines(self.file_string)

        return f"Content written to '{self.filename_without_extension}{self.filename_extension}'."


class ReadTextFile(BaseModel):
    """
    Reads the text content of a specified file and returns it.
    """

    directory: str = Field(
        description="Path to the directory containing the file. Without filename !!!!"
    )

    file_name: str = Field(
        ...,
        description="The name of the file to be read, including its extension (e.g., 'document.txt')."
    )

    def run(self):
        try:
            if self.directory.endswith(f"{self.file_name}"):
                self.directory = self.directory.replace(f"{self.file_name}", "")
            if not os.path.exists(f"{base_folder}/{self.directory}/{self.file_name}"):
                return f"File '{self.directory}/{self.file_name}' doesn't exists!"
            with open(f"{base_folder}/{self.directory}/{self.file_name}", "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip() == "":
                return f"File '{self.file_name}' is empty!"
        except Exception as e:
            return f"Error reading file '{self.file_name}': {e}"
        return f"File '{self.file_name}':\n{content}"


class GetFileList(BaseModel):
    """
    Scans a specified directory and creates a list of all files within that directory, including files in its subdirectories.
    """

    directory: str = Field(

        description="Path to the directory where files will be listed. This path can include subdirectories to be scanned."
    )

    def run(self):
        filenames = "File List:\n"
        counter = 1
        base_path = Path(base_folder) / self.directory

        for root, _, files in os.walk(os.path.join(base_folder, self.directory)):
            for file in files:
                relative_root = Path(root).relative_to(base_path)
                filenames += f"{counter}. {relative_root / file}\n"
                counter += 1

        if counter == 1:
            return f"Directory '{self.directory}' is empty!"
        return filenames


class SendMessageToUser(BaseModel):
    """
    Send a message to the User.
    """

    message: str = Field(..., description="Message you want to send to the user.")

    def run(self):
        print("Message: " + self.message)

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

### Agent core memory example (Editable by agent)
Agent core memory example
```python
from llama_cpp import Llama, LlamaGrammar

from llama_cpp_agent.llm_agent import LlamaCppAgent

from example_agent_models_auto_coder import SendMessageToUser
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.memory_tools import AgentCoreMemory
from llama_cpp_agent.function_call_tools import LlamaCppFunctionTool

agent_core_memory = AgentCoreMemory()

function_tools = [LlamaCppFunctionTool(SendMessageToUser)]

function_tools.extend(agent_core_memory.get_tool_list())
function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

main_model = Llama(
    "../gguf-models/openhermes-2.5-mistral-7b.Q8_0.gguf",
    n_gpu_layers=45,
    f16_kv=True,
    offload_kqv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=True,
    seed=-1,
)

llama_cpp_agent = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt=system_prompt2,
                              predefined_messages_formatter_type=MessagesFormatterType.CHATML)

user_input = 'Add "Be friendly" under "Core-Guidelines" to your core memory.'
while True:

    if user_input is None:
        user_input = "Hello."

    user_input = llama_cpp_agent.get_chat_response(
        user_input,
        system_prompt=f"You are a advanced helpful AI assistant interacting through calling functions in form of JSON objects.\n\n{agent_core_memory.get_core_memory_manager().build_core_memory_context()}\n\nHere are your available functions:\n\n" + function_tool_registry.get_documentation(),
        temperature=1.25, function_tool_registry=function_tool_registry)


```

### Agent retrieval memory example
Agent retrieval memory example
```python
from llama_cpp import Llama, LlamaGrammar

from llama_cpp_agent.llm_agent import LlamaCppAgent

from example_agent_models_auto_coder import SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.memory_tools import AgentRetrievalMemory
from llama_cpp_agent.function_call_tools import LlamaCppFunctionTool

agent_retrieval_memory = AgentRetrievalMemory()

function_tools = [LlamaCppFunctionTool(SendMessageToUser)]

function_tools.extend(agent_retrieval_memory.get_tool_list())
function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

main_model = Llama(
    "../gguf-models/openhermes-2.5-mistral-7b.Q8_0.gguf",
    n_gpu_layers=45,
    f16_kv=True,
    offload_kqv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=True,
    seed=-1,
)

llama_cpp_agent = LlamaCppAgent(main_model, debug_output=True,
                                system_prompt=system_prompt2,
                                predefined_messages_formatter_type=MessagesFormatterType.CHATML)

user_input = 'Add my Birthday the 1991.12.11 to the retrieval memory.'
while True:

    if user_input is None:
        user_input = "Hello."

    user_input = llama_cpp_agent.get_chat_response(
        user_input,
        system_prompt=f"You are a advanced helpful AI assistant interacting through calling functions in form of JSON objects.\n\nHere are your available functions:\n\n" + function_tool_registry.get_documentation(),
        temperature=1.25, function_tool_registry=function_tool_registry)

```
## Additional Information
- **Dependencies**: pydantic for grammars based generation and of course llama-cpp-python.
