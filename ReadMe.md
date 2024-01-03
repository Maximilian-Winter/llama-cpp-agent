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
