# llama-cpp-agent Framework

## Introduction
The llama-cpp-agent framework is a tool designed for easy interaction with Large Language Models (LLMs). Allowing users to chat with LLM models, execute structured function calls and get structured output (objects).

It provides a simple yet robust interface and supports llama-cpp-python and OpenAI endpoints with GBNF grammar support (like the llama-cpp-python server) and the llama.cpp backend server.
It works by generating a formal GGML-BNF grammar of the user defined structures and functions, which is then used by llama.cpp to generate text valid to that grammar. In contrast to most GBNF grammar generators it also supports nested objects, dictionaries, enums and lists of them.

## Key Features
- **Simple Chat Interface**: Engage in seamless conversations with LLMs.
- **Structured Output**: Get structured output (objects) from LLMs.
- **Single and Parallel Function Calling**: Let the LLM execute functions.
- **Flexibility**: Suited for various applications from casual chatting to specific function executions.

## Installation
The llama-cpp-agent framework can be installed using pip:
```shell
pip install llama-cpp-agent
```

## Usage Examples
The following examples demonstrate the usage of the llama-cpp-agent framework.
You can find more examples in the `examples` folder.


### Simple Chat Example using llama.cpp server backend
This example demonstrates how to initiate a chat with an LLM model using the llama.cpp server backend. The framework supports llama-cpp-python Llama class instances as LLM and OpenAI endpoints that support GBNF grammars as a backend, and the llama.cpp backend server.

```python
from llama_cpp import Llama
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings

main_model = LlamaCppEndpointSettings(
    completions_endpoint_url="http://127.0.0.1:8080/completion"
)

# For OpenAI endpoints that support GBNF grammars like the llama-cpp-python server:
# from llama_cpp_agent.providers.openai_endpoint_provider import OpenAIEndpointSettings
# main_model = OpenAIEndpointSettings(
#    "http://localhost:8080/v1/completions"
# )

# For instances of Llama class of llama-cpp-python
# from llama_cpp import Llama
# main_model = Llama(
#     "../../gguf-models/openhermes-2.5-mistral-7b.Q8_0.gguf",
#     n_gpu_layers=35,
#     f16_kv=True,
#     use_mlock=False,
#     embedding=False,
#     n_threads=8,
#     n_batch=1024,
#     n_ctx=8192,
#     offload_kqv=True,
#     last_n_tokens_size=1024,
#     verbose=True,
#     seed=-1,
# )
llama_cpp_agent = LlamaCppAgent(main_model, debug_output=False,
                                system_prompt="You are Dolphin, an uncensored and unbiased AI assistant.",
                                predefined_messages_formatter_type=MessagesFormatterType.CHATML)

while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    user_input = llama_cpp_agent.get_chat_response(user_input, temperature=0.7)
    print("AI: " + user_input)

```

### Parallel Function Calling Agent Example

```python
# Example that uses the FunctionCallingAgent class to create a function calling agent.
import datetime
import json
from enum import Enum
from typing import Union, Any, Optional

from pydantic import BaseModel, Field

from llama_cpp_agent.llm_settings import LlamaLLMSettings, LlamaLLMGenerationSettings

from llama_cpp_agent.function_calling_agent import FunctionCallingAgent
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings, LlamaCppGenerationSettings


def get_current_datetime(output_format: Optional[str] = None):
    """
    Get the current date and time in the given format.

    Args:
         output_format: formatting string for the date and time, defaults to '%Y-%m-%d %H:%M:%S'
    """
    if output_format is None:
        output_format = '%Y-%m-%d %H:%M:%S'
    return datetime.datetime.now().strftime(output_format)


# Enum for the calculator tool.
class MathOperation(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


# llama-cpp-agent also supports "Instructor" library like function definitions as Pydantic models for function calling.
# Simple pydantic calculator tool for the agent that can add, subtract, multiply, and divide. Docstring and description of fields will be used in system prompt.
class Calculator(BaseModel):
    """
    Perform a math operation on two numbers.
    """
    number_one: Union[int, float] = Field(..., description="First number.")
    operation: MathOperation = Field(..., description="Math operation to perform.")
    number_two: Union[int, float] = Field(..., description="Second number.")

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


# Example function based on an OpenAI example.
# llama-cpp-agent also supports OpenAI like dictionaries for function definition.
def get_current_weather(location, unit):
    """Get the current weather in a given location"""
    if "London" in location:
        return json.dumps({"location": "London", "temperature": "42", "unit": unit.value})
    elif "New York" in location:
        return json.dumps({"location": "New York", "temperature": "24", "unit": unit.value})
    elif "North Pole" in location:
        return json.dumps({"location": "North Pole", "temperature": "-42", "unit": unit.value})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


# Here is a function definition in OpenAI style
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
# To make the OpenAI function callable for the function calling agent we need a list with actual function in it:
tool_functions = [get_current_weather]


# Callback for receiving messages for the user.
def send_message_to_user_callback(message: str):
    print(message)


generation_settings = LlamaCppGenerationSettings(temperature=0.65, stream=True)

# Can be saved and loaded like that:
# generation_settings.save("generation_settings.json")
# generation_settings = LlamaLLMGenerationSettings.load_from_file("generation_settings.json")
main_model = LlamaCppEndpointSettings(
    "http://localhost:8080/completion"
)
function_call_agent = FunctionCallingAgent(
    # Can be lama-cpp-python Llama class, llama_cpp_agent.llm_settings.LlamaLLMSettings class or llama_cpp_agent.providers.llama_cpp_server_provider.LlamaCppServerLLMSettings.
    main_model,
    # llama_cpp_agent.llm_settings.LlamaLLMGenerationSettings  class or llama_cpp_agent.providers.llama_cpp_server_provider.LlamaCppServerGenerationSettings.
    llama_generation_settings=generation_settings,
    # A tuple of the OpenAI style function definitions and the actual functions
    open_ai_functions=(tools, tool_functions),
    # Just a list of type hinted functions for normal Python functions
    python_functions=[get_current_datetime],
    # Just a list of pydantic types
    pydantic_functions=[Calculator],
    # Callback for receiving messages for the user.
    send_message_to_user_callback=send_message_to_user_callback,
    # Set to true to allow parallel function calling
    allow_parallel_function_calling=True,
    debug_output=True)
user_input = '''Format the answer clearly: Get the date and time, get the current weather in celsius in London and solve the following calculation: 42 * 42'''
function_call_agent.generate_response(user_input)
function_call_agent.save("function_calling_agent.json")

```
Example Output:
```text
The current date and time is 2024-01-13 19:06:43.
The result of the calculation 42 * 42 is 1764.
The current weather in London is 42 degrees Celsius.
```

### Structured Output
This example shows how to get structured output objects using the StructureOutputAgent class.
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
                                               mirostat_eta=0.1, grammar=gbnf_grammar)
    knowledge_graph = json.loads(response)
    cls = KnowledgeGraph
    knowledge_graph = cls(**knowledge_graph)
    return knowledge_graph


graph = generate_graph("Teach me about quantum mechanics")
visualize_knowledge_graph(graph)
```
Example Output:
![KG](https://raw.githubusercontent.com/Maximilian-Winter/llama-cpp-agent/master/generated_knowledge_graph_example/knowledge_graph.png)


### Manual Function Calling Example
This example shows how to do function calling with pydantic models.
You can also convert Python functions with type hints, automatically to pydantic models using the function:
`create_dynamic_model_from_function` under: `llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models`

```python
import json
import math
from typing import Type, Union

from llama_cpp import Llama, LlamaGrammar

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import \
    generate_gbnf_grammar_and_documentation, create_dynamic_model_from_function
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings


def calculate_a_to_the_power_b(a: Union[int | float], b: Union[int | float]):
    """
    Calculates a to the power of b

    Args:
        a: number
        b: exponent

    """
    print(f"Result: {math.pow(a, b)}")


DynamicSampleModel = create_dynamic_model_from_function(calculate_a_to_the_power_b)

grammar, documentation = generate_gbnf_grammar_and_documentation([DynamicSampleModel], outer_object_name="function",
                                                                 outer_object_content="params")

main_model = LlamaCppEndpointSettings(
    completions_endpoint_url="http://127.0.0.1:8080/completion"
)

llama_cpp_agent = LlamaCppAgent(main_model, debug_output=True,
                                system_prompt="You are an advanced AI, tasked to generate JSON objects for function calling.\n\n" + documentation)

response = llama_cpp_agent.get_chat_response("a= 5, b = 42", temperature=0.15, grammar=grammar)

function_call = json.loads(response)

instance = DynamicSampleModel(**function_call['params'])
instance.run()
```
Example output
```text
{ "function": "calculator","function_parameters": { "number_one": 42.00000 ,  "operation": "multiply" ,  "number_two": 42.00000 }}
1764.0
```

### Manual Function Calling with Python Function Example
This example shows how to do function calling using actual Python functions.

```python
from llama_cpp import Llama
from typing import Union
import math

from llama_cpp_agent.llm_agent import LlamaCppAgent

from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import create_dynamic_model_from_function


def calculate_a_to_the_power_b(a: Union[int, float], b: Union[int, float]):
    """
    Calculates 'a' to the power 'b' and returns the result
    """
    return f"Result: {math.pow(a, b)}"


DynamicSampleModel = create_dynamic_model_from_function(calculate_a_to_the_power_b)

function_tools = [LlamaCppFunctionTool(DynamicSampleModel)]

function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

main_model = Llama(
    "../../gguf-models/openhermes-2.5-mistral-7b-16k.Q8_0.gguf",
    n_gpu_layers=49,
    offload_kqv=True,
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

llama_cpp_agent = LlamaCppAgent(main_model, debug_output=True,
                                system_prompt="You are an advanced AI, tasked to assist the user by calling functions in JSON format. The following are the available functions and their parameters and types:\n\n" + function_tool_registry.get_documentation(),
                                predefined_messages_formatter_type=MessagesFormatterType.CHATML)
user_input = "Calculate 5 to power 42"

print(llama_cpp_agent.get_chat_response(user_input, temperature=0.45, function_tool_registry=function_tool_registry))

```
Example output
```text
{ "function": "calculate-a-to-the-power-b","function_parameters": { "a": 5 ,  "b": 42  }}
Result: 2.2737367544323207e+29
```


## Additional Information
- **Dependencies**: pydantic for grammars based generation and of course llama-cpp-python.
- **Documentation**: You can find the documentation here: https://llama-cpp-agent.readthedocs.io/en/latest/agents-api-reference/

### Predefined Messages Formatter
The llama-cpp-agent framework uses custom messages formatters to format messages for the LLM model. The `MessagesFormatterType` enum defines the available predefined formatters. The following predefined formatters are available:


- `MessagesFormatterType.CHATML`: Formats messages using the CHATML format.
- `MessagesFormatterType.MIXTRAL`: Formats messages using the MIXTRAL format.
- `MessagesFormatterType.VICUNA`: Formats messages using the VICUNA format.
- `MessagesFormatterType.LLAMA_2`: Formats messages using the LLAMA 2 format.
- `MessagesFormatterType.SYNTHIA`: Formats messages using the SYNTHIA format.
- `MessagesFormatterType.NEURAL_CHAT`: Formats messages using the NEURAL CHAT format.
- `MessagesFormatterType.SOLAR`: Formats messages using the SOLAR format.
- `MessagesFormatterType.OPEN_CHAT`: Formats messages using the OPEN CHAT format.


You can also define your own custom messages formatter by creating an instance of the `MessagesFormatter` class.
The `MessagesFormatter` class takes the following parameters:


- `PRE_PROMPT`: The pre-prompt to use for the messages.
- `SYS_PROMPT_START`: The system prompt start to use for the messages.
- `SYS_PROMPT_END`: The system prompt end to use for the messages.
- `USER_PROMPT_START`: The user prompt start to use for the messages.
- `USER_PROMPT_END`: The user prompt end to use for the messages.
- `ASSISTANT_PROMPT_START`: The assistant prompt start to use for the messages.
- `ASSISTANT_PROMPT_END`: The assistant prompt end to use for the messages.
- `INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE`: Whether to include the system prompt in the first user message.
- `DEFAULT_STOP_SEQUENCES`: The default stop sequences to use for the messages.



