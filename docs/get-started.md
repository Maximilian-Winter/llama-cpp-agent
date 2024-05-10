# Get Started with llama-cpp-agent

Welcome to the llama-cpp-agent framework! This guide will help you set up and start interacting with Large Language Models (LLMs) using our framework. Whether you're looking to engage in seamless conversations, execute function calls, or generate structured outputs, this document will guide you through the setup and basic operations.

# Table of Contents

1. [Get Started with llama-cpp-agent](#get-started-with-llama-cpp-agent)
    - [Welcome](#get-started-with-llama-cpp-agent)
2. [Installation](#installation)
    - [Install via pip](#installation)
3. [Setting Up Providers](#setting-up-providers)
    - [llama-cpp-python provider](#llama-cpp-python-provider)
    - [llama.cpp server provider](#llamacpp-server-provider)
    - [text-generation-inference (TGI) server provider](#text-generation-inference-tgi-server-provider)
    - [vllm OpenAI compatible server provider](#vllm-openai-compatible-server-provider)
4. [Define the agent](#define-the-agent)
    - [Create agent](#create-agent)
    - [Customize agent](#customize-agent)
5. [Talk to the agent](#talk-to-the-agent)
    - [Get chat response](#talk-to-the-agent)
    - [Sampling parameters](#sampling-parameters)
6. [Let the agent use tools](#let-the-agent-use-tools)
    - [Define function tool](#let-the-agent-use-tools)
    - [Using the FunctionCallingAgent class](#using-the-functioncallingagent-class)
7. [Let the agent generate objects](#let-the-agent-generate-objects)
    - [Generate a book entry dataset](#let-the-agent-generate-objects)
    - [Using the StructuredOutputAgent](#using-the-structuredoutputagent)


## Installation

Before you begin, ensure that you have Python installed on your system. You can install the llama-cpp-agent framework and its dependencies through pip:

```bash
pip install llama-cpp-agent
```

## Setting Up Providers
### llama-cpp-python provider
```python
# Import the Llama class of llama-cpp-python and the LlamaCppPythonProvider of llama-cpp-agent
from llama_cpp import Llama
from llama_cpp_agent.providers.llama_cpp_python import LlamaCppPythonProvider

# Create an instance of the Llama class and load the model
llama_model = Llama(r"C:\gguf-models\mistral-7b-instruct-v0.2.Q6_K.gguf", n_batch=1024, n_threads=10, n_gpu_layers=40)

# Create the provider by passing the Llama class instance to the LlamaCppPythonProvider class
provider = LlamaCppPythonProvider(llama_model)
```

### llama.cpp server provider
```python
# Import the LlamaCppServerProvider of llama-cpp-agent
from llama_cpp_agent.providers.llama_cpp_server import LlamaCppServerProvider

# Create the provider by passing the server URL to the LlamaCppServerProvider class, you can also pass an API key for authentication and a flag to use a llama-cpp-python server.
provider = LlamaCppServerProvider("http://127.0.0.1:8080")
```

### text-generation-inference (TGI) server provider
```python
# Import the TGIServerProvider of llama-cpp-agent
from llama_cpp_agent.providers.tgi_server import TGIServerProvider

# Create the provider by passing the server URL to the TGIServerProvider class, you can also pass an API key for authentication.
provider = TGIServerProvider("http://localhost:8080")
```

### vllm OpenAI compatible server provider
```python
# Import the VLLMServerProvider of llama-cpp-agent
from llama_cpp_agent.providers.vllm_server import VLLMServerProvider

# Create the provider by passing the server URL and the used model to the VLLMServerProvider class, you can also pass an API key for authentication.
provider = VLLMServerProvider("http://localhost:8000/v1", "TheBloke/Llama-2-7b-Chat-AWQ", "token-abc123")
```

## Define the agent
The next step is to define and create the agent. You simply have to pass the provider of the previous step to the `LlamaCppAgent` class.

### Create agent
```python
# Import the LlamaCppAgent class of the framework
from llama_cpp_agent.llm_agent import LlamaCppAgent

# Create the provider like in the previous step.
provider = ...

# Pass the provider to the LlamaCppAgentClass
agent = LlamaCppAgent(provider)
```

### Customize agent
We can also change the chat formatter and the system message like showed below.


```python
# Import the LlamaCppAgent of the framework and the predefined chat message formatter.
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
# Create the provider like in the previous step.
provider = ...

# Pass the provider to the LlamaCppAgentClass and define the system prompt and predefined message formatter
agent = LlamaCppAgent(provider,
                      system_prompt="You are a helpful assistant.",
                      predefined_messages_formatter_type=MessagesFormatterType.CHATML)
```

## Talk to the agent
We can talk with agent by calling the `get_chat_response` method on the agent we created before.

```python
agent = ...

agent_output = agent.get_chat_response("Hello, World!")
print(f"Agent: {agent_output.strip()}")
```

### Sampling parameters
We can change the generation and samplings parameters by passing a ´LlmSamplingSettings´ instance to the `get_chat_response` method of the agent. We can get the default samplings settings of the provider by calling the `get_provider_default_settings` method on it.

```python
provider = ...
agent = ...

settings = provider.get_provider_default_settings()

settings.temperature = 0.65

agent_output = agent.get_chat_response("Hello, World!", llm_samplings_settings=settings)
print(f"Agent: {agent_output.strip()}")
```

## Let the agent use tools
To let the agent use tools and call function, we need to pass an instance of the `LlmStructuredOutputSettings` class to the `get_chat_response` method.

The llama-cpp-agent framework supports python functions as tools, pydantic tools, llama-index tools and OpenAI function schemas together with a function as tools.

Below we will use a python function as a tool. It is important that the docstring of the function includes a general description of the function and includes all arguments. These information can be used to generate a documentation for the llm on how to use these functions.

```python
# Import the LlmStructuredOutputSettings
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings

provider = ...

# Lets define a simple function tool
def calculate_a_to_the_power_b(a: Union[int, float], b: Union[int, float]):
    """
    Calculates a to the power of b

    Args:
        a: number
        b: exponent

    """
    return f"Result: {math.pow(a, b)}"

# Now let's create an instance of the LlmStructuredOutput class by calling the `from_functions` function of it and passing it a list of functions.

output_settings = LlmStructuredOutputSettings.from_functions([calculate_a_to_the_power_b], allow_parallel_function_calling=True)

# Create a LlamaCppAgent instance as before, including a system message with information about the tools available for the LLM agent.
llama_cpp_agent = LlamaCppAgent(
    provider,
    debug_output=True,
    system_prompt=f"You are an advanced AI, tasked to assist the user by calling functions in JSON format. The following are the available functions and their parameters and types:\n\n{output_settings.get_llm_documentation(provider)}",
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

# Define some user input
user_input = "Calculate a to the power of b: a = 2, b = 3"

# Pass the user input together with output settings to `get_chat_response` method.
# This will print the result of the function the LLM will call, it is a list of dictionaries containing the result.
print(
    llama_cpp_agent.get_chat_response(
        user_input, structured_output_settings=output_settings
    )
)
```


This will print the result of the function the LLM will call, it is a list of dictionaries containing the result.
It will look like this:
```json
[{"function": "calculate_a_to_the_power_b", "params": {"a": 2, "b": 3}, "return_value": "Result: 8.0"}]
```
To let the agent automatically generate an answer based on the function call results, you can use the FunctionCallingAgent class. It is described next and handles passing the results back to the agent to generate an answer. 


### Using the FunctionCallingAgent class
The FunctionCallingAgent class enables function calling by llms and it will handle the answer generation. It is a wrapper around a LlamaCppAgent instance.

We will define a pydantic tool, which is a pydantic model with a run method. When the agent uses these tools, the run method will get executed.

The following code shows how to create the pydantic tool, the creation of the function calling agent and using it with parallel function calling.

```python
# Import the necessary classes for the pydantic tool and the agent
from enum import Enum
from typing import Union

from pydantic import BaseModel, Field

from llama_cpp_agent.function_calling_agent import FunctionCallingAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.providers.tgi_server import TGIServerProvider

# Set up the provider
provider = TGIServerProvider("http://localhost:8080")


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

    number_one: Union[int, float] = Field(
        ...,
        description="First number.")
    number_two: Union[int, float] = Field(
        ...,
        description="Second number.")
    operation: MathOperation = Field(..., description="Math operation to perform.")

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


# Callback for receiving messages for the user.
def send_message_to_user_callback(message: str):
    print(message)


# Create a list of function call tools.
function_tools = [LlamaCppFunctionTool(Calculator)]

# Create the function calling agent. We are passing the provider, the tool list, send message to user callback and the chat message formatter. Also, we allow parallel function calling.
function_call_agent = FunctionCallingAgent(
    provider,
    llama_cpp_function_tools=function_tools,
    allow_parallel_function_calling=True,
    send_message_to_user_callback=send_message_to_user_callback,
    messages_formatter_type=MessagesFormatterType.CHATML)

# Define the user input.
user_input = "Solve the following calculations: 42 * 42, 24 * 24, 5 * 5, 89 * 75, 42 * 46, 69 * 85, 422 * 420, 753 * 321, 72 * 55, 240 * 204, 789 * 654, 123 * 321, 432 * 89, 564 * 321?"
function_call_agent.generate_response(user_input)

```

## Let the agent generate objects
To let the agent generate objects, we need to pass an instance of the `LlmStructuredOutputSettings` class to the `get_chat_response` method.

The llama-cpp-agent framework supports object generation based on pydantic models.

The following code shows the creation of a dataset entry for a book, based on a short summary.

```python
# Import necessary libraries of pydantic and the llama-cpp-agent framework.
from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent

from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType
from llama_cpp_agent.providers.tgi_server import TGIServerProvider

# Create the provider.
provider = TGIServerProvider("http://localhost:8080")



# An enum for the book category
class Category(Enum):
    """
    The category of the book.
    """

    Fiction = "Fiction"
    NonFiction = "Non-Fiction"

# The class representing the database entry we want to generate.
class Book(BaseModel):
    """
    Represents an entry about a book.
    """

    title: str = Field(..., description="Title of the book.")
    author: str = Field(..., description="Author of the book.")
    published_year: int = Field(..., description="Publishing year of the book.")
    keywords: List[str] = Field(..., description="A list of keywords.")
    category: Category = Field(..., description="Category of the book.")
    summary: str = Field(..., description="Summary of the book.")


# We create an instance of the LlmStructuredOutputSettings class by calling its from_pydantic_models method and specify the output type.
output_settings = LlmStructuredOutputSettings.from_pydantic_models([Book], output_type=LlmStructuredOutputType.list_of_objects)

# We are creating the agent with a custom system prompt including information about the dataset entry and its structure.
llama_cpp_agent = LlamaCppAgent(
    provider,
    system_prompt="You are an advanced AI, tasked to create JSON database entries for books.\n\n\n" + output_settings.get_llm_documentation(provider),
)

# We define the input information for the agent.
text = """The Feynman Lectures on Physics is a physics textbook based on some lectures by Richard Feynman, a Nobel laureate who has sometimes been called "The Great Explainer". The lectures were presented before undergraduate students at the California Institute of Technology (Caltech), during 1961–1963. The book's co-authors are Feynman, Robert B. Leighton, and Matthew Sands."""

# We call get_chat_response with output_settings. This will return an instance of the dataset entry class 'Book'.
book_dataset_entry = llama_cpp_agent.get_chat_response(text, structured_output_settings=output_settings)
print(book_dataset_entry)

```
This will output something like this:
```
[Book(title='The Feynman Lectures on Physics', author='Richard Feynman, Robert B. Leighton, Matthew Sands', published_year=1963, keywords=['Physics', 'Textbook', 'Lectures', 'Richard Feynman'], category=<Category.NonFiction: 'Non-Fiction'>, summary="The Feynman Lectures on Physics is a physics textbook based on the lectures given by Nobel laureate Richard Feynman at the California Institute of Technology (Caltech) between 1961 and 1963. The book's co-authors are Feynman, Robert B. Leighton, and Matthew Sands.")]
```

### Using the StructuredOutputAgent
You can also use the StructuredOutputAgent class to simplify the process a little bit.

```python

# Example that uses the StructuredOutputAgent class to create a dataset entry of a book, out of unstructured data.

from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.structured_output_agent import StructuredOutputAgent
from llama_cpp_agent.providers.tgi_server import TGIServerProvider

model = TGIServerProvider("http://localhost:8080")


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
    keywords: List[str] = Field(..., description="A list of keywords.")
    category: Category = Field(..., description="Category of the book.")
    summary: str = Field(..., description="Summary of the book.")


structured_output_agent = StructuredOutputAgent(
    model, debug_output=True,
    messages_formatter_type=MessagesFormatterType.CHATML
)

text = """The Feynman Lectures on Physics is a physics textbook based on some lectures by Richard Feynman, a Nobel laureate who has sometimes been called "The Great Explainer". The lectures were presented before undergraduate students at the California Institute of Technology (Caltech), during 1961–1963. The book's co-authors are Feynman, Robert B. Leighton, and Matthew Sands."""
print(structured_output_agent.create_object(Book, text))

```


