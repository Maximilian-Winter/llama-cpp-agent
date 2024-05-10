# Get Started

## Define the provider
llama-cpp-agent is compatible with the llama.cpp server, llama-cpp-python and its server, the TGI server, and the vllm OpenAI server.

You can create a new provider by importing the corresponding class. Then you instantiate an object of the class and pass the needed parameters as below.

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
print(
    llama_cpp_agent.get_chat_response(
        user_input, structured_output_settings=output_settings
    )
)
```




