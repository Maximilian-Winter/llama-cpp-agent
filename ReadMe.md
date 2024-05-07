# llama-cpp-agent

<img src="https://github.com/Maximilian-Winter/llama-cpp-agent/blob/db41b3184ebc902f50edbd3d27f7a3a1128b7d7d/logo/logo_orange.png" alt="llama-cpp-agent logo" width="400"/>

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
- [Documentation](#documentation)
- [Getting Started](#getting-started)
- [Discord Community](#discord-community)
- [Usage Examples](#usage-examples)
    - [Simple Chat](#simple-chat-example-using-llamacpp-server-backend)
    - [Parallel Function Calling](#parallel-function-calling-agent-example)
    - [Structured Output](#structured-output)
    - [RAG - Retrieval Augmented Generation](#rag---retrieval-augmented-generation)
    - [llama-index Tools](#llama-index-tools-example)
    - [Sequential Chain](#sequential-chain-example)
    - [Mapping Chain](#mapping-chain-example)
    - [Knowledge Graph Creation](#knowledge-graph-creation-example)
- [Additional Information](#additional-information)
    - [Predefined Messages Formatter](#predefined-messages-formatter)
    - [Creating Custom Messages Formatter](#creating-custom-messages-formatter)
- [Contributing](#contributing)
- [License](#license)
- [FAQ](#faq)

## Introduction
The llama-cpp-agent framework is a tool designed to simplify interactions with Large Language Models (LLMs). It provides an interface for chatting with LLMs, executing function calls, generating structured output, performing retrieval augmented generation, and processing text using agentic chains with tools. The framework integrates seamlessly with the llama.cpp server, llama-cpp-python and OpenAI endpoints that support grammar, offering flexibility and extensibility.

## Key Features
- **Simple Chat Interface**: Engage in seamless conversations with LLMs.
- **Structured Output**: Generate structured output (objects) from LLMs.
- **Single and Parallel Function Calling**: Execute functions using LLMs.
- **RAG - Retrieval Augmented Generation**: Perform retrieval augmented generation with colbert reranking.
- **Agent Chains**: Process text using agent chains with tools, supporting Conversational, Sequential, and Mapping Chains.
- **Compatibility**: Works with llama-index tools and OpenAI tool schemas.
- **Flexibility**: Suitable for various applications, from casual chatting to specific function executions.

## Installation
Install the llama-cpp-agent framework using pip:
```shell
pip install llama-cpp-agent
```
## Documentation
You can find the latest documentation [here!](https://llama-cpp-agent.readthedocs.io/en/latest/)

## Getting Started
1. Ensure you have the required dependencies installed, including pydantic and llama-cpp-python.
2. Import the necessary classes and functions from the llama-cpp-agent framework.
3. Set up your LLM provider (e.g., llama-cpp-python Llama class, OpenAI endpoint with grammar support like llama-cpp-python server, or llama.cpp server).
4. Create an instance of the desired agent class (e.g., LlamaCppAgent, FunctionCallingAgent, StructuredOutputAgent).
5. Interact with the agent using the provided methods and examples.

Here's a basic example of using the LlamaCppAgent for a simple chat:

```python
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings

main_model = LlamaCppEndpointSettings(
    completions_endpoint_url="http://127.0.0.1:8080/completion"
)

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

## Discord Community
Join the Discord Community [here](https://discord.gg/6tGznupZGX)

## Usage Examples
The llama-cpp-agent framework provides a wide range of examples demonstrating its capabilities. Here are some key examples:

### Simple Chat Example using llama.cpp server backend
This example demonstrates how to initiate a chat with an LLM model using the llama.cpp server backend. It supports llama-cpp-python Llama class instances, OpenAI endpoints with GBNF grammar support, and the llama.cpp backend server.

[View Example](https://llama-cpp-agent.readthedocs.io/en/latest/simple-chat-example/)

### Parallel Function Calling Agent Example
This example showcases parallel function calling using the FunctionCallingAgent class. It demonstrates how to define and execute multiple functions concurrently.

[View Example](https://llama-cpp-agent.readthedocs.io/en/latest/parallel_function_calling/)

### Structured Output
This example illustrates how to generate structured output objects using the StructuredOutputAgent class. It shows how to create a dataset entry of a book from unstructured data.

[View Example](https://llama-cpp-agent.readthedocs.io/en/latest/structured-output-example/)

### RAG - Retrieval Augmented Generation
This example demonstrates Retrieval Augmented Generation (RAG) with colbert reranking. It requires installing the optional rag dependencies (ragatouille).

[View Example](https://llama-cpp-agent.readthedocs.io/en/latest/rag/)

### llama-index Tools Example
This example shows how to use llama-index tools and query engines with the FunctionCallingAgent class.

[View Example](https://llama-cpp-agent.readthedocs.io/en/latest/llama_index_tool_use/)

### Sequential Chain Example
This example demonstrates how to create a complete product launch campaign using a sequential chain.

[View Example](https://llama-cpp-agent.readthedocs.io/en/latest/sequential_chain/)

### Mapping Chain Example
This example illustrates how to create a mapping chain to summarize multiple articles into a single summary.

[View Example](https://llama-cpp-agent.readthedocs.io/en/latest/map_chain/)

### Knowledge Graph Creation Example
This example, based on an example from the Instructor library for OpenAI, shows how to create a knowledge graph using the llama-cpp-agent framework.

[View Example](https://llama-cpp-agent.readthedocs.io/en/latest/knowledge-graph-example/)


## Additional Information

### Predefined Messages Formatter
The llama-cpp-agent framework provides predefined message formatters to format messages for the LLM model. The `MessagesFormatterType` enum defines the available formatters:

- `MessagesFormatterType.CHATML`: Formats messages using the CHATML format.
- `MessagesFormatterType.MIXTRAL`: Formats messages using the MIXTRAL format.
- `MessagesFormatterType.VICUNA`: Formats messages using the VICUNA format.
- `MessagesFormatterType.LLAMA_2`: Formats messages using the LLAMA 2 format.
- `MessagesFormatterType.SYNTHIA`: Formats messages using the SYNTHIA format.
- `MessagesFormatterType.NEURAL_CHAT`: Formats messages using the NEURAL CHAT format.
- `MessagesFormatterType.SOLAR`: Formats messages using the SOLAR format.
- `MessagesFormatterType.OPEN_CHAT`: Formats messages using the OPEN CHAT format.

### Creating Custom Messages Formatter
You can create your own custom messages formatter by instantiating the `MessagesFormatter` class with the desired parameters:

```python
from llama_cpp_agent.messages_formatter import MessagesFormatter

custom_formatter = MessagesFormatter(
    PRE_PROMPT="",
    SYS_PROMPT_START="<|system|>",
    SYS_PROMPT_END="<|endsystem|>",
    USER_PROMPT_START="<|user|>",
    USER_PROMPT_END="<|enduser|>",
    ASSISTANT_PROMPT_START="<|assistant|>",
    ASSISTANT_PROMPT_END="<|endassistant|>",
    INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE=False,
    DEFAULT_STOP_SEQUENCES=["<|endsystem|>", "<|enduser|>", "<|endassistant|>"]
)
```

## Contributing
We welcome contributions to the llama-cpp-agent framework! If you'd like to contribute, please follow these guidelines:

1. Fork the repository and create your branch from `master`.
2. Ensure your code follows the project's coding style and conventions.
3. Write clear, concise commit messages and pull request descriptions.
4. Test your changes thoroughly before submitting a pull request.
5. Open a pull request to the `master` branch.

If you encounter any issues or have suggestions for improvements, please open an issue on the [GitHub repository](https://github.com/Maximilian-Winter/llama-cpp-agent/issues).

## License
The llama-cpp-agent framework is released under the [MIT License](https://github.com/Maximilian-Winter/llama-cpp-agent/blob/master/LICENSE).

## FAQ

**Q: How do I install the optional dependencies for RAG?**  
A: To use the RAGColbertReranker class and the RAG example, you need to install the optional rag dependencies (ragatouille). You can do this by running `pip install llama-cpp-agent[rag]`.

**Q: Can I contribute to the llama-cpp-agent project?**  
A: Absolutely! We welcome contributions from the community. Please refer to the [Contributing](#contributing) section for guidelines on how to contribute.

**Q: Is llama-cpp-agent compatible with the latest version of llama-cpp-python?**  
A: Yes, llama-cpp-agent is designed to work with the latest version of llama-cpp-python. However, if you encounter any compatibility issues, please open an issue on the GitHub repository.
