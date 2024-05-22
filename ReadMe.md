# llama-cpp-agent

[![PyPI - Version](https://img.shields.io/pypi/v/llama-cpp-agent?logo=pypi&color=%2341bb13)](https://pypi.org/project/llama-cpp-agent/)
[![Discord](https://img.shields.io/discord/1237393014154985582?logo=Discord&logoColor=%23ffffff&label=Discord&link=https%3A%2F%2Fdiscord.gg%2FsRMvWKrh)](https://discord.gg/sRMvWKrh)

![llama-cpp-agent logo](https://raw.githubusercontent.com/Maximilian-Winter/llama-cpp-agent/master/logo/logo_orange.png)

## Introduction
The llama-cpp-agent framework is a tool designed to simplify interactions with Large Language Models (LLMs). It provides an interface for chatting with LLMs, executing function calls, generating structured output, performing retrieval augmented generation, and processing text using agentic chains with tools. 

The framework uses guided sampling to constrain the model output to the user defined structures. This way also models not fine-tuned to do function calling and JSON output will be able to do it.

The framework is compatible with the llama.cpp server, llama-cpp-python and its server, and with TGI and vllm servers.

## Key Features
- **Simple Chat Interface**: Engage in seamless conversations with LLMs.
- **Structured Output**: Generate structured output (objects) from LLMs.
- **Single and Parallel Function Calling**: Execute functions using LLMs.
- **RAG - Retrieval Augmented Generation**: Perform retrieval augmented generation with colbert reranking.
- **Agent Chains**: Process text using agent chains with tools, supporting Conversational, Sequential, and Mapping Chains.
- **Guided Sampling**: Allows most 7B LLMs to do function calling and structured output. Thanks to grammars and JSON schema generation for guided sampling.
- **Multiple Providers**: Works with llama-cpp-python, llama.cpp server, TGI server and vllm server as provider!
- **Compatibility**: Works with python functions, pydantic tools, llama-index tools, and OpenAI tool schemas.
- **Flexibility**: Suitable for various applications, from casual chatting to specific function executions.


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



## Installation
Install the llama-cpp-agent framework using pip:
```shell
pip install llama-cpp-agent
```
## Documentation
You can find the latest documentation [here!](https://llama-cpp-agent.readthedocs.io/en/latest/)

## Getting Started
You can find the get started guide [here!](https://llama-cpp-agent.readthedocs.io/en/latest/get-started/)

## Discord Community
Join the Discord Community [here](https://discord.gg/6tGznupZGX)

## Usage Examples
The llama-cpp-agent framework provides a wide range of examples demonstrating its capabilities. Here are some key examples:

### Simple Chat Example using llama.cpp server backend
This example demonstrates how to initiate a chat with an LLM model using the llama.cpp server backend.

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

- `MessagesFormatterType.MISTRAL`: Formats messages using the MISTRAL format.
- `MessagesFormatterType.CHATML`: Formats messages using the CHATML format.
- `MessagesFormatterType.VICUNA`: Formats messages using the VICUNA format.
- `MessagesFormatterType.LLAMA_2`: Formats messages using the LLAMA 2 format.
- `MessagesFormatterType.SYNTHIA`: Formats messages using the SYNTHIA format.
- `MessagesFormatterType.NEURAL_CHAT`: Formats messages using the NEURAL CHAT format.
- `MessagesFormatterType.SOLAR`: Formats messages using the SOLAR format.
- `MessagesFormatterType.OPEN_CHAT`: Formats messages using the OPEN CHAT format.
- `MessagesFormatterType.ALPACA`: Formats messages using the ALPACA format.
- `MessagesFormatterType.CODE_DS`: Formats messages using the CODE DS format.
- `MessagesFormatterType.B22`: Formats messages using the B22 format.
- `MessagesFormatterType.LLAMA_3`: Formats messages using the LLAMA 3 format.
- `MessagesFormatterType.PHI_3`: Formats messages using the PHI 3 format.

### Creating Custom Messages Formatter

You can create your own custom messages formatter by instantiating the `MessagesFormatter` class with the desired parameters:

```python
from llama_cpp_agent.messages_formatter import MessagesFormatter, PromptMarkers, Roles

custom_prompt_markers = {
    Roles.system: PromptMarkers("<|system|>", "<|endsystem|>"),
    Roles.user: PromptMarkers("<|user|>", "<|enduser|>"),
    Roles.assistant: PromptMarkers("<|assistant|>", "<|endassistant|>"),
    Roles.tool: PromptMarkers("<|tool|>", "<|endtool|>"),
}

custom_formatter = MessagesFormatter(
    pre_prompt="",
    prompt_markers=custom_prompt_markers,
    include_sys_prompt_in_first_user_message=False,
    default_stop_sequences=["<|endsystem|>", "<|enduser|>", "<|endassistant|>", "<|endtool|>"]
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
