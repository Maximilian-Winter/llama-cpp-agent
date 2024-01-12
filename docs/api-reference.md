---
title: API Reference
---

## Agents

### LlamaCppAgent
The following is the basic agent that is also used by other agents.

::: llama_cpp_agent.llm_agent

### FunctionCallingAgent    
A predefined agent that performs function calling based on LlamaCppAgent. Can pydantic "Instructor" like functions, OpenAI function definitions and normal Python functions.

::: llama_cpp_agent.function_calling_agent

### StructuredOutputAgent
A predefined agent that can create objects based on pydantic models out of unstructured text or random.

::: llama_cpp_agent.structured_output_agent


## LLM Settings

### llama-cpp-python backend
Settings for the llama-cpp-python backend.

::: llama_cpp_agent.llm_settings

### llama.cpp server backend
Settings for the llama.cpp server backend.

::: llama_cpp_agent.providers.llama_cpp_server_provider


## Grammar Generator
The grammar generator that is used by framework in the background.

::: llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models

