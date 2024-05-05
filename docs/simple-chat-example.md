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