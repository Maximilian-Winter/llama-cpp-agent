from llama_cpp import Llama

from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppPythonProvider

llama_model = Llama(r"C:\AI\Agents\gguf-models\mistral-7b-instruct-v0.2.Q6_K.gguf", n_batch=1024, n_threads=10, n_gpu_layers=33, n_ctx=8192, verbose=False)

provider = LlamaCppPythonProvider(llama_model)

agent = LlamaCppAgent(
    provider,
    system_prompt="You are a helpful assistant.",
    predefined_messages_formatter_type=MessagesFormatterType.MISTRAL,
    debug_output=True
)

settings = provider.get_provider_default_settings()
settings.max_tokens = 2000
settings.stream = True
while True:
    agent_output = agent.get_chat_response("Hello!", llm_sampling_settings=settings)
    print(f"Agent: {agent_output.strip()}")
