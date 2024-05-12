from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.providers import VLLMServerProvider

provider = VLLMServerProvider("http://localhost:8000/v1", "TheBloke/Llama-2-7b-Chat-AWQ", "token-abc123")

agent = LlamaCppAgent(
    provider,
    system_prompt="You are a helpful assistant.",
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

settings = provider.get_provider_default_settings()
settings.max_tokens = 512
settings.temperature = 0.65

while True:
    user_input = input(">")
    if user_input == "exit":
        break
    agent_output = agent.get_chat_response(user_input, llm_sampling_settings=settings)
    print(f"Agent: {agent_output.strip()}")
