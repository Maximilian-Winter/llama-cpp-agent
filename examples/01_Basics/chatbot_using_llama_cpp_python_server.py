from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_server import LlamaCppServerProvider

provider = LlamaCppServerProvider("http://127.0.0.1:8080", llama_cpp_python_server=True)

agent = LlamaCppAgent(
    provider,
    system_prompt="You are a helpful assistant.",
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

settings = provider.get_provider_default_settings()
settings.n_predict = 512
settings.temperature = 0.65

while True:
    user_input = input(">")
    if user_input == "exit":
        break
    agent_output = agent.get_chat_response(user_input, llm_samplings_settings=settings)
    print(f"Agent: {agent_output.strip()}")
