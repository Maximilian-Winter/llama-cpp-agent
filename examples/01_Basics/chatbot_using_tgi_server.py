from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.tgi_server import TGIServerProvider

provider = TGIServerProvider("http://localhost:8080")

agent = LlamaCppAgent(
    provider,
    system_prompt="You are a helpful assistant.",
    predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL,
)

settings = provider.get_provider_default_settings()
settings.max_new_tokens = 512
settings.temperature = 0.65
settings.do_sample = True

while True:
    user_input = input(">")
    if user_input == "exit":
        break

    agent_output = agent.get_chat_response(user_input, llm_samplings_settings=settings)
    print(f"Agent: {agent_output.strip()}")
