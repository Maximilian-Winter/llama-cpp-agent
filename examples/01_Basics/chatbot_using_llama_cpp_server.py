from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppServerProvider

provider = LlamaCppServerProvider("http://127.0.0.1:8080")

agent = LlamaCppAgent(
    provider,
    system_prompt="You are a helpful assistant.",
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

settings = provider.get_provider_default_settings()
settings.n_predict = 512
settings.temperature = 0.65
agent.chat_history.get_message_store().load_from_json("test.json")
while True:
    user_input = input(">")
    if user_input == "exit":
        break
    agent_output = agent.get_chat_response(user_input, llm_sampling_settings=settings)
    agent.chat_history.get_message_store().save_to_json("test.json")
    print(f"Agent: {agent_output.strip()}")
