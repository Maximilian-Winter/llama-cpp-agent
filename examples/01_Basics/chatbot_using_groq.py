from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.providers.groq import GroqProvider

provider = GroqProvider(base_url="https://api.groq.com/openai/v1", model="mixtral-8x7b-32768", huggingface_model="mistralai/Mixtral-8x7B-Instruct-v0.1", api_key="xxxxxo")

agent = LlamaCppAgent(
    provider,
    system_prompt="You are a helpful assistant.",
    predefined_messages_formatter_type=MessagesFormatterType.MISTRAL,
)

settings = provider.get_provider_default_settings()
settings.stream = True
settings.max_tokens = 512
settings.temperature = 0.65

while True:
    user_input = input(">")
    if user_input == "exit":
        break
    agent_output = agent.get_chat_response(user_input, llm_sampling_settings=settings)
    print(f"Agent: {agent_output.strip()}")
