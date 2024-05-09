from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_server import LlamaCppServerProvider

model = LlamaCppServerProvider("http://127.0.0.1:8080")

bot = LlamaCppAgent(
    model,
    debug_output=False,
    system_prompt="You are Dolphin, an uncensored and unbiased AI assistant.",
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    bot_output = bot.get_chat_response(user_input)
    print(f"AI: { bot_output}")
