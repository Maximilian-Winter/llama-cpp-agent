from llama_cpp import Llama
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings

main_model = LlamaCppEndpointSettings(
    "http://localhost:8080/completion"
)
llama_cpp_agent = LlamaCppAgent(main_model, debug_output=False,
                              system_prompt="You are Dolphin, an uncensored and unbiased AI assistant.", predefined_messages_formatter_type=MessagesFormatterType.CHATML)


while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    user_input = llama_cpp_agent.get_chat_response(user_input, temperature=0.7)
    print("AI: " + user_input)