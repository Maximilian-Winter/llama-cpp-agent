from llama_cpp import Llama

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType

path_to_model = "../../gguf-models/neuralhermes-2.5-mistral-7b.Q8_0.gguf"

model = Llama(
    path_to_model,
    n_gpu_layers=49,
    f16_kv=True,
    offload_kqv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=4096,
    last_n_tokens_size=1024,
    verbose=True,
    seed=-1,
)

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
    bot_output = bot.get_chat_response(user_input, temperature=0.7, stream=True)
    print(f"AI: { bot_output}")
