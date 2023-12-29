import json
from llama_cpp import Llama
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
main_model = Llama(
    "../gguf-models/airoboros-l2-13b-3.1.1.Q4_K_M.gguf",
    n_gpu_layers=35,
    f16_kv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=False,
    seed=42,
)
wrapped_model = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt="You are an advanced AI assistant.", predefined_messages_formatter_type=MessagesFormatterType.LLAMA_2)

wrapped_model.get_chat_response('Write a long poem about the USA.', temperature=0.7)
