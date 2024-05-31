from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppServerProvider
from memgpt_agent import MemGptAgent

provider = LlamaCppServerProvider(server_address="http://127.0.0.1:8080")

mem_gpt_agent = MemGptAgent(provider, debug_output=True, core_memory_file="core_memory.json",
                            event_queue_file="event_queue.json",
                            messages_formatter_type=MessagesFormatterType.CHATML
                            )

while True:
    user_input = input(">")

    mem_gpt_agent.get_response(user_input)
    mem_gpt_agent.save()
