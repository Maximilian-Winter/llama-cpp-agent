from llama_cpp import Llama

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_python import LlamaCppPythonProvider


llama_model = Llama(r"C:\AI\Agents\gguf-models\mistral-7b-instruct-v0.2.Q6_K.gguf", n_batch=1024, n_threads=10,
                    n_gpu_layers=40)
provider = LlamaCppPythonProvider(llama_model)

agent = LlamaCppAgent(
    provider,
    system_prompt="You are a helpful assistant.",
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

while True:
    user_input = input(">")
    if user_input == "exit":
        break
    agent_output = agent.get_chat_response(user_input, print_output=False)
    print(f"Agent: {agent_output.strip()}")
