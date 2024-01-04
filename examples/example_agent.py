from llama_cpp import Llama

from llama_cpp_agent.llm_agent import LlamaCppAgent

from example_agent_models_auto_coder import SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile
from llama_cpp_agent.messages_formatter import MessagesFormatterType

from llama_cpp_agent.function_call_tools import LlamaCppFunctionTool

function_tools = [LlamaCppFunctionTool(SendMessageToUser), LlamaCppFunctionTool(GetFileList), LlamaCppFunctionTool(ReadTextFile),
                  LlamaCppFunctionTool(WriteTextFile, True)]

function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

main_model = Llama(
    "../../gguf-models/nous-hermes-2-solar-10.7b.Q6_K.gguf",
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

system_prompt = f'''You are an advanced AI agent called AutoCoder. As AutoCoder your primary task is to autonomously plan, outline and implement complete software projects based on user specifications. You can execute functions to interact with the user's file-system. 

# OUTPUT CONSTRAINTS FOR DEEP THOUGHT
Your output is constrained to multi-line JSON objects, which represent function calls. You can use the following functions to interact with the user's file-system:
{function_tool_registry.get_documentation()}'''.strip()

llama_cpp_agent = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt=system_prompt,
                              predefined_messages_formatter_type=MessagesFormatterType.CHATML)

user_input = 'Write a chat bot frontend in HTML, CSS and Javascript with a dark UI using TailwindCSS under the "./workspace" folder.'
while True:

    if user_input is None:
        user_input = "Proceed."

    user_input = llama_cpp_agent.get_chat_response(
        user_input,
        temperature=0.35, mirostat_mode=0, mirostat_tau=3.0, mirostat_eta=0.1, function_tool_registry=function_tool_registry)

