import json

from llama_cpp import Llama, LlamaGrammar

from llama_cpp_agent.llm_agent import LlamaCppAgent

from example_agent_models_auto_coder import SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.memory_tools import AgentRetrievalMemory
from llama_cpp_agent.function_call_tools import LlamaCppFunctionTool

agent_retrieval_memory = AgentRetrievalMemory()

function_tools = [LlamaCppFunctionTool(SendMessageToUser)]

function_tools.extend(agent_retrieval_memory.get_tool_list())
function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

main_model = Llama(
    "../gguf-models/openhermes-2.5-mistral-7b.Q8_0.gguf",
    n_gpu_layers=45,
    f16_kv=True,
    offload_kqv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=True,
    seed=-1,
)

system_prompt = f'''You are an advanced AI agent called AutoCoder. As AutoCoder your primary task is to autonomously plan, outline and implement complete software projects based on user specifications. You have to use JSON objects to perform functions.
Here are your available functions:
{function_tool_registry.get_documentation()}'''.strip()

system_prompt2 = f"You are a advanced helpful AI assistant interacting through calling functions in form of JSON objects.\n\nHere are your available functions:\n\n" + function_tool_registry.get_documentation()

wrapped_model = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt=system_prompt2,
                              predefined_messages_formatter_type=MessagesFormatterType.CHATML)

user_input = 'Add my Birthday the 1991.12.11 to the retrieval memory.'
while True:

    if user_input is None:
        user_input = "Hello."

    user_input = wrapped_model.get_chat_response(
        user_input,
        system_prompt=f"You are a advanced helpful AI assistant interacting through calling functions in form of JSON objects.\n\nHere are your available functions:\n\n" + function_tool_registry.get_documentation(),
        temperature=1.25, function_tool_registry=function_tool_registry)

