from llama_cpp import Llama

from llama_cpp_agent.llm_agent import LlamaCppAgent


from example_function_call_models import SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_call_tools import LlamaCppFunctionTool

function_tools = [LlamaCppFunctionTool(SendMessageToUser), LlamaCppFunctionTool(GetFileList), LlamaCppFunctionTool(ReadTextFile),
                  LlamaCppFunctionTool(WriteTextFile)]

function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

main_model = Llama(
    "../gguf-models/dolphin-2.6-mistral-7b-Q8_0.gguf",
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
                              system_prompt="You are an advanced AI, tasked to assist the user by calling functions in JSON format.\n\n\n" + function_tool_registry.get_documentation(),
                              predefined_messages_formatter_type=MessagesFormatterType.CHATML)

response = wrapped_model.get_chat_response('Write a long poem about the USA in the "HelloUSA.txt" file under "./".',
                                           temperature=0.75, function_tool_registry=function_tool_registry)

