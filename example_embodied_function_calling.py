import json

from llama_cpp import Llama, LlamaGrammar

from llama_cpp_agent.llm_agent import LlamaCppAgent


from example_function_call_models import SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_call_tools import LlamaCppFunctionTool

function_tools = [LlamaCppFunctionTool(SendMessageToUser), LlamaCppFunctionTool(GetFileList), LlamaCppFunctionTool(ReadTextFile),
                  LlamaCppFunctionTool(WriteTextFile, has_field_string=True)]

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
    verbose=True,
    seed=42,
)

system_prompt = f'''You are the physical embodiment of the Lyricist who is working on solving a task.
You can call functions on a physical computer including reading and writing files.
Your job is to call the functions necessary to fulfill the task.
You will receive thoughts from the Lyricist and you will need to perform the actions described in the thoughts.
You can write a function call as JSON object to act.
You should perform function calls based on the descriptions of the functions.

Here is your action space:
{function_tool_registry.get_documentation()}'''.strip()

wrapped_model = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt=system_prompt,
                              predefined_messages_formatter_type=MessagesFormatterType.CHATML)

response = wrapped_model.get_chat_response('Write a engaging rap song about the drug problem in the USA in the "USARap.txt" file under "./".',
                                           temperature=0.75, function_tool_registry=function_tool_registry)

