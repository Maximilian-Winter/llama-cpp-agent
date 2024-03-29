import datetime

from llama_cpp import Llama

from llama_cpp_agent.llm_agent import LlamaCppAgent

from example_agent_models_auto_coder import SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile, \
    agent_dev_folder_setup
from llama_cpp_agent.messages_formatter import MessagesFormatterType

from llama_cpp_agent.function_calling import LlamaCppFunctionTool

function_tools = [LlamaCppFunctionTool(GetFileList), LlamaCppFunctionTool(ReadTextFile),
                  LlamaCppFunctionTool(WriteTextFile, has_triple_quoted_string=True, triple_quoted_string_field_name="file_content")]

function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

main_model = Llama(
    "../../gguf-models/neuralhermes-2.5-mistral-7b.Q8_0.gguf",
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

# OUTPUT CONSTRAINTS
Your output is constrained to JSON formatted strings, except for file content, which is placed in a markdown code block. You can use the following functions to interact with the user's file-system:
{function_tool_registry.get_documentation()}



```'''.strip()

llama_cpp_agent = LlamaCppAgent(main_model, debug_output=True,
                                system_prompt=system_prompt,
                                predefined_messages_formatter_type=MessagesFormatterType.CHATML)
timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")

agent_dev_folder_setup(f"dev_{timestamp}")
user_input = 'Write a chat bot frontend in HTML, CSS and Javascript with a dark UI using Bootstrap under the "./workspace" folder.'
while True:

    if "None" in user_input:
        user_input = "Proceed."

    user_input = llama_cpp_agent.get_chat_response(
        user_input,
        temperature=0.45, repeat_penalty=1.0, function_tool_registry=function_tool_registry)

    user_input = '\n'.join([str(output) for output in user_input])