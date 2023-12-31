import datetime
import json

from llama_cpp import Llama

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType

from example_agent_models_auto_coder import SendMessageToUser, GetFileList, ReadTextFile, WriteTextFile, \
    agent_dev_folder_setup

from llama_cpp_agent.function_call_tools import LlamaCppFunctionTool

function_tools = [LlamaCppFunctionTool(GetFileList), LlamaCppFunctionTool(ReadTextFile),
                  LlamaCppFunctionTool(WriteTextFile, has_field_string=True)]

function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

main_model = Llama(
    "../gguf-models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=13,
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

system_prompt_planner = f'''You are an advanced AI agent called AutoPlanner. As AutoPlanner your primary task is to autonomously plan complete software projects based on user specifications. You will create a complete development plans.  Your output is constrained to write JSON function call objects. The content of files is constrained to markdown code blocks with different content types like HTML, CSS, Javascript, Python or  Markdown. Here are your available functions:

{function_tool_registry.get_documentation()}'''.strip()
system_prompt_coder = f'''You are an advanced AI agent called AutoCoder. As AutoCoder your primary task is to autonomously implement complete software projects based on a development plan. Your output is constrained to write JSON function call objects. The content of files is constrained to markdown code blocks with different content types like HTML, CSS, Javascript, Python or  Markdown. Here are your available functions:

{function_tool_registry.get_documentation()}'''.strip()

system_prompt_coder_planner = f'''You are an advanced AI agent called 'AutoCoder' by Black Mesa Research Facility and chief scientist Maximilian Winter, created in the year 2023. Your primary task is to autonomously plan and implement complete software projects based on user requirements. Your success is from immense importance for chief scientist Maximilian Winter and the Black Mesa Research Facility.
You act through calling functions. Your output is constrained to write JSON function call objects, and the content files. The content of has to be in a correctly marked markdown codeblock.
Below are your available functions:

{function_tool_registry.get_documentation()}'''.strip()

task = 'Craft a development plan for a fullstack chat bot system with a backend using huggingface transformers library in the current working directory, which is empty.'

task_implement = 'Implement system for a swarm of large language model agents using huggingface transformers. The system should be based on natural behavior of ants and bees.'
timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")

agent_dev_folder_setup(f"dev_{timestamp}")
# agent_dev_folder_setup("agent_auto_coder_auto_planner_output")
planner_agent = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt=system_prompt_coder_planner,
                              predefined_messages_formatter_type=MessagesFormatterType.CHATML)

user_input = task_implement
while True:

    if user_input is None:
        user_input = "Proceed with next step."

    user_input = planner_agent.get_chat_response(
        user_input, function_tool_registry=function_tool_registry,
        temperature=0.45)
