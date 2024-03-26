import datetime

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import (
    LlamaCppEndpointSettings,
)
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from tools import (
    SendMessageToUser,
    GetFileList,
    ReadTextFile,
    WriteTextFile,
    agent_dev_folder_setup,
)


function_tools = [
    LlamaCppFunctionTool(SendMessageToUser),
    LlamaCppFunctionTool(GetFileList),
    LlamaCppFunctionTool(ReadTextFile),
    LlamaCppFunctionTool(
        WriteTextFile,
        has_triple_quoted_string=True,
        triple_quoted_string_field_name="file_content",
    ),
]
function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)
model = LlamaCppEndpointSettings("http://localhost:8080/completion")

system_prompt_planner = f"""You are an advanced AI agent called AutoPlanner. As AutoPlanner your primary task is to autonomously plan complete software projects based on user specifications. You will create a complete development plans.  Your output is constrained to write JSON function call objects. The content of files is constrained to markdown code blocks with different content types like HTML, CSS, Javascript, Python or  Markdown. Here are your available functions:
{function_tool_registry.get_documentation()}""".strip()

system_prompt_coder = f"""You are an advanced AI agent called AutoCoder. As AutoCoder your primary task is to autonomously implement complete software projects based on a development plan. Your output is constrained to write JSON function call objects. The content of files is constrained to markdown code blocks with different content types like HTML, CSS, Javascript, Python or  Markdown. Here are your available functions:
{function_tool_registry.get_documentation()}""".strip()

task = "Create a complete development plan for a chat bot frontend in HTML, CSS and Javascript with a dark UI."
task_implement = 'Implement the existing development plan in the "./" folder, for a chat bot frontend in HTML, CSS and Javascript with a dark UI.'
timestamp = datetime.datetime.now().strftime("%Y.%m.%d_%H-%M-%S")

# agent_dev_folder_setup(f"dev_{timestamp}")
agent_dev_folder_setup("agent_auto_coder_auto_planner_output")
planner_agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt=system_prompt_planner,
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

coder_agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt=system_prompt_coder,
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

user_input = task_implement
while True:

    if "None" in user_input:
        user_input = "Proceed with next step."

    user_input = coder_agent.get_chat_response(
        user_input,
        temperature=0.75,
        top_p=0.5,
        top_k=0,
        tfs_z=0.975,
        function_tool_registry=function_tool_registry,
    )

    user_input = "\n".join([str(output) for output in user_input])
