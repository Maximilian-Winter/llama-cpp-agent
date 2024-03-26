from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.tools.web import ScrapeWebsite
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import (
    LlamaCppEndpointSettings,
)
from tools import (
    SendMessageToUser,
    GetFileList,
    ReadTextFile,
    WriteTextFile,
)


function_tools = [
    LlamaCppFunctionTool(SendMessageToUser),
    LlamaCppFunctionTool(ScrapeWebsite),
]

function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

model = LlamaCppEndpointSettings(
    completions_endpoint_url="http://127.0.0.1:8080/completion"
)

llama_cpp_agent = LlamaCppAgent(
    model,
    debug_output=True,
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

while True:
    user_input = input("User> ")
    if "/exit" in user_input:
        break

    user_input = llama_cpp_agent.get_chat_response(
        user_input,
        system_prompt=f"You are a advanced helpful AI assistant interacting through calling functions in form of JSON objects. \n\nHere are your available functions:\n\n{function_tool_registry.get_documentation()}",
        temperature=0.65,
        function_tool_registry=function_tool_registry,
    )
