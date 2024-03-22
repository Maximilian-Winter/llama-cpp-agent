from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.agent_memory.memory_tools import AgentCoreMemory
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import (
    LlamaCppEndpointSettings,
)


class SendMessageToUser(BaseModel):
    """
    Send a message to the User.
    """

    message: str = Field(..., description="Message you want to send to the user.")

    def run(self):
        print("Message: " + self.message)


function_tools = [LlamaCppFunctionTool(SendMessageToUser)]
agent_core_memory = AgentCoreMemory()q
function_tools.extend(agent_core_memory.get_tool_list())
function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

main_model = LlamaCppEndpointSettings(
    completions_endpoint_url="http://127.0.0.1:8080/completion"
)

llama_cpp_agent = LlamaCppAgent(
    main_model,
    debug_output=True,
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

# Have a dialog with Cory.
# When you tell Cory to remember something, Cory will store it in its Core Memory. Beware that regular message caching has been deactivated for demonstration purposes.
# That means Cory will not remember your messages or its responses. It only remembers facts in its core memory.

while True:
    user_input = input("USER> ")

    if "exit" in user_input:
        break

    output = llama_cpp_agent.get_chat_response(
        user_input,
        system_prompt=f"You are Cory, a advanced helpful AI assistant interacting through calling functions in form of JSON objects.\n\n{agent_core_memory.get_core_memory_manager().build_core_memory_context()}\n\nHere are your available functions:\n\n{function_tool_registry.get_documentation()}",
        add_message_to_chat_history=False,
        add_response_to_chat_history=False,
        temperature=0.65,
        function_tool_registry=function_tool_registry,
    )
