import os

from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.agent_memory.memory_tools import AgentCoreMemory
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.providers import LlamaCppServerProvider

provider = LlamaCppServerProvider("http://hades.hq.solidrust.net:8084")


class SendMessageToUser(BaseModel):
    """
    Send a message to the User.
    """

    message: str = Field(..., description="Message you want to send to the user.")

    def run(self):
        print("Message: " + self.message)


function_tools = [LlamaCppFunctionTool(SendMessageToUser)]
agent_core_memory = AgentCoreMemory()

if os.path.exists("core_memory.json"):
    agent_core_memory.load_core_memory("core_memory.json")

function_tools.extend(agent_core_memory.get_tool_list())


llama_cpp_agent = LlamaCppAgent(
    provider,
    debug_output=True,
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)
output_settings = LlmStructuredOutputSettings.from_llama_cpp_function_tools(function_tools, add_thoughts_and_reasoning_field=True)
# Have a dialog with Cory.
# When you tell Cory to remember something, Cory will store it in its Core Memory. Beware that regular message caching has been deactivated for demonstration purposes.
# That means Cory will not remember your messages or its responses. It only remembers facts in its core memory.

while True:
    user_input = input("USER> ")

    if "exit" in user_input:
        break

    output = llama_cpp_agent.get_chat_response(
        user_input,
        system_prompt=f"You are Cory, an advanced AI assistant. You can remember things between chat session by reading and writing to your core memory.",
        add_message_to_chat_history=False,
        add_response_to_chat_history=False,
        structured_output_settings=output_settings,
    )
    agent_core_memory.save_core_memory("core_memory.json")