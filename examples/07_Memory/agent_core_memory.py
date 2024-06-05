import datetime
import os

from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent, SystemPromptModules, SystemPromptModule
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.agent_memory.memory_tools import AgentCoreMemory
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.providers import LlamaCppServerProvider, VLLMServerProvider

provider = LlamaCppServerProvider("http://localhost:8080")


class SendMessageToUser(BaseModel):
    """
    Send a message to the User.
    """

    message: str = Field(..., description="Message you want to send to the user.")

    def run(self):
        print("Message: " + self.message)


function_tools = [LlamaCppFunctionTool(SendMessageToUser)]
agent_core_memory = AgentCoreMemory(["persona", "human"])

if os.path.exists("core_memory.json"):
    agent_core_memory.load_core_memory("core_memory.json")

function_tools.extend(agent_core_memory.get_tool_list())


llama_cpp_agent = LlamaCppAgent(
    provider,
    debug_output=True,
    predefined_messages_formatter_type=MessagesFormatterType.CHATML
)
output_settings = LlmStructuredOutputSettings.from_llama_cpp_function_tools(function_tools, add_thoughts_and_reasoning_field=True)
llm_settings = provider.get_provider_default_settings()
llm_settings.n_predict = 1024
llm_settings.temperature = 0.35
llm_settings.top_k = 0
llm_settings.top_p = 1.0

core_memory_section = SystemPromptModule("core_memory", "The following section shows the current content of your core memory with information about your persona and the human you are interacting with:")
date_time_section = SystemPromptModule("current_date_time", "The following section shows the current date and time:")
while True:
    user_input = input("USER> ")

    if "exit" in user_input:
        break

    core_memory_section.set_content(agent_core_memory.get_core_memory_view().strip())
    date_time_section.set_content(datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S") + "\nFormat: dd.mm.yyyy HH:mm:ss")

    output = llama_cpp_agent.get_chat_response(
        user_input,
        llm_sampling_settings=llm_settings,
        system_prompt=f"You are an advanced AI assistant. You have access to a core memory section, which is always visible to you and you can write to it.",
        system_prompt_additions=SystemPromptModules([core_memory_section, date_time_section]),
        structured_output_settings=output_settings,
    )

    agent_core_memory.save_core_memory("core_memory.json")