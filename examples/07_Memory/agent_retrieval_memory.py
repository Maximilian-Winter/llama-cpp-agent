from llama_cpp import Llama
from pydantic import BaseModel, Field

from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings

from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.agent_memory.memory_tools import AgentRetrievalMemory
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.providers import LlamaCppServerProvider


class SendMessageToUser(BaseModel):
    """
    Send a message to the User.
    """

    message: str = Field(..., description="Message you want to send to the user.")

    def run(self):
        print("Message: " + self.message)


agent_retrieval_memory = AgentRetrievalMemory()

function_tools = [LlamaCppFunctionTool(SendMessageToUser)]

function_tools.extend(agent_retrieval_memory.get_tool_list())
structured_output_settings = LlmStructuredOutputSettings.from_llama_cpp_function_tools(function_tools)

provider = LlamaCppServerProvider("http://localhost:8080")

llama_cpp_agent = LlamaCppAgent(provider, debug_output=True,
                                predefined_messages_formatter_type=MessagesFormatterType.CHATML)

user_input = 'Add my Birthday the 1991.12.11 to the retrieval memory.'

user_input = llama_cpp_agent.get_chat_response(
    user_input,
    system_prompt=f"You are a advanced helpful AI assistant interacting through calling functions in form of JSON objects.",
    structured_output_settings=structured_output_settings)
role = Roles.tool
while True:

    if user_input[0]["function"] == "SendMessageToUser":
        user_input = input("Input your message: ")
        role = Roles.user
    else:
        role = Roles.tool
    if isinstance(user_input, str):
        user_input = llama_cpp_agent.get_chat_response(
            user_input,
            role=role,
            system_prompt=f"You are a advanced helpful AI assistant interacting through calling functions in form of JSON objects.",
            structured_output_settings=structured_output_settings)
    else:
        user_input = llama_cpp_agent.get_chat_response(
            user_input[0]["return_value"],
            role=role,
            system_prompt=f"You are a advanced helpful AI assistant interacting through calling functions in form of JSON objects.",
            structured_output_settings=structured_output_settings)
