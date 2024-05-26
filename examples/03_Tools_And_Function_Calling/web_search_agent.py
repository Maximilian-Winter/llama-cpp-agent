from llama_cpp_agent import MessagesFormatterType, LlamaCppAgent
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent.providers import LlamaCppServerProvider
from llama_cpp_agent.tools import WebSearchTool


def send_message_to_user(message: str):
    """
    Send a message to user.
    Args:
        message (str): Message to send.
    """
    print(message)


provider = LlamaCppServerProvider("http://localhost:8080")
agent = LlamaCppAgent(
    provider,
    debug_output=True,
    system_prompt="You are a helpful assistant. Use additional available information you have access to when giving a response. Always give detailed and long responses. Format your response, well structured in markdown format.",
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
    add_tools_and_structures_documentation_to_system_prompt=True,
)

search_tool = WebSearchTool(provider, MessagesFormatterType.CHATML, max_tokens_search_results=20000)

settings = provider.get_provider_default_settings()

settings.temperature = 0.65
# settings.top_p = 0.85
# settings.top_k = 60
# settings.tfs_z = 0.95
settings.max_tokens = 2048
output_settings = LlmStructuredOutputSettings.from_functions(
    [search_tool.get_tool(), send_message_to_user])
user = input(">")
result = agent.get_chat_response(user, prompt_suffix="\n```json\n",
                                 llm_sampling_settings=settings, structured_output_settings=output_settings)
while True:
    if result[0]["function"] == "send_message_to_user":
        user = input(">")
        result = agent.get_chat_response(user, prompt_suffix="\n```json\n", structured_output_settings=output_settings,
                                         llm_sampling_settings=settings)
    else:
        result = agent.get_chat_response(result[0]["return_value"], role=Roles.tool, prompt_suffix="\n```json\n",
                                         structured_output_settings=output_settings, llm_sampling_settings=settings)
