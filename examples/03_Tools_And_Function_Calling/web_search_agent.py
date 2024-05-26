from llama_cpp_agent import MessagesFormatterType, LlamaCppAgent
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent.prompt_templates import web_search_system_prompt
from llama_cpp_agent.providers import LlamaCppServerProvider
from llama_cpp_agent.tools import WebSearchTool

provider = LlamaCppServerProvider("http://hades.hq.solidrust.net:8084")
agent = LlamaCppAgent(
    provider,
    debug_output=True,
    system_prompt=web_search_system_prompt,
    predefined_messages_formatter_type=MessagesFormatterType.MISTRAL,
    add_tools_and_structures_documentation_to_system_prompt=True,
)


def write_message_to_user():
    """
    Let you write a message to the user.
    """
    return "Please write the message to the user."


search_tool = WebSearchTool(provider, MessagesFormatterType.MISTRAL, max_tokens_search_results=20000)

settings = provider.get_provider_default_settings()

settings.temperature = 0.65
# settings.top_p = 0.85
# settings.top_k = 60
# settings.tfs_z = 0.95
settings.max_tokens = 2048
output_settings = LlmStructuredOutputSettings.from_functions(
    [search_tool.get_tool(), write_message_to_user])


def run_web_search_agent():
    user = input(">")
    if user == "exit":
        return
    result = agent.get_chat_response(user,
                                     llm_sampling_settings=settings, structured_output_settings=output_settings)
    while True:
        if result[0]["function"] == "write_message_to_user":
            break
        else:
            result = agent.get_chat_response(result[0]["return_value"], role=Roles.tool,
                                             structured_output_settings=output_settings, llm_sampling_settings=settings)

    result = agent.get_chat_response(result[0]["return_value"], role=Roles.tool,
                                     llm_sampling_settings=settings)

    print(result)
    run_web_search_agent()

run_web_search_agent()