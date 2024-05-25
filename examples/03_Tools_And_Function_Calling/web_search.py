import json

from duckduckgo_search import DDGS
from trafilatura import fetch_url, extract

from llama_cpp_agent import LlamaCppAgent, MessagesFormatterType
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent.providers import LlamaCppServerProvider

def send_message_to_user(message: str):
    """
    Send a message to user.
    Args:
        message (str): Message to send.
    """
    print(message)

class DDGWebSearch:

    def __init__(self, llm_provider, message_formatter_type):
        self.summarising_agent = LlamaCppAgent(llm_provider, debug_output=True,
                                               system_prompt="You are a text summarization and information extraction specialist and you are able to summarize and filter out information relevant to a specific query.",
                                               predefined_messages_formatter_type=message_formatter_type)

    def get_website_content_from_url(self, url: str) -> str:
        """
        Get website content from a URL using Selenium and BeautifulSoup for improved content extraction and filtering.

        Args:
            url (str): URL to get website content from.

        Returns:
            str: Extracted content including title, main text, and tables.
        """

        try:
            downloaded = fetch_url(url)

            result = extract(downloaded, include_formatting=True, include_links=True, output_format='json', url=url)

            if result:
                result = json.loads(result)
                return f'=========== Website Title: {result["title"]} ===========\n\n=========== Website URL: {url} ===========\n\n=========== Website Content ===========\n\n{result["raw_text"]}\n\n=========== Website Content End ===========\n\n'
            else:
                return ""
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def search_web(self, search_query: str):
        """
        Search the web for information.
        Args:
            search_query (str): Search query to search for.
        """
        results = DDGS().text(search_query, region='wt-wt', safesearch='off', max_results=4)
        result_string = ''
        for res in results:
            web_info = self.get_website_content_from_url(res['href'])
            if web_info != "":
                web_info = self.summarising_agent.get_chat_response(
                    f"Please summarize the following Website content and extract relevant information to this query:'{search_query}'.\n\n" + web_info,
                    add_response_to_chat_history=False, add_message_to_chat_history=False)
                result_string += web_info

        res = result_string.strip()
        return "Based on the following results, answer the previous user query:\nResults:\n\n" + res

    def get_tool(self):
        return self.search_web


provider = LlamaCppServerProvider("http://hades.hq.solidrust.net:8084")
#provider = LlamaCppServerProvider("http://localhost:8080")
agent = LlamaCppAgent(
    provider,
    debug_output=True,
    system_prompt="You are a helpful assistant. Use additional available information you have access to when giving a response. Always give detailed and long responses. Format your response, well structured in markdown format.",
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
    add_tools_and_structures_documentation_to_system_prompt=True,
)

search_tool = DDGWebSearch(provider, MessagesFormatterType.CHATML)

settings = provider.get_provider_default_settings()

settings.temperature = 0.45
settings.max_tokens = 1024
output_settings = LlmStructuredOutputSettings.from_functions(
    [search_tool.get_tool(), send_message_to_user])
user = input(">")
result = agent.get_chat_response(user,
                                 llm_sampling_settings=settings, structured_output_settings=output_settings)
while True:
    if result[0]["function"] == "send_message_to_user":
        user = input(">")
        result = agent.get_chat_response(user, structured_output_settings=output_settings,
                                         llm_sampling_settings=settings)
    else:
        result = agent.get_chat_response(result[0]["return_value"], role=Roles.tool,
                                         structured_output_settings=output_settings, llm_sampling_settings=settings)
