import json

from duckduckgo_search import DDGS

from trafilatura import fetch_url, extract

from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppServerProvider


def get_website_content_from_url(url: str) -> str:
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


def search_web(search_query: str):
    """
    Search the web for information.
    Args:
        search_query (str): Search query to search for.
    """
    results = DDGS().text(search_query, region='wt-wt', safesearch='off', timelimit='y', max_results=3)
    result_string = ''
    for res in results:
        web_info = get_website_content_from_url(res['href'])
        if web_info != "":
            result_string += web_info

    res = result_string.strip()
    return "Based on the following results, answer the previous user query:\nResults:\n\n" + res


def send_message_to_user(message: str):
    """
    Send a message to user.
    Args:
        message (str): Message to send.
    """
    print(message)


def chat_with_agent():
    provider = LlamaCppServerProvider("http://127.0.0.1:8080")

    # result = search_web_agent.generate_response("Research the web on how to use react native and give me a summary.")
    agent = LlamaCppAgent(provider,
                          system_prompt="You are a helpful assistant. Use additional available information you have access to when giving a response. Always give detailed and long responses. Format your response, well structured in markdown format.",
                          predefined_messages_formatter_type=MessagesFormatterType.MISTRAL)
    settings = provider.get_provider_default_settings()
    settings.n_predict = 2048
    settings.temperature = 0.45
    settings.top_p = 1.0
    settings.top_k = 0
    settings.min_p = 0.1
    output_settings = LlmStructuredOutputSettings.from_functions(
        [search_web, send_message_to_user])
    user = input(">")
    result = agent.get_chat_response(user,
                                     llm_sampling_settings=settings, structured_output_settings=output_settings)
    while True:
        if result[0]["function"] == "send_message_to_user":
            user = input(">")
            result = agent.get_chat_response(user, structured_output_settings=output_settings)
        else:
            result = agent.get_chat_response(result[0]["return_value"], role=Roles.tool,
                                             structured_output_settings=output_settings)


if __name__ == '__main__':
    chat_with_agent()
