from llama_cpp_agent import LlamaCppAgent, MessagesFormatterType
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent.providers import LlamaCppServerProvider
from llama_cpp_agent.providers.provider_base import LlmProvider, LlmProviderId
from web_search_interfaces import WebCrawler, WebSearchProvider
from default_web_crawlers import TrafilaturaWebCrawler
from default_web_search_providers import DDGWebSearchProvider


class WebSearchTool:

    def __init__(self, llm_provider: LlmProvider, message_formatter_type: MessagesFormatterType, context_character_limit: int = 7500,
                 web_crawler: WebCrawler = None, web_search_provider: WebSearchProvider = None, temperature: int = 0.45,
                 top_p: int = 0.95,
                 top_k: int = 40,
                 max_tokens_per_summary: int = 750):
        self.summarising_agent = LlamaCppAgent(llm_provider, debug_output=True,
                                               system_prompt="You are a text summarization and information extraction specialist and you are able to summarize and filter out information relevant to a specific query.",
                                               predefined_messages_formatter_type=message_formatter_type)
        if web_crawler is None:
            self.web_crawler = TrafilaturaWebCrawler()
        else:
            self.web_crawler = web_crawler

        if web_search_provider is None:
            self.web_search_provider = DDGWebSearchProvider()
        else:
            self.web_search_provider = web_search_provider

        self.context_character_limit = context_character_limit
        settings = llm_provider.get_provider_default_settings()
        provider_id = llm_provider.get_provider_identifier()
        settings.temperature = temperature
        settings.top_p = top_p
        settings.top_k = top_k

        if provider_id == LlmProviderId.llama_cpp_server:
            settings.n_predict = max_tokens_per_summary
        elif provider_id == LlmProviderId.tgi_server:
            settings.max_new_tokens = max_tokens_per_summary
        else:
            settings.max_tokens = max_tokens_per_summary

        self.settings = settings

    def search_web(self, search_query: str):
        """
        Search the web for information.
        Args:
            search_query (str): Search query to search for.
        """
        results = self.web_search_provider.search_web(search_query)
        result_string = ''
        for res in results:
            web_info = self.web_crawler.get_website_content_from_url(res)
            if web_info != "":
                web_info = self.summarising_agent.get_chat_response(
                    f"Please summarize the following Website content and extract relevant information to this query:'{search_query}'.\n\n" + web_info,
                    add_response_to_chat_history=False, add_message_to_chat_history=False, llm_sampling_settings=self.settings)
                result_string += web_info

        res = result_string.strip()
        return "Based on the following results, answer the previous user query:\nResults:\n\n" + res[:self.context_character_limit]

    def get_tool(self):
        return self.search_web



