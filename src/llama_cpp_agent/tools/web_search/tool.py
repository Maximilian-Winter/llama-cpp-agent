from typing import List

import pypdf

from llama_cpp_agent import LlamaCppAgent, MessagesFormatterType
from llama_cpp_agent.providers.provider_base import LlmProvider, LlmProviderId
from .web_search_interfaces import WebCrawler, WebSearchProvider
from .default_web_crawlers import TrafilaturaWebCrawler, ReadabilityWebCrawler
from .default_web_search_providers import DDGWebSearchProvider, GoogleWebSearchProvider
from ...llm_prompt_template import PromptTemplate
from ...prompt_templates import website_summarizing_system_prompt, general_summarizing_system_prompt, \
    summarizing_system_prompt_ocr


class WebSearchTool:

    def __init__(self, llm_provider: LlmProvider, message_formatter_type: MessagesFormatterType,
                 web_crawler: WebCrawler = None, web_search_provider: WebSearchProvider = None, temperature: int = 0.45,
                 top_p: int = 0.95,
                 top_k: int = 40,
                 model_max_context_tokens=8192,
                 max_tokens_search_results: int = 7500,
                 max_tokens_per_summary: int = 750,
                 number_of_search_results: int = 3):
        self.llm_provider = llm_provider
        self.summarising_agent = LlamaCppAgent(llm_provider, debug_output=True,
                                               system_prompt="",
                                               predefined_messages_formatter_type=message_formatter_type)
        if web_crawler is None:
            self.web_crawler = TrafilaturaWebCrawler()
        else:
            self.web_crawler = web_crawler

        if web_search_provider is None:
            self.web_search_provider = DDGWebSearchProvider()
        else:
            self.web_search_provider = web_search_provider
        self.number_of_search_results = number_of_search_results
        self.max_tokens_search_results = max_tokens_search_results
        settings = llm_provider.get_provider_default_settings()
        provider_id = llm_provider.get_provider_identifier()
        settings.temperature = temperature
        settings.top_p = top_p
        settings.top_k = top_k
        self.model_max_context_tokens = model_max_context_tokens
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
        results = self.web_search_provider.search_web(search_query, self.number_of_search_results)
        result_string = ''
        for res in results:
            web_info = self.web_crawler.get_website_content_from_url(res)
            if web_info != "":
                tokens = self.llm_provider.tokenize(web_info)
                original_prompt_token_count = len(tokens)
                remove_char_count = 0
                has_remove_char = False
                if original_prompt_token_count > (self.model_max_context_tokens - 512):
                    has_remove_char = True
                    while True:
                        if self.max_tokens_search_results >= len(tokens):
                            break
                        else:
                            remove_char_count += 50
                            tokens = self.llm_provider.tokenize(web_info[:remove_char_count])
                if has_remove_char:
                    web_info = web_info[:remove_char_count]
                web_info = self.summarising_agent.get_chat_response(
                    web_info, system_prompt=PromptTemplate.from_string(website_summarizing_system_prompt).generate_prompt({"QUERY": search_query, "WEBSITE_URL": res}),
                    add_response_to_chat_history=False, add_message_to_chat_history=False,
                    llm_sampling_settings=self.settings)
                result_string += f"\n{web_info.strip()}"

        result_string = result_string.strip()
        tokens = self.llm_provider.tokenize(result_string)
        original_prompt_token_count = len(tokens)
        remove_char_count = 0
        has_remove_char = False
        if original_prompt_token_count > self.max_tokens_search_results:
            has_remove_char = True
            while True:
                if self.max_tokens_search_results >= len(tokens):
                    break
                else:
                    remove_char_count += 50
                    tokens = self.llm_provider.tokenize(result_string[:remove_char_count])
        if not has_remove_char:
            return result_string
        return result_string[:remove_char_count]

    def get_tool(self):
        return self.search_web


