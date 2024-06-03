from enum import Enum
from typing import List

from llama_cpp_agent import LlamaCppAgent, MessagesFormatterType
from llama_cpp_agent.llm_prompt_template import PromptTemplate
from llama_cpp_agent.prompt_templates import summarizing_system_prompt_ocr, general_summarizing_system_prompt, \
    website_summarizing_system_prompt
from llama_cpp_agent.providers.provider_base import LlmProviderId, LlmProvider


class TextType(Enum):
    ocr = 'ocr'
    web = 'web'
    text = 'text'


class SummarizerTool:

    def __init__(self, llm_provider: LlmProvider, message_formatter_type: MessagesFormatterType,
                 temperature: int = 0.45,
                 top_p: int = 0.95,
                 top_k: int = 40,
                 model_max_context_tokens=8192,
                 max_tokens_per_summary: int = 750):
        self.llm_provider = llm_provider
        self.summarising_agent = LlamaCppAgent(llm_provider, debug_output=True,
                                               system_prompt="",
                                               predefined_messages_formatter_type=message_formatter_type)

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
        self.max_tokens_per_summary = max_tokens_per_summary
        self.settings = settings

    def summarize_text(self, user_query: str, input_texts: List[str], text_type: TextType = TextType.text) -> List[str]:
        """
        Summarizes the list of input texts.
        Args:
            user_query (str): The initial query of the user to focus on in the summarization process.
            input_texts (str): A list of texts to summarize.
            text_type (TextType): The type of input text. Can be either TextType.text or TextType.ocr or TextType.web
        """
        result_strings = []
        for input in input_texts:
            if input != "":
                tokens = self.llm_provider.tokenize(input)
                original_prompt_token_count = len(tokens)
                remove_char_count = 0
                has_remove_char = False
                if original_prompt_token_count > (self.model_max_context_tokens - self.max_tokens_per_summary):
                    has_remove_char = True
                    while True:
                        if (self.model_max_context_tokens - self.max_tokens_per_summary) >= len(tokens):
                            break
                        else:
                            remove_char_count += 50
                            tokens = self.llm_provider.tokenize(input[:remove_char_count])
                if has_remove_char:
                    input = input[:remove_char_count]

                template = general_summarizing_system_prompt

                if text_type == TextType.ocr:
                    template = summarizing_system_prompt_ocr
                elif text_type == TextType.web:
                    template = website_summarizing_system_prompt
                elif text_type == TextType.text:
                    template = general_summarizing_system_prompt

                summary = self.summarising_agent.get_chat_response(
                    input, system_prompt=PromptTemplate.from_string(template).generate_prompt(
                        {"QUERY": user_query}),
                    add_response_to_chat_history=False, add_message_to_chat_history=False,
                    llm_sampling_settings=self.settings)
                result_strings.append(f"{summary.strip()}")

        return result_strings

    def get_tool(self):
        return self.summarize_text
