import json
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union

from llama_cpp_agent.llm_output_settings import (
    LlmStructuredOutputSettings,
    LlmStructuredOutputType,
)
from llama_cpp_agent.providers.provider_base import (
    LlmProvider,
    LlmProviderId,
    LlmSamplingSettings,
)


@dataclass
class VLLMServerSamplingSettings(LlmSamplingSettings):
    """
    VLLMServerSamplingSettings dataclass
    """

    best_of: Optional[int] = None
    use_beam_search = False
    top_k: float = -1
    top_p: float = 1
    min_p: float = 0.0
    temperature: float = 0.7
    max_tokens: int = 16
    repetition_penalty: Optional[float] = 1.0
    length_penalty: Optional[float] = 1.0
    early_stopping: Optional[bool] = False
    ignore_eos: Optional[bool] = False
    min_tokens: Optional[int] = 0
    stop_token_ids: Optional[List[int]] = field(default_factory=list)
    skip_special_tokens: Optional[bool] = True
    spaces_between_special_tokens: Optional[bool] = True
    stream: bool = False

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.vllm_server

    def get_additional_stop_sequences(self) -> Union[List[str], None]:
        return None

    def add_additional_stop_sequences(self, sequences: List[str]):
        pass

    def is_streaming(self):
        return self.stream

    @staticmethod
    def load_from_dict(settings: dict) -> "VLLMServerSamplingSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            LlamaCppSamplingSettings: The loaded settings.
        """
        return VLLMServerSamplingSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        return self.__dict__


class VLLMServerProvider(LlmProvider):
    def __init__(self, base_url: str, model: str, huggingface_model: str, api_key: str = None):
        from openai import OpenAI
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key if api_key else "xxx-xxxxxxxx",
        )
        self.model = model

    def is_using_json_schema_constraints(self):
        return True

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.vllm_server

    def get_provider_default_settings(self) -> VLLMServerSamplingSettings:
        return VLLMServerSamplingSettings()

    def create_completion(
        self,
        prompt: str,
        structured_output_settings: LlmStructuredOutputSettings,
        settings: VLLMServerSamplingSettings,
        bos_token: str,
    ):
        grammar = None
        if (
            structured_output_settings.output_type
            != LlmStructuredOutputType.no_structured_output
        ):
            grammar = structured_output_settings.get_json_schema()
        top_p = settings.top_p
        stream = settings.stream
        temperature = settings.temperature
        max_tokens = settings.max_tokens

        settings_dict = deepcopy(settings.as_dict())
        settings_dict.pop("top_p")
        settings_dict.pop("stream")
        settings_dict.pop("temperature")
        settings_dict.pop("max_tokens")
        if grammar is not None:
            settings_dict["guided_json"] = grammar

        if settings.stream:
            result = self.client.completions.create(
                prompt=prompt,
                model=self.model,
                extra_body=settings_dict,
                top_p=top_p,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            def generate_chunks():
                for chunk in result:
                    if chunk.choices is not None:
                        if chunk.choices[0].text is not None:
                            yield {"choices": [{"text": chunk.choices[0].text}]}

            return generate_chunks()
        else:
            result = self.client.completions.create(
                prompt=prompt,
                model=self.model,
                extra_body=settings_dict,
                top_p=top_p,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return {"choices": [{"text": result.choices[0].text}]}

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        structured_output_settings: LlmStructuredOutputSettings,
        settings: VLLMServerSamplingSettings
    ):
        grammar = None
        if (
            structured_output_settings.output_type
            != LlmStructuredOutputType.no_structured_output
        ):
            grammar = structured_output_settings.get_json_schema()

        top_p = settings.top_p
        stream = settings.stream
        temperature = settings.temperature
        max_tokens = settings.max_tokens

        settings_dict = copy(settings.as_dict())
        settings_dict.pop("top_p")
        settings_dict.pop("stream")
        settings_dict.pop("temperature")
        settings_dict.pop("max_tokens")
        if grammar is not None:
            settings_dict["guided_json"] = grammar

        if settings.stream:
            result = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                extra_body=settings_dict,
                top_p=top_p,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            def generate_chunks():
                for chunk in result:
                    if chunk.choices[0].delta.content is not None:
                        yield {"choices": [{"text": chunk.choices[0].delta.content}]}

            return generate_chunks()
        else:
            result = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                extra_body=settings_dict,
                top_p=top_p,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return {"choices": [{"text": result.choices[0].message.content}]}

    def tokenize(self, prompt: str) -> list[int]:
        result = self.tokenizer.encode(text=prompt)
        return result
