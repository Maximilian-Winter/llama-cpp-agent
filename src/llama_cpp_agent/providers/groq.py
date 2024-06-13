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
class GroqSamplingSettings(LlmSamplingSettings):
    """
    GroqSamplingSettings dataclass
    """

    top_p: float = 1
    temperature: float = 0.7
    max_tokens: int = 16
    stream: bool = False

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.groq

    def get_additional_stop_sequences(self) -> Union[List[str], None]:
        return None

    def add_additional_stop_sequences(self, sequences: List[str]):
        pass

    def is_streaming(self):
        return self.stream

    @staticmethod
    def load_from_dict(settings: dict) -> "GroqSamplingSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            LlamaCppSamplingSettings: The loaded settings.
        """
        return GroqSamplingSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        return self.__dict__


class GroqProvider(LlmProvider):
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
        return LlmProviderId.groq

    def get_provider_default_settings(self) -> GroqSamplingSettings:
        return GroqSamplingSettings()

    def create_completion(
            self,
            prompt: str | list[dict],
            structured_output_settings: LlmStructuredOutputSettings,
            settings: GroqSamplingSettings,
            bos_token: str,
    ):
        tools = None
        if (
                structured_output_settings.output_type
                == LlmStructuredOutputType.function_calling
                or structured_output_settings.output_type == LlmStructuredOutputType.parallel_function_calling
        ):
            tools = [tool.to_openai_tool() for tool in structured_output_settings.function_tools]
        top_p = settings.top_p
        stream = settings.stream
        temperature = settings.temperature
        max_tokens = settings.max_tokens

        settings_dict = deepcopy(settings.as_dict())
        settings_dict.pop("top_p")
        settings_dict.pop("stream")
        settings_dict.pop("temperature")
        settings_dict.pop("max_tokens")

        if settings.stream:
            result = self.client.chat.completions.create(
                messages=prompt,
                model=self.model,
                extra_body=settings_dict,
                tools=tools,
                top_p=top_p,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            def generate_chunks():
                for chunk in result:
                    if chunk.choices[0].delta.tool_calls is not None:
                        if tools is not None:
                            args = chunk.choices[0].delta.tool_calls[0].function.arguments
                            args_loaded = json.loads(args)
                            function_name = chunk.choices[0].delta.tool_calls[0].function.name
                            function_dict = {structured_output_settings.function_calling_name_field_name: function_name, structured_output_settings.function_calling_content: args_loaded}
                            yield {"choices": [{"text": json.dumps(function_dict)}]}
                    if chunk.choices[0].delta.content is not None:
                        yield {"choices": [{"text": chunk.choices[0].delta.content}]}

            return generate_chunks()
        else:
            result = self.client.chat.completions.create(
                messages=prompt,
                model=self.model,
                extra_body=settings_dict,
                tools=tools,
                top_p=top_p,
                stream=stream,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if tools is not None:
                args = result.choices[0].message.tool_calls[0].function.arguments
                args_loaded = json.loads(args)
                function_name = result.choices[0].message.tool_calls[0].function.name
                function_dict = {structured_output_settings.function_calling_name_field_name: function_name, structured_output_settings.function_calling_content: args_loaded}
                return {"choices": [{"text": json.dumps(function_dict)}]}
            return {"choices": [{"text": result.choices[0].message.content}]}

    def create_chat_completion(
            self,
            messages: List[Dict[str, str]],
            structured_output_settings: LlmStructuredOutputSettings,
            settings: GroqSamplingSettings
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
