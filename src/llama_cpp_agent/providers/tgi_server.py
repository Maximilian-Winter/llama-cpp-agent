import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union

import requests

from llama_cpp_agent.llm_output_settings import (
    LlmStructuredOutputType,
    LlmStructuredOutputSettings,
)
from llama_cpp_agent.providers.provider_base import (
    LlmProviderId,
    LlmProvider,
    LlmSamplingSettings,
)


@dataclass
class TGIServerSamplingSettings(LlmSamplingSettings):
    """
    TGIServerSamplingSettings dataclass
    """

    best_of: Optional[int] = field(default=None, metadata={"minimum": 0})
    decoder_input_details: bool = False
    details: bool = True
    do_sample: bool = False
    frequency_penalty: Optional[float] = field(
        default=None, metadata={"exclusiveMinimum": -2}
    )
    grammar: Optional[dict] = None
    max_new_tokens: Optional[int] = field(default=None, metadata={"minimum": 0})
    repetition_penalty: Optional[float] = field(
        default=None, metadata={"exclusiveMinimum": 0}
    )
    return_full_text: Optional[bool] = field(default=None)
    seed: Optional[int] = field(default=None, metadata={"minimum": 0})
    stop: Optional[List[str]] = field(default_factory=list)
    temperature: Optional[float] = field(default=None, metadata={"exclusiveMinimum": 0})
    top_k: Optional[int] = field(default=None, metadata={"exclusiveMinimum": 0})
    top_n_tokens: Optional[int] = field(
        default=None, metadata={"minimum": 0, "exclusiveMinimum": 0}
    )
    top_p: Optional[float] = field(
        default=None, metadata={"maximum": 1, "exclusiveMinimum": 0}
    )
    truncate: Optional[int] = field(default=None, metadata={"minimum": 0})
    typical_p: Optional[float] = field(
        default=None, metadata={"maximum": 1, "exclusiveMinimum": 0}
    )
    watermark: bool = False
    stream: bool = False

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.tgi_server

    def get_additional_stop_sequences(self) -> Union[List[str], None]:
        return self.stop

    def add_additional_stop_sequences(self, sequences: List[str]):
        self.stop.extend(sequences)

    def is_streaming(self):
        return self.stream

    @staticmethod
    def load_from_dict(settings: dict) -> "TGIServerSamplingSettings":
        """
        Load the settings from a dictionary.

        Args:
            settings (dict): The dictionary containing the settings.

        Returns:
            LlamaCppSamplingSettings: The loaded settings.
        """
        return TGIServerSamplingSettings(**settings)

    def as_dict(self) -> dict:
        """
        Convert the settings to a dictionary.

        Returns:
            dict: The dictionary representation of the settings.
        """
        return self.__dict__


class TGIServerProvider(LlmProvider):

    def __init__(self, server_address: str, api_key: str = None):
        self.server_address = server_address
        self.server_completion_endpoint = self.server_address + "/generate"
        self.server_streaming_completion_endpoint = (
                self.server_address + "/generate_stream"
        )
        self.server_chat_completion_endpoint = (
                self.server_address + "/v1/chat/completions"
        )
        self.server_tokenize_endpoint = self.server_address + "/tokenize"
        self.api_key = api_key

    def is_using_json_schema_constraints(self):
        return True

    def get_provider_identifier(self) -> LlmProviderId:
        return LlmProviderId.tgi_server

    def get_provider_default_settings(self) -> TGIServerSamplingSettings:
        return TGIServerSamplingSettings()

    def create_completion(
            self,
            prompt: str,
            structured_output_settings: LlmStructuredOutputSettings,
            settings: TGIServerSamplingSettings,
            bos_token: str,
    ):
        if self.api_key is not None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",  # Add API key as bearer token
            }
        else:
            headers = {"Content-Type": "application/json"}

        settings_dict = deepcopy(settings.as_dict())

        data = {"parameters": settings_dict, "inputs": prompt}
        grammar = None
        if (
                structured_output_settings.output_type
                != LlmStructuredOutputType.no_structured_output
        ):
            grammar = structured_output_settings.get_json_schema()

        if grammar is not None:
            data["parameters"]["grammar"] = {"type": "json", "value": grammar}
        if settings.stream:
            return self.get_response_stream(
                headers, data, self.server_streaming_completion_endpoint
            )

        response = requests.post(
            self.server_completion_endpoint, headers=headers, json=data
        )
        data = response.json()

        returned_data = {"choices": [{"text": data["generated_text"]}]}
        return returned_data

    def create_chat_completion(
            self,
            messages: List[Dict[str, str]],
            structured_output_settings: LlmStructuredOutputSettings,
            settings: TGIServerSamplingSettings
    ):
        if self.api_key is not None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",  # Add API key as bearer token
            }
        else:
            headers = {"Content-Type": "application/json"}
        data = deepcopy(settings.as_dict())
        data["messages"] = messages
        data["model"] = "tgi"
        grammar = None
        if (
                structured_output_settings.output_type
                != LlmStructuredOutputType.no_structured_output
        ):
            grammar = structured_output_settings.get_json_schema()

        if grammar is not None:
            data["parameters"]["grammar"] = {"type": "json", "value": grammar}
        if settings.stream:
            return self.get_response_stream(
                headers, data, self.server_chat_completion_endpoint
            )
        response = requests.post(
            self.server_chat_completion_endpoint, headers=headers, json=data
        )
        data = response.json()

        returned_data = data
        return returned_data

    def tokenize(self, prompt: str) -> list[int]:
        if self.api_key is not None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",  # Add API key as bearer token
            }
        else:
            headers = {"Content-Type": "application/json"}
        response = requests.post(
            self.server_tokenize_endpoint, headers=headers, json={"inputs": prompt}
        )
        if response.status_code == 200:
            tokens = response.json()
            return [tok["id"] for tok in tokens]
        else:
            raise Exception(
                f"Tokenization request failed. Status code: {response.status_code}\nResponse: {response.text}"
            )

    def get_response_stream(self, headers, data, endpoint_address):
        response = requests.post(
            endpoint_address,
            headers=headers,
            json=data,
            stream=True,
        )

        # Check if the request was successful
        response.raise_for_status()

        # Define a generator function to yield text chunks
        def generate_text_chunks():
            try:
                decoded_chunk = ""
                for chunk in response.iter_lines():
                    if chunk:
                        decoded_chunk += chunk.decode("utf-8")
                        new_data = json.loads(decoded_chunk.replace("data:", ""))
                        if "token" in new_data and (
                                new_data["token"]["text"] is not None
                        ):
                            returned_data = {
                                "choices": [{"text": new_data["token"]["text"]}]
                            }
                            yield returned_data
                        elif (
                                "choices" in new_data
                                and (new_data["choices"][0]["delta"] is not None)
                                and ("content" in new_data["choices"][0]["delta"])
                        ):
                            returned_data = {
                                "choices": [
                                    {"text": new_data["choices"][0]["delta"]["content"]}
                                ]
                            }
                            yield returned_data
                        decoded_chunk = ""

            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")

        return generate_text_chunks()
