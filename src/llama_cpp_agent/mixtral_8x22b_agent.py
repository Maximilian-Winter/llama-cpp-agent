import json
import random
import string
import uuid
from enum import Enum

from llama_cpp import Llama
from mistral_common.protocol.instruct.messages import (
    UserMessage,
    SystemMessage,
    AssistantMessage,
    ToolMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import ToolCall, FunctionCall
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.providers.provider_base import LlmProvider, LlmSamplingSettings


def generate_id(length=8):
    # Characters to use in the ID
    characters = string.ascii_letters + string.digits
    # Random choice of characters
    return "".join(random.choice(characters) for _ in range(length))


class Mixtral8x22BAgent:
    def __init__(
        self,
        provider: LlmProvider,
        system_prompt: str = None,
        debug_output: bool = False,
    ):
        self.messages: list[
            SystemMessage | UserMessage | AssistantMessage | ToolMessage
        ] = []
        self.agent = LlamaCppAgent(provider, debug_output=debug_output)
        self.debug_output = debug_output
        if system_prompt is not None:
            self.messages.append(SystemMessage(content=system_prompt))
        self.tokenizer_v3 = MistralTokenizer.v3()

    def get_response(
        self,
        message=None,
        tools: list[LlamaCppFunctionTool] = None,
        llm_sampling_settings: LlmSamplingSettings = None,
    ):
        if tools is None:
            tools = []
        if message is not None:
            msg = UserMessage(content=message)
            self.messages.append(msg)
        mistral_tools = []
        mistral_tool_mapping = {}
        for tool in tools:
            mistral_tools.append(tool.to_mistral_tool())
            mistral_tool_mapping[tool.model.__name__] = tool
        request = ChatCompletionRequest(
            tools=mistral_tools,
            messages=self.messages,
            model="open-mistral-7b",
        )
        tokenized = self.tokenizer_v3.encode_chat_completion(request)
        tokens, text = tokenized.tokens, tokenized.text
        if self.debug_output:
            print(text)
        result = self.agent.get_text_response(
            text,
            llm_sampling_settings=llm_sampling_settings,
            print_output=self.debug_output,
        )
        if result.strip().startswith("[TOOL_CALLS]"):
            tool_calls = []

            result = result.replace("[TOOL_CALLS]", "")
            function_calls = json.loads(result.strip())
            tool_messages = []
            for function_call in function_calls:
                tool = mistral_tool_mapping[function_call["name"]]
                cls = tool.model
                call_parameters = function_call["arguments"]
                call = cls(**call_parameters)
                output = call.run(**tool.additional_parameters)
                tool_call_id = generate_id(length=9)
                tool_calls.append(
                    ToolCall(
                        function=FunctionCall(
                            name=function_call["name"],
                            arguments=json.dumps(call_parameters),
                        ),
                        id=tool_call_id,
                    )
                )
                tool_messages.append(
                    ToolMessage(content=str(output), tool_call_id=tool_call_id)
                )
            self.messages.append(AssistantMessage(content=None, tool_calls=tool_calls))
            self.messages.extend(tool_messages)
            return self.get_response()
        else:
            self.messages.append(AssistantMessage(content=result.strip()))
            return result.strip()
