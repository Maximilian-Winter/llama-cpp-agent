import json
import random
import re
import string
import uuid
from enum import Enum
from typing import List, Dict, Literal, Tuple

from llama_cpp import Llama
from pydantic import ValidationError
from transformers import AutoTokenizer

from llama_cpp_agent.chat_history import ChatMessage, SystemMessage, UserMessage
from llama_cpp_agent.chat_history.messages import convert_messages_to_list_of_dictionaries, ToolCall, FunctionCall, \
    ToolMessage, AssistantMessage
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent.llm_prompt_template import PromptTemplate
from llama_cpp_agent.messages_formatter import MessagesFormatter

from llama_cpp_agent.providers.provider_base import LlmProvider


def generate_id(length=8):
    # Characters to use in the ID
    characters = string.ascii_letters + string.digits
    # Random choice of characters
    return "".join(random.choice(characters) for _ in range(length))


def parse_tool_calls(text):
    # List to hold all extracted dictionaries
    json_dicts = []

    # Regular expression to find <tool_call>...</tool_call> patterns
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"

    # Find all occurrences of the pattern
    tool_calls = re.findall(tool_call_pattern, text, re.DOTALL)

    # Process each JSON within the found tags
    for json_text in tool_calls:
        json_text = json_text.strip()
        try:
            json_dict = json.loads(json_text)
            json_dicts.append(json_dict)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")

    return json_dicts


class Hermes2ProMessageFormatter(MessagesFormatter):
    """
    Class representing a messages formatter for LLMs.
    """

    def __init__(self, PRE_PROMPT: str = "", SYS_PROMPT_START: str = "", SYS_PROMPT_END: str = "", USER_PROMPT_START: str = "",
                 USER_PROMPT_END: str = "", ASSISTANT_PROMPT_START: str = "", ASSISTANT_PROMPT_END: str = "",
                 INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE: bool = False, DEFAULT_STOP_SEQUENCES: List[str] = None):
        """
        Initializes a new MessagesFormatter object.
        """
        super().__init__(PRE_PROMPT, SYS_PROMPT_START, SYS_PROMPT_END, USER_PROMPT_START, USER_PROMPT_END,
                         ASSISTANT_PROMPT_START, ASSISTANT_PROMPT_END, INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE,
                         DEFAULT_STOP_SEQUENCES)
        SYS_PROMPT_START_MIXTRAL = """"""
        SYS_PROMPT_END_MIXTRAL = """\n\n"""
        USER_PROMPT_START_MIXTRAL = """[INST] """
        USER_PROMPT_END_MIXTRAL = """ """
        ASSISTANT_PROMPT_START_MIXTRAL = """[/INST] """
        ASSISTANT_PROMPT_END_MIXTRAL = """</s>"""
        FUNCTION_PROMPT_START_MIXTRAL = """"""
        FUNCTION_PROMPT_END_MIXTRAL = """"""
        DEFAULT_MIXTRAL_STOP_SEQUENCES = ["</s>"]
        self.PRE_PROMPT = ""
        self.SYS_PROMPT_START = "<|im_start|>system\n"
        self.SYS_PROMPT_END = "<|im_end|>\n"
        self.USER_PROMPT_START = "<|im_start|>user\n"
        self.USER_PROMPT_END = "<|im_end|>\n"
        self.ASSISTANT_PROMPT_START = "<|im_start|>assistant\n"
        self.ASSISTANT_PROMPT_END = "<|im_end|>\n"
        self.INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE = False

        self.DEFAULT_STOP_SEQUENCES = [
            "<|im_end|>",
            "<|im_start|>assistant",
            "<|im_start|>user",
            "<|im_start|>system"
        ]
        self.FUNCTION_PROMPT_START = "<|im_start|>tool\n"
        self.FUNCTION_PROMPT_END = "<|im_end|>\n"
        self.USE_USER_ROLE_FUNCTION_CALL_RESULT = False
        self.STRIP_PROMPT = True

    def format_messages(
        self,
        messages: List[Dict[str, str]],
        response_role: Literal["user", "assistant"] | None = None,
    ) -> Tuple[str, str]:
        """
        Formats a list of messages into a conversation string.

        Args:
            messages (List[Dict[str, str]]): List of messages with role and content.
            response_role(Literal["system", "user", "assistant", "function"]|None): Forces the response role to be "system", "user" or "assistant".
        Returns:
            Tuple[str, str]: Formatted conversation string and the role of the last message.
        """
        formatted_messages = self.PRE_PROMPT
        last_role = "assistant"

        no_user_prompt_start = False
        for message in messages:
            if message["role"] == "system":
                formatted_messages += (
                    self.SYS_PROMPT_START + message["content"] + self.SYS_PROMPT_END
                )
                last_role = "system"
                if self.INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE:
                    formatted_messages = self.USER_PROMPT_START + formatted_messages
                    no_user_prompt_start = True
            elif message["role"] == "user":
                if no_user_prompt_start:
                    no_user_prompt_start = False
                    formatted_messages += message["content"] + self.USER_PROMPT_END
                else:
                    formatted_messages += (
                        self.USER_PROMPT_START
                        + message["content"]
                        + self.USER_PROMPT_END
                    )
                last_role = "user"
            elif message["role"] == "assistant":
                if self.STRIP_PROMPT:
                    message["content"] = message["content"].strip()
                formatted_messages += (
                    self.ASSISTANT_PROMPT_START
                    + message["content"]
                    + self.ASSISTANT_PROMPT_END
                )
                last_role = "assistant"
            elif message["role"] == "tool":
                if isinstance(message["content"], list):
                    message["content"] = "\n".join(
                        [json.dumps(m, indent=2) for m in message["content"]]
                    )
                if self.USE_USER_ROLE_FUNCTION_CALL_RESULT:
                    formatted_messages += (
                        self.USER_PROMPT_START
                        + message["content"]
                        + self.USER_PROMPT_END
                    )
                    last_role = "user"
                else:
                    formatted_messages += (
                        self.FUNCTION_PROMPT_START
                        + message["content"]
                        + self.FUNCTION_PROMPT_END
                    )
                    last_role = "tool"
        if last_role == "system" or last_role == "user" or last_role == "tool":
            if self.STRIP_PROMPT:
                if response_role is not None:
                    if response_role == "assistant":
                        return (
                            formatted_messages + self.ASSISTANT_PROMPT_START.strip(),
                            "assistant",
                        )
                    if response_role == "user":
                        return (
                            formatted_messages + self.USER_PROMPT_START.strip(),
                            "user",
                        )
                else:
                    return (
                        formatted_messages + self.ASSISTANT_PROMPT_START.strip(),
                        "assistant",
                    )
            else:
                if response_role is not None:
                    if response_role == "assistant":
                        return (
                            formatted_messages + self.ASSISTANT_PROMPT_START,
                            "assistant",
                        )
                    if response_role == "user":
                        return formatted_messages + self.USER_PROMPT_START, "user"
                else:
                    return formatted_messages + self.ASSISTANT_PROMPT_START, "assistant"
        if self.STRIP_PROMPT:
            if response_role is not None:
                if response_role == "assistant":
                    return (
                        formatted_messages + self.ASSISTANT_PROMPT_START.strip(),
                        "assistant",
                    )
                if response_role == "user":
                    return formatted_messages + self.USER_PROMPT_START.strip(), "user"
            else:
                return formatted_messages + self.USER_PROMPT_START.strip(), "user"
        else:
            if response_role is not None:
                if response_role == "assistant":
                    return formatted_messages + self.ASSISTANT_PROMPT_START, "assistant"
                if response_role == "user":
                    return formatted_messages + self.USER_PROMPT_START, "user"
            else:
                return formatted_messages + self.USER_PROMPT_START, "user"


class Hermes2ProAgent:
    def __init__(
        self,
        provider: LlmProvider,
        system_prompt: str = None,
        debug_output: bool = False,
    ):
        self.messages: list[ChatMessage] = []
        self.messages_formatter = Hermes2ProMessageFormatter()
        self.agent = LlamaCppAgent(provider, debug_output=debug_output, custom_messages_formatter=self.messages_formatter)
        self.debug_output = debug_output
        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = "You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.\nHere are the available tools:"
        self.messages.append(SystemMessage(content="system_prompt"))

        self.sys_prompt_template = """{system_prompt} <tools> {tools} </tools>
Use the following pydantic model json schema for each tool call you will make: {"properties": {"arguments": {"title": "Arguments", "type": "object"}, "name": {"title": "Name", "type": "string"}}, "required": ["arguments", "name"], "title": "FunctionCall", "type": "object"} 
For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"arguments": <args-dict>, "name": <function-name>}
</tool_call>
"""
        self.system_prompter = PromptTemplate.from_string(self.sys_prompt_template)

    def get_response(
        self,
        message=None,
        structured_output_settings: LlmStructuredOutputSettings = None,
    ):
        if message is not None:
            msg = UserMessage(content=message)
            self.messages.append(msg)
        openai_tools = []
        openai_tool_mapping = {}
        for tool in structured_output_settings.function_tools:
            openai_tools.append(tool.to_openai_tool())
            openai_tool_mapping[tool.model.__name__] = tool

        self.messages[0].content = self.system_prompter.generate_prompt(
            {"tools": json.dumps(openai_tools), "system_prompt": self.system_prompt}
        )
        text, role = self.messages_formatter.format_messages(
            convert_messages_to_list_of_dictionaries(self.messages)
        )

        if self.debug_output:
            print(text)
        result = self.agent.get_text_response(
            text,

            print_output=self.debug_output,
        )

        if "<tool_call>" in result:
            tool_calls = []
            if result.strip().endswith("</tool_call"):
                result += ">"
            function_calls = parse_tool_calls(result)
            tool_messages = []
            for function_call in function_calls:
                tool = openai_tool_mapping[function_call["name"]]
                cls = tool.model
                call_parameters = function_call["arguments"]
                try:
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
                except ValidationError as e:
                    tool_messages.append(ToolMessage(content=str(e), tool_call_id="-1"))
                    self.messages.append(
                        AssistantMessage(content=result, tool_calls=tool_calls)
                    )
                    self.messages.extend(tool_messages)
            self.messages.append(
                AssistantMessage(content=result, tool_calls=tool_calls)
            )
            self.messages.extend(tool_messages)
            return self.get_response(structured_output_settings=structured_output_settings)
        else:
            self.messages.append(AssistantMessage(content=result.strip()))
            return result.strip()
