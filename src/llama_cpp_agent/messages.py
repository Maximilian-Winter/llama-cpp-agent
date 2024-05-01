import string
from enum import Enum
import random
from typing import Literal, Union, List, Optional, Annotated, Dict, Any

from pydantic import BaseModel, Field

from llama_cpp_agent.providers.provider_base import LLMProviderBase


def generate_id(length=9):
    # Characters to use in the ID
    characters = string.ascii_letters + string.digits
    # Random choice of characters
    return "".join(random.choice(characters) for _ in range(length))


class ToolType(Enum):
    function = "function"


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: ToolType = ToolType.function
    function: FunctionCall


class Roles(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class BaseMessage(BaseModel):
    role: Literal[Roles.system, Roles.user, Roles.assistant, Roles.tool]


class UserMessage(BaseMessage):
    role: Literal[Roles.user] = Roles.user
    content: str


class SystemMessage(BaseMessage):
    role: Literal[Roles.system] = Roles.system
    content: str


class AssistantMessage(BaseMessage):
    role: Literal[Roles.assistant] = Roles.assistant
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ToolMessage(BaseMessage):
    content: str
    role: Literal[Roles.tool] = Roles.tool
    tool_call_id: str


ChatMessage = Annotated[Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage], Field(discriminator="role")]


# Function to convert messages to dictionary format
def convert_messages_to_dict(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    result = []
    for message in messages:
        # Determine the appropriate content to include
        content = ""
        if isinstance(message, AssistantMessage):
            if message.content is not None:
                content = message.content
            elif message.tool_calls is not None:
                if len(message.tool_calls) > 1:
                    content = "Function Calls:\n"
                    count = 1
                    for tool_call in message.tool_calls:
                        content += f"{count}. Function Call Id: {tool_call.id}\nFunction: {tool_call.function.name}\nArguments: {tool_call.function.arguments}\n"
                        count += 1
                else:
                    content = f"Function Call:\nFunction Call Id: {message.tool_calls[0].id}\nFunction: {message.tool_calls[0].function.name}\nArguments: {message.tool_calls[0].function.arguments}\n"
        elif isinstance(message, ToolMessage):
            content = f"Function Call Result:\nFunction Call Id: {message.tool_call_id}\nResult: {message.content}\n"
        else:
            content = f"{message.content}"
        # Construct the dictionary for the current message
        msg_dict = {
            'role': message.role.value,
            'content': content
        }
        result.append(msg_dict)
    return result


class ChatMemoryStrategy(Enum):
    last_k_messages: str = "last_k_messages"
    last_k_tokens: str = "last_k_tokens"


class ChatMemory:
    def __init__(self, memory_strategy: ChatMemoryStrategy = ChatMemoryStrategy.last_k_messages, k: int = 10, llm_provider: LLMProviderBase = None,

                 messages: List[ChatMessage] = None):
        if messages is None:
            messages = []
        self.messages: List[ChatMessage] = messages
        self.k = k
        self.strategy = memory_strategy
        self.llm_provider = llm_provider
        if memory_strategy == ChatMemoryStrategy.last_k_tokens and llm_provider is None:
            raise Exception("LLM provider needed when using last k tokens as memory strategy!")

    def add_message(self, message: ChatMessage):
        self.messages.append(message)

    def add_user_message(self, message: str):
        self.messages.append(UserMessage(role=Roles.user, content=message))

    def add_system_message(self, message: str):
        self.messages.append(SystemMessage(role=Roles.system, content=message))

    def remove_last_message(self):
        self.messages = self.messages[:-1]

    def remove_last_k_message(self, k: int):
        self.messages = self.messages[:-k]

    def pop_message(self, k: int) -> ChatMessage:
        return self.messages.pop(k)

    def get_message(self, k: int) -> ChatMessage:
        return self.messages[k]

    def get_last_message(self) -> ChatMessage:
        return self.messages[-1]

    def get_last_k_messages(self, k: int) -> List[ChatMessage]:
        return self.messages[-k:]

    def get_chat(self) -> List[Dict[str, str]]:
        if self.strategy == ChatMemoryStrategy.last_k_messages:
            return convert_messages_to_dict(self.get_last_k_messages(self.k))
        elif self.strategy == ChatMemoryStrategy.last_k_tokens:
            total_tokens = 0
            selected_messages = []
            converted_messages = convert_messages_to_dict(self.messages)
            sys_message = None
            if converted_messages[0]['role'] == Roles.system:
                sys_message = converted_messages.pop(0)
            for message in reversed(converted_messages):
                tokens = self.llm_provider.tokenize(message["content"])
                total_tokens += len(tokens)
                if total_tokens >= self.k:
                    if len(selected_messages) == 0:
                        selected_messages.append(message)
                    break
                else:
                    selected_messages.append(message)
            if sys_message is not None:
                selected_messages.append(sys_message)
            return list(reversed(selected_messages))
        return []

