import json
import string
from abc import ABC, abstractmethod
from enum import Enum
import random
from typing import Literal, Union, List, Optional, Annotated, Dict, Any

from pydantic import BaseModel, Field, parse_obj_as

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
    tool_call_id: str
    role: Literal[Roles.tool] = Roles.tool
    content: str


ChatMessage = Annotated[Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage], Field(discriminator="role")]


class ChatHistory(ABC):
    @abstractmethod
    def get_chat_messages(self):
        pass


class ChatMessageStore(ABC):

    @abstractmethod
    def add_message(self, message: ChatMessage):
        pass

    @abstractmethod
    def add_user_message(self, message: str):
        pass

    @abstractmethod
    def add_system_message(self, message: str):
        pass

    @abstractmethod
    def remove_last_message(self):
        pass

    @abstractmethod
    def remove_last_k_message(self, k: int):
        pass

    @abstractmethod
    def pop_message(self, k: int) -> ChatMessage:
        pass

    @abstractmethod
    def get_message(self, k: int) -> ChatMessage:
        pass

    @abstractmethod
    def get_last_message(self) -> ChatMessage:
        pass

    @abstractmethod
    def get_last_k_messages(self, k: int) -> List[ChatMessage]:
        pass

    @abstractmethod
    def get_messages(self, k: int) -> List[ChatMessage]:
        pass

    @abstractmethod
    def get_all_messages(self) -> List[ChatMessage]:
        pass


# Function to convert messages to list of dictionary format
def convert_messages_to_list_of_dictionaries(messages: List[ChatMessage]) -> List[Dict[str, str]]:
    """
    Converts a list of messages to a list of dictionaries.
    Args:
        messages (List[ChatMessage]): The list of messages.
    Returns:
        List[Dict[str, str]]: A list of dictionaries.
    """
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
                        content += f"{count}. Function: {tool_call.function.name}\nArguments: {tool_call.function.arguments}\n"
                        count += 1
                else:
                    content = f"Function Call:\nFunction: {message.tool_calls[0].function.name}\nArguments: {message.tool_calls[0].function.arguments}\n"
        elif isinstance(message, ToolMessage):
            content = f"Function Call Result:\nResult: {message.content}\n"
        else:
            content = f"{message.content}"
        # Construct the dictionary for the current message
        msg_dict = {
            'role': message.role.value,
            'content': content
        }
        result.append(msg_dict)
    return result


class BasicChatMessageStore(ChatMessageStore):

    def __init__(self, messages: List[ChatMessage] = None):
        if messages is None:
            messages = []
        self.messages: List[ChatMessage] = messages

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

    def get_messages(self, k: int) -> List[ChatMessage]:
        return self.messages[k:]

    def get_all_messages(self) -> List[ChatMessage]:
        return self.messages

    def save_to_json(self, file_path: str):
        # Convert messages to a list of dictionaries using pydantic's model_dump() method
        messages_dict = [message.model_dump() for message in self.messages]
        # Write the list of dictionaries to a JSON file
        with open(file_path, 'w') as file:
            json.dump(messages_dict, file, indent=4)

    def load_from_json(self, file_path: str):
        # Read the list of dictionaries from a JSON file
        with open(file_path, 'r') as file:
            messages_dict = json.load(file)
        # Convert dictionaries back to ChatMessage instances
        self.messages = [parse_obj_as(ChatMessage, message) for message in messages_dict]


class BasicChatHistoryStrategy(Enum):
    last_k_messages: str = "last_k_messages"
    last_k_tokens: str = "last_k_tokens"


class BasicChatHistory(ChatHistory):
    def __init__(self, chat_history_strategy: BasicChatHistoryStrategy = BasicChatHistoryStrategy.last_k_messages,
                 k: int = 10,
                 llm_provider: LLMProviderBase = None, message_store: ChatMessageStore = None):
        if message_store is None:
            message_store = BasicChatMessageStore()
        self.message_store: ChatMessageStore = message_store
        self.k = k
        self.strategy = chat_history_strategy
        self.llm_provider = llm_provider
        if chat_history_strategy == BasicChatHistoryStrategy.last_k_tokens and llm_provider is None:
            raise Exception("Please pass a LLM provider to BasicChatHistory when using last k tokens as memory strategy!")

    def get_chat_messages(self) -> List[Dict[str, str]]:
        if self.strategy == BasicChatHistoryStrategy.last_k_messages:
            return convert_messages_to_list_of_dictionaries(self.message_store.get_last_k_messages(self.k))
        elif self.strategy == BasicChatHistoryStrategy.last_k_tokens:
            total_tokens = 0
            selected_messages = []
            converted_messages = convert_messages_to_list_of_dictionaries(self.message_store.get_all_messages())
            sys_message = None
            if converted_messages[0]['role'] == "system":
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
