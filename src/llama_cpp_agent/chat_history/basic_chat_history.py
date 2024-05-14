import json
from enum import Enum
from typing import List, Dict

from pydantic import parse_obj_as

from llama_cpp_agent.chat_history.chat_history_base import ChatMessageStore, ChatHistory
from llama_cpp_agent.chat_history.messages import (
    ChatMessage,
    UserMessage,
    AssistantMessage,
    SystemMessage,
    Roles,
    convert_messages_to_list_of_dictionaries, ToolMessage, generate_function_call_id,
)
from llama_cpp_agent.providers.provider_base import LlmProvider


class BasicChatMessageStore(ChatMessageStore):
    def __init__(self, messages: List[ChatMessage] = None):
        if messages is None:
            messages = []
        self.messages: List[ChatMessage] = messages

    def get_messages_count(self):
        return len(self.messages)

    def add_message(self, message: ChatMessage):
        self.messages.append(message)

    def edit_message(self, index: int, edited_message: ChatMessage):
        self.messages[index] = edited_message

    def add_user_message(self, message: str):
        self.messages.append(UserMessage(role=Roles.user, content=message))

    def add_assistant_message(self, message: str):
        self.messages.append(AssistantMessage(role=Roles.assistant, content=message))

    def add_system_message(self, message: str):
        self.messages.append(SystemMessage(role=Roles.system, content=message))

    def remove_last_message(self):
        self.messages = self.messages[:-1]

    def remove_last_k_messages(self, k: int):
        self.messages = self.messages[:-k]

    def get_message(self, index: int) -> ChatMessage:
        return self.messages[index]

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
        messages_dict = [{"role": message.role.value, "content": message.content} for message in self.messages]
        # Write the list of dictionaries to a JSON file
        with open(file_path, "w") as file:
            json.dump(messages_dict, file, indent=4)

    def load_from_json(self, file_path: str):
        # Read the list of dictionaries from a JSON file
        with open(file_path, "r") as file:
            messages_dict = json.load(file)
        # Convert dictionaries back to ChatMessage instances
        self.messages = [SystemMessage(content=message["content"]) if message["role"] == "system" else
                         UserMessage(content=message["content"]) if message["role"] == "user" else
                         AssistantMessage(content=message["content"]) if message["role"] == "assistant" else
                         ToolMessage(content=message["content"]) for message in messages_dict]


class BasicChatHistoryStrategy(Enum):
    last_k_messages: str = "last_k_messages"
    last_k_tokens: str = "last_k_tokens"


class BasicChatHistory(ChatHistory):

    def __init__(
            self,
            chat_history_strategy: BasicChatHistoryStrategy = BasicChatHistoryStrategy.last_k_messages,
            k: int = 20,
            llm_provider: LlmProvider = None,
            message_store: ChatMessageStore = None,
    ):
        if message_store is None:
            message_store = BasicChatMessageStore()
        self.message_store: ChatMessageStore = message_store
        self.k = k
        self.strategy = chat_history_strategy
        self.llm_provider = llm_provider
        if (
                chat_history_strategy == BasicChatHistoryStrategy.last_k_tokens
                and llm_provider is None
        ):
            raise Exception(
                "Please pass a LLM provider to BasicChatHistory when using last k tokens as memory strategy!"
            )

    def get_message_store(self) -> BasicChatMessageStore:
        return self.message_store

    def get_chat_messages(self) -> List[Dict[str, str]]:
        if self.strategy == BasicChatHistoryStrategy.last_k_messages:
            converted_messages = convert_messages_to_list_of_dictionaries(
                self.message_store.get_last_k_messages(self.k - 1)
            )
            if len(converted_messages) == self.k and converted_messages[0]["role"] != "system":
                messages = [convert_messages_to_list_of_dictionaries(self.message_store.get_message(0))]
                messages.extend(converted_messages[1:])
                return messages
            return converted_messages
        elif self.strategy == BasicChatHistoryStrategy.last_k_tokens:
            total_tokens = 0
            selected_messages = []
            converted_messages = convert_messages_to_list_of_dictionaries(
                self.message_store.get_all_messages()
            )
            sys_message = None
            if converted_messages[0]["role"] == "system":
                sys_message = converted_messages.pop(0)
            if sys_message is not None:
                total_tokens = len(self.llm_provider.tokenize(sys_message["content"]))
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

    def add_message(self, message: Dict[str, str]):
        if message["role"] == Roles.system:
            self.message_store.add_message(
                SystemMessage(content=message["content"])
            )
        elif message["role"] == Roles.user:
            self.message_store.add_message(
                UserMessage(content=message["content"])
            )
        elif message["role"] == Roles.assistant:
            self.message_store.add_message(
                AssistantMessage(content=message["content"])
            )
        elif message["role"] == Roles.tool:
            self.message_store.add_message(
                ToolMessage(tool_call_id=generate_function_call_id(), content=message["content"])
            )

    def get_messages_count(self):
        return self.message_store.get_messages_count()

    def edit_message(self, index: int, message: Dict[str, str]):
        if message["role"] == Roles.system:
            self.message_store.edit_message(
                index,
                SystemMessage(content=message["content"])
            )
        elif message["role"] == Roles.user:
            self.message_store.edit_message(
                index,
                UserMessage(content=message["content"])
            )
        elif message["role"] == Roles.assistant:
            self.message_store.edit_message(
                index,
                AssistantMessage(content=message["content"])
            )
        elif message["role"] == Roles.tool:
            self.message_store.edit_message(
                index,
                ToolMessage(tool_call_id=generate_function_call_id(), content=message["content"])
            )

    def get_message(self, index) -> Dict[str, str]:
        message = self.message_store.get_message(index)
        return {"role": message.role, "content": message.content}
