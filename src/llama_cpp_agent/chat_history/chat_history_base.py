import json
from abc import abstractmethod, ABC
from enum import Enum
from typing import List, Dict

from llama_cpp_agent.chat_history.messages import ChatMessage


class ChatMessageStore(ABC):
    """
    An abstract base class for storing and managing chat messages.
    """

    @abstractmethod
    def get_messages_count(self):
        """
        Get the total count of messages in the store.
        """
        pass

    @abstractmethod
    def add_message(self, message: ChatMessage):
        """
        Add a new message to the store.

        :param message: The ChatMessage object to add.
        """
        pass

    @abstractmethod
    def edit_message(self, index: int, edited_message: ChatMessage):
        """
        Edit a message at a specific index in the store.

        :param index: The index of the message to edit.
        :param edited_message: The edited ChatMessage object.
        """
        pass

    @abstractmethod
    def add_user_message(self, message: str):
        """
        Add a user message to the store.

        :param message: The content of the user message.
        """
        pass

    @abstractmethod
    def add_assistant_message(self, message: str):
        """
        Add an assistant message to the store.

        :param message: The content of the assistant message.
        """
        pass

    @abstractmethod
    def add_system_message(self, message: str):
        """
        Add a system message to the store.

        :param message: The content of the system message.
        """
        pass

    @abstractmethod
    def remove_last_message(self):
        """
        Remove the last message from the store.
        """
        pass

    @abstractmethod
    def remove_last_k_messages(self, k: int):
        """
        Remove the last k messages from the store.

        :param k: The number of messages to remove.
        """
        pass

    @abstractmethod
    def get_message(self, index: int) -> ChatMessage:
        """
        Get a message at a specific index from the store.

        :param index: The index of the message to retrieve.
        :return: The ChatMessage object at the specified index.
        """
        pass

    @abstractmethod
    def get_last_message(self) -> ChatMessage:
        """
        Get the last message from the store.

        :return: The last ChatMessage object in the store.
        """
        pass

    @abstractmethod
    def get_last_k_messages(self, k: int) -> List[ChatMessage]:
        """
        Get the last k messages from the store.

        :param k: The number of messages to retrieve.
        :return: A list of the last k ChatMessage objects.
        """
        pass

    @abstractmethod
    def get_messages(self, k: int) -> List[ChatMessage]:
        """
        Get the messages starting from index k.

        :param k: The starting index.
        :return: A list of ChatMessage objects starting from index k.
        """
        pass

    @abstractmethod
    def get_all_messages(self) -> List[ChatMessage]:
        """
        Get all messages from the store.

        :return: A list of all ChatMessage objects in the store.
        """
        pass

    @abstractmethod
    def save_to_json(self, file_path: str):
        """
        Save the messages to a JSON file.

        :param file_path: The path to the JSON file.
        """
        pass

    @abstractmethod
    def load_from_json(self, file_path: str):
        """
        Load messages from a JSON file.

        :param file_path: The path to the JSON file.
        """
        pass


class ChatHistory(ABC):
    """
    An abstract base class for managing chat history.
    """

    @abstractmethod
    def get_message_store(self) -> ChatMessageStore:
        """
        Get the message store associated with the chat history.

        :return: The ChatMessageStore object.
        """
        pass

    @abstractmethod
    def get_chat_messages(self) -> List[Dict[str, str]]:
        """
        Get the chat messages as a list of dictionaries.

        :return: A list of dictionaries representing the chat messages.
        """
        pass

    @abstractmethod
    def add_message(self, message: Dict[str, str]):
        """
        Add a message to the chat history.

        :param message: A dictionary representing the message to add.
        """
        pass

    @abstractmethod
    def get_messages_count(self):
        """
        Get the total count of messages in the chat history.

        :return: The count of messages.
        """
        pass

    @abstractmethod
    def edit_message(self, index: int, edited_message: Dict[str, str]):
        """
        Edit a message at a specific index in the chat history.

        :param index: The index of the message to edit.
        :param edited_message: A dictionary representing the edited message.
        """
        pass

    @abstractmethod
    def get_message(self, index) -> Dict[str, str]:
        """
        Get a message at a specific index from the chat history.

        :param index: The index of the message to retrieve.
        :return: A dictionary representing the message at the specified index.
        """
        pass
