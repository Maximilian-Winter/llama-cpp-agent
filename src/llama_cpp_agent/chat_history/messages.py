import string
from enum import Enum
import random
from typing import Literal, Union, List, Optional, Annotated, Dict

from pydantic import BaseModel, Field


def generate_function_call_id(length=9):
    # Characters to use in the ID
    characters = string.ascii_letters + string.digits
    # Random choice of characters
    return "".join(random.choice(characters) if (idx % 500 != 0) else "\n" for idx in range(length))


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


ChatMessage = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage],
    Field(discriminator="role"),
]


# Function to convert messages to list of dictionary format
def convert_messages_to_list_of_dictionaries(
    messages: List[ChatMessage],
) -> List[Dict[str, str]]:
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
            content = f"{message.content}\n"
        else:
            content = f"{message.content}"
        # Construct the dictionary for the current message
        msg_dict = {"role": message.role.value, "content": content}
        result.append(msg_dict)
    return result
