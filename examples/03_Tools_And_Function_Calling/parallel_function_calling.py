# Import the necessary classes for the pydantic tool and the agent
from enum import Enum
from typing import Union

from pydantic import BaseModel, Field

from llama_cpp_agent.function_calling_agent import FunctionCallingAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.providers.tgi_server import TGIServerProvider

# Set up the provider
provider = TGIServerProvider("http://localhost:8080")


# Simple calculator tool for the agent that can add, subtract, multiply, and divide.
class MathOperation(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


class Calculator(BaseModel):
    """
    Perform a math operation on two numbers.
    """

    number_one: Union[int, float] = Field(
        ...,
        description="First number.")
    number_two: Union[int, float] = Field(
        ...,
        description="Second number.")
    operation: MathOperation = Field(..., description="Math operation to perform.")

    def run(self):
        if self.operation == MathOperation.ADD:
            return self.number_one + self.number_two
        elif self.operation == MathOperation.SUBTRACT:
            return self.number_one - self.number_two
        elif self.operation == MathOperation.MULTIPLY:
            return self.number_one * self.number_two
        elif self.operation == MathOperation.DIVIDE:
            return self.number_one / self.number_two
        else:
            raise ValueError("Unknown operation.")


# Callback for receiving messages for the user.
def send_message_to_user_callback(message: str):
    print(message)


# Create a list of function call tools.
function_tools = [LlamaCppFunctionTool(Calculator)]

# Create the function calling agent. We are passing the provider, the tool list, send message to user callback and the chat message formatter. Also we allow parallel function calling.
function_call_agent = FunctionCallingAgent(
    provider,
    llama_cpp_function_tools=function_tools,
    allow_parallel_function_calling=True,
    send_message_to_user_callback=send_message_to_user_callback,
    messages_formatter_type=MessagesFormatterType.CHATML)

# Define the user input.
user_input = "Solve the following calculations: 42 * 42, 24 * 24, 5 * 5, 89 * 75, 42 * 46, 69 * 85, 422 * 420, 753 * 321, 72 * 55, 240 * 204, 789 * 654, 123 * 321, 432 * 89, 564 * 321?"
function_call_agent.generate_response(user_input)
