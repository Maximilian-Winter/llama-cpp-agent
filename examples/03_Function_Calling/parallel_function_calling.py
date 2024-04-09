from enum import Enum
from typing import Union

from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import (
    LlamaCppEndpointSettings,
)


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
        description="First number.",
        max_precision=5,
        min_precision=2,
    )
    number_two: Union[int, float] = Field(
        ...,
        description="Second number.",
        max_precision=5,
        min_precision=2,
    )
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


function_tools = [LlamaCppFunctionTool(Calculator)]
function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools, True)
model = LlamaCppEndpointSettings("http://localhost:8080/completion")
llama_cpp_agent = LlamaCppAgent(
    model,
    debug_output=False,
    system_prompt=f"You are an advanced AI, tasked to assist the user by calling functions in JSON format. You can also perform parallel function calls.\n\n\n{function_tool_registry.get_documentation()}",
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

user_input = "Solve the following calculations: 42 * 42, 24 * 24, 5 * 5, 89 * 75, 42 * 46, 69 * 85, 422 * 420, 753 * 321, 72 * 55, 240 * 204, 789 * 654, 123 * 321, 432 * 89, 564 * 321?"
print(
    llama_cpp_agent.get_chat_response(
        user_input,
        temperature=0.45,
        function_tool_registry=function_tool_registry,
    )
)
