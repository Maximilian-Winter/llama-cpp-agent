from enum import Enum
from typing import Union

from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.providers.llama_cpp_server import LlamaCppServerProvider

model = LlamaCppServerProvider("http://127.0.0.1:8080")


# Simple pydantic calculator tool for the agent that can add, subtract, multiply, and divide.
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

# Create a list of function call tools.
function_tools = [LlamaCppFunctionTool(Calculator)]

output_settings = LlmStructuredOutputSettings.from_llama_cpp_function_tools(function_tools, output_type=LlmStructuredOutputType.function_calling)
llama_cpp_agent = LlamaCppAgent(
    model,
    debug_output=False,
    system_prompt=f"You are an advanced AI, tasked to assist the user by calling functions in JSON format.\n\n\n{output_settings.get_llm_documentation()}",
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

user_input = "What is 42 + 42?"
print(
    llama_cpp_agent.get_chat_response(
        user_input,
        structured_output_settings=output_settings
    )
)
