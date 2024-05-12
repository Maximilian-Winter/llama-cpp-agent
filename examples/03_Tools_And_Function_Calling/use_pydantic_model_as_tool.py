from enum import Enum
from typing import Union

from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.providers.tgi_server import TGIServerProvider

provider = TGIServerProvider("http://localhost:8080")


# Simple pydantic calculator tool for the agent that can add, subtract, multiply, and divide.
class MathOperation(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


# The Pydantic Calculator tool needs a run which executes the tool.
class Calculator(BaseModel):
    """
    Perform a math operation on two numbers.
    """

    number_one: Union[int, float] = Field(
        ...,
        description="First number."
    )
    number_two: Union[int, float] = Field(
        ...,
        description="Second number."
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

output_settings = LlmStructuredOutputSettings.from_llama_cpp_function_tools(function_tools,
                                                                            allow_parallel_function_calling=True)
llama_cpp_agent = LlamaCppAgent(
    provider,
    debug_output=False,
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

user_input = "What is 42 + 42?"
print(
    llama_cpp_agent.get_chat_response(
        user_input,
        structured_output_settings=output_settings
    )
)
