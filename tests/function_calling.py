from enum import Enum
from typing import Union

from pydantic import BaseModel, Field

from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent import LlamaCppFunctionTool
from llama_cpp_agent.providers import TGIServerProvider, LlamaCppServerProvider, VLLMServerProvider

provider = VLLMServerProvider("http://localhost:8123/v1", "TitanML/Mistral-7B-Instruct-v0.2-AWQ-4bit","TitanML/Mistral-7B-Instruct-v0.2-AWQ-4bit", "token-abc123")


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

output_settings = LlmStructuredOutputSettings.from_llama_cpp_function_tools(function_tools, add_thoughts_and_reasoning_field=True, add_heartbeat_field=True)
#output_settings.add_function_name_to_heartbeat_list(Calculator.__name__)
llama_cpp_agent = LlamaCppAgent(
    provider,
    debug_output=True,
    predefined_messages_formatter_type=MessagesFormatterType.MISTRAL,
)

user_input = "What is 71549 * 75312?"

print("Agent Input: " + user_input + "\n\nAgent Output:")

llm_settings = provider.get_provider_default_settings()
llm_settings.max_tokens = 1024

llama_cpp_agent.get_chat_response(
    user_input,
    llm_sampling_settings=llm_settings,
    structured_output_settings=output_settings,
    print_output=True
)
