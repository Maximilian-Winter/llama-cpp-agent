import datetime

from typing import Optional
from pydantic import BaseModel, Field
from typing import Union, get_type_hints
from enum import Enum

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.tool_calls import Tool, Function

from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.mixtral_8x22b_agent import Mixtral8x22BAgent
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings


def get_current_datetime(output_format: Optional[str] = None):
    """
    Get the current date and time in the given format.

    Args:
         output_format: Formatting string for the date and time, defaults to '%Y-%m-%d %H:%M:%S'
    """
    if output_format is None:
        output_format = '%Y-%m-%d %H:%M:%S'
    return datetime.datetime.now().strftime(output_format)


# Enum for the calculator tool.
class MathOperation(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


# llama-cpp-agent also supports "Instructor" library like function definitions as Pydantic models for function calling.
# Simple pydantic calculator tool for the agent that can add, subtract, multiply, and divide. Docstring and description of fields will be used in system prompt.
class calculator(BaseModel):
    """
    Perform a math operation on two numbers.
    """
    number_one: Union[int, float] = Field(..., description="First number.")
    operation: MathOperation = Field(..., description="Math operation to perform.")
    number_two: Union[int, float] = Field(..., description="Second number.")

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


# Example function based on an OpenAI example.
# llama-cpp-agent also supports OpenAI like dictionaries for function definition.
def get_current_weather(location, unit):
    """Get the current weather in a given location"""
    if "London" in location:
        return f"Weather in {location}: {22}° {unit.value}"
    elif "New York" in location:
        return f"Weather in {location}: {24}° {unit.value}"
    elif "North Pole" in location:
        return f"Weather in {location}: {-42}° {unit.value}"
    else:
        return f"Weather in {location}: unknown"


# Here is a function definition in OpenAI style
open_ai_tool_definition = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit, e.g. celsius, fahrenheit",
                },
            },
            "required": ["location"],
        },
    },
}

# First we create a calculator tool by passing a pydantic class with a run method representing a calculator, to the LlamaCppFunctionTool constructor.
calculator_function_tool = LlamaCppFunctionTool(calculator)

# Next we create a current datetime tool by passing a function to the LlamaCppFunctionTool constructor.
current_datetime_function_tool = LlamaCppFunctionTool(get_current_datetime)

# For OpenAI tool definitions, we pass a OpenAI function definition with actual function in a tuple to the LlamaCppFunctionTool constructor.
get_weather_function_tool = LlamaCppFunctionTool((open_ai_tool_definition, get_current_weather))

model = LlamaCppEndpointSettings(
    "http://localhost:8080/completion"
)

agent = Mixtral8x22BAgent(model=model)

result = agent.get_response("Get the date and time in '%d-%m-%Y %H:%M' format.", tools=[current_datetime_function_tool])

print(result)

result = agent.get_response("Solve the following calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6  and 96/8.", tools=[calculator_function_tool])

print(result)

result = agent.get_response("Get the current weather in celsius in London, New York and at the North Pole.", tools=[get_weather_function_tool])

print(result)