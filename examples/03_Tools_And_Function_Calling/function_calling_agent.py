# Example that uses the FunctionCallingAgent class to create a function calling agent.
import datetime
from enum import Enum
from typing import Union, Optional

from pydantic import BaseModel, Field

from llama_cpp_agent import LlamaCppFunctionTool
from llama_cpp_agent import FunctionCallingAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppServerProvider

model = LlamaCppServerProvider("http://localhost:8080")


# Simple tool for the agent, to get the current date and time in a specific format.
def get_current_datetime(output_format: Optional[str] = None):
    """
    Get the current date and time in the given format.

    Args:
         output_format: formatting string for the date and time, defaults to '%Y-%m-%d %H:%M:%S'
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
# llama-cpp-agent supports OpenAI like schemas for function definition.
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
open_ai_tool_spec = {
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
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
        },
    },
}


# Callback for receiving messages for the user.
def send_message_to_user_callback(message: str):
    print("Assistant: " + message.strip())

# First we create the calculator tool.
calculator_function_tool = LlamaCppFunctionTool(calculator)

# Next we create the current datetime tool.
current_datetime_function_tool = LlamaCppFunctionTool(get_current_datetime)

# The from_openai_tool function of the LlamaCppFunctionTool class converts an OpenAI tool schema and a callable function into a LlamaCppFunctionTool
get_weather_function_tool = LlamaCppFunctionTool.from_openai_tool(open_ai_tool_spec, get_current_weather)

# Create the function calling agent. We are passing the provider, the tool list, send message to user callback and the chat message formatter. Also, we allow parallel function calling.
function_call_agent = FunctionCallingAgent(
    model,
    llama_cpp_function_tools=[calculator_function_tool, current_datetime_function_tool, get_weather_function_tool],
    send_message_to_user_callback=send_message_to_user_callback,
    allow_parallel_function_calling=True,
    debug_output=False,
    messages_formatter_type=MessagesFormatterType.CHATML)

user_input = '''Get the date and time in '%d-%m-%Y %H:%M' format. Get the current weather in celsius in London, New York and at the North Pole. Solve the following calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6  and 96/8.'''
print("User: " + user_input)

settings = model.get_provider_default_settings()
settings.add_additional_stop_sequences(["<|end|>"])
settings.stream = False
settings.temperature = 0.65

function_call_agent.generate_response(user_input, llm_sampling_settings=settings)

