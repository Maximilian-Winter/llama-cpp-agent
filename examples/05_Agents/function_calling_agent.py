# Example that uses the FunctionCallingAgent class to create a function calling agent.
import datetime
import json
from enum import Enum
from typing import Union, Any, Optional

from pydantic import BaseModel, Field

from llama_cpp_agent.llm_settings import LlamaLLMSettings, LlamaLLMGenerationSettings

from llama_cpp_agent.function_calling_agent import FunctionCallingAgent
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings, LlamaCppGenerationSettings


# llama-cpp-agent supports type hinted function definitions for function calling.
# Write to file function that can be used by the agent. Docstring will be used in system prompt.
def write_to_file(chain_of_thought: str, file_path: str, file_content: str):
    """
    Write file to the user filesystem.
    :param chain_of_thought: Your chain of thought while writing the file.
    :param file_path: The file path includes the filename and file ending.
    :param file_content: The actual content to write.
    """
    print(chain_of_thought)
    with open(file_path, mode="w", encoding="utf-8") as file:
        file.write(file_content)
    return f"File {file_path} successfully written."


# Read file function that can be used by the agent. Docstring will be used in system prompt.
def read_file(file_path: str):
    """
    Read file from the user filesystem.
    :param file_path: The file path includes the filename and file ending.
    :return: File content.
    """
    output = ""
    with open(file_path, mode="r", encoding="utf-8") as file:
        output = file.read()
    return f"Content of file '{file_path}':\n\n{output}"


def get_current_datetime(output_format: Optional[str] = None):
    """
    Get the current date and time in the given format.

    Args:
        output_format: format string for the date and time. Defaults to '%Y-%m-%d %H:%M:%S'
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
class Calculator(BaseModel):
    """
    Perform a math operation on two numbers.
    """
    number_one: Any = Field(..., description="First number.")
    operation: MathOperation = Field(..., description="Math operation to perform.")
    number_two: Any = Field(..., description="Second number.")

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
        return json.dumps({"location": "London", "temperature": "42", "unit": unit.value})
    elif "New York" in location:
        return json.dumps({"location": "New York", "temperature": "24", "unit": unit.value})
    elif "North Pole" in location:
        return json.dumps({"location": "North Pole", "temperature": "-42", "unit": unit.value})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


# Here is a function definition in OpenAI style
tools = [
    {
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
                "required": ["location"],
            },
        },
    }
]
# To make the OpenAI function callable for the function calling agent we need a list with actual function in it:
tool_functions = [get_current_weather]


# Callback for receiving messages for the user.
def send_message_to_user_callback(message: str):
    print(message)


generation_settings = LlamaCppGenerationSettings(temperature=0.65, stream=True)

# Can be saved and loaded like that:
# generation_settings.save("generation_settings.json")
# generation_settings = LlamaLLMGenerationSettings.load_from_file("generation_settings.json")
main_model = LlamaCppEndpointSettings(
    "http://localhost:8080/completion"
)


function_call_agent = FunctionCallingAgent(
    # Can be lama-cpp-python Llama class, llama_cpp_agent.llm_settings.LlamaLLMSettings class or llama_cpp_agent.providers.llama_cpp_server_provider.LlamaCppServerLLMSettings.
    main_model,
    # llama_cpp_agent.llm_settings.LlamaLLMGenerationSettings  class or llama_cpp_agent.providers.llama_cpp_server_provider.LlamaCppServerGenerationSettings.
    llama_generation_settings=generation_settings,
    # A tuple of the OpenAI style function definitions and the actual functions
    open_ai_functions=(tools, tool_functions),
    # Just a list of type hinted functions for normal Python functions
    python_functions=[write_to_file, read_file, get_current_datetime],
    # Just a list of pydantic types
    pydantic_functions=[Calculator],
    # Callback for receiving messages for the user.
    send_message_to_user_callback=send_message_to_user_callback,
    # Set to true to allow parallel function calling
    allow_parallel_function_calling=True,
    debug_output=True)


user_input = 'Solve the following calculations: 42 * 42, 24 * 24, 5 * 5'
function_call_agent.generate_response(user_input)
function_call_agent.save("function_calling_agent.json")
