# Example that uses the FunctionCallingAgent class to create a function calling agent.

from enum import Enum
from typing import Union

from llama_cpp import Llama
from pydantic import BaseModel, Field

from llama_cpp_agent.llm_settings import LlamaLLMSettings, LlamaLLMGenerationSettings

from llama_cpp_agent.function_calling_agent import FunctionCallingAgent


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


# Enum for the calculator tool.
class MathOperation(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


# Simple pydantic calculator tool for the agent that can add, subtract, multiply, and divide. Docstring and description of fields will be used in system prompt.
class Calculator(BaseModel):
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


# Callback for receiving messages for the user.
def send_message_to_user_callback(message: str):
    print(message)

generation_settings = LlamaLLMGenerationSettings(temperature=0.65, top_p=0.5, tfs_z=0.975)

# Can be saved and loaded like that:
# generation_settings.save("generation_settings.json")
# generation_settings = LlamaLLMGenerationSettings.load_from_file("generation_settings.json")

function_call_agent = FunctionCallingAgent(LlamaLLMSettings.load_from_file("openhermes-2.5-mistral-7b.Q8_0.json"),  # Can lama-cpp-python Llama class or LlamaLLMSettings class.
                                           llama_generation_settings=generation_settings,
                                           python_functions=[write_to_file, read_file],
                                           pydantic_functions=[Calculator],
                                           send_message_to_user_callback=send_message_to_user_callback)

while True:
    user_input = input(">")
    function_call_agent.generate_response(user_input)
    function_call_agent.save("function_calling_agent.json")
