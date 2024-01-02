from enum import Enum

from llama_cpp import Llama
from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent

from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_call_tools import LlamaCppFunctionTool


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
    number_one: float = Field(..., description="First number.", max_precision=5, min_precision=2)
    operation: MathOperation = Field(..., description="Math operation to perform.")
    number_two: float = Field(..., description="Second number.", max_precision=5, min_precision=2)

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

function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

main_model = Llama(
    "../gguf-models/dolphin-2.6-mistral-7b-Q8_0.gguf",
    n_gpu_layers=35,
    f16_kv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=False,
    seed=42,
)
llama_cpp_agent = LlamaCppAgent(main_model, debug_output=False,
                                system_prompt="You are an advanced AI, tasked to assist the user by calling functions in JSON format.\n\n\n" + function_tool_registry.get_documentation(),
                                predefined_messages_formatter_type=MessagesFormatterType.CHATML)
user_input = 'What is 42 * 42?'
print(llama_cpp_agent.get_chat_response(user_input, temperature=0.45, function_tool_registry=function_tool_registry))
