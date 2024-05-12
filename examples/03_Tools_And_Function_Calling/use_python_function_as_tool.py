from typing import Union
import math

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputType, LlmStructuredOutputSettings
from llama_cpp_agent.messages_formatter import MessagesFormatterType

from llama_cpp_agent.providers.tgi_server import TGIServerProvider

provider = TGIServerProvider("http://localhost:8080")


def calculate_a_to_the_power_b(a: Union[int, float], b: Union[int, float]):
    """
    Calculates a to the power of b

    Args:
        a: number
        b: exponent

    """
    return f"Result: {math.pow(a, b)}"


output_settings = LlmStructuredOutputSettings.from_functions([calculate_a_to_the_power_b], allow_parallel_function_calling=True)
llama_cpp_agent = LlamaCppAgent(
    provider,
    debug_output=True,
    system_prompt=f"You are an advanced AI, tasked to assist the user by calling functions in JSON format.",
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

user_input = "Calculate a to the power of b: a = 2, b = 3"

print(
    llama_cpp_agent.get_chat_response(
        user_input, structured_output_settings=output_settings
    )
)
