from typing import Union
import math

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import (
    create_dynamic_model_from_function,
)
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import (
    LlamaCppEndpointSettings,
)

model = LlamaCppEndpointSettings(
    completions_endpoint_url="http://127.0.0.1:8080/completion"
)


def calculate_a_to_the_power_b(a: Union[int, float], b: Union[int, float]):
    """
    Calculates a to the power of b

    Args:
        a: number
        b: exponent

    """
    return f"Result: {math.pow(a, b)}"


DynamicSampleModel = create_dynamic_model_from_function(calculate_a_to_the_power_b)
function_tools = [LlamaCppFunctionTool(DynamicSampleModel)]
function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

llama_cpp_agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt=f"You are an advanced AI, tasked to assist the user by calling functions in JSON format. The following are the available functions and their parameters and types:\n\n{function_tool_registry.get_documentation()}",
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

user_input = "a = 2, b = 3"

print(
    llama_cpp_agent.get_chat_response(
        user_input, temperature=0.45, function_tool_registry=function_tool_registry
    )
)
