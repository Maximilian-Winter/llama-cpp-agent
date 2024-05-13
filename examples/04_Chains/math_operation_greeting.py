# Example: Math Operation with Greeting Generation.
from enum import Enum

from llama_cpp_agent import LlamaCppFunctionTool
from llama_cpp_agent import AgentChainElement, AgentChain
from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.providers import TGIServerProvider

model = TGIServerProvider("http://127.0.0.1:8080")

agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt="",
    predefined_messages_formatter_type=MessagesFormatterType.MISTRAL
)


class MathOps(str, Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


def math_operation(operation: MathOps, num1: float, num2: float) -> float:
    """
    Performs math operations on two numbers.

    Args:
        operation (MathOps): Math operation to perform
        num1 (float): first number
        num2 (float): second number
    Returns:
        float: result of math operation
    """
    if operation == MathOps.ADD:
        return num1 + num2
    elif operation == MathOps.SUBTRACT:
        return num1 - num2
    elif operation == MathOps.MULTIPLY:
        return num1 * num2
    elif operation == MathOps.DIVIDE:
        return num1 / num2


math_tool = LlamaCppFunctionTool(math_operation)


def postprocess_math_result(sys_prompt, prompt, outputs, response):
    return f"The result of the math operation is: {response}"


math_element = AgentChainElement(
    output_identifier="math_result",
    system_prompt="You are a math assistant that performs mathematical operations.",
    prompt="Perform the following math operation: {operation} {num1} and {num2}",
    tools=[math_tool],
    postprocessor=postprocess_math_result
)

greeting_element = AgentChainElement(
    output_identifier="greeting",
    system_prompt="You are a greeting assistant that generates personalized greetings.",
    prompt="Generate a personalized greeting for a person named {name} who just received the following math result: {math_result}"
)

chain = [math_element, greeting_element]

agent_chain = AgentChain(agent, chain)

output, _ = agent_chain.run_chain(additional_fields={
    "operation": "multiply",
    "num1": 5,
    "num2": 3,
    "name": "Alice"
})

print(output)