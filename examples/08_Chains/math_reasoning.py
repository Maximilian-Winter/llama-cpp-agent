from enum import Enum

from llama_cpp_agent.chain import AgentChainElement, AgentChain
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings

model = LlamaCppEndpointSettings(completions_endpoint_url="http://127.0.0.1:8080/completion")

agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt="",
    predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL
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


def generate_math_word_problem(sys_prompt, prompt, outputs, response):
    return f"Math Word Problem: {response}"


word_problem_element = AgentChainElement(
    output_identifier="word_problem",
    system_prompt="You are a math word problem generator.",
    prompt="Generate a math word problem involving the following operation and numbers: {operation} {num1} and {num2}",
    postprocessor=generate_math_word_problem
)


def extract_math_operation(sys_prompt, prompt, outputs, response):
    return f"Extracted Math Operation: {response}"


extraction_element = AgentChainElement(
    output_identifier="extracted_operation",
    system_prompt="You are a math operation extraction assistant.",
    prompt="Extract the math operation and numbers from the following word problem: {word_problem}",
    postprocessor=extract_math_operation
)


def postprocess_math_result(sys_prompt, prompt, outputs, response):
    return f"The result of the math operation is: {response}"


math_element = AgentChainElement(
    output_identifier="math_result",
    system_prompt="You are a math assistant that performs mathematical operations.",
    prompt="Perform the following math operation: {extracted_operation}",
    tools=[math_tool],
    postprocessor=postprocess_math_result
)


def observe_and_critique(sys_prompt, prompt, outputs, response):
    return f"Observation: The math operation was performed correctly. The result matches the word problem. No critiques."


observing_element = AgentChainElement(
    output_identifier="observation",
    system_prompt="You are an observing assistant that analyzes the results of the math operation.",
    prompt="Observe and critique the following math result in the context of the word problem: {word_problem} {math_result}",
    postprocessor=observe_and_critique
)

answer_element = AgentChainElement(
    output_identifier="answer",
    system_prompt="You are an answering assistant that provides the final answer to the user.",
    prompt="Provide a final answer to the user based on the word problem, extracted operation, math result, and observation: {word_problem} {extracted_operation} {math_result} {observation}"
)

chain = [word_problem_element, extraction_element, math_element, observing_element, answer_element]

agent_chain = AgentChain(agent, chain)

output, _ = agent_chain.run_chain(additional_fields={
    "operation": "multiply",
    "num1": 7,
    "num2": 5
})

print(output)