import json
import math
from typing import Type, Union

from llama_cpp import Llama, LlamaGrammar

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import \
    generate_gbnf_grammar_and_documentation, create_dynamic_model_from_function
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings


def calculate_a_to_the_power_b(a: Union[int | float], b: Union[int | float]):
    """
    Calculates a to the power of b

    Args:
        a: number
        b: exponent

    """
    print(f"Result: {math.pow(a, b)}")


DynamicSampleModel = create_dynamic_model_from_function(calculate_a_to_the_power_b)

grammar, documentation = generate_gbnf_grammar_and_documentation([DynamicSampleModel], outer_object_name="function",
                                                                 outer_object_content="params")

main_model = LlamaCppEndpointSettings(
    completions_endpoint_url="http://127.0.0.1:8080/completion"
)

llama_cpp_agent = LlamaCppAgent(main_model, debug_output=True,
                                system_prompt="You are an advanced AI, tasked to generate JSON objects for function calling.\n\n" + documentation)

response = llama_cpp_agent.get_chat_response("a= 5, b = 42", temperature=0.15, grammar=grammar)

function_call = json.loads(response)

instance = DynamicSampleModel(**function_call['params'])
instance.run()
