# Example: Product Comparison and Recommendation
from llama_cpp_agent.chain import AgentChainElement, AgentChain
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

product_comparison = AgentChainElement(
    output_identifier="out_0",
    system_prompt="You are a product comparison expert",
    prompt="Compare the features and specifications of {product1} and {product2} in the {category} category."
)

product_recommendation = AgentChainElement(
    output_identifier="out_1",
    system_prompt="You are a product recommendation assistant",
    prompt="Based on the following product comparison, provide a recommendation on which product is better suited for {user_profile}:\n--\n{out_0}"
)

chain = [product_comparison, product_recommendation]
agent_chain = AgentChain(agent, chain)
agent_chain.run_chain(additional_fields={"product1": "iPhone 13", "product2": "Samsung Galaxy S22", "category": "Smartphones", "user_profile": "a professional photographer"})

