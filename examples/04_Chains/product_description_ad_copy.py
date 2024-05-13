# Example: Product Description and Ad Copy
from llama_cpp_agent import AgentChainElement, AgentChain
from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppServerProvider

model = LlamaCppServerProvider("http://127.0.0.1:8080")

agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt="",
    predefined_messages_formatter_type=MessagesFormatterType.MISTRAL
)

product_description = AgentChainElement(
    output_identifier="out_0",
    system_prompt="You are a product description writer for an e-commerce website",
    prompt="Write a detailed product description for a {product_name} in the {category} category. Include features, benefits, and specifications."
)

ad_copy = AgentChainElement(
    output_identifier="out_1",
    system_prompt="You are an advertising copywriter",
    prompt="Create a short, engaging ad copy based on the following product description:\n--\n{out_0}"
)

chain = [product_description, ad_copy]
agent_chain = AgentChain(agent, chain)
agent_chain.run_chain(additional_fields={"product_name": "Bluetooth Wireless Headphones", "category": "Electronics"})

