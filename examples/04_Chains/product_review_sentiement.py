# Example: Product Review Summary and Sentiment Analysis
from llama_cpp_agent import AgentChainElement, AgentChain
from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppServerProvider

model = LlamaCppServerProvider("http://127.0.0.1:8080")

agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt="",
    predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL
)

review_summary = AgentChainElement(
    output_identifier="out_0",
    system_prompt="You are a product review summarization assistant",
    prompt="Summarize the main points and opinions expressed in the following customer reviews:\n--\n{review_text}"
)

sentiment_analysis = AgentChainElement(
    output_identifier="out_1",
    system_prompt="You are a sentiment analysis expert",
    prompt="Analyze the overall sentiment (positive, negative, or neutral) of the following review summary:\n--\n{out_0}"
)

chain = [review_summary, sentiment_analysis]
agent_chain = AgentChain(agent, chain)
agent_chain.run_chain(additional_fields={"review_text": "..."})  # Replace "..." with the actual review text

