# Example: Article Summary and Social Media Post
from llama_cpp_agent.chain import AgentChainElement, AgentChain
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_server import LlamaCppServerProvider

model = LlamaCppServerProvider("http://127.0.0.1:8080")

agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt="",
    predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL
)

article_summary = AgentChainElement(
    output_identifier="out_0",
    system_prompt="You are an article summarization assistant",
    prompt="Summarize the key points of the following article in 3-4 sentences:\n--\n{article_text}"
)

social_media_post = AgentChainElement(
    output_identifier="out_1",
    system_prompt="You are a social media manager",
    prompt="Create an engaging social media post based on the following article summary. Include relevant hashtags:\n--\n{out_0}"
)

chain = [article_summary, social_media_post]
agent_chain = AgentChain(agent, chain)
agent_chain.run_chain(additional_fields={"article_text": "..."})  # Replace "..." with the actual article text
