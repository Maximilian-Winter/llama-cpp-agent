# Example: Blog Post Outline and Introduction
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

blog_post_outline = AgentChainElement(
    output_identifier="out_0",
    system_prompt="You are a blog post outliner",
    prompt="Create an outline for a blog post about {topic}. Include main sections and subpoints."
)

blog_post_intro = AgentChainElement(
    output_identifier="out_1",
    system_prompt="You are a blog post writer",
    prompt="Write an engaging introduction for a blog post based on the following outline:\n--\n{out_0}"
)

chain = [blog_post_outline, blog_post_intro]
agent_chain = AgentChain(agent, chain)
agent_chain.run_chain(additional_fields={"topic": "The Benefits of Meditation"})

