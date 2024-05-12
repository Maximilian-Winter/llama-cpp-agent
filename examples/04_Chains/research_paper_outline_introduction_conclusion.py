# Example: Research Paper Outline, Introduction, and Conclusion
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

research_paper_outline = AgentChainElement(
    output_identifier="out_0",
    system_prompt="You are a research paper outliner",
    prompt="Create a detailed outline for a research paper on {topic}. Include main sections, subsections, and key points to cover."
)

research_paper_intro = AgentChainElement(
    output_identifier="out_1",
    system_prompt="You are a research paper introduction writer",
    prompt="Write an engaging introduction for a research paper based on the following outline:\n--\n{out_0}"
)

research_paper_conclusion = AgentChainElement(
    output_identifier="out_2",
    system_prompt="You are a research paper conclusion writer",
    prompt="Write a compelling conclusion for a research paper based on the following outline and introduction:\n--\nOutline:\n{out_0}\n\nIntroduction:\n{out_1}"
)

research_paper_abstract = AgentChainElement(
    output_identifier="out_3",
    system_prompt="You are a research paper abstract writer",
    prompt="Create a concise abstract for a research paper based on the following outline, introduction, and conclusion:\n--\nOutline:\n{out_0}\n\nIntroduction:\n{out_1}\n\nConclusion:\n{out_2}"
)

chain = [research_paper_outline, research_paper_intro, research_paper_conclusion, research_paper_abstract]
agent_chain = AgentChain(agent, chain)
agent_chain.run_chain(additional_fields={"topic": "The Impact of Social Media on Mental Health"})

