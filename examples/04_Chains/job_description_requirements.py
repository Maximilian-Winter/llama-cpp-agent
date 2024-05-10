# Example: Job Description and Candidate Requirements
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

job_description = AgentChainElement(
    output_identifier="out_0",
    system_prompt="You are a human resources assistant",
    prompt="Create a detailed job description for a {job_title} position in the {industry} industry. Include key responsibilities and qualifications."
)

candidate_requirements = AgentChainElement(
    output_identifier="out_1",
    system_prompt="You are a talent acquisition specialist",
    prompt="Based on the following job description, provide a list of essential candidate requirements and preferred skills:\n--\n{out_0}"
)

chain = [job_description, candidate_requirements]
agent_chain = AgentChain(agent, chain)
agent_chain.run_chain(additional_fields={"job_title": "Data Scientist", "industry": "Technology"})

