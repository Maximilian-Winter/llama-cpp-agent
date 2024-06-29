from typing import List, Any
from .llm_agent import LlamaCppAgent


class MixtureOfAgents:
    def __init__(self, agents: List[LlamaCppAgent], final_agent: LlamaCppAgent):
        self.agents = agents
        self.final_agent = final_agent

    def get_response(self, input_message: str, **kwargs) -> Any:
        # Collect responses from all agents
        agent_responses = []
        for i, agent in enumerate(self.agents):
            response = agent.get_chat_response(message=input_message, **kwargs)
            agent_responses.append(f"Agent {i + 1} response: {response}")

        # Combine all responses into a single message for the final agent
        combined_responses = "\n\n".join(agent_responses)
        final_prompt = f"""You are a meta-agent tasked with analyzing and synthesizing responses from multiple AI agents to produce a final, comprehensive answer. 

Here are the responses from various agents to the following input: "{input_message}"

{combined_responses}

Please analyze these responses, identify key insights, reconcile any contradictions, and compose a final answer that incorporates the best elements from each response while adding your own insights. Your goal is to provide the most accurate, comprehensive, and useful response possible.

Your final answer:"""

        # Get the final response from the final agent
        final_response = self.final_agent.get_chat_response(message=final_prompt, prompt_suffix="\nMy final answer:", **kwargs)

        return final_response

    def add_agent(self, agent: LlamaCppAgent):
        self.agents.append(agent)

    def remove_agent(self, index: int):
        if 0 <= index < len(self.agents):
            del self.agents[index]
        else:
            raise IndexError("Agent index out of range")

    def set_final_agent(self, agent: LlamaCppAgent):
        self.final_agent = agent
