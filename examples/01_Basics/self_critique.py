import json
from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from llama_cpp_agent import LlamaCppAgent, MessagesFormatterType
from llama_cpp_agent.chat_history import BasicChatHistory
from llama_cpp_agent.chat_history.messages import Roles
from llama_cpp_agent.providers import LlamaCppServerProvider


class ReflectionState(Enum):
    approved = "approved"
    rejected = "rejected"


class Reflection(BaseModel):
    analysis: str = Field(..., description="Analysis of the latest response")
    critique: str = Field(..., description="Critique of the latest response")
    things_to_improve: List[str] = Field(default_factory=list, description="List of things to improve")
    response_state: ReflectionState = Field(..., description="The decision if this response is approved or rejected")


class AgenticReflection:
    def __init__(self, llm_provider, generator_prompt: str = None, reflection_prompt: str = None):
        self.chat_history = BasicChatHistory(k=35)

        self.generator_agent = LlamaCppAgent(
            llm_provider,
            debug_output=True,
            system_prompt="You are a misinformed AI agent tha",
            predefined_messages_formatter_type=MessagesFormatterType.CHATML,
            chat_history=self.chat_history
        )

        self.reflection = LlamaCppAgent(
            llm_provider,
            system_prompt="Your task is to analyze, provide feedback and critique on an AI agent's latest response to a user in an ongoing conversation. You then decide if the latest response is approved or rejected.",
            debug_output=True,
            predefined_messages_formatter_type=MessagesFormatterType.CHATML
        )

    def get_response(self, input_message: str):
        approved = False
        while not approved:
            self.generator_agent.get_chat_response(input_message)
            messages = self.generator_agent.chat_history.get_chat_messages()
            ctx = ""
            for message in messages:
                ctx += f"{json.dumps(message, indent=2)}\n\n"

            self.reflection.get_chat_response(ctx)

provider = LlamaCppServerProvider("http://127.0.0.1:8080")
reflection = AgenticReflection(provider)

reflection.get_response("Write a summary about the independence war of america against england.")