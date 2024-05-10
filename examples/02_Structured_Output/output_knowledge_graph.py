# Based on an example of the Instructor library for OpenAI


import json
from typing import List

from graphviz import Digraph
from pydantic import BaseModel, Field

from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import (
    generate_gbnf_grammar_and_documentation,
)
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.tgi_server import TGIServerProvider

provider = TGIServerProvider("http://localhost:8080")


class Node(BaseModel):
    id: int
    label: str
    color: str


class Edge(BaseModel):
    source: int
    target: int
    label: str
    color: str = "black"


class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(..., default_factory=list)
    edges: List[Edge] = Field(..., default_factory=list)

output_settings = LlmStructuredOutputSettings.from_pydantic_models([KnowledgeGraph], output_type=LlmStructuredOutputType.object_instance)

agent = LlamaCppAgent(
    provider,
    debug_output=True,
    system_prompt="You are an advanced AI assistant responding in JSON format.\n\nAvailable JSON response models:\n\n"
    + output_settings.get_llm_documentation(provider=provider),
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)


def visualize_knowledge_graph(kg):
    dot = Digraph(comment="Knowledge Graph")

    # Add nodes
    for node in kg.nodes:
        dot.node(str(node.id), node.label, color=node.color)

    # Add edges
    for edge in kg.edges:
        dot.edge(str(edge.source), str(edge.target), label=edge.label, color=edge.color)

    # Render the graph
    dot.render("knowledge_graph6.gv", view=True)


def generate_graph(user_input: str):
    prompt = f"""Help me understand the following by describing it as a extremely detailed knowledge graph with at least 20 nodes: {user_input}""".strip()
    response = agent.get_chat_response(
        message=prompt,
        structured_output_settings=output_settings
    )

    return response


graph = generate_graph("Teach me about quantum mechanics")
visualize_knowledge_graph(graph)
