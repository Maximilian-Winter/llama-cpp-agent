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
from llama_cpp_agent.providers.llama_cpp_server import LlamaCppServerProvider

model = LlamaCppServerProvider("http://127.0.0.1:8080")


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


gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(
    [KnowledgeGraph],
    model_prefix="Response Model",
    fields_prefix="Response Model Field",
)

output_settings = LlmStructuredOutputSettings.from_pydantic_models([KnowledgeGraph], output_type=LlmStructuredOutputType.object_instance)

agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt="You are an advanced AI assistant responding in JSON format.\n\nAvailable JSON response models:\n\n"
    + documentation,
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)


def visualize_knowledge_graph(kg: KnowledgeGraph):
    dot = Digraph(comment="Knowledge Graph")

    # Add nodes
    for node in kg.nodes:
        dot.node(str(node.id), node.label, color=node.color)

    # Add edges
    for edge in kg.edges:
        dot.edge(str(edge.source), str(edge.target), label=edge.label, color=edge.color)

    # Render the graph
    dot.render("knowledge_graph6.gv", view=True)


def generate_graph(user_input: str) -> KnowledgeGraph:
    prompt = f"""Help me understand the following by describing it as a extremely detailed knowledge graph with at least 20 nodes: {user_input}""".strip()
    response = agent.get_chat_response(
        message=prompt,
        structured_output_settings=output_settings
    )
    knowledge_graph = json.loads(response)
    cls = KnowledgeGraph
    knowledge_graph = cls(**knowledge_graph)
    return knowledge_graph


graph = generate_graph("Teach me about quantum mechanics")
visualize_knowledge_graph(graph)
