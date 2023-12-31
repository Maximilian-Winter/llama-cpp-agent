# Based on an example of the Instructor library for OpenAI


import json
from typing import List

from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import generate_gbnf_grammar_and_documentation

main_model = Llama(
    "../gguf-models/solarc-moe-10.7bx4.Q4_K_M.gguf",
    n_gpu_layers=23,
    f16_kv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    offload_kqv=True,
    last_n_tokens_size=1024,
    verbose=True,
    seed=42,
)

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




gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([KnowledgeGraph],False)

print(gbnf_grammar)
grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=True)


wrapped_model = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt="You are an advanced AI assistant, developed by OpenAI, responding in JSON format.\n\nAvailable JSON response models:\n\n" + documentation,
                              predefined_messages_formatter_type=MessagesFormatterType.SOLAR)


from graphviz import Digraph


def visualize_knowledge_graph(kg: KnowledgeGraph):
    dot = Digraph(comment="Knowledge Graph")

    # Add nodes
    for node in kg.nodes:
        dot.node(str(node.id), node.label, color=node.color)

    # Add edges
    for edge in kg.edges:
        dot.edge(str(edge.source), str(edge.target), label=edge.label, color=edge.color)

    # Render the graph
    dot.render("knowledge_graph4.gv", view=True)


def generate_graph(user_input: str) -> KnowledgeGraph:
    prompt = f'''Help me understand the following by describing it as a extremely detailed knowledge graph with at least 20 nodes: {user_input}'''.strip()
    response = wrapped_model.get_chat_response(message=prompt, temperature=0.25, mirostat_mode=0, mirostat_tau=4.0,
                                               mirostat_eta=0.1, grammar=grammar)
    knowledge_graph = json.loads(response)
    cls = KnowledgeGraph
    knowledge_graph = cls(**knowledge_graph)
    return knowledge_graph


graph = generate_graph("Quantum Mechanics")
visualize_knowledge_graph(graph)
