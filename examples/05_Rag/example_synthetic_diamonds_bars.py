import json

from ragatouille.utils import get_wikipedia_page

from llama_cpp_agent.messages_formatter import MessagesFormatterType

from typing import List

from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import (
    generate_gbnf_grammar_and_documentation,
)

from llama_cpp_agent.rag.rag_colbert_reranker import RAGColbertReranker
from llama_cpp_agent.rag.text_utils import RecursiveCharacterTextSplitter


# Initialize the chromadb vector database with a colbert reranker.
rag = RAGColbertReranker(persistent=False)

# Initialize a recursive character text splitter with the correct chunk size of the embedding model.
length_function = len
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=512,
    chunk_overlap=0,
    length_function=length_function,
    keep_separator=True
)

# Use the ragatouille helper function to get the content of a wikipedia page.
page = get_wikipedia_page("Synthetic_diamond")

# Split the text of the wikipedia page into chunks for the vector database.
splits = splitter.split_text(page)

# Add the splits into the vector database
for split in splits:
    rag.add_document(split)

# Define a llamacpp server endpoint.
from llama_cpp_agent.providers.llama_cpp_server import LlamaCppServerProvider

model = LlamaCppServerProvider("http://127.0.0.1:8080")

# Define a test agent to see the answer without retrieved information.
agent_without_rag_information = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt="You are an advanced AI assistant, trained by OpenAI.",
    predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL
)

# Define the query we want to ask based on the retrieved information
query = "What is a BARS apparatus?"

# Ask the query without retrieved information.
agent_without_rag_information.get_chat_response(query)


# Define a pydantic class to represent a query extension as additional queries to the original query.
class QueryExtension(BaseModel):
    """
    Represents an extension of a query as additional queries.
    """
    queries: List[str] = Field(default_factory=list, description="List of queries.")


# Generate a grammar and documentation of the query extension model.
grammar, docs = generate_gbnf_grammar_and_documentation([QueryExtension])

# Define a query extension agent which will extend the query with additional queries.
query_extension_agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt="You are a world class query extension algorithm capable of extending queries by writing new queries. Do not answer the queries, simply provide a list of additional queries in JSON format. Structure your output according to the following model:\n\n" + docs.strip(),
    predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL
)

# Perform the query extension with the agent.
output = query_extension_agent.get_chat_response(
    f"Consider the following query: {query}", grammar=grammar)

# Load the query extension in JSON format and create an instance of the query extension model.
queries = QueryExtension.model_validate(json.loads(output))

# Define the final prompt for the query with the retrieved information
prompt = "Consider the following context:\n==========Context===========\n"

# Retrieve the most fitting document chunks based on the original query and add them to the prompt.
documents = rag.retrieve_documents(query, k=3)
for doc in documents:
    prompt += doc["content"] + "\n\n"

# Retrieve the most fitting document chunks based on the extended queries and add them to the prompt.
for qu in queries.queries:
    documents = rag.retrieve_documents(qu, k=3)
    for doc in documents:
        if doc["content"] not in prompt:
            prompt += doc["content"] + "\n\n"
prompt += "\n======================\nQuestion: " + query

# Define a new agent to answer the original query based on the retrieved information.
agent_with_rag_information = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt="You are an advanced AI assistant, trained by OpenAI. Only answer question based on the context information provided.",
    predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL
)

# Ask the agent the original query with the generated prompt that contains the retrieved information.
agent_with_rag_information.get_chat_response(prompt)
