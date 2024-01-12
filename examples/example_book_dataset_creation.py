from enum import Enum

from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import generate_gbnf_grammar_and_documentation
from llama_cpp_agent.providers.llama_cpp_server_provider import LlamaCppServerLLMSettings

main_model = LlamaCppServerLLMSettings(
    completions_endpoint_url="http://127.0.0.1:8080/completion"
)


text = """The Feynman Lectures on Physics is a physics textbook based on some lectures by Richard Feynman, a Nobel laureate who has sometimes been called "The Great Explainer". The lectures were presented before undergraduate students at the California Institute of Technology (Caltech), during 1961–1963. The book's co-authors are Feynman, Robert B. Leighton, and Matthew Sands."""


class Category(Enum):
    """
    The category of the book.
    """
    Fiction = "Fiction"
    NonFiction = "Non-Fiction"


class Book(BaseModel):
    """
    Represents an entry about a book.
    """
    title: str = Field(..., description="Title of the book.")
    author: str = Field(..., description="Author of the book.")
    published_year: int = Field(..., description="Publishing year of the book.")
    keywords: list[str] = Field(..., description="A list of keywords.")
    category: Category = Field(..., description="Category of the book.")
    summary: str = Field(..., description="Summary of the book.")


gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([Book])

llama_cpp_agent = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt="You are an advanced AI, tasked to create JSON database entries for books.\n\n\n" + documentation)


print(llama_cpp_agent.get_chat_response(text, temperature=0.15, grammar=gbnf_grammar))

