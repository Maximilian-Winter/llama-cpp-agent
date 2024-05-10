from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent

from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType
from llama_cpp_agent.providers.tgi_server import TGIServerProvider

model = TGIServerProvider("http://localhost:8080")

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
    keywords: List[str] = Field(..., description="A list of keywords.")
    category: Category = Field(..., description="Category of the book.")
    summary: str = Field(..., description="Summary of the book.")


output_settings = LlmStructuredOutputSettings.from_pydantic_models([Book], output_type=LlmStructuredOutputType.list_of_objects)

llama_cpp_agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt="You are an advanced AI, tasked to create JSON database entries for books.\n\n\n" + output_settings.get_llm_documentation(),
)

print(llama_cpp_agent.get_chat_response(text, structured_output_settings=output_settings))
