# Import necessary libraries of pydantic and the llama-cpp-agent framework.
from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent

from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType
from llama_cpp_agent.providers.tgi_server import TGIServerProvider

# Create the provider.
provider = TGIServerProvider("http://localhost:8080")


# An enum for the book category
class Category(Enum):
    """
    The category of the book.
    """

    Fiction = "Fiction"
    NonFiction = "Non-Fiction"


# The class representing the database entry we want to generate.
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


# We create an instance of the LlmStructuredOutputSettings class by calling its from_pydantic_models method and specify the output type.
output_settings = LlmStructuredOutputSettings.from_pydantic_models([Book],
                                                                   output_type=LlmStructuredOutputType.list_of_objects)

# We are creating the agent with a custom system prompt including information about the dataset entry and its structure.
llama_cpp_agent = LlamaCppAgent(
    provider,
    system_prompt="You are an advanced AI, tasked to create JSON database entries for books.\n\n\n" + output_settings.get_llm_documentation(
        provider),
)

# We define the input information for the agent.
text = """The Feynman Lectures on Physics is a physics textbook based on some lectures by Richard Feynman, a Nobel laureate who has sometimes been called "The Great Explainer". The lectures were presented before undergraduate students at the California Institute of Technology (Caltech), during 1961–1963. The book's co-authors are Feynman, Robert B. Leighton, and Matthew Sands."""

# We call get_chat_response with output_settings. This will return an instance of the dataset entry class 'Book'.
book_dataset_entry = llama_cpp_agent.get_chat_response(text, structured_output_settings=output_settings)
print(book_dataset_entry)
