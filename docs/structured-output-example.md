### Structured Output
This example shows how to get structured output objects using the StructureOutputAgent class.
```python

# Example that uses the StructuredOutputAgent class to create a dataset entry of a book, out of unstructured data.

from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent import StructuredOutputAgent
from llama_cpp_agent.providers import TGIServerProvider

model = TGIServerProvider("http://localhost:8080")


# Example enum for our output model
class Category(Enum):
    Fiction = "Fiction"
    NonFiction = "Non-Fiction"


# Example output model
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


structured_output_agent = StructuredOutputAgent(
    model, debug_output=True,
    messages_formatter_type=MessagesFormatterType.CHATML
)

text = """The Feynman Lectures on Physics is a physics textbook based on some lectures by Richard Feynman, a Nobel laureate who has sometimes been called "The Great Explainer". The lectures were presented before undergraduate students at the California Institute of Technology (Caltech), during 1961–1963. The book's co-authors are Feynman, Robert B. Leighton, and Matthew Sands."""
print(structured_output_agent.create_object(Book, text))
```
Example output
```text
 { "title": "The Feynman Lectures on Physics"  ,  "author": "Richard Feynman, Robert B. Leighton, Matthew Sands"  ,  "published_year": 1963 ,  "keywords": [ "physics" , "textbook" , "Nobel laureate" , "The Great Explainer" , "California Institute of Technology" , "undergraduate" , "lectures"  ] ,  "category": "Non-Fiction" ,  "summary": "The Feynman Lectures on Physics is a physics textbook based on lectures by Nobel laureate Richard Feynman, known as 'The Great Explainer'. The lectures were presented to undergraduate students at Caltech between 1961 and 1963. Co-authors of the book are Feynman, Robert B. Leighton, and Matthew Sands."  }


title='The Feynman Lectures on Physics' author='Richard Feynman, Robert B. Leighton, Matthew Sands' published_year=1963 keywords=['physics', 'textbook', 'Nobel laureate', 'The Great Explainer', 'California Institute of Technology', 'undergraduate', 'lectures'] category=<Category.NonFiction: 'Non-Fiction'> summary="The Feynman Lectures on Physics is a physics textbook based on lectures by Nobel laureate Richard Feynman, known as 'The Great Explainer'. The lectures were presented to undergraduate students at Caltech between 1961 and 1963. Co-authors of the book are Feynman, Robert B. Leighton, and Matthew Sands."

```