# llama-cpp-agent

<img src="https://github.com/Maximilian-Winter/llama-cpp-agent/blob/db41b3184ebc902f50edbd3d27f7a3a1128b7d7d/logo/logo_orange.png" alt="llama-cpp-agent logo" width="400"/>

## Introduction
The llama-cpp-agent framework is a tool designed for easy interaction with Large Language Models (LLMs). Allowing users to chat with LLM models, execute structured function calls, get structured output (objects), do retrieval augmented generation, and process text in agentic chains with tools.

It provides a simple yet robust interface and supports llama-cpp-python and OpenAI endpoints with GBNF grammar support (like the llama-cpp-python server) and the llama.cpp backend server.
It works by generating a formal GGML-BNF grammar of the user defined structures and functions, which is then used by llama.cpp to generate text valid to that grammar. In contrast to most GBNF grammar generators it also supports nested objects, dictionaries, enums and lists of them.

## Key Features
- **Simple Chat Interface**: Engage in seamless conversations with LLMs.
- **Structured Output**: Get structured output (objects) from LLMs.
- **Single and Parallel Function Calling**: Let the LLM execute functions.
- **RAG - Retrieval Augmented Generation**: Perform retrieval augmented generation with colbert reranking.
- **Agent Chains**: Supports processing of text in agent chains with tools. Supports Conversational, Sequential and Mapping Chains.
- **Flexibility**: Suited for various applications from casual chatting to specific function executions.

## Installation
The llama-cpp-agent framework can be installed using pip:
```shell
pip install llama-cpp-agent
```

## Usage Examples
The following examples demonstrate the usage of the llama-cpp-agent framework.
You can find more examples in the `examples` folder.


### Simple Chat Example using llama.cpp server backend
This example demonstrates how to initiate a chat with an LLM model using the llama.cpp server backend. The framework supports llama-cpp-python Llama class instances as LLM and OpenAI endpoints that support GBNF grammars as a backend, and the llama.cpp backend server.

```python
from llama_cpp import Llama
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings

main_model = LlamaCppEndpointSettings(
    completions_endpoint_url="http://127.0.0.1:8080/completion"
)

# For OpenAI endpoints that support GBNF grammars like the llama-cpp-python server:
# from llama_cpp_agent.providers.openai_endpoint_provider import OpenAIEndpointSettings
# main_model = OpenAIEndpointSettings(
#    "http://localhost:8080/v1/completions"
# )

# For instances of Llama class of llama-cpp-python
# from llama_cpp import Llama
# main_model = Llama(
#     "../../gguf-models/openhermes-2.5-mistral-7b.Q8_0.gguf",
#     n_gpu_layers=35,
#     f16_kv=True,
#     use_mlock=False,
#     embedding=False,
#     n_threads=8,
#     n_batch=1024,
#     n_ctx=8192,
#     offload_kqv=True,
#     last_n_tokens_size=1024,
#     verbose=True,
#     seed=-1,
# )
llama_cpp_agent = LlamaCppAgent(main_model, debug_output=False,
                                system_prompt="You are Dolphin, an uncensored and unbiased AI assistant.",
                                predefined_messages_formatter_type=MessagesFormatterType.CHATML)

while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    user_input = llama_cpp_agent.get_chat_response(user_input, temperature=0.7)
    print("AI: " + user_input)

```

### Parallel Function Calling Agent Example

```python
# Example that uses the FunctionCallingAgent class to create a function calling agent.
import datetime
from enum import Enum
from typing import Union, Optional

from llama_cpp import Llama
from pydantic import BaseModel, Field

from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.function_calling_agent import FunctionCallingAgent
from llama_cpp_agent.llm_settings import LlamaLLMGenerationSettings
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings, LlamaCppGenerationSettings


def get_current_datetime(output_format: Optional[str] = None):
    """
    Get the current date and time in the given format.

    Args:
         output_format: formatting string for the date and time, defaults to '%Y-%m-%d %H:%M:%S'
    """
    if output_format is None:
        output_format = '%Y-%m-%d %H:%M:%S'
    return datetime.datetime.now().strftime(output_format)


# Enum for the calculator tool.
class MathOperation(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


# llama-cpp-agent also supports "Instructor" library like function definitions as Pydantic models for function calling.
# Simple pydantic calculator tool for the agent that can add, subtract, multiply, and divide. Docstring and description of fields will be used in the system prompt.
class calculator(BaseModel):
    """
    Perform a math operation on two numbers.
    """
    number_one: Union[int, float] = Field(..., description="First number.")
    operation: MathOperation = Field(..., description="Math operation to perform.")
    number_two: Union[int, float] = Field(..., description="Second number.")

    def run(self):
        if self.operation == MathOperation.ADD:
            return self.number_one + self.number_two
        elif self.operation == MathOperation.SUBTRACT:
            return self.number_one - self.number_two
        elif self.operation == MathOperation.MULTIPLY:
            return self.number_one * self.number_two
        elif self.operation == MathOperation.DIVIDE:
            return self.number_one / self.number_two
        else:
            raise ValueError("Unknown operation.")


# Example function based on an OpenAI example.
# llama-cpp-agent also supports OpenAI like dictionaries for function definition.
def get_current_weather(location, unit):
    """Get the current weather in a given location"""
    if "London" in location:
        return f"Weather in {location}: {22}° {unit.value}"
    elif "New York" in location:
        return f"Weather in {location}: {24}° {unit.value}"
    elif "North Pole" in location:
        return f"Weather in {location}: {-42}° {unit.value}"
    else:
        return f"Weather in {location}: unknown"


# Here is a function definition in OpenAI style
open_ai_tool_spec = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}


# Callback for receiving messages for the user.
def send_message_to_user_callback(message: str):
    print(message)


path_to_model = "../../../gguf-models/mistral-7b-instruct-v0.2.Q6_K.gguf"

model = Llama(
    path_to_model,
    n_gpu_layers=49,
    f16_kv=True,
    offload_kqv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=4096,
    last_n_tokens_size=1024,
    verbose=True,
    seed=-1,
)
generation_settings = LlamaLLMGenerationSettings(
    temperature=0.4, top_k=0, top_p=1.0, repeat_penalty=1.1,
    min_p=0.1, tfs_z=0.95, stream=True)
# Can be saved and loaded like that:
# generation_settings.save("generation_settings.json")
# generation_settings = LlamaLLMGenerationSettings.load_from_file("generation_settings.json")

# To make the function tools available to our agent, we have to create a list of LlamaCppFunctionTool instances.

# First we create the calculator tool.
calculator_function_tool = LlamaCppFunctionTool(calculator)

# Next we create the current datetime tool.
current_datetime_function_tool = LlamaCppFunctionTool(get_current_datetime)

# For OpenAI tool specifications, we pass the specification with actual function in a tuple to the LlamaCppFunctionTool constructor.
get_weather_function_tool = LlamaCppFunctionTool((open_ai_tool_spec, get_current_weather))


function_call_agent = FunctionCallingAgent(
    # Can be lama-cpp-python Llama class, llama_cpp_agent.llm_settings.LlamaLLMSettings class or llama_cpp_agent.providers.llama_cpp_server_provider.LlamaCppServerLLMSettings.
    model,
    # llama_cpp_agent.llm_settings.LlamaLLMGenerationSettings  class or llama_cpp_agent.providers.llama_cpp_server_provider.LlamaCppServerGenerationSettings.
    llama_generation_settings=generation_settings,
    # Pass the LlamaCppFunctionTool instances as a list to the agent.
    llama_cpp_function_tools=[calculator_function_tool, current_datetime_function_tool, get_weather_function_tool],
    # Callback for receiving messages for the user.
    send_message_to_user_callback=send_message_to_user_callback,
    # Set to true to allow parallel function calling
    allow_parallel_function_calling=True,
    messages_formatter_type=MessagesFormatterType.CHATML,
    debug_output=True)

user_input = '''Get the date and time in '%d-%m-%Y %H:%M' format. Get the current weather in celsius in London, New York and at the North Pole. Solve the following calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6  and 96/8.'''
function_call_agent.generate_response(user_input)


```
Example Output:
```text
The current date and time is 10-04-2024 07:58. The weather in London is 22 degrees Celsius, in New York it's 24 degrees Celsius, and at the North Pole it's -42 degrees Celsius. The calculations are as follows:

- 42 * 42 = 1764
- 74 + 26 = 100
- 7 * 26 = 182
- 4 + 6 = 10
- 96 / 8 = 12
```

### Structured Output
This example shows how to get structured output objects using the StructureOutputAgent class.
```python
# Example agent that uses the StructuredOutputAgent class to create a dataset entry of a book out of unstructured data.

from enum import Enum

from llama_cpp import Llama
from pydantic import BaseModel, Field

from llama_cpp_agent.structured_output_agent import StructuredOutputAgent


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
    keywords: list[str] = Field(..., description="A list of keywords.")
    category: Category = Field(..., description="Category of the book.")
    summary: str = Field(..., description="Summary of the book.")


main_model = Llama(
    "../gguf-models/nous-hermes-2-solar-10.7b.Q6_K.gguf",
    n_gpu_layers=49,
    offload_kqv=True,
    f16_kv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=4096,
    last_n_tokens_size=1024,
    verbose=False,
    seed=42,
)

structured_output_agent = StructuredOutputAgent(main_model, debug_output=True)

text = """The Feynman Lectures on Physics is a physics textbook based on some lectures by Richard Feynman, a Nobel laureate who has sometimes been called "The Great Explainer". The lectures were presented before undergraduate students at the California Institute of Technology (Caltech), during 1961–1963. The book's co-authors are Feynman, Robert B. Leighton, and Matthew Sands."""
print(structured_output_agent.create_object(Book, text))
```
Example output
```text
 { "title": "The Feynman Lectures on Physics"  ,  "author": "Richard Feynman, Robert B. Leighton, Matthew Sands"  ,  "published_year": 1963 ,  "keywords": [ "physics" , "textbook" , "Nobel laureate" , "The Great Explainer" , "California Institute of Technology" , "undergraduate" , "lectures"  ] ,  "category": "Non-Fiction" ,  "summary": "The Feynman Lectures on Physics is a physics textbook based on lectures by Nobel laureate Richard Feynman, known as 'The Great Explainer'. The lectures were presented to undergraduate students at Caltech between 1961 and 1963. Co-authors of the book are Feynman, Robert B. Leighton, and Matthew Sands."  }


title='The Feynman Lectures on Physics' author='Richard Feynman, Robert B. Leighton, Matthew Sands' published_year=1963 keywords=['physics', 'textbook', 'Nobel laureate', 'The Great Explainer', 'California Institute of Technology', 'undergraduate', 'lectures'] category=<Category.NonFiction: 'Non-Fiction'> summary="The Feynman Lectures on Physics is a physics textbook based on lectures by Nobel laureate Richard Feynman, known as 'The Great Explainer'. The lectures were presented to undergraduate students at Caltech between 1961 and 1963. Co-authors of the book are Feynman, Robert B. Leighton, and Matthew Sands."

```

### RAG - Retrieval Augmented Generation
This example shows how to do RAG with colbert reranking. You have to install the optional rag dependencies (ragatouille) to use the RAGColbertReranker class and this example. 
```python
import json

from ragatouille.utils import get_wikipedia_page

from llama_cpp_agent.messages_formatter import MessagesFormatterType

from typing import List

from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import (
    generate_gbnf_grammar_and_documentation,
)
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import (
    LlamaCppEndpointSettings,
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

# Define the query we want to ask based on the retrieved information
query = "What is a BARS apparatus?"

# Define a pydantic class to represent a query extension as additional queries to the original query.
class QueryExtension(BaseModel):
    """
    Represents an extension of a query as additional queries.
    """
    queries: List[str] = Field(default_factory=list, description="List of queries.")


# Generate a grammar and documentation of the query extension model.
grammar, docs = generate_gbnf_grammar_and_documentation([QueryExtension])

# Define a llamacpp server endpoint.
main_model = LlamaCppEndpointSettings(completions_endpoint_url="http://127.0.0.1:8080/completion")

# Define a query extension agent which will extend the query with additional queries.
query_extension_agent = LlamaCppAgent(
    main_model,
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
    main_model,
    debug_output=True,
    system_prompt="You are an advanced AI assistant, trained by OpenAI. Only answer question based on the context information provided.",
    predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL
)

# Ask the agent the original query with the generated prompt that contains the retrieved information.
agent_with_rag_information.get_chat_response(prompt)

```
Example output
```text
 BARS (Bridgman-Anvil High Pressure Reactor System) apparatus is a type of diamond-producing press used in the HPHT (High Pressure High Temperature) method for synthetic diamond growth. It consists of a ceramic cylindrical "synthesis capsule" placed in a cube of pressure-transmitting material, which is pressed by inner anvils and outer anvils. The whole assembly is locked in a disc-type barrel filled with oil, which pressurizes upon heating, and the oil pressure is transferred to the central cell. The BARS apparatus is claimed to be the most compact, efficient, and economical press design for diamond synthesis.
```

### Sequential Chain Example
This example demonstrates how to create a complete product launch campaign with help of a sequential chain.
```python
# Example: Product Launch Campaign (Product Description, USP, Target Audience, Marketing Channels, Ad Copy, Landing Page, Email Campaign, Social Media Posts, Press Release, and Performance Metrics)
from llama_cpp_agent.chain import AgentChainElement, AgentChain
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings

model = LlamaCppEndpointSettings(completions_endpoint_url="http://127.0.0.1:8080/completion")

agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt="",
    predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL
)

product_description = AgentChainElement(
    output_identifier="out_0",
    system_prompt="You are a product description writer",
    prompt="Write a detailed product description for {product_name}, including its features and benefits."
)

product_usp = AgentChainElement(
    output_identifier="out_1",
    system_prompt="You are a unique selling proposition (USP) creator",
    prompt="Create a compelling USP for {product_name} based on the following product description:\n--\n{out_0}"
)

target_audience = AgentChainElement(
    output_identifier="out_2",
    system_prompt="You are a target audience identifier",
    prompt="Identify the target audience for {product_name} based on the following product description and USP:\n--\nProduct Description:\n{out_0}\nUSP:\n{out_1}"
)

marketing_channels = AgentChainElement(
    output_identifier="out_3",
    system_prompt="You are a marketing channel strategist",
    prompt="Suggest the most effective marketing channels to promote {product_name} based on the following target audience:\n--\n{out_2}"
)

ad_copy = AgentChainElement(
    output_identifier="out_4",
    system_prompt="You are an advertising copywriter",
    prompt="Write engaging ad copy for {product_name} based on the following product description, USP, and target audience:\n--\nProduct Description:\n{out_0}\nUSP:\n{out_1}\nTarget Audience:\n{out_2}"
)

landing_page = AgentChainElement(
    output_identifier="out_5",
    system_prompt="You are a landing page designer",
    prompt="Create a high-converting landing page structure for {product_name} based on the following product description, USP, target audience, and ad copy:\n--\nProduct Description:\n{out_0}\nUSP:\n{out_1}\nTarget Audience:\n{out_2}\nAd Copy:\n{out_4}"
)

email_campaign = AgentChainElement(
    output_identifier="out_6",
    system_prompt="You are an email marketing specialist",
    prompt="Develop an email campaign for {product_name} based on the following product description, USP, target audience, and landing page structure:\n--\nProduct Description:\n{out_0}\nUSP:\n{out_1}\nTarget Audience:\n{out_2}\nLanding Page Structure:\n{out_5}"
)

social_media_posts = AgentChainElement(
    output_identifier="out_7",
    system_prompt="You are a social media content creator",
    prompt="Create a series of engaging social media posts for {product_name} based on the following product description, USP, target audience, and ad copy:\n--\nProduct Description:\n{out_0}\nUSP:\n{out_1}\nTarget Audience:\n{out_2}\nAd Copy:\n{out_4}"
)

press_release = AgentChainElement(
    output_identifier="out_8",
    system_prompt="You are a press release writer",
    prompt="Write a compelling press release announcing the launch of {product_name} based on the following product description, USP, and target audience:\n--\nProduct Description:\n{out_0}\nUSP:\n{out_1}\nTarget Audience:\n{out_2}"
)

performance_metrics = AgentChainElement(
    output_identifier="out_9",
    system_prompt="You are a marketing performance analyst",
    prompt="Identify the key performance metrics to track the success of the {product_name} launch campaign based on the following marketing channels, ad copy, landing page, email campaign, social media posts, and press release:\n--\nMarketing Channels:\n{out_3}\nAd Copy:\n{out_4}\nLanding Page Structure:\n{out_5}\nEmail Campaign:\n{out_6}\nSocial Media Posts:\n{out_7}\nPress Release:\n{out_8}"
)

chain = [product_description, product_usp, target_audience, marketing_channels, ad_copy, landing_page, email_campaign, social_media_posts, press_release, performance_metrics]
agent_chain = AgentChain(agent, chain)
agent_chain.run_chain(additional_fields={"product_name": "Smart Fitness Tracker"})


```

### Mapping Chain Example
This example demonstrates how to create a mapping chain to summarize 3 articles into one summary.
```python
from llama_cpp_agent.chain import AgentChainElement, MapChain
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings

model = LlamaCppEndpointSettings(completions_endpoint_url="http://127.0.0.1:8080/completion")

agent = LlamaCppAgent(
    model,
    debug_output=True,
    system_prompt="",
    predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL
)

summary_chain = AgentChainElement("out_0", system_prompt="You are an advanced AI agent for summarizing articles", prompt="Summarize this article into bullet points:\n{item}")

combine_chain = AgentChainElement("out_1", system_prompt="You are an advanced AI agent that summarizes text", prompt="Please combine the bullet points of different summaries below, into one summary as bullet points:\n{map_output}")

map_chain = MapChain(agent, [summary_chain], [combine_chain])


article_list = [
    """### 1. Quantum Computing: The Next Frontier in Computational Power

**Introduction**
Quantum computing represents a revolutionary approach to information processing, leveraging the principles of quantum mechanics to solve problems that are intractable for classical computers. This article explores the fundamental concepts of quantum computing, its potential applications, and the challenges it faces.

**Quantum Mechanics and Computing**
Quantum computers use quantum bits, or qubits, which can exist in multiple states simultaneously, thanks to superposition. This capability, combined with entanglement—where the state of one qubit can depend on the state of another, no matter the distance between them—allows quantum computers to process a vast number of possibilities concurrently.

**Quantum Algorithms**
Several algorithms have been developed for quantum computers that show significant speed-ups over their classical counterparts. Shor’s Algorithm, for instance, can factorize large integers exponentially faster than the best-known classical algorithms, which has profound implications for cryptography. Grover's Algorithm offers a quadratic speedup for unstructured search problems.

**Applications**
Quantum computing has potential applications across various fields:
- **Cryptography**: Secure communication through quantum key distribution.
- **Drug Discovery**: Modeling molecular interactions at quantum levels to predict drug efficacy and side effects.
- **Optimization Problems**: Enhancing solutions in logistics, finance, and materials science.

**Challenges**
Despite its potential, quantum computing faces several hurdles:
- **Qubit Coherence**: Maintaining the state of qubits for sufficient time is challenging due to decoherence.
- **Error Rates**: Quantum gates are prone to errors significantly higher than conventional binary computing gates.
- **Scalability**: Building machines with enough qubits to be useful for complex problems is currently beyond our reach.

**Conclusion**
Quantum computing is still in its infancy, but it holds the promise of massive computational power. The coming decades are likely to see significant advancements in this field as researchers overcome its current limitations.""",
    """### 2. Machine Learning: Transforming Data into Insights

**Introduction**
Machine Learning (ML) is a branch of artificial intelligence that focuses on building applications that can learn from data and improve their accuracy over time without being explicitly programmed. This article delves into the types of ML, key algorithms, applications, and future prospects.

**Types of Machine Learning**
- **Supervised Learning**: Models predict outputs based on input data, and learning is guided by comparing actual and predicted outputs.
- **Unsupervised Learning**: Algorithms identify patterns in data without reference to known or labeled outcomes.
- **Reinforcement Learning**: Models learn to make sequences of decisions by receiving rewards or penalties.

**Key Algorithms**
- **Linear Regression** and **Logistic Regression** for predictive modeling.
- **Decision Trees** and **Random Forests** for classification and regression tasks.
- **Neural Networks**: Used in deep learning for complex pattern recognition, such as in image and speech recognition.

**Applications**
- **Healthcare**: From diagnosing diseases to personalized medicine.
- **Finance**: For credit scoring, algorithmic trading, and risk assessment.
- **Retail**: Enhancing customer experience through personalized recommendations.

**Challenges and Future Prospects**
The field of ML is not without challenges, including data privacy concerns, the need for large labeled datasets, and the risk of creating biased models. However, ongoing research in areas like unsupervised learning, transfer learning, and the development of more robust models promises to mitigate these issues.

**Conclusion**
Machine learning continues to be a dynamic field of research and application, with the potential to impact numerous sectors profoundly.""",
    """### 3. Blockchain Technology: Decentralizing Trust

**Introduction**
Blockchain technology is best known as the backbone of cryptocurrencies like Bitcoin, but its applications extend far beyond. This article outlines the technology’s fundamentals, applications beyond finance, and the challenges it faces.

**Blockchain Basics**
A blockchain is a decentralized ledger of all transactions across a network. Each transaction is added to a "block" and linked to the previous block, forming a "chain." This structure, combined with cryptographic techniques, makes it secure and immutable.

**Key Features**
- **Decentralization**: No single point of control or failure.
- **Transparency**: Changes to the public blockchain are viewable by all parties creating transparency.
- **Immutability**: Once a transaction is recorded, it cannot be altered, increasing trust.

**Applications**
- **Supply Chain Management**: Enhancing transparency and traceability.
- **Healthcare**: Secure sharing of medical records.
- **Smart Contracts**: Automatically executing contracts when conditions are met.

**Challenges**
- **Scalability**: Current blockchain solutions, like Bitcoin, have limitations on transaction speed and volume.
- **Regulatory Issues**: Balancing the need for regulation with the ethos of decentralization.
- **Energy Consumption**: The energy requirement for "mining" transactions, particularly in networks like Bitcoin, is substantial.

**Conclusion**
Blockchain technology holds great promise for creating a more transparent and efficient world, but significant challenges must be addressed to realize its full potential."""
]


map_chain.run_map_chain(items_to_map=article_list)
```

### Knowledge Graph Creation Example
This example, based on an example of the Instructor library for OpenAI,
demonstrates how to create a knowledge graph using the llama-cpp-agent framework.
```python
import json
from typing import List

from enum import Enum

from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import generate_gbnf_grammar_and_documentation

main_model = Llama(
    "../gguf-models/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=13,
    f16_kv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
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




llama_cpp_agent = LlamaCppAgent(main_model, debug_output=True,
                              system_prompt="You are an advanced AI assistant responding in JSON format.\n\nAvailable JSON response models:\n\n" + documentation)


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
    dot.render("knowledge_graph.gv", view=True)


def generate_graph(user_input: str) -> KnowledgeGraph:
    prompt = f'''Help me understand the following by describing it as a detailed knowledge graph: {user_input}'''.strip()
    response = llama_cpp_agent.get_chat_response(message=prompt, temperature=0.65, mirostat_mode=0, mirostat_tau=3.0,
                                               mirostat_eta=0.1, grammar=gbnf_grammar)
    knowledge_graph = json.loads(response)
    cls = KnowledgeGraph
    knowledge_graph = cls(**knowledge_graph)
    return knowledge_graph


graph = generate_graph("Teach me about quantum mechanics")
visualize_knowledge_graph(graph)
```
Example Output:
![KG](https://raw.githubusercontent.com/Maximilian-Winter/llama-cpp-agent/master/generated_knowledge_graph_example/knowledge_graph.png)


### Manual Function Calling Example
This example shows how to do function calling with pydantic models.
You can also convert Python functions with type hints, automatically to pydantic models using the function:
`create_dynamic_model_from_function` under: `llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models`

```python
import json
import math
from typing import Type, Union

from llama_cpp import Llama, LlamaGrammar

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import \
    generate_gbnf_grammar_and_documentation, create_dynamic_model_from_function
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings


def calculate_a_to_the_power_b(a: Union[int | float], b: Union[int | float]):
    """
    Calculates a to the power of b

    Args:
        a: number
        b: exponent

    """
    print(f"Result: {math.pow(a, b)}")


DynamicSampleModel = create_dynamic_model_from_function(calculate_a_to_the_power_b)

grammar, documentation = generate_gbnf_grammar_and_documentation([DynamicSampleModel], outer_object_name="function",
                                                                 outer_object_content="params")

main_model = LlamaCppEndpointSettings(
    completions_endpoint_url="http://127.0.0.1:8080/completion"
)

llama_cpp_agent = LlamaCppAgent(main_model, debug_output=True,
                                system_prompt="You are an advanced AI, tasked to generate JSON objects for function calling.\n\n" + documentation)

response = llama_cpp_agent.get_chat_response("a= 5, b = 42", temperature=0.15, grammar=grammar)

function_call = json.loads(response)

instance = DynamicSampleModel(**function_call['params'])
instance.run()
```
Example output
```text
{ "function": "calculator","function_parameters": { "number_one": 42.00000 ,  "operation": "multiply" ,  "number_two": 42.00000 }}
1764.0
```

### Manual Function Calling with Python Function Example
This example shows how to do function calling using actual Python functions.

```python
from llama_cpp import Llama
from typing import Union
import math

from llama_cpp_agent.llm_agent import LlamaCppAgent

from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import create_dynamic_model_from_function


def calculate_a_to_the_power_b(a: Union[int, float], b: Union[int, float]):
    """
    Calculates 'a' to the power 'b' and returns the result
    """
    return f"Result: {math.pow(a, b)}"


DynamicSampleModel = create_dynamic_model_from_function(calculate_a_to_the_power_b)

function_tools = [LlamaCppFunctionTool(DynamicSampleModel)]

function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)

main_model = Llama(
    "../../gguf-models/openhermes-2.5-mistral-7b-16k.Q8_0.gguf",
    n_gpu_layers=49,
    offload_kqv=True,
    f16_kv=True,
    use_mlock=False,
    embedding=False,
    n_threads=8,
    n_batch=1024,
    n_ctx=8192,
    last_n_tokens_size=1024,
    verbose=True,
    seed=42,
)

llama_cpp_agent = LlamaCppAgent(main_model, debug_output=True,
                                system_prompt="You are an advanced AI, tasked to assist the user by calling functions in JSON format. The following are the available functions and their parameters and types:\n\n" + function_tool_registry.get_documentation(),
                                predefined_messages_formatter_type=MessagesFormatterType.CHATML)
user_input = "Calculate 5 to power 42"

print(llama_cpp_agent.get_chat_response(user_input, temperature=0.45, function_tool_registry=function_tool_registry))

```
Example output
```text
{ "function": "calculate-a-to-the-power-b","function_parameters": { "a": 5 ,  "b": 42  }}
Result: 2.2737367544323207e+29
```


## Additional Information
- **Dependencies**: pydantic for grammars based generation and of course llama-cpp-python.
- **Documentation**: You can find the documentation here: https://llama-cpp-agent.readthedocs.io/en/latest/agents-api-reference/

### Predefined Messages Formatter
The llama-cpp-agent framework uses custom messages formatters to format messages for the LLM model. The `MessagesFormatterType` enum defines the available predefined formatters. The following predefined formatters are available:


- `MessagesFormatterType.CHATML`: Formats messages using the CHATML format.
- `MessagesFormatterType.MIXTRAL`: Formats messages using the MIXTRAL format.
- `MessagesFormatterType.VICUNA`: Formats messages using the VICUNA format.
- `MessagesFormatterType.LLAMA_2`: Formats messages using the LLAMA 2 format.
- `MessagesFormatterType.SYNTHIA`: Formats messages using the SYNTHIA format.
- `MessagesFormatterType.NEURAL_CHAT`: Formats messages using the NEURAL CHAT format.
- `MessagesFormatterType.SOLAR`: Formats messages using the SOLAR format.
- `MessagesFormatterType.OPEN_CHAT`: Formats messages using the OPEN CHAT format.


You can also define your own custom messages formatter by creating an instance of the `MessagesFormatter` class.
The `MessagesFormatter` class takes the following parameters:


- `PRE_PROMPT`: The pre-prompt to use for the messages.
- `SYS_PROMPT_START`: The system prompt start to use for the messages.
- `SYS_PROMPT_END`: The system prompt end to use for the messages.
- `USER_PROMPT_START`: The user prompt start to use for the messages.
- `USER_PROMPT_END`: The user prompt end to use for the messages.
- `ASSISTANT_PROMPT_START`: The assistant prompt start to use for the messages.
- `ASSISTANT_PROMPT_END`: The assistant prompt end to use for the messages.
- `INCLUDE_SYS_PROMPT_IN_FIRST_USER_MESSAGE`: Whether to include the system prompt in the first user message.
- `DEFAULT_STOP_SEQUENCES`: The default stop sequences to use for the messages.



