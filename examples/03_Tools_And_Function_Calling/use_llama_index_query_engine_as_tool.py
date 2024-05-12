# Example that uses the FunctionCallingAgent class to use llama_index tools and query engines. This is based on a llama-index example

# To get the PDFs used in this example:
# mkdir -p 'data/10k/'
# wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
# wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'


# Import necessary classes of llama-cpp-agent
from llama_cpp_agent import LlamaCppFunctionTool
from llama_cpp_agent import FunctionCallingAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.providers import TGIServerProvider

model = TGIServerProvider("http://127.0.0.1:8080")
# Code taken from llama-index example to create a query engine for asking questions
# https://docs.llamaindex.ai/en/stable/examples/agent/react_agent_with_query_engine/

# Import necessary classes of llama-index
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# Setting the default llm of llama-index to None, llama-index will throw error otherwise!
Settings.llm = None


# load data
lyft_docs = SimpleDirectoryReader(
    input_files=["./data/10k/lyft_2021.pdf"]
).load_data()
uber_docs = SimpleDirectoryReader(
    input_files=["./data/10k/uber_2021.pdf"]
).load_data()

# build index
lyft_index = VectorStoreIndex.from_documents(lyft_docs, embed_model="local")
uber_index = VectorStoreIndex.from_documents(uber_docs, embed_model="local")

# Create the query engines for lyft and uber.

lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
uber_engine = uber_index.as_query_engine(similarity_top_k=3)

# Create a list query engine tools.
query_engine_tools = [
    QueryEngineTool(
        query_engine=lyft_engine,
        metadata=ToolMetadata(
            name="lyft_10k",
            description=(
                "Provides information about Lyft financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
    QueryEngineTool(
        query_engine=uber_engine,
        metadata=ToolMetadata(
            name="uber_10k",
            description=(
                "Provides information about Uber financials for year 2021. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

# Creating LlamaCppFunctionTool instances out of the llama-index query engine tools.
# We pass the llama-index query engine tools to the from_llama_index_tool function of the LlamaCppFunctionTool class and create the llama-cpp-agent tools.
lyft_query_engine_tool = LlamaCppFunctionTool.from_llama_index_tool(query_engine_tools[0])

uber_query_engine_tool = LlamaCppFunctionTool.from_llama_index_tool(query_engine_tools[1])


function_call_agent = FunctionCallingAgent(
    model,
    # Pass the LlamaCppFunctionTool instances as a list to the agent.
    llama_cpp_function_tools=[lyft_query_engine_tool, uber_query_engine_tool],
    allow_parallel_function_calling=False,
    messages_formatter_type=MessagesFormatterType.CHATML,
    debug_output=True)

settings = model.get_provider_default_settings()
settings.max_new_tokens = 512
settings.temperature = 0.65
settings.do_sample = True

user_input = "What was Lyft's revenue growth in 2021?"
function_call_agent.generate_response(user_input, llm_sampling_settings=settings)
