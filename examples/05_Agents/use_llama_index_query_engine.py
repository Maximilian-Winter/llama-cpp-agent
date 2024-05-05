# Example that uses the FunctionCallingAgent class to use llama_index tools and query engines.

# Import necessary classes of llama-cpp-agent
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.function_calling_agent import FunctionCallingAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings, LlamaCppGenerationSettings

# Code taken from llama-index example to create a query engine for asking questions
# https://docs.llamaindex.ai/en/stable/examples/agent/react_agent_with_query_engine/

# Import necessary classes of llama-index
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage, Settings,
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
uber_index = VectorStoreIndex(uber_docs, embed_model="local")

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

# Initialize the llama-cpp-agent LLM and the generation parameters.
generation_settings = LlamaCppGenerationSettings(temperature=0.45, top_p=1.0, top_k=0, stream=True)
main_model = LlamaCppEndpointSettings("http://localhost:8080/completion")

# Creating LlamaCppFunctionTool instances out of the llama-index query engine tools.
# We pass the llama-index query engine tools to the from_llama_index_tool function of the LlamaCppFunctionTool class and create the llama-cpp-agent tools.
lyft_query_engine_tool = LlamaCppFunctionTool.from_llama_index_tool(query_engine_tools[0])

uber_query_engine_tool = LlamaCppFunctionTool.from_llama_index_tool(query_engine_tools[1])


function_call_agent = FunctionCallingAgent(
    main_model,
    llama_generation_settings=generation_settings,
    # Pass the LlamaCppFunctionTool instances as a list to the agent.
    llama_cpp_function_tools=[lyft_query_engine_tool, uber_query_engine_tool],
    allow_parallel_function_calling=True,
    messages_formatter_type=MessagesFormatterType.MIXTRAL,
    debug_output=True)

user_input = "What was Lyft's revenue growth in 2021?"
function_call_agent.generate_response(user_input)
