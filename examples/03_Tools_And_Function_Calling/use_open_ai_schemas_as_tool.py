from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent import LlamaCppFunctionTool
from llama_cpp_agent.providers import TGIServerProvider

provider = TGIServerProvider("http://localhost:8080")

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
            "required": ["location", "unit"],
        },
    },
}


# Example function based on an OpenAI example.
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


# Create a list of function call tools.
function_tools = [LlamaCppFunctionTool((open_ai_tool_spec, get_current_weather))]

output_settings = LlmStructuredOutputSettings.from_llama_cpp_function_tools(function_tools,
                                                                            allow_parallel_function_calling=True)
llama_cpp_agent = LlamaCppAgent(
    provider,
    debug_output=False,
    system_prompt=f"You are an advanced AI, tasked to assist the user by calling functions in JSON format.",
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)

user_input = "What is the weather in New York?"
print(
    llama_cpp_agent.get_chat_response(
        user_input,
        structured_output_settings=output_settings
    )
)
