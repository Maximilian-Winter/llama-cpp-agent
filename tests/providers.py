import datetime
from enum import Enum
from typing import Optional, Union, List

from llama_cpp import Llama
from pydantic import BaseModel, Field

from llama_cpp_agent import LlamaCppAgent, FunctionCallingAgent
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers.llama_cpp_server import LlmProviderId, LlamaCppServerProvider
from llama_cpp_agent.providers.tgi_server import LlmProviderId, TGIServerProvider
from llama_cpp_agent.providers.vllm_server import LlmProviderId, VLLMServerProvider
from llama_cpp_agent.providers.llama_cpp_python import LlmProviderId, LlamaCppPythonProvider

if __name__ == "__main__":
    test_endpoint = LlmProviderId.tgi_server
    text_test = "[INST] Hello World! [/INST]"
    messages_test = [{"role": "user", "content": "Hello World!"}]
    structured_output_settings_test = LlmStructuredOutputSettings(
        output_type=LlmStructuredOutputType.no_structured_output)
    provider = None
    if test_endpoint is LlmProviderId.llama_cpp_server:
        provider = LlamaCppServerProvider("http://localhost:8080")
    elif test_endpoint is LlmProviderId.llama_cpp_python:
        llama_model = Llama(r"C:\AI\Agents\gguf-models\mistral-7b-instruct-v0.2.Q6_K.gguf", n_batch=1024, n_threads=10,
                            n_gpu_layers=40)
        provider = LlamaCppPythonProvider(llama_model)
    elif test_endpoint is LlmProviderId.tgi_server:
        provider = TGIServerProvider("http://localhost:8080")
    elif test_endpoint is LlmProviderId.vllm_server:
        provider = VLLMServerProvider("http://localhost:8000/v1", "TheBloke/Llama-2-7b-Chat-AWQ", "token-abc123")
    sampling_settings_test = provider.get_provider_default_settings()

    # Test text completion
    result_test = provider.create_completion(text_test, structured_output_settings_test, sampling_settings_test, "<s>")
    print(result_test["choices"][0]["text"])

    # Test chat completion
    provider.create_chat_completion(messages_test, structured_output_settings_test, sampling_settings_test)
    print(result_test["choices"][0]["text"])

    # Test text completion streaming
    sampling_settings_test.stream = True
    for t in provider.create_completion(text_test, structured_output_settings_test, sampling_settings_test, "<s>"):
        print(t["choices"][0]["text"], end="")
    print("")

    # Test chat completion streaming
    for t in provider.create_chat_completion(messages_test, structured_output_settings_test, sampling_settings_test):
        print(t["choices"][0]["text"], end="")
    print("")


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
    # Simple pydantic calculator tool for the agent that can add, subtract, multiply, and divide. Docstring and description of fields will be used in system prompt.
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

    # First we create the calculator tool.
    calculator_function_tool = LlamaCppFunctionTool(calculator)

    # Next we create the current datetime tool.
    current_datetime_function_tool = LlamaCppFunctionTool(get_current_datetime)

    # The from_openai_tool function of the LlamaCppFunctionTool class converts an OpenAI tool schema and a callable function into a LlamaCppFunctionTool
    get_weather_function_tool = LlamaCppFunctionTool.from_openai_tool(open_ai_tool_spec, get_current_weather)

    tools = [calculator_function_tool, current_datetime_function_tool, get_weather_function_tool]

    settings = LlmStructuredOutputSettings(output_type=LlmStructuredOutputType.parallel_function_calling,
                                           function_tools=tools)

    prompt_test = f"""[INST] You are Funky, an AI assistant that calls functions to perform tasks.

To call functions, you respond with a JSON object containing three fields:
"001_thoughts_and_reasoning": Your thoughts and reasoning behind the function call.
"002_function": The name of the function you want to call.
"003_arguments": The arguments required for the function.

After performing a function call, you will receive a response containing the return values of the function calls.

### Functions:
Below is a list of functions you can use to interact with the system. Each function has specific parameters and requirements. Make sure to follow the instructions for each function carefully.
Choose the appropriate function based on the task you want to perform. Provide your function calls in JSON format.

{settings.get_llm_documentation(provider).strip()}

Solve the following calculation: 42 * 42. [/INST]"""
    sampling_settings_test.stream = False
    sampling_settings_test.max_tokens = 256
    test = provider.create_completion(prompt_test, structured_output_settings=settings, settings=sampling_settings_test,
                                      bos_token="<s>")
    print(test["choices"][0]["text"])

    agent = LlamaCppAgent(
        provider,
        debug_output=True,
        system_prompt="You are a helpful assistant.",
        predefined_messages_formatter_type=MessagesFormatterType.CHATML,
    )

    settings = provider.get_provider_default_settings()
    settings.stream = True
    settings.max_new_tokens = 512
    settings.temperature = 0.65
    settings.do_sample = True

    agent.get_chat_response("Hello!", llm_sampling_settings=settings)


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

    # We define the input information for the agent.
    text = """The Feynman Lectures on Physics is a physics textbook based on some lectures by Richard Feynman, a Nobel laureate who has sometimes been called "The Great Explainer". The lectures were presented before undergraduate students at the California Institute of Technology (Caltech), during 1961–1963. The book's co-authors are Feynman, Robert B. Leighton, and Matthew Sands."""

    print(agent.get_chat_response(text, llm_sampling_settings=settings, structured_output_settings=output_settings))


    class Calculator(BaseModel):
        """
        Perform a math operation on two numbers.
        """

        number_one: Union[int, float] = Field(
            ...,
            description="First number."
        )
        number_two: Union[int, float] = Field(
            ...,
            description="Second number."
        )
        operation: MathOperation = Field(..., description="Math operation to perform.")

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


    # Create a list of function call tools.
    function_tools = [LlamaCppFunctionTool(Calculator)]

    output_settings = LlmStructuredOutputSettings.from_llama_cpp_function_tools(function_tools,
                                                                                allow_parallel_function_calling=True)

    user_input = "What is 42 + 42?"

    agent.get_chat_response(user_input, llm_sampling_settings=settings, structured_output_settings=output_settings)


    # Simple tool for the agent, to get the current date and time in a specific format.
    def get_current_datetime(output_format: Optional[str] = None):
        """
        Get the current date and time in the given format.

        Args:
             output_format: formatting string for the date and time, defaults to '%Y-%m-%d %H:%M:%S'
        """
        if output_format is None:
            output_format = '%Y-%m-%d %H:%M:%S'
        return datetime.datetime.now().strftime(output_format)


    # Simple pydantic calculator tool for the agent that can add, subtract, multiply, and divide. Docstring and description of fields will be used in system prompt.
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
    # llama-cpp-agent supports OpenAI like schemas for function definition.
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
                "required": ["location", "unit"],
            },
        },
    }


    # Callback for receiving messages for the user.
    def send_message_to_user_callback(message: str):
        print(message)


    # First we create the calculator tool.
    calculator_function_tool = LlamaCppFunctionTool(calculator)

    # Next we create the current datetime tool.
    current_datetime_function_tool = LlamaCppFunctionTool(get_current_datetime)

    # The from_openai_tool function of the LlamaCppFunctionTool class converts an OpenAI tool schema and a callable function into a LlamaCppFunctionTool
    get_weather_function_tool = LlamaCppFunctionTool.from_openai_tool(open_ai_tool_spec, get_current_weather)

    # Create the function calling agent. We are passing the provider, the tool list, send message to user callback and the chat message formatter. Also, we allow parallel function calling.
    function_call_agent = FunctionCallingAgent(
        provider,
        debug_output=True,
        llama_cpp_function_tools=[calculator_function_tool, current_datetime_function_tool, get_weather_function_tool],
        send_message_to_user_callback=send_message_to_user_callback,
        allow_parallel_function_calling=True,
        messages_formatter_type=MessagesFormatterType.CHATML)

    user_input = '''Get the date and time in '%d-%m-%Y %H:%M' format. Get the current weather in celsius in London, New York and at the North Pole. Solve the following calculations: 42 * 42, 74 + 26, 7 * 26, 4 + 6  and 96/8.'''
    function_call_agent.generate_response(user_input)
