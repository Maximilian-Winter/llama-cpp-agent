import requests

from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.function_calling import LlamaCppFunctionTool
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import (
    LlamaCppEndpointSettings,
)
from tools import (
    SendMessageToUser,
)


class ScrapeWebsite(BaseModel):
    """
    Scrape the content of a website given its URL.
    """

    url: str = Field(
        description="The URL of the website to scrape.",
    )

    def run(self) -> str:
        response = requests.get(self.url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            html_string = str(soup)
            return html_string
        else:
            print("Failed to retrieve page:", response.status_code)
            return ""


function_tools = [
    LlamaCppFunctionTool(SendMessageToUser),
    LlamaCppFunctionTool(ScrapeWebsite),
]
function_tool_registry = LlamaCppAgent.get_function_tool_registry(function_tools)
model = LlamaCppEndpointSettings(completions_endpoint_url="http://127.0.0.1:8080/completion")
llama_cpp_agent = LlamaCppAgent(
    model,
    debug_output=True,
    predefined_messages_formatter_type=MessagesFormatterType.CHATML,
)


while True:
    user_input = input("User> ")
    if "/exit" in user_input:
        break

    response = llama_cpp_agent.get_chat_response(
        user_input,
        system_prompt=f"You are a advanced helpful AI assistant interacting through calling functions in form of JSON objects. \n\nHere are your available functions:\n\n{function_tool_registry.get_documentation()}",
        temperature=0.65,
        function_tool_registry=function_tool_registry,
    )

    print(response)
