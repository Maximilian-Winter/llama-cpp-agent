# Based on an example of the Instructor library for OpenAI
from typing import List, Any

from llama_cpp_agent.llm_output_settings import LlmStructuredOutputSettings, LlmStructuredOutputType
from llama_cpp_agent.messages_formatter import MessagesFormatterType

from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.output_parser import extract_object_from_response

from llama_cpp_agent.providers.tgi_server import TGIServerProvider

model = TGIServerProvider("http://localhost:8080")



class RowData(BaseModel):
    row: List[Any] = Field(..., description="The values for each row")
    citation: str = Field(
        ..., description="The citation for this row from the original source data"
    )


class Dataframe(BaseModel):
    """
    Class representing a dataframe. This class is used to convert
    data into a frame that can be used by pandas.
    """

    name: str = Field(..., description="The name of the dataframe")
    data: List[RowData] = Field(
        ...,
        description="Correct rows of data aligned to column names, Nones are allowed",
    )
    columns: List[str] = Field(
        ...,
        description="Column names relevant from source data, should be in snake_case",
    )

    def to_pandas(self):
        import pandas as pd

        columns = self.columns + ["citation"]
        data = [row.row + [row.citation] for row in self.data]

        return pd.DataFrame(data=data, columns=columns)


class Database(BaseModel):
    """
    A set of correct named and defined tables as dataframes
    """

    tables: List[Dataframe] = Field(
        ...,
        description="List of tables in the database",
    )


output_settings = LlmStructuredOutputSettings.from_pydantic_models([Database], output_type=LlmStructuredOutputType.object_instance)

llama_cpp_agent = LlamaCppAgent(model, debug_output=True,
                                system_prompt="""You are an advanced AI assistant, responding in JSON format.

Available JSON response models:\n\n""" + output_settings.get_llm_documentation() + """""",
                                predefined_messages_formatter_type=MessagesFormatterType.MIXTRAL)


def dataframe(data: str) -> Database:
    prompt = data
    parameters = model.get_provider_default_settings()
    parameters.do_sample = True
    parameters.temperature = 0.1
    response = llama_cpp_agent.get_chat_response(message=prompt, structured_output_settings=output_settings)
    return response


dfs = dataframe(
    """Map this data into a database: "My name is John and I am 25 years old. I live in New York and I like to play basketball. His name is Mike and he is 30 years old. He lives in San Francisco and he likes to play baseball. Sarah is 20 years old and she lives in Los Angeles. She likes to play tennis. Her name is Mary and she is 35 years old. She lives in Chicago. On one team 'Tigers' the captain is John and there are 12 players. On the other team 'Lions' the captain is Mike and there are 10 players." """)

for df in dfs.tables:
    print(df.name)
    print(df.to_pandas())
