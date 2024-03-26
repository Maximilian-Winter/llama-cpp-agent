# Based on an example of the Instructor library for OpenAI


import json
from typing import List, Any

from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp import Llama, LlamaGrammar
from pydantic import BaseModel, Field

from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.output_parser import extract_object_from_response
from llama_cpp_agent.gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import \
    generate_gbnf_grammar_and_documentation
from llama_cpp_agent.providers.llama_cpp_endpoint_provider import LlamaCppEndpointSettings

main_model = LlamaCppEndpointSettings(
    completions_endpoint_url="http://127.0.0.1:8080/completion"
)


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


gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation([Database],
                                                           model_prefix="Response Model",
                                                           fields_prefix="Response Model Field")

llama_cpp_agent = LlamaCppAgent(main_model, debug_output=True,
                                system_prompt="You are an advanced AI assistant, responding in JSON format. \n\nAvailable JSON response models:\n\n" + documentation + """""",
                                predefined_messages_formatter_type=MessagesFormatterType.CHATML)




def dataframe(data: str) -> Database:
    prompt = data
    response = llama_cpp_agent.get_chat_response(message=prompt, temperature=0.65, grammar=gbnf_grammar)

    database = extract_object_from_response(response, Database)
    return database


dfs = dataframe(
    """Map this data into a database: "My name is John and I am 25 years old. I live in New York and I like to play basketball. His name is Mike and he is 30 years old. He lives in San Francisco and he likes to play baseball. Sarah is 20 years old and she lives in Los Angeles. She likes to play tennis. Her name is Mary and she is 35 years old. She lives in Chicago. On one team 'Tigers' the captain is John and there are 12 players. On the other team 'Lions' the captain is Mike and there are 10 players." """)

for df in dfs.tables:
    print(df.name)
    print(df.to_pandas())
