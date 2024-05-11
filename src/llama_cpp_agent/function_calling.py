import inspect
import json
import typing
from enum import Enum
from types import GenericAlias
from typing import Type, List, Callable, Any, Union, Tuple, Dict

from llama_cpp import LlamaGrammar
from pydantic import BaseModel

from .output_parser import (
    parse_json_response_with_markdown_code_block_or_triple_quoted_string,
    parse_json_response,
)
from .gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import (
    format_model_and_field_name,
    generate_gbnf_grammar_and_documentation,
    create_dynamic_model_from_function,
    create_dynamic_models_from_dictionaries,
    add_run_method_to_dynamic_model,
)

from docstring_parser import parse


def get_enum_type(enum):
    """Determine the JSON schema type for an enum based on its members."""
    enum_values = [e.value for e in enum]
    if all(isinstance(e, int) for e in enum_values):
        return {"type": "integer", "enum": enum_values}
    elif all(isinstance(e, float) for e in enum_values):
        return {"type": "number", "enum": enum_values}
    else:
        return {"type": "string", "enum": enum_values}


def py_type_to_json_type(schema):
    type_map = {
        Any: {"type": "any"},
        str: {"type": "string"},
        float: {"type": "number"},
        int: {"type": "integer"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
    }
    return type_map[schema]


def get_openai_type(py_type):
    """Map Python types to JSON schema types and handle special cases like Enums, Lists, and Unions."""
    if inspect.isclass(py_type) and issubclass(py_type, Enum):
        # Handle Enum types by determining their actual value types
        return get_enum_type(py_type)
    elif inspect.isclass(py_type) and issubclass(py_type, BaseModel):
        # Handle nested Pydantic models by recursive call
        return {
            "type": "object",
            "properties": pydantic_model_to_openai_function_definition(py_type)[
                "function"
            ]["parameters"]["properties"],
        }
    elif hasattr(py_type, "__origin__"):
        if py_type.__origin__ is Union:
            # Filter out NoneType to handle optional fields
            non_none_types = [t for t in py_type.__args__ if t is not type(None)]
            return get_openai_type(non_none_types[0])
        elif py_type.__origin__ is List or py_type.__origin__ is list:
            # Handle lists by identifying the type of list items
            return {"type": "array", "items": get_openai_type(py_type.__args__[0])}
    else:
        # Default type handling
        return py_type_to_json_type(py_type)


def pydantic_model_to_openai_function_definition(pydantic_model: Type[BaseModel]):
    model_schema = pydantic_model.schema()
    properties = model_schema["properties"]
    required_fields = model_schema.get("required", [])
    class_doc = inspect.getdoc(pydantic_model)
    base_class_doc = inspect.getdoc(BaseModel)
    class_description = class_doc if class_doc and class_doc != base_class_doc else ""

    function_definition = {
        "type": "function",
        "function": {
            "name": pydantic_model.__name__.lower(),
            "description": class_description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": required_fields,
            },
        },
    }

    type_hints = typing.get_type_hints(pydantic_model)
    for prop_name, prop_meta in properties.items():
        prop_type = type_hints[prop_name]

        openai_type = get_openai_type(prop_type)
        field_info = pydantic_model.model_fields.get(prop_name)
        field_description = (
            field_info.description if field_info and field_info.description else ""
        )
        if isinstance(openai_type, dict) and "union" in openai_type.get("type", ""):
            # Handling Union types specifically
            function_definition["function"]["parameters"]["properties"][prop_name] = {
                "type": "union",
                "options": openai_type["options"],
                "description": field_description,
            }
        else:
            function_definition["function"]["parameters"]["properties"][prop_name] = {
                **openai_type,
                "description": field_description,
            }

    return function_definition


class LlamaCppFunctionTool:
    """
    Callable class representing a tool for handling function calls in the LlamaCpp environment.

    Args:
        function_tool (Union[Type[BaseModel], Callable, Tuple[Dict[str, Any], Callable]]): The function tool, can be either a pydantic model with run method, a python function or a tuple of a OpenAI tool specification and a function as callback.
        has_markdown_code_block (bool): Flag indicating whether the response contains an extra markdown code block field.
        has_triple_quoted_string (bool): Flag indicating whether the response contains an extra triple-quoted string field.
        **additional_parameters: Additional parameters to pass to the Pydantic model during function call.

    Attributes:
        model (Type[BaseModel]): The Pydantic model representing the function parameters.
        look_for_field_string (bool): Flag indicating whether to look for a field string in the response.
        has_markdown_code_block (bool): Flag indicating whether the response contains a markdown code block.
        has_triple_quoted_string (bool): Flag indicating whether the response contains a triple-quoted string.
        additional_parameters (dict): Additional parameters to pass to the Pydantic model during function call.

    Methods:
        __call__(*args, **kwargs): Calls the Pydantic model with the provided keyword arguments.
    """

    def __init__(
        self,
        function_tool: Union[BaseModel, Callable, Tuple[Dict[str, Any], Callable]],
        add_params_to_result=False,
        has_markdown_code_block=False,
        has_triple_quoted_string=False,
        markdown_code_block_field_name=None,
        triple_quoted_string_field_name=None,
        add_outer_request_heartbeat_field=True,
        **additional_parameters,
    ):
        # Determine the type of function_tool and set up the appropriate handling
        if isinstance(function_tool, type) and issubclass(function_tool, BaseModel):
            # Handle BaseModel subclass
            self.model = function_tool  # instantiate the model if needed
        elif (
            isinstance(function_tool, tuple)
            and len(function_tool) == 2
            and isinstance(function_tool[0], dict)
            and callable(function_tool[1])
        ):
            # Handle OpenAI functions
            models = create_dynamic_models_from_dictionaries([function_tool[0]])
            self.model = add_run_method_to_dynamic_model(models[0], function_tool[1])
        elif callable(function_tool):
            # Handle simple callable
            self.model = create_dynamic_model_from_function(function_tool)
        else:
            raise ValueError("Invalid function_tool type provided")
        self.add_params_to_result = add_params_to_result
        self.look_for_field_string = has_markdown_code_block or has_triple_quoted_string
        self.has_markdown_code_block = has_markdown_code_block
        self.has_triple_quoted_string = has_triple_quoted_string
        self.markdown_code_block_field_name = markdown_code_block_field_name
        self.triple_quoted_string_field_name = triple_quoted_string_field_name
        self.additional_parameters = (
            additional_parameters if additional_parameters else {}
        )
        self.add_outer_request_heartbeat_field = add_outer_request_heartbeat_field

    @staticmethod
    def from_pydantic_model_and_callable(
        pydantic_model: BaseModel, tool_function: Callable
    ):
        """
        Converts an OpenAI tool schema and a callable function into a LlamaCppFunctionTool
        Args:
            pydantic_model(BaseModel): Pydantic Model representing the arguments to the tool.
            tool_function(Callable): Callable function that will be invoked when the agent uses it and will be passed the fields of the pydantic model.

        Returns:
            LlamaCppFunctionTool: The LlamaCppFunctionTool instance.
        """
        pydantic_model = add_run_method_to_dynamic_model(pydantic_model, tool_function)
        return LlamaCppFunctionTool(pydantic_model)

    @staticmethod
    def from_openai_tool(openai_tool_schema: dict, tool_function: Callable):
        """
        Converts an OpenAI tool schema and a callable function into a LlamaCppFunctionTool
        Args:
            openai_tool_schema(dict): OpenAI tool description dictionary.
            tool_function(Callable): Callable function that will be invoked when the agent uses it.

        Returns:
            LlamaCppFunctionTool: The LlamaCppFunctionTool instance.
        """
        models = create_dynamic_models_from_dictionaries([openai_tool_schema])
        model = add_run_method_to_dynamic_model(models[0], tool_function)
        return LlamaCppFunctionTool(model)

    @staticmethod
    def from_llama_index_tool(llama_index_tool):
        """
        Converts a llama-index tool into a LlamaCppFunctionTool
        Args:
            llama_index_tool(["BaseTool"]): OpenAI tool description dictionary.

        Returns:
            LlamaCppFunctionTool: The LlamaCppFunctionTool instance.
        """
        models = create_dynamic_models_from_dictionaries(
            [llama_index_tool.metadata.to_openai_tool()]
        )
        model = add_run_method_to_dynamic_model(models[0], llama_index_tool.call)
        return LlamaCppFunctionTool(model)

    def to_mistral_tool(self):
        from mistral_common.protocol.instruct.tool_calls import Tool, Function

        root = pydantic_model_to_openai_function_definition(self.model)
        return Tool(
            function=Function(
                name=root["function"]["name"],
                description=root["function"]["description"],
                parameters=root["function"]["parameters"],
            )
        )

    def to_openai_tool(self):
        root = pydantic_model_to_openai_function_definition(self.model)
        return root

    def __call__(self, *args, **kwargs):
        """
        Calls the Pydantic model with the provided keyword arguments.

        Returns:
            BaseModel: An instance of the Pydantic model.
        """
        return self.model(**kwargs)
