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
        function_tool: Union[
            Type[BaseModel], Callable, Tuple[Dict[str, Any], Callable]
        ],
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


class LlamaCppFunctionToolRegistry:
    """
    Registry for managing LlamaCppFunctionTool instances and generating GBNF grammar.

    Methods:
        register_function_tool(function_tool: LlamaCppFunctionTool): Register a function tool.
        get_function_tool(function_name: str): Get a registered function tool by name.
        finalize(): Finalize the registry, generating the GBNF grammar and documentation.
        get_grammar() -> LlamaGrammar: Get the generated GBNF grammar.
        get_documentation() -> str: Get the documentation for the generated GBNF grammar.
        handle_function_call(function_call_response: str): Handle a function call response and return the output.

    Attributes:
        tool_root (str): Root element for the GBNF grammar.
        tool_rule_content (str): Content rule for the GBNF grammar.
        model_prefix (str): Prefix for documentation of function models.
        fields_prefix (str): Prefix for documentation of function parameter fields.
        function_tools (dict): Dictionary of registered LlamaCppFunctionTool instances.
        function_tools_containing_field_string (dict): Dictionary of registered function tools with extra markdown block or extra triple quoted strings.
        grammar (LlamaGrammar): Generated LlamaGrammar instance.
        grammar_documentation (str): Documentation for the generated GBNF grammar.
        gbnf_grammar (str): Generated GBNF grammar as a string.
        allow_parallel_function_calling (bool): Flag indicating whether to allow parallel function calling.
        add_inner_thoughts (bool): Flag indicating whether to add inner thoughts to the GBNF grammar.
        allow_inner_thoughts_only (bool): Flag indicating whether to allow only inner thoughts in the GBNF grammar.
        add_request_heartbeat (bool): Flag indicating whether to add a request heartbeat field to the GBNF grammar.
        inner_thoughts_field_name (str): Field name for inner thoughts in the GBNF grammar.
        request_heartbeat_field_name (str): Field name for request heartbeat in the GBNF grammar.

    """

    def __init__(
        self,
        allow_parallel_function_calling,
        add_inner_thoughts=True,
        allow_inner_thoughts_only=True,
        add_request_heartbeat=True,
        tool_root="function",
        tool_rule_content="arguments",
        model_prefix="function",
        fields_prefix="parameters",
        inner_thoughts_field_name="thoughts_and_reasoning",
        request_heartbeat_field_name="request_heartbeat",
        add_tool_root_content_to_all_results=True
    ):
        """
        Initialize the LlamaCppFunctionToolRegistry.

        Args:
            allow_parallel_function_calling (bool): Flag indicating whether to allow parallel function calling.
            add_inner_thoughts (bool): Flag indicating whether to add inner thoughts to the GBNF grammar.
            allow_inner_thoughts_only (bool): Flag indicating whether to allow only inner thoughts in the GBNF grammar.
            add_request_heartbeat (bool): Flag indicating whether to add a request heartbeat field to the GBNF grammar.
            tool_root (str): Root element for the GBNF grammar.
            tool_rule_content (str): Content rule for the GBNF grammar.
            model_prefix (str): Prefix for documentation of function models.
            fields_prefix (str): Prefix for documentation of function parameter fields.
            inner_thoughts_field_name (str): Field name for inner thoughts in the GBNF grammar.
            request_heartbeat_field_name (str): Field name for request heartbeat in the GBNF grammar.
        """
        self.tool_root = tool_root
        self.tool_rule_content = tool_rule_content

        self.model_prefix = model_prefix
        self.fields_prefix = fields_prefix
        self.function_tools = {}
        self.function_tools_containing_field_string = {}
        self.grammar = None
        self.grammar_documentation = None
        self.gbnf_grammar = None
        self.allow_parallel_function_calling = allow_parallel_function_calling
        self.add_inner_thoughts = add_inner_thoughts
        self.allow_inner_thoughts_only = allow_inner_thoughts_only
        self.add_request_heartbeat = add_request_heartbeat
        self.inner_thoughts_field_name = inner_thoughts_field_name
        self.request_heartbeat_field_name = request_heartbeat_field_name
        self.add_tool_root_content_to_all_results = add_tool_root_content_to_all_results

    def register_function_tool(self, function_tool: LlamaCppFunctionTool):
        """
        Register a function tool in the registry.

        Args:
            function_tool (LlamaCppFunctionTool): The function tool to register.
        """
        function_name = function_tool.model.__name__
        if function_tool.look_for_field_string:
            self.function_tools_containing_field_string[function_name] = function_tool
        else:
            self.function_tools[function_name] = function_tool

    def get_function_tool(self, function_name: str):
        """
        Get a registered function tool by name.

        Args:
            function_name (str): The name of the function tool.

        Returns:
            LlamaCppFunctionTool: The registered function tool, or None if not found.
        """
        if function_name in self.function_tools:
            return self.function_tools[function_name]
        elif function_name in self.function_tools_containing_field_string:
            return self.function_tools_containing_field_string[function_name]
        else:
            return None

    def finalize(self):
        """
        Finalize the registry, generating the GBNF grammar and documentation.
        """
        pydantic_function_models = []
        request_heartbeat_list = []
        for function_tool in self.function_tools.values():
            pydantic_function_models.append(function_tool.model)
            if function_tool.add_outer_request_heartbeat_field:
                request_heartbeat_list.append(function_tool.model.__name__)
        for function_tool in self.function_tools_containing_field_string.values():
            pydantic_function_models.append(function_tool.model)

        gbnf_grammar, documentation = generate_gbnf_grammar_and_documentation(
            pydantic_function_models,
            self.tool_root,
            self.tool_rule_content,
            self.model_prefix,
            self.fields_prefix,
            self.allow_parallel_function_calling,
            inner_thoughts_field_name=self.inner_thoughts_field_name,
            request_heartbeat_field_name=self.request_heartbeat_field_name,
            add_inner_thoughts=self.add_inner_thoughts,
            allow_only_inner_thoughts=self.allow_inner_thoughts_only,
            add_request_heartbeat=self.add_request_heartbeat,
            request_heartbeat_models=request_heartbeat_list,
        )

        self.grammar = LlamaGrammar.from_string(gbnf_grammar, verbose=False)
        self.grammar_documentation = documentation
        self.gbnf_grammar = gbnf_grammar

    def get_grammar(self):
        """
        Get the generated LlamaGrammar instance.

        Returns:
           LlamaGrammar: The LlamaGrammar instance.
        """
        return self.grammar

    def get_documentation(self):
        """
        Get the documentation for the provided functions for the LLM as guidance.

        Returns:
            str: The documentation for the provided functions for the LLM as guidance.
        """
        return self.grammar_documentation

    def handle_function_call(self, function_call_response: str):
        """
        Handle a function call response and return the output.

        Args:
            function_call_response (str): The function call response.

        Returns:
            str: The output of the function call or an error message.
        """
        try:
            for name, tool in self.function_tools_containing_field_string.items():
                if (
                    f'"function": "{name}"' in function_call_response
                    or f'"function":"{name}"' in function_call_response
                ):
                    marker = (
                        "'''"
                        if self.function_tools_containing_field_string[
                            name
                        ].has_triple_quoted_string
                        else "```"
                    )
                    (
                        function_call,
                        content,
                    ) = parse_json_response_with_markdown_code_block_or_triple_quoted_string(
                        function_call_response, marker
                    )
                    if self.function_tools_containing_field_string[
                        function_call[self.tool_root]
                    ].has_markdown_code_block:
                        function_call[self.tool_rule_content][
                            tool.markdown_code_block_field_name
                        ] = content
                    elif self.function_tools_containing_field_string[
                        function_call[self.tool_root]
                    ].has_triple_quoted_string:
                        function_call[self.tool_rule_content][
                            tool.triple_quoted_string_field_name
                        ] = content
                    if not self.allow_parallel_function_calling:
                        output = self.intern_function_call(
                            function_call, with_markdown_code_block=True
                        )
                    else:
                        output = self.intern_parallel_function_call(
                            function_call, with_markdown_code_block=True
                        )
                    return output

            function_call = parse_json_response(function_call_response)
            if function_call is None:
                return "Error: Invalid function call response."
            if not self.allow_parallel_function_calling:
                output = self.intern_function_call(function_call)
            else:
                output = self.intern_parallel_function_call(function_call)

            return output

        except AttributeError as e:
            return f"Error: {e}"

    def intern_function_call(self, function_call: dict, with_markdown_code_block=False):
        """
        Internal method to handle a function call and return the output.

        Args:
            function_call (dict): The function call dictionary.
            with_markdown_code_block (bool): Flag indicating whether the response contains a markdown code block.

        Returns:
            str: The output of the function call or an error message.
        """
        function_tool = None
        if with_markdown_code_block:
            if self.tool_root in function_call:
                function_tool = self.function_tools_containing_field_string[
                    function_call[self.tool_root]
                ]
        else:
            if self.tool_root in function_call:
                function_tool = self.function_tools[function_call[self.tool_root]]
        try:
            if self.tool_root in function_call:
                cls = function_tool.model
                call_parameters = function_call[self.tool_rule_content]
                call = cls(**call_parameters)
                output = call.run(**function_tool.additional_parameters)
                if function_tool.add_params_to_result or self.add_tool_root_content_to_all_results:
                    if self.add_request_heartbeat:
                        return [
                            {
                                "function": function_tool.model.__name__,
                                "params": call_parameters,
                                "return_value": output,
                                "request_heartbeat": function_call["request_heartbeat"]
                                if "request_heartbeat" in function_call
                                else None,
                            }
                        ]
                    return [
                        {
                            "function": function_tool.model.__name__,
                            "params": call_parameters,
                            "return_value": output,
                            "request_heartbeat": None,
                        }
                    ]
                else:
                    if self.add_request_heartbeat:
                        return [
                            {
                                "function": function_tool.model.__name__,
                                "return_value": output,
                                "request_heartbeat": function_call["request_heartbeat"]
                                if "request_heartbeat" in function_call
                                else None,
                            }
                        ]
                    return [
                        {
                            "function": function_tool.model.__name__,
                            "return_value": output,
                            "request_heartbeat": None,
                        }
                    ]
        except AttributeError as e:
            return f"Error: {e}"

    def intern_parallel_function_call(
        self, function_calls: List[dict], with_markdown_code_block=False
    ):
        """
        Internal method to handle a function call and return the output.

        Args:
            function_calls List[dict]: The function call dictionary.
            with_markdown_code_block (bool): Flag indicating whether the response contains a markdown code block.

        Returns:
            str: The output of the function call or an error message.
        """
        result = []
        if not with_markdown_code_block:
            for function_call in function_calls:
                if self.tool_root in function_call:
                    function_tool = self.function_tools[function_call[self.tool_root]]
                    try:
                        cls = function_tool.model
                        call_parameters = function_call[self.tool_rule_content]
                        call = cls(**call_parameters)
                        output = call.run(**function_tool.additional_parameters)
                        if function_tool.add_params_to_result or self.add_tool_root_content_to_all_results:
                            if self.add_request_heartbeat:
                                result.append(
                                    {
                                        "function": function_tool.model.__name__,
                                        "params": call_parameters,
                                        "return_value": output,
                                        "request_heartbeat": function_call[
                                            "request_heartbeat"
                                        ]
                                        if "request_heartbeat" in function_call
                                        else None,
                                    }
                                )
                            else:
                                result.append(
                                    {
                                        "function": function_tool.model.__name__,
                                        "params": call_parameters,
                                        "return_value": output,
                                    }
                                )
                        else:
                            if self.add_request_heartbeat:
                                result.append(
                                    {
                                        "function": function_tool.model.__name__,
                                        "return_value": output,
                                        "request_heartbeat": function_call[
                                            "request_heartbeat"
                                        ]
                                        if "request_heartbeat" in function_call
                                        else None,
                                    }
                                )
                            else:
                                result.append(
                                    {
                                        "function": function_tool.model.__name__,
                                        "return_value": output,
                                    }
                                )
                    except AttributeError as e:
                        return f"Error: {e}"
        else:
            for function_call in function_calls:
                if self.tool_root in function_call:
                    function_tool = self.function_tools_containing_field_string[
                        function_call[self.tool_root]
                    ]
                    try:
                        cls = function_tool.model
                        call_parameters = function_call[self.tool_rule_content]
                        call = cls(**call_parameters)
                        output = call.run(**function_tool.additional_parameters)
                        if function_tool.add_params_to_result:
                            if self.add_request_heartbeat:
                                result.append(
                                    {
                                        "function": function_tool.model.__name__,
                                        "params": call_parameters,
                                        "return_value": output,
                                        "request_heartbeat": function_call[
                                            "request_heartbeat"
                                        ]
                                        if "request_heartbeat" in function_call
                                        else None,
                                    }
                                )
                            else:
                                result.append(
                                    {
                                        "function": function_tool.model.__name__,
                                        "params": call_parameters,
                                        "return_value": output,
                                    }
                                )
                        else:
                            if self.add_request_heartbeat:
                                result.append(
                                    {
                                        "function": function_tool.model.__name__,
                                        "return_value": output,
                                        "request_heartbeat": function_call[
                                            "request_heartbeat"
                                        ]
                                        if "request_heartbeat" in function_call
                                        else None,
                                    }
                                )
                            else:
                                result.append(
                                    {
                                        "function": function_tool.model.__name__,
                                        "return_value": output,
                                    }
                                )
                    except AttributeError as e:
                        return f"Error: {e}"
        return result
