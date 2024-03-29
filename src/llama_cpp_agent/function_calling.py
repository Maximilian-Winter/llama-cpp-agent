import json
from typing import Type, List

from llama_cpp import LlamaGrammar
from pydantic import BaseModel

from .output_parser import (
    parse_json_response_with_markdown_code_block_or_triple_quoted_string,
    parse_json_response,
)
from .gbnf_grammar_generator.gbnf_grammar_from_pydantic_models import (
    format_model_and_field_name,
    generate_gbnf_grammar_and_documentation,
)


class LlamaCppFunctionTool:
    """
    Callable class representing a tool for handling function calls in the LlamaCpp environment.

    Args:
        pydantic_model (Type[BaseModel]): The Pydantic model representing the function.
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
        pydantic_model: Type[BaseModel],
        add_params_to_result=False,
        has_markdown_code_block=False,
        has_triple_quoted_string=False,
        markdown_code_block_field_name=None,
        triple_quoted_string_field_name=None,
        add_outer_request_heartbeat_field=True,
        **additional_parameters,
    ):
        self.model = pydantic_model
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
    """

    def __init__(
        self,
        allow_parallel_function_calling,
        add_inner_thoughts=True,
        allow_inner_thoughts_only=True,
        add_request_heartbeat=True,
    ):
        self.tool_root = "function"
        self.tool_rule_content = "params"
        self.model_prefix = "#### Function"
        self.fields_prefix = "Parameters"
        self.function_tools = {}
        self.function_tools_containing_field_string = {}
        self.grammar = None
        self.grammar_documentation = None
        self.gbnf_grammar = None
        self.allow_parallel_function_calling = allow_parallel_function_calling
        self.add_inner_thoughts = add_inner_thoughts
        self.allow_inner_thoughts_only = allow_inner_thoughts_only
        self.add_request_heartbeat = add_request_heartbeat

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
                if name in function_call_response:
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
        if with_markdown_code_block:
            function_tool = self.function_tools_containing_field_string[
                function_call[self.tool_root]
            ]
        else:
            function_tool = self.function_tools[function_call[self.tool_root]]
        try:
            cls = function_tool.model
            call_parameters = function_call[self.tool_rule_content]
            call = cls(**call_parameters)
            output = call.run(**function_tool.additional_parameters)
            if function_tool.add_params_to_result:
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
                function_tool = self.function_tools[function_call[self.tool_root]]
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
        else:
            for function_call in function_calls:
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
