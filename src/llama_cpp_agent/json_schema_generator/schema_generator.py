from inspect import isclass
from pydantic import BaseModel, Field
from enum import Enum
from typing import Union, get_origin, get_args, List, Set, Dict, Any
from types import NoneType

from llama_cpp_agent.llm_documentation import generate_text_documentation


def custom_json_schema(model: BaseModel):
    def get_type_str(annotation):
        """Resolve the JSON type string from the Python annotation."""
        basic_types = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean",
            NoneType: "null",
        }
        return basic_types.get(annotation, None)

    def refine_schema(schema, model, index):
        """Refine the generated schema based on the model's annotations and field details."""
        if "properties" in schema:
            for name, prop in schema["properties"].items():
                field = model.__fields__[name]
                prop["title"] = name.replace("_", " ").title()
                prop["description"] = field.description or ""

                # Handle Enums
                if isclass(field.annotation) and issubclass(field.annotation, Enum):
                    if "allOf" in prop:
                        prop.pop("allOf")
                    prop["enum"] = [e.value for e in field.annotation]
                    prop["type"] = get_type_str(
                        type(next(iter(field.annotation)).value)
                    )

                # Handle Unions, including Optional
                origin = get_origin(field.annotation)
                if origin is Union:
                    types = get_args(field.annotation)
                    new_anyof = []
                    for sub_type in types:
                        type_str = get_type_str(sub_type)
                        if sub_type is NoneType:
                            new_anyof.append({"type": type_str})
                        elif isclass(sub_type) and issubclass(sub_type, BaseModel):
                            new_anyof.append(
                                refine_schema(sub_type.schema(), sub_type, index + 1)
                            )
                        elif type_str:
                            new_anyof.append({"type": type_str})
                    prop["anyOf"] = new_anyof

                # Handle lists and sets containing Pydantic models or basic types
                elif origin in [list, set]:
                    item_type = get_args(field.annotation)[0]
                    if isclass(item_type) and issubclass(item_type, BaseModel):
                        prop["items"] = refine_schema(
                            item_type.schema(), item_type, index + 1
                        )
                    elif item_type is Any:
                        prop["items"] = {
                            "type": "object",
                            "anyOf": [
                                {"type": "boolean"},
                                {"type": "number"},
                                {"type": "null"},
                                {"type": "string"},
                            ],
                        }
                    else:
                        origin = get_origin(item_type)
                        if origin is Union:
                            types = get_args(item_type)
                            new_anyof = []
                            for sub_type in types:
                                type_str = get_type_str(sub_type)
                                if sub_type is NoneType:
                                    new_anyof.append({"type": type_str})
                                elif isclass(sub_type) and issubclass(
                                    sub_type, BaseModel
                                ):
                                    new_anyof.append(
                                        refine_schema(
                                            sub_type.schema(), sub_type, index + 1
                                        )
                                    )
                                elif type_str:
                                    new_anyof.append({"type": type_str})
                            prop["items"]["anyOf"] = new_anyof
                        else:
                            type_str = get_type_str(item_type)
                            if type_str:
                                prop["items"] = {"type": type_str}

                    prop["minItems"] = 1

                # Handle dictionaries
                elif origin is dict:
                    key_type, value_type = get_args(field.annotation)
                    key_type_str = get_type_str(key_type)
                    if isclass(value_type) and issubclass(value_type, BaseModel):
                        prop["additionalProperties"] = refine_schema(
                            value_type.schema(), value_type, index + 1
                        )
                    else:
                        value_type_str = get_type_str(value_type)
                        prop["additionalProperties"] = {"type": value_type_str}
                    prop["type"] = "object"

                # Handle nested Pydantic models
                elif isclass(field.annotation) and issubclass(
                    field.annotation, BaseModel
                ):
                    prop.update(
                        refine_schema(
                            field.annotation.schema(), field.annotation, index + 1
                        )
                    )

        new_schema = None
        if "properties" in schema:
            new_schema = {}
            count = 1
            for name, prop in schema["properties"].items():
                if name == "$defs":
                    continue
                else:
                    if "required" in prop:
                        new_schema["required"] = [
                            "{:03}".format(count) + "_" + name
                            for name in prop["required"]
                        ]
                    else:
                        new_schema["{:03}".format(count) + "_" + name] = prop
                    count += 1
            schema["properties"] = new_schema
            if "required" in schema:
                schema["required"] = [
                    "{:03}".format(idx + 1) + "_" + name
                    for idx, name in enumerate(schema["required"])
                ]
        schema["title"] = model.__name__
        schema["description"] = model.__doc__.strip() if model.__doc__ else ""

        if "$defs" in schema:
            schema.pop("$defs")
        return schema

    return refine_schema(model.schema(), model, 0)


def generate_list(
    models: List[BaseModel],
    outer_object_name=None,
    outer_object_properties_name=None,
    add_inner_thoughts: bool = False,
    inner_thoughts_name: str = "thoughts_and_reasoning",
    add_heartbeat: bool = False,
    heartbeat_name: str = "heartbeat",
    heartbeat_list: List[str] = None,
    min_items: int = 1,
    max_items: int = 1000,
):
    if max_items == 1:
        list_of_models = []
        for model in models:
            schema = custom_json_schema(model)
            if (
                outer_object_name is not None
                and outer_object_properties_name is not None
            ):
                function_name_object = {"enum": [model.__name__], "type": "string"}
                model_schema_object = schema
                if (add_inner_thoughts and not add_heartbeat) or (add_inner_thoughts and add_heartbeat and model.__name__ not in heartbeat_list):
                    # Create a wrapper object that contains the function name and the model schema
                    wrapper_object = {
                        "type": "object",
                        "properties": {
                            inner_thoughts_name: {"type": "string"},
                            outer_object_name: function_name_object,
                            outer_object_properties_name: model_schema_object,
                        },
                        "required": [
                            inner_thoughts_name,
                            outer_object_name,
                            outer_object_properties_name,
                        ],
                    }
                elif add_inner_thoughts and add_heartbeat and model.__name__ in heartbeat_list:
                    # Create a wrapper object that contains the function name and the model schema
                    wrapper_object = {
                        "type": "object",
                        "properties": {
                            inner_thoughts_name: {"type": "string"},
                            outer_object_name: function_name_object,
                            outer_object_properties_name: model_schema_object,
                            heartbeat_name: {"type": "boolean"},
                        },
                        "required": [
                            inner_thoughts_name,
                            outer_object_name,
                            outer_object_properties_name,
                            heartbeat_name
                        ],
                    }
                elif not add_inner_thoughts and add_heartbeat and model.__name__ in heartbeat_list:
                    # Create a wrapper object that contains the function name and the model schema
                    wrapper_object = {
                        "type": "object",
                        "properties": {
                            outer_object_name: function_name_object,
                            outer_object_properties_name: model_schema_object,
                            heartbeat_name: {"type": "boolean"},
                        },
                        "required": [
                            outer_object_name,
                            outer_object_properties_name,
                            heartbeat_name
                        ],
                    }
                else:
                    # Create a wrapper object that contains the function name and the model schema
                    wrapper_object = {
                        "type": "object",
                        "properties": {
                            outer_object_name: function_name_object,
                            outer_object_properties_name: model_schema_object,
                        },
                        "required": [
                            outer_object_name,
                            outer_object_properties_name,
                        ],
                    }
                list_of_models.append(wrapper_object)
        return {"type": "object", "anyOf": list_of_models}
    list_object = {"type": "array", "items": {"type": "object", "anyOf": []}}

    for model in models:
        schema = custom_json_schema(model)
        outer_object = {}

        if outer_object_name is not None and outer_object_properties_name is not None:
            function_name_object = {"enum": [model.__name__], "type": "string"}
            model_schema_object = schema

            if (add_inner_thoughts and not add_heartbeat) or (add_inner_thoughts and add_heartbeat and model.__name__ not in heartbeat_list):
                # Create a wrapper object that contains the function name and the model schema
                wrapper_object = {
                    "type": "object",
                    "properties": {
                        inner_thoughts_name: {"type": "string"},
                        outer_object_name: function_name_object,
                        outer_object_properties_name: model_schema_object,
                    },
                    "required": [
                        inner_thoughts_name,
                        outer_object_name,
                        outer_object_properties_name,
                    ],
                }

            elif add_inner_thoughts and add_heartbeat and model.__name__ in heartbeat_list:
                # Create a wrapper object that contains the function name and the model schema
                wrapper_object = {
                    "type": "object",
                    "properties": {
                        inner_thoughts_name: {"type": "string"},
                        outer_object_name: function_name_object,
                        outer_object_properties_name: model_schema_object,
                        heartbeat_name: {"type": "boolean"},
                    },
                    "required": [
                        inner_thoughts_name,
                        outer_object_name,
                        outer_object_properties_name,
                        heartbeat_name
                    ],
                }

            elif not add_inner_thoughts and add_heartbeat and model.__name__ in heartbeat_list:
                # Create a wrapper object that contains the function name and the model schema
                wrapper_object = {
                    "type": "object",
                    "properties": {
                        outer_object_name: function_name_object,
                        outer_object_properties_name: model_schema_object,
                        heartbeat_name: {"type": "boolean"},
                    },
                    "required": [
                        outer_object_name,
                        outer_object_properties_name,
                        heartbeat_name
                    ],
                }
            else:
                # Create a wrapper object that contains the function name and the model schema
                wrapper_object = {
                    "type": "object",
                    "properties": {
                        outer_object_name: function_name_object,
                        outer_object_properties_name: model_schema_object,
                    },
                    "required": [
                        outer_object_name,
                        outer_object_properties_name,
                    ],
                }

            outer_object.update(wrapper_object)
        else:
            outer_object = schema
        list_object["items"]["anyOf"].append(outer_object)
        list_object["minItems"] = min_items

    return list_object


def generate_json_schemas(
    models: List[BaseModel],
    outer_object_name=None,
    outer_object_properties_name=None,
    allow_list=False,
    add_inner_thoughts: bool = False,
    inner_thoughts_name: str = "thoughts_and_reasoning",
    add_heartbeat: bool = False,
    heartbeat_name: str = "heartbeat",
    heartbeat_list=None,
):
    if heartbeat_list is None:
        heartbeat_list = []
    if allow_list:
        model_schema_list = generate_list(
            models,
            outer_object_name,
            outer_object_properties_name,
            add_inner_thoughts,
            inner_thoughts_name,
            add_heartbeat=add_heartbeat,
            heartbeat_name=heartbeat_name,
            heartbeat_list=heartbeat_list
        )
    else:
        model_schema_list = generate_list(
            models,
            outer_object_name,
            outer_object_properties_name,
            add_inner_thoughts,
            inner_thoughts_name,
            add_heartbeat=add_heartbeat,
            heartbeat_name=heartbeat_name,
            heartbeat_list=heartbeat_list,
            max_items=1,
        )
    return model_schema_list
