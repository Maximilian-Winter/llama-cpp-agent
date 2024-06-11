import json
from enum import Enum
from inspect import isclass, getdoc
from types import UnionType, GenericAlias
from typing import get_args, get_origin, Union, Any

from pydantic import BaseModel


def generate_markdown_documentation(
    pydantic_models: list[type[BaseModel]],
    model_prefix="Model",
    fields_prefix="Fields",
    documentation_with_field_description=True,
    ordered_json_mode=False,
) -> str:
    """
    Generate markdown documentation for a list of Pydantic models.

    Args:
        pydantic_models (list[type[BaseModel]]): list of Pydantic model classes.
        model_prefix (str): Prefix for the model section.
        fields_prefix (str): Prefix for the fields section.
        documentation_with_field_description (bool): Include field descriptions in the documentation.

    Returns:
        str: Generated text documentation.
    """
    documentation = ""
    pyd_models = [(model, True) for model in pydantic_models]
    for model, add_prefix in pyd_models:
        if add_prefix:
            documentation += f"### {model_prefix} '{model.__name__}'\n"
        else:
            documentation += f"### '{model.__name__}'\n"

        # Handling multi-line model description with proper indentation

        class_doc = getdoc(model)
        base_class_doc = getdoc(BaseModel)
        class_description = (
            class_doc if class_doc and class_doc != base_class_doc else ""
        )
        if class_description != "":
            documentation += format_multiline_description(class_description, 0) + "\n"

        if add_prefix:
            # Indenting the fields section
            documentation += f"#### {fields_prefix}\n"
        else:
            documentation += f"#### Fields\n"
        if isclass(model) and issubclass(model, BaseModel):
            count = 1
            for name, field_type in model.__annotations__.items():
                # if name == "markdown_code_block":
                #    continue
                if get_origin(field_type) == list:
                    element_type = get_args(field_type)[0]
                    if isclass(element_type) and issubclass(element_type, BaseModel):
                        pyd_models.append((element_type, False))
                if get_origin(field_type) == Union:
                    element_types = get_args(field_type)
                    for element_type in element_types:
                        if isclass(element_type) and issubclass(
                            element_type, BaseModel
                        ):
                            pyd_models.append((element_type, False))
                documentation += generate_field_markdown(
                    name
                    if not ordered_json_mode
                    else "{:03}".format(count) + "_" + name,
                    field_type,
                    model,
                    documentation_with_field_description=documentation_with_field_description,
                )
                count += 1
            if add_prefix:
                if documentation.endswith(f"#### {fields_prefix}\n"):
                    documentation += "none\n"
            else:
                if documentation.endswith("#### Fields\n"):
                    documentation += "none\n"
            documentation += "\n"

        if (
            hasattr(model, "Config")
            and hasattr(model.Config, "json_schema_extra")
            and "example" in model.Config.json_schema_extra
        ):
            documentation += f"  Expected Example Output for {model.__name__}:\n"
            json_example = json.dumps(model.Config.json_schema_extra["example"])
            documentation += format_multiline_description(json_example, 2) + "\n"

    return documentation


def generate_field_markdown(
    field_name: str,
    field_type: type[Any],
    model: type[BaseModel],
    depth=1,
    documentation_with_field_description=True,
) -> str:
    """
    Generate markdown documentation for a Pydantic model field.

    Args:
        field_name (str): Name of the field.
        field_type (type[Any]): Type of the field.
        model (type[BaseModel]): Pydantic model class.
        depth (int): Indentation depth in the documentation.
        documentation_with_field_description (bool): Include field descriptions in the documentation.

    Returns:
        str: Generated text documentation for the field.
    """
    indent = ""

    field_info = model.model_fields.get(field_name)
    field_description = (
        field_info.description if field_info and field_info.description else ""
    )
    is_enum = False
    enum_values = None
    if get_origin(field_type) == list:
        element_type = get_args(field_type)[0]
        field_text = (
            f"{indent}{field_name} ({field_type.__name__} of {element_type.__name__})"
        )
        if field_description != "":
            field_text += ": "
        else:
            field_text += "\n"
    elif get_origin(field_type) == Union or isinstance(field_type, UnionType):
        element_types = get_args(field_type)
        types = []
        for element_type in element_types:
            if element_type.__name__ == "NoneType":
                types.append("null")
            else:
                types.append(element_type.__name__)
        field_text = f"{indent}{field_name} ({' or '.join(types)})"
        if field_description != "":
            field_text += ": "
        else:
            field_text += "\n"

    elif issubclass(field_type, Enum):
        enum_values = [f"'{str(member.value)}'" for member in field_type]
        is_enum = True
        field_text = f"{indent}{field_name}"
        if field_description != "":
            field_text += ": "
        else:
            field_text += "\n"

    else:
        field_text = f"{indent}{field_name} ({field_type.__name__})"
        if field_description != "":
            field_text += ": "
        else:
            field_text += "\n"

    if not documentation_with_field_description:
        return field_text

    if is_enum:

        field_text = field_text.strip() + field_description.strip() + f" Can be one of the following values: {' or '.join(enum_values)}" + "\n"
    elif field_description != "":
        field_text += field_description + "\n"


    # Check for and include field-specific examples if available
    if (
        hasattr(model, "Config")
        and hasattr(model.Config, "json_schema_extra")
        and "example" in model.Config.json_schema_extra
    ):
        field_example = model.Config.json_schema_extra["example"].get(field_name)
        if field_example is not None:
            example_text = (
                f"'{field_example}'"
                if isinstance(field_example, str)
                else field_example
            )
            field_text += f"{indent}  Example: {example_text}\n"

    if isclass(field_type) and issubclass(field_type, BaseModel):
        field_text += f"{indent}  Details:\n"
        for name, type_ in field_type.__annotations__.items():
            field_text += generate_field_markdown(name, type_, field_type, depth + 2)

    return field_text


def format_json_example(example: dict[str, Any], depth: int) -> str:
    """
    Format a JSON example into a readable string with indentation.

    Args:
        example (dict): JSON example to be formatted.
        depth (int): Indentation depth.

    Returns:
        str: Formatted JSON example string.
    """
    indent = "    " * depth
    formatted_example = "{\n"
    for key, value in example.items():
        value_text = f"'{value}'" if isinstance(value, str) else value
        formatted_example += f"{indent}{key}: {value_text},\n"
    formatted_example = formatted_example.rstrip(",\n") + "\n" + indent + "}"
    return formatted_example


def generate_text_documentation(
    pydantic_models: list[BaseModel],
    model_prefix="Output Model",
    fields_prefix="Fields",
    documentation_with_field_description=True,
    ordered_json_mode=False,
) -> str:
    """
    Generate markdown documentation for a list of Pydantic models.

    Args:
        pydantic_models (list[type[BaseModel]]): list of Pydantic model classes.
        model_prefix (str): Prefix for the model section.
        fields_prefix (str): Prefix for the fields section.
        documentation_with_field_description (bool): Include field descriptions in the documentation.
        ordered_json_mode (bool): Add ordering prefix for JSON schemas
    Returns:
        str: Generated text documentation.
    """
    documentation = ""
    pyd_models = [(model, True) for model in pydantic_models]
    for model, add_prefix in pyd_models:
        if add_prefix:
            documentation += f"{model_prefix}: {model.__name__}\n"
        else:
            documentation += f"Model: {model.__name__}\n"

        # Handling multi-line model description with proper indentation

        class_doc = getdoc(model)
        base_class_doc = getdoc(BaseModel)
        class_description = (
            class_doc if class_doc and class_doc != base_class_doc else ""
        )
        if class_description != "":
            documentation += "  Description: "
            documentation += format_multiline_description(class_description, 2) + "\n"

        if add_prefix:
            # Indenting the fields section
            documentation += f"  {fields_prefix}:\n"
        else:
            documentation += f"  Fields:\n"
        if isclass(model) and issubclass(model, BaseModel):
            count = 1
            for name, field_type in model.__annotations__.items():
                # if name == "markdown_code_block":
                #    continue
                if get_origin(field_type) == list:
                    element_type = get_args(field_type)[0]
                    if isclass(element_type) and issubclass(element_type, BaseModel):
                        pyd_models.append((element_type, False))
                    if get_origin(element_type) == Union or isinstance(
                        element_type, UnionType
                    ):
                        element_types = get_args(element_type)
                        for element_type in element_types:
                            if isclass(element_type) and issubclass(
                                element_type, BaseModel
                            ):
                                pyd_models.append((element_type, False))
                            if get_origin(element_type) == list:
                                element_type = get_args(element_type)[0]
                                if isclass(element_type) and issubclass(
                                    element_type, BaseModel
                                ):
                                    pyd_models.append((element_type, False))
                if get_origin(field_type) == Union or isinstance(field_type, UnionType):
                    element_types = get_args(field_type)
                    for element_type in element_types:
                        if isclass(element_type) and issubclass(
                            element_type, BaseModel
                        ):
                            pyd_models.append((element_type, False))
                        if get_origin(element_type) == list:
                            element_type = get_args(element_type)[0]
                            if isclass(element_type) and issubclass(
                                element_type, BaseModel
                            ):
                                pyd_models.append((element_type, False))
                if isclass(field_type) and issubclass(field_type, BaseModel):
                    pyd_models.append((field_type, False))
                documentation += generate_field_text(
                    name if not ordered_json_mode
                    else "{:03}".format(count) + "_" + name,
                    name,
                    field_type,
                    model,
                    documentation_with_field_description=documentation_with_field_description,
                )
                count += 1
            if add_prefix:
                if documentation.endswith(f"{fields_prefix}:\n"):
                    documentation += "    none\n"
            else:
                if documentation.endswith("fields:\n"):
                    documentation += "    none\n"
            documentation += "\n"

        if (
            hasattr(model, "Config")
            and hasattr(model.Config, "json_schema_extra")
            and "example" in model.Config.json_schema_extra
        ):
            documentation += f"  Expected Example Output for {model.__name__}:\n"
            json_example = json.dumps(model.Config.json_schema_extra["example"])
            documentation += format_multiline_description(json_example, 2) + "\n"

    return documentation


def generate_field_text(
    field_name: str,
    field_real_name: str,
    field_type: type[Any],
    model: type[BaseModel],
    depth=1,
    documentation_with_field_description=True,
) -> str:
    """
    Generate markdown documentation for a Pydantic model field.

    Args:
        field_name (str): Output Name of the field.
        field_real_name (str): Real Name of the field.:
        field_type (type[Any]): Type of the field.

        model (type[BaseModel]): Pydantic model class.
        depth (int): Indentation depth in the documentation.
        documentation_with_field_description (bool): Include field descriptions in the documentation.

    Returns:
        str: Generated text documentation for the field.

    """
    indent = "    " * depth

    field_info = model.model_fields.get(field_real_name)
    field_description = (
        field_info.description if field_info and field_info.description else ""
    )
    field_text = ""
    is_enum = False
    if get_origin(field_type) == list:
        element_type = get_args(field_type)[0]
        if get_origin(element_type) == Union or isinstance(element_type, UnionType):
            element_types = get_args(element_type)
            types = []
            for element_type in element_types:
                if element_type.__name__ == "NoneType":
                    types.append("null")
                else:
                    if isclass(element_type) and issubclass(element_type, Enum):
                        enum_values = [
                            f"'{str(member.value)}'" for member in element_type
                        ]
                        for enum_value in enum_values:
                            types.append(enum_value)

                    else:
                        if get_origin(element_type) == list:
                            element_type = get_args(element_type)[0]
                            types.append(f"(list of {element_type.__name__})")
                        else:
                            types.append(element_type.__name__)
            field_text = f"({' or '.join(types)})"
            field_text = f"{indent}{field_name} ({field_type.__name__} of {field_text})"
            if field_description != "":
                field_text += ": "
            else:
                field_text += "\n"
        else:
            field_text = f"{indent}{field_name} ({field_type.__name__} of {element_type.__name__})"
            if field_description != "":
                field_text += ": "
            else:
                field_text += "\n"
    elif get_origin(field_type) == Union:
        element_types = get_args(field_type)
        types = []
        for element_type in element_types:
            if element_type.__name__ == "NoneType":
                types.append("null")
            else:
                if isclass(element_type) and issubclass(element_type, Enum):
                    enum_values = [f"'{str(member.value)}'" for member in element_type]
                    for enum_value in enum_values:
                        types.append(enum_value)

                else:
                    if get_origin(element_type) == list:
                        element_type = get_args(element_type)[0]
                        types.append(f"(list of {element_type.__name__})")
                    else:
                        types.append(element_type.__name__)
        field_text = f"{indent}{field_name} ({' or '.join(types)})"
        if field_description != "":
            field_text += ": "
        else:
            field_text += "\n"
    elif isinstance(field_type, GenericAlias):
        if field_type.__origin__ == dict:
            key_type, value_type = get_args(field_type)

            additional_key_type = key_type.__name__
            additional_value_type = value_type.__name__
            field_text = f"{indent}{field_name} (dict of {additional_key_type} to {additional_value_type})"
            if field_description != "":
                field_text += ": "
            else:
                field_text += "\n"
        elif field_type.__origin__ == list:
            element_type = get_args(field_type)[0]
            field_text = f"{indent}{field_name} (list of {element_type.__name__})"
            if field_description != "":
                field_text += ": "
            else:
                field_text += "\n"
    elif issubclass(field_type, Enum):
        enum_values = [f"'{str(member.value)}'" for member in field_type]
        is_enum = True
        field_text = f"{indent}{field_name} (enum)"
        if field_description != "":
            field_text += ": "
        else:
            field_text += "\n"

    else:
        field_text = f"{indent}{field_name} ({field_type.__name__})"
        if field_description != "":
            field_text += ": "
        else:
            field_text += "\n"

    if not documentation_with_field_description:
        return field_text

    if is_enum:

        field_text = field_text + field_description.strip() + f" Can be one of the following values: {' or '.join(enum_values)}" + "\n"
    elif field_description != "":
        field_text += field_description + "\n"

    # Check for and include field-specific examples if available
    if (
        hasattr(model, "Config")
        and hasattr(model.Config, "json_schema_extra")
        and "example" in model.Config.json_schema_extra
    ):
        field_example = model.Config.json_schema_extra["example"].get(field_real_name)
        if field_example is not None:
            example_text = (
                f"'{field_example}'"
                if isinstance(field_example, str)
                else field_example
            )
            field_text += f"{indent}  Example: {example_text}\n"

    return field_text


def format_multiline_description(description: str, indent_level: int) -> str:
    """
    Format a multiline description with proper indentation.

    Args:
        description (str): Multiline description.
        indent_level (int): Indentation level.

    Returns:
        str: Formatted multiline description.
    """
    indent = "  " * indent_level
    return description.replace("\n", "\n" + indent)
