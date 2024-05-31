import re
from dataclasses import dataclass
from typing import List, Dict, Union


@dataclass
class PromptTemplateField:
    """
    Data class representing a field in a prompt template.

    Attributes:
        name (str): The name of the template field.
        value (str): The value associated with the template field.
    """

    name: str
    value: str


class PromptTemplateFields:
    """
    Class representing a collection of PromptTemplateField objects.

    Methods:
        add_field(name: str, value: str): Add a new field to the collection.
        remove_field(name: str): Remove a field by name from the collection.
        edit_field(name: str, new_value: str): Edit the value of an existing field.
        find_field(name: str) -> PromptTemplateField: Find and return a field by name.
        list_fields() -> List[PromptTemplateField]: Get a list of all fields in the collection.
        get_fields_dict() -> Dict[str, str]: Get a dictionary representation of the fields.
        set_fields_from_dict(field_dict: Dict[str, str]): Set the fields using a dictionary.

    Attributes:
        fields (List[PromptTemplateField]): List of PromptTemplateField objects.
    """

    def __init__(self):
        self.fields: List[PromptTemplateField] = []

    def add_field(self, name: str, value: str):
        """Add a new field to the collection."""
        self.fields.append(PromptTemplateField(name, value))

    def remove_field(self, name: str):
        """Remove a field by name from the collection."""
        self.fields = [field for field in self.fields if field.name != name]

    def edit_field(self, name: str, new_value: str):
        """Edit the value of an existing field."""
        field = self.find_field(name)
        if field:
            field.value = new_value
        else:
            raise ValueError(f"Field '{name}' not found.")

    def find_field(self, name: str) -> PromptTemplateField:
        """Find and return a field by name."""
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def list_fields(self):
        """Get a list of all fields in the collection."""
        return self.fields

    def get_fields_dict(self) -> Dict[str, str]:
        """Get a dictionary representation of the fields."""
        return {field.name: field.value for field in self.fields}

    def set_fields_from_dict(self, field_dict: Dict[str, str]):
        """Set the fields using a dictionary."""
        self.fields.clear()
        for name, value in field_dict.items():
            self.add_field(name, value)


class PromptTemplate:
    """
    Class representing a prompt template.

    Methods:
        generate_prompt(template_fields: Union[dict, PromptTemplateFields], remove_empty_template_field=True) -> str:
        Generate a prompt by replacing placeholders in the template with values.

    Class Methods:
        from_string(template_string: str) -> PromptTemplate:
        Create a PromptTemplate from a string.
        from_file(template_file: str) -> PromptTemplate:
        Create a PromptTemplate from a file.

    Attributes:
        template (str): The template string containing placeholders.
    """

    def __init__(self, template_file=None, template_string=None):
        """
        Initialize a PromptTemplate instance.

        Args:
            template_file (str): The path to a file containing the template.
            template_string (str): The template string.
        """
        if template_file:
            with open(template_file, "r") as file:
                self.template = file.read()
        elif template_string:
            self.template = template_string
        else:
            raise ValueError(
                "Either 'template_file' or 'template_string' must be provided"
            )

    @classmethod
    def from_string(cls, template_string):
        """
        Create a PromptTemplate instance from a string.

        Args:
            template_string (str): The template string.

        Returns:
            PromptTemplate: Created PromptTemplate instance.
        """
        return cls(template_string=template_string)

    @classmethod
    def from_file(cls, template_file):
        """
        Create a PromptTemplate instance from a file.

        Args:
            template_file (str): The path to a file containing the template.

        Returns:
            PromptTemplate: Created PromptTemplate instance.
        """
        with open(template_file, "r") as file:
            template_string = file.read()
        return cls(template_string=template_string)

    def _remove_empty_placeholders(self, text):
        """
        Remove lines that contain only the empty placeholder.

        Args:
            text (str): The text containing placeholders.

        Returns:
            str: Text with empty placeholders removed.
        """
        # Split text into lines
        lines = text.split('\n')
        # Process each line individually
        processed_lines = []
        for line in lines:
            if '__EMPTY_TEMPLATE_FIELD__' in line:
                new_line = line.replace('__EMPTY_TEMPLATE_FIELD__', '')
                if new_line.strip():
                    processed_lines.append(new_line)
            else:
                processed_lines.append(line)
        # Join the lines back into a single string
        return '\n'.join(processed_lines)

    def generate_prompt(
        self,
        template_fields: Union[dict, PromptTemplateFields],
        remove_empty_template_field=True,
    ) -> str:
        """
        Generate a prompt by replacing placeholders in the template with values.

        Args:
            template_fields (Union[dict, PromptTemplateFields]): The template fields.
            remove_empty_template_field (bool): If True, removes lines with empty placeholders.

        Returns:
            str: The generated prompt.
        """
        cleaned_fields = {}
        for key, value in template_fields.items():
            cleaned_fields[key] = str(value) if not isinstance(value, str) else value

        template_fields = cleaned_fields
        if isinstance(template_fields, PromptTemplateFields):
            template_fields = template_fields.get_fields_dict()

        if not remove_empty_template_field:

            def replace_placeholder(match):
                placeholder = match.group(1)
                return template_fields.get(placeholder, match.group(0))

            prompt = re.sub(r"\{(\w+)\}", replace_placeholder, self.template)
            return prompt

        def replace_placeholder(match):
            placeholder = match.group(1)
            if template_fields.get(placeholder, match.group(0)) != "":
                return template_fields.get(placeholder, match.group(0))
            return "__EMPTY_TEMPLATE_FIELD__"

        # Initial placeholder replacement
        prompt = re.sub(r"\{(\w+)\}", replace_placeholder, self.template)

        return self._remove_empty_placeholders(prompt)
