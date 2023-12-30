import re
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class PromptTemplateField:
    name: str
    value: str


class PromptTemplateFields:
    def __init__(self):
        self.fields: List[PromptTemplateField] = []

    def add_field(self, name: str, value: str):
        self.fields.append(PromptTemplateField(name, value))

    def remove_field(self, name: str):
        self.fields = [field for field in self.fields if field.name != name]

    def edit_field(self, name: str, new_value: str):
        field = self.find_field(name)
        if field:
            field.value = new_value
        else:
            raise ValueError(f"Field '{name}' not found.")

    def find_field(self, name: str) -> PromptTemplateField:
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def list_fields(self):
        return self.fields

    def get_fields_dict(self) -> Dict[str, str]:
        return {field.name: field.value for field in self.fields}

    def set_fields_from_dict(self, field_dict: Dict[str, str]):
        self.fields.clear()
        for name, value in field_dict.items():
            self.add_field(name, value)


class Prompter:
    def __init__(self, template_file=None, template_string=None):
        if template_file:
            with open(template_file, "r") as file:
                self.template = file.read()
        elif template_string:
            self.template = template_string
        else:
            raise ValueError("Either 'template_file' or 'template_string' must be provided")

    @classmethod
    def from_string(cls, template_string):
        return cls(template_string=template_string)

    @classmethod
    def from_file(cls, template_file):
        with open(template_file, "r") as file:
            template_string = file.read()
        return cls(template_string=template_string)

    def generate_prompt(self, template_fields):
        def replace_placeholder(match):
            placeholder = match.group(1)
            return template_fields.get(placeholder, match.group(0))

        prompt = re.sub(r"\{(\w+)}", replace_placeholder, self.template)
        return prompt
