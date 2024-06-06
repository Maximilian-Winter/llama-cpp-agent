import datetime
import json


class CoreMemoryManager:
    def __init__(self, core_memory: dict):
        self.core_memory = core_memory
        self.last_modified = "Never"

    def add_to_core_memory(self, key: str, child_key: str, value) -> str:
        """
        Adds or updates an entry in the core memory.
        """

        if key not in self.core_memory:
            self.core_memory[key] = {}
        self.core_memory[key][child_key] = value
        self.last_modified = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        return f"Core memory updated. Key: {key}, Child Key: {child_key}"

    def replace_in_core_memory(self, key: str, child_key: str, new_value) -> str:
        """
        Replaces an existing entry in the core memory.
        """

        if key in self.core_memory and child_key in self.core_memory[key]:
            self.core_memory[key][child_key] = new_value
            self.last_modified = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
            return f"Core memory replaced. Key: {key}, Child Key: {child_key}"
        else:
            return "Key or child key not found in Core memory."

    def remove_from_core_memory(self, key: str, child_key: str) -> str:
        """
        Removes a specific field from a core memory entry.
        """

        if key in self.core_memory and child_key in self.core_memory[key]:
            del self.core_memory[key][child_key]
            self.last_modified = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
            return f"Core memory entry removed. Key: {key}, Child Key: {child_key}"
        else:
            return "Key or child key not found in Core memory."

    def build_core_memory_context(self):
        context = ""
        for key, item in self.core_memory.items():
            context += f"""<{key}>\n"""
            for key2, item2 in item.items():
                context += f"""  <{key2}>{self.format_multiline_description(item2.strip(), 2)}</{key2}>\n"""
            context += f"</{key}>\n"
        if context == "":
            context = "No Core Memories!"

        return context

    def format_multiline_description(self, description: str, indent_level: int) -> str:
        """
        Format a multiline description with proper indentation.

        Args:
            description (str): Multiline description.
            indent_level (int): Indentation level.

        Returns:
            str: Formatted multiline description.
        """
        indent = " " * indent_level
        return description.replace("\n", "\n" + indent)

    def load(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            self.core_memory = json.load(file)
        self.last_modified = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

    def save(self, filepath):
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(self.core_memory, file, indent=4)
