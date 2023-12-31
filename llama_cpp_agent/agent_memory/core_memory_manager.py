import json

from pydantic import BaseModel, Field


class CoreMemoryManager:
    def __init__(self, core_memory: dict):
        self.core_memory = core_memory

    def add_to_core_memory(self, key: str, child_key: str, value) -> str:
        """
        Adds or updates an entry in the core memory.
        """
        if key not in self.core_memory:
            self.core_memory[key] = {}
        self.core_memory[key][child_key] = value
        return f"Core memory updated. Key: {key}, Child Key: {child_key}"

    def replace_in_core_memory(self, key: str, child_key: str, new_value) -> str:
        """
        Replaces an existing entry in the core memory.
        """
        if key in self.core_memory and child_key in self.core_memory[key]:
            self.core_memory[key][child_key] = new_value
            return f"Core memory replaced. Key: {key}, Child Key: {child_key}"
        else:
            return "Key or child key not found in core memory."

    def remove_from_core_memory(self, key: str, child_key: str) -> str:
        """
        Removes a specific field from a core memory entry.
        """
        if key in self.core_memory and child_key in self.core_memory[key]:
            del self.core_memory[key][child_key]
            return f"Core memory entry removed. Key: {key}, Child Key: {child_key}"
        else:
            return "Key or child key not found in core memory."

    def build_core_memory_context(self):
        output = json.dumps(self.core_memory, indent=4)
        context = f"# Core-Memory:\n{output if output != '{}' else 'Empty'}"
        return context


class AddCoreMemoryModel(BaseModel):
    """
    Add a new entry to the core memory.
    """
    key: str = Field(..., description="The key identifier for the core memory entry.")
    field: str = Field(..., description="A secondary key or field within the core memory entry.")
    value: str = Field(..., description="The value or data to be stored in the specified core memory entry.")

    def run(self, core_memory_manager: CoreMemoryManager):
        return core_memory_manager.add_to_core_memory(self.key, self.field, self.value)


# Replace Core Memory Model
class ReplaceCoreMemoryModel(BaseModel):
    """
    Replace an entry in the core memory.
    """
    key: str = Field(..., description="The key identifier for the core memory entry.")
    field: str = Field(..., description="The specific field within the core memory entry to be replaced.")
    new_value: str = Field(...,
                           description="The new value to replace the existing data in the specified core memory field.")

    def run(self, core_memory_manager: CoreMemoryManager):
        return core_memory_manager.replace_in_core_memory(self.key, self.field, self.value)


class RemoveCoreMemoryModel(BaseModel):
    """
    Remove an entry in the core memory.
    """
    key: str = Field(..., description="The key identifier for the core memory entry to be removed.")
    field: str = Field(..., description="The specific field within the core memory entry to be removed.")

    def run(self, core_memory_manager: CoreMemoryManager):
        return core_memory_manager.remove_from_core_memory(self.key, self.field)
