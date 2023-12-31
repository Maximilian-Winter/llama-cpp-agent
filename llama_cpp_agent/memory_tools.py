from pydantic import BaseModel, Field

from .function_call_tools import LlamaCppFunctionTool
from .agent_memory.core_memory_manager import CoreMemoryManager
from .agent_memory.retrieval_memory_manager import RetrievalMemoryManager, RetrievalMemory


class AddCoreMemory(BaseModel):
    """
    Add a new entry to the core memory.
    """
    key: str = Field(..., description="The key identifier for the core memory entry.")
    field: str = Field(..., description="A secondary key or field within the core memory entry.")
    value: str = Field(..., description="The value or data to be stored in the specified core memory entry.")

    def run(self, core_memory_manager: CoreMemoryManager):
        return core_memory_manager.add_to_core_memory(self.key, self.field, self.value)


# Replace Core Memory Model
class ReplaceCoreMemory(BaseModel):
    """
    Replace an entry in the core memory.
    """
    key: str = Field(..., description="The key identifier for the core memory entry.")
    field: str = Field(..., description="The specific field within the core memory entry to be replaced.")
    new_value: str = Field(...,
                           description="The new value to replace the existing data in the specified core memory field.")

    def run(self, core_memory_manager: CoreMemoryManager):
        return core_memory_manager.replace_in_core_memory(self.key, self.field, self.value)


class RemoveCoreMemory(BaseModel):
    """
    Remove an entry in the core memory.
    """
    key: str = Field(..., description="The key identifier for the core memory entry to be removed.")
    field: str = Field(..., description="The specific field within the core memory entry to be removed.")

    def run(self, core_memory_manager: CoreMemoryManager):
        return core_memory_manager.remove_from_core_memory(self.key, self.field)


class RetrieveMemories(BaseModel):
    """
    Retrieve memories from the retrieval memory based on a query.
    """
    query: str = Field(..., description="The query to be used to retrieve memories from the retrieval memory.")

    def run(self, retrieval_memory_manager: RetrievalMemoryManager):
        return retrieval_memory_manager.retrieve_memories(self.query)


class AddRetrievalMemory(BaseModel):
    """
    Add memory to the retrieval memory.
    """
    memory: str = Field(..., description="The memory to be added to the retrieval memory.")
    importance: float = Field(..., description="The importance of the memory to be added to the retrieval memory.")

    def run(self, retrieval_memory_manager: RetrievalMemoryManager):
        return retrieval_memory_manager.add_memory_to_retrieval(self.memory, self.importance)


class AgentRetrievalMemory:
    def __init__(self, persistent_db_path="./retrieval_memory", embedding_model_name="all-MiniLM-L6-v2",
                 collection_name="retrieval_memory_collection"):
        self.retrieval_memory = RetrievalMemory(persistent_db_path, embedding_model_name, collection_name)
        self.retrieval_memory_manager = RetrievalMemoryManager(self.retrieval_memory)
        self.retrieve_memories_tool = LlamaCppFunctionTool(RetrieveMemories,
                                                           retrieval_memory_manager=self.retrieval_memory_manager)
        self.add_retrieval_memory_tool = LlamaCppFunctionTool(AddRetrievalMemory,
                                                              retrieval_memory_manager=self.retrieval_memory_manager)

    def get_tool_list(self):
        return [self.add_retrieval_memory_tool, self.retrieve_memories_tool]

    def get_retrieve_memories_tool(self):
        return self.retrieve_memories_tool

    def get_add_retrieval_memory_tool(self):
        return self.add_retrieval_memory_tool


class AgentCoreMemory:
    def __init__(self, core_memory=None):
        if core_memory is None:
            core_memory = {}
        self.core_memory_manager = CoreMemoryManager(core_memory)
        self.add_core_memory_tool = LlamaCppFunctionTool(AddCoreMemory,
                                                         core_memory_manager=self.core_memory_manager)
        self.replace_core_memory_tool = LlamaCppFunctionTool(ReplaceCoreMemory,
                                                             core_memory_manager=self.core_memory_manager)
        self.remove_core_memory_tool = LlamaCppFunctionTool(RemoveCoreMemory,
                                                            core_memory_manager=self.core_memory_manager)

    def get_core_memory_manager(self):
        return self.core_memory_manager

    def get_tool_list(self):
        return [self.add_core_memory_tool, self.replace_core_memory_tool, self.remove_core_memory_tool]

    def get_add_core_memory_tool(self):
        return self.add_core_memory_tool

    def get_replace_core_memory_tool(self):
        return self.replace_core_memory_tool

    def get_remove_core_memory_tool(self):
        return self.remove_core_memory_tool

    def save_core_memory(self, file_path):
        self.core_memory_manager.save(file_path)

    def load_core_memory(self, file_path):
        self.core_memory_manager.load(file_path)