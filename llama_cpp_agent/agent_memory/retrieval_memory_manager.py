from .retrieval_memory import RetrievalMemory


class RetrievalMemoryManager:
    def __init__(self, retrieval_memory: RetrievalMemory):
        self.retrieval_memory = retrieval_memory

    def add_memory_to_retrieval(self, description: str, importance: float = 1.0) -> str:
        """
        Adds a memory with a given description and importance to the memory stream.
        """
        self.retrieval_memory.add_memory(description, importance=importance)
        return f"Memory with description '{description}' added to the stream."

    def retrieve_memories(self, query: str, max_results: int = 5) -> str:
        """
        Retrieves memories from the memory stream based on a query.
        """
        memories = self.retrieval_memory.retrieve_memories(query, max_results)
        formatted_memories = "\n".join([str(memory) for memory in memories])
        return formatted_memories if formatted_memories else "No memories found matching the query."
