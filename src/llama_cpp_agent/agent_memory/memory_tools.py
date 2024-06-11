from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

from .event_memory import Base
from .event_memory_manager import EventMemoryManager
from ..chat_history.messages import Roles
from ..function_calling import LlamaCppFunctionTool
from .core_memory_manager import CoreMemoryManager
from .retrieval_memory_manager import RetrievalMemoryManager, RetrievalMemory


class CoreMemoryKey(Enum):
    PERSONA: str = "persona"
    HUMAN: str = "human"


class conversation_search(BaseModel):
    """
    Search prior conversation history using case-insensitive string matching.
    """

    keywords: List[str] = Field(
        ...,
        description='Keywords that the messages have to contain. Eg. ["hello", "world"]',
    )
    page: Optional[int] = Field(
        ...,
        description="Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).",
    )

    def run(self, event_memory_manager: EventMemoryManager):
        parsed_start_datetime = None
        parsed_end_datetime = None
        if self.page is None:
            self.page = 0
        return event_memory_manager.query_events(
            event_types=[
                Roles.user,
                Roles.assistant,
                Roles.system,
                Roles.tool,
            ],
            content_keywords=self.keywords,
            start_date=parsed_start_datetime,
            end_date=parsed_end_datetime,
            page=self.page,
        )


class conversation_search_date(BaseModel):
    """
    Search prior conversation history using a date range.
    """

    start_date: str = Field(
        ...,
        description='The start of the date range to search, in the format "dd/mm/YYYY, H:M:S" eg. "01/01/2024, 08:00:30"',
    )
    end_date: str = Field(
        ...,
        description='The end of the date range to search, in the format "dd/mm/YYYY, H:M:S" eg. "04/02/2024, 18:57:29"',
    )
    page: Optional[int] = Field(
        ...,
        description="Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).",
    )

    def run(self, event_memory_manager: EventMemoryManager):
        parsed_start_datetime = None
        parsed_end_datetime = None
        if self.start_date:
            parsed_start_datetime = datetime.strptime(
                self.start_date, "%d/%m/%Y, %H:%M:%S"
            )
        if self.end_date:
            parsed_end_datetime = datetime.strptime(self.end_date, "%d/%m/%Y, %H:%M:%S")
        if self.page is None:
            self.page = 0

        return event_memory_manager.query_events(
            event_types=[
                Roles.user,
                Roles.assistant,
                Roles.system,
                Roles.tool,
            ],
            start_date=parsed_start_datetime,
            end_date=parsed_end_datetime,
            page=self.page,
        )


class archival_memory_search(BaseModel):
    """
    Search archival memory using semantic (embedding-based) search.
    """

    query: str = Field(
        ...,
        description="String to search for. The search will return the most semantically similar memories to this query.",
    )
    page: Optional[int] = Field(
        ...,
        description="Allows you to page through results. Only use on a follow-up query. Defaults to 0 (first page).",
    )

    def run(self, retrieval_memory_manager: RetrievalMemoryManager):
        return retrieval_memory_manager.retrieve_memories(self.query)


class archival_memory_insert(BaseModel):
    """
    Add to archival memory. Make sure to phrase the memory contents such that it can be easily queried later.
    """

    memory: str = Field(
        ...,
        description="Content to write to the memory. All unicode (including emojis) are supported.",
    )
    importance: float = Field(
        ...,
        description="A value from 1.0 to 10.0 indicating the importance of the memory.",
    )

    def run(self, retrieval_memory_manager: RetrievalMemoryManager):
        return retrieval_memory_manager.add_memory_to_retrieval(
            self.memory, self.importance
        )


class AgentRetrievalMemory:
    def __init__(
            self,
            persistent_db_path="./retrieval_memory",
            embedding_model_name="all-MiniLM-L6-v2",
            collection_name="retrieval_memory_collection",
    ):
        self.retrieval_memory = RetrievalMemory(
            persistent_db_path, embedding_model_name, collection_name
        )
        self.retrieval_memory_manager = RetrievalMemoryManager(self.retrieval_memory)
        self.retrieve_memories_tool = LlamaCppFunctionTool(
            archival_memory_search,
            retrieval_memory_manager=self.retrieval_memory_manager,
        )
        self.add_retrieval_memory_tool = LlamaCppFunctionTool(
            archival_memory_insert,
            retrieval_memory_manager=self.retrieval_memory_manager,
        )

    def get_tool_list(self):
        return [self.retrieve_memories_tool, self.add_retrieval_memory_tool]

    def get_retrieve_memories_tool(self):
        return self.retrieve_memories_tool

    def get_add_retrieval_memory_tool(self):
        return self.add_retrieval_memory_tool


def create_enum(enum_name, enum_values):
    return Enum(enum_name, {value: value for value in enum_values})


class AgentCoreMemory:
    def __init__(self, core_memory_sections: list[str], core_memory=None, core_memory_file=None):
        if core_memory is None:
            core_memory = {}
        custom_enum = create_enum("core_memory_sections", core_memory_sections)

        class core_memory_append(BaseModel):
            """
            Append a new entry to the Core Memory.
            """

            key: custom_enum = Field(..., description="The key identifier of the core memory.")
            field: str = Field(..., description="The field within the core memory.")
            value: str = Field(
                ...,
                description="The value or data to be stored in the specified core memory entry.",
            )

            def run(self, core_memory_manager: CoreMemoryManager):
                return core_memory_manager.add_to_core_memory(
                    self.key.value, self.field, self.value
                )

        class core_memory_replace(BaseModel):
            """
            Replace an entry in the Core Memory.
            """

            key: custom_enum = Field(..., description="The key identifier of the core memory.")
            field: str = Field(..., description="The field within the core memory.")
            new_value: str = Field(
                ...,
                description="The new value to replace with the existing data in the specified Core Memory field.",
            )

            def run(self, core_memory_manager: CoreMemoryManager):
                return core_memory_manager.replace_in_core_memory(
                    self.key.value, self.field, self.new_value
                )

        class core_memory_remove(BaseModel):
            """
            Remove an entry from the Core Memory.
            """

            key: custom_enum = Field(..., description="The key identifier of the core memory.")
            field: str = Field(..., description="The field within the core memory.")

            def run(self, core_memory_manager: CoreMemoryManager):
                return core_memory_manager.remove_from_core_memory(self.key.value, self.field)

        self.core_memory_manager = CoreMemoryManager(core_memory)
        if core_memory_file is not None:
            self.core_memory_manager.load(core_memory_file)

        self.add_core_memory_tool = LlamaCppFunctionTool(
            core_memory_append, core_memory_manager=self.core_memory_manager
        )
        self.remove_core_memory_tool = LlamaCppFunctionTool(
            core_memory_remove, core_memory_manager=self.core_memory_manager
        )
        self.replace_core_memory_tool = LlamaCppFunctionTool(
            core_memory_replace, core_memory_manager=self.core_memory_manager
        )

    def get_core_memory_view(self):
        return self.core_memory_manager.build_core_memory_context()

    def get_core_memory_manager(self):
        return self.core_memory_manager

    def get_tool_list(self):
        return [
            self.add_core_memory_tool,
            self.remove_core_memory_tool,
            self.replace_core_memory_tool,
        ]

    def get_add_core_memory_tool(self):
        return self.add_core_memory_tool

    def get_remove_core_memory_tool(self):
        return self.remove_core_memory_tool

    def get_replace_core_memory_tool(self):
        return self.replace_core_memory_tool

    def save_core_memory(self, file_path):
        self.core_memory_manager.save(file_path)

    def load_core_memory(self, file_path):
        self.core_memory_manager.load(file_path)


class AgentEventMemory:
    def __init__(self, event_queue_file=None, db_path="sqlite:///events.db"):
        self.engine = create_engine(db_path)
        session_factory = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
        self.Session = scoped_session(session_factory)
        self.session = self.Session()
        self.event_memory_manager = EventMemoryManager(self.session)

        if event_queue_file is not None:
            self.event_memory_manager.load_event_queue(event_queue_file)
        self.search_event_memory_manager_tool = LlamaCppFunctionTool(
            conversation_search, event_memory_manager=self.event_memory_manager
        )

        self.search_event_memory_manager_tool_date = LlamaCppFunctionTool(
            conversation_search_date, event_memory_manager=self.event_memory_manager
        )

    def add_event(self, role: Roles, content:str):
        self.event_memory_manager.add_event_to_queue(role, content, {})
    def get_event_memory_manager(self):
        return self.event_memory_manager

    def get_tool_list(self):
        return [
            self.search_event_memory_manager_tool,
            self.search_event_memory_manager_tool_date,
        ]

    def get_search_event_memory_manager_tool(self):
        return self.search_event_memory_manager_tool
