from sqlalchemy.orm import Session
from .event_memory import Event
import datetime
import json

from ..chat_history import BasicChatHistory
from ..chat_history.messages import Roles


class EventMemoryManager:
    def __init__(self, session: Session, event_queue_limit: int = 10):
        self.session = session
        self.event_queue: list[Event] = []
        self.event_queue_limit = event_queue_limit

    def build_event_memory_context(self):
        messages = []
        for event in self.event_queue:
            messages.append({"role": Roles(event.event_type.value), "content": event.content})
        return messages

    def build_chat_history(self):
        history = BasicChatHistory(k=self.event_queue_limit)
        messages = self.build_event_memory_context()
        for message in messages:
            history.add_message(message)
        return history

    def add_event_to_queue(self, event_type: Roles, content: str, metadata: dict):
        new_event = Event(
            event_type=event_type,
            timestamp=datetime.datetime.now(),
            content=content,
            event_keywords=json.dumps(metadata),
        )
        self.event_queue.append(new_event)

        if len(self.event_queue) > self.event_queue_limit:
            self.commit_oldest_event()

    def commit_oldest_event(self):
        if self.event_queue:
            oldest_event = self.event_queue.pop(0)
            try:
                self.session.add(oldest_event)
                self.session.commit()
                return "Oldest event committed successfully."
            except Exception as e:
                self.session.rollback()
                return f"Error committing oldest event: {e}"
        else:
            return "Skipped committing event to database."

    def modify_event_in_queue(self, modification, event_index=-1):
        if not self.event_queue:
            return "Event queue is empty."

        if event_index < -len(self.event_queue) or event_index >= len(self.event_queue):
            return "Invalid event index."

        event_to_modify = self.event_queue[event_index]
        for key, value in modification.items():
            if hasattr(event_to_modify, key):
                setattr(event_to_modify, key, value)

        return "Event modified successfully."

    def query_events(
        self,
        event_types: list = None,
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
        content_keywords: list = None,
        keywords: list = None,
        page: int = 1,
        page_size: int = 5,
    ) -> str:
        query = self.session.query(Event)

        # Filtering based on provided criteria
        if event_types:
            query = query.filter(Event.event_type.in_(event_types))
        if start_date and end_date:
            query = query.filter(Event.timestamp.between(start_date, end_date))
        if content_keywords:
            for keyword in content_keywords:
                query = query.filter(Event.content.contains(keyword))
        if keywords:
            for value in keywords:
                query = query.filter(Event.event_keywords.contains(value))

        # Calculate offset for paging
        offset_value = (page - 1) * page_size
        # Apply limit and offset to the query for paging
        events = query.limit(page_size).offset(offset_value).all()

        formatted_events = "\n".join([json.dumps(event, indent=2) for event in events])

        if formatted_events:
            formatted_events += f"\n\nPage {page} of {query.count() // page_size + 1}"

        return (
            formatted_events
            if formatted_events
            else "No recall memories found matching the query."
        )

    def save_event_queue(self, filepath):
        with open(filepath, "w") as file:
            json.dump([event.to_dict() for event in self.event_queue], file)

    def load_event_queue(self, filepath):
        with open(filepath, "r") as file:
            self.event_queue = [
                Event.from_dict(event_dict) for event_dict in json.load(file)
            ]
