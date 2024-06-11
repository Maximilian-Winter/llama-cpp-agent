import datetime

from sqlalchemy import Column, Integer, Text, DateTime, Enum
from sqlalchemy.ext.declarative import declarative_base

from enum import Enum as PyEnum
import json

from llama_cpp_agent.chat_history.messages import Roles


Base = declarative_base()


class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    event_type = Column(Enum(Roles))
    timestamp = Column(DateTime, index=True)
    content = Column(Text)
    event_keywords = Column(Text)  # Storing keywords as JSON string

    def __str__(self):
        content = (
            f'Timestamp: {self.timestamp.strftime("%Y-%m-%d %H:%M")}\nType: {self.event_type.value}\n\n{self.content}'
        )
        return content

    def add_keyword(self, keyword):
        """Add a keyword to the event."""
        if self.event_keywords:
            keywords = json.loads(self.event_keywords)
        else:
            keywords = []
        keywords.append(keyword)
        self.event_keywords = json.dumps(keywords)

    def to_dict(self):
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "event_keywords": self.event_keywords,
        }

    @staticmethod
    def from_dict(data):
        return Event(
            event_type=data["event_type"],
            timestamp=datetime.datetime.fromisoformat(data["timestamp"]),
            content=data["content"],
            event_keywords=data["metadata"],
        )
