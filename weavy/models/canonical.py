from typing import Literal

from pydantic import BaseModel

from weavy.timeutil import UTCDatetime


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Session(BaseModel):
    id: str  # s:N
    timestamp: UTCDatetime
    messages: list[ChatMessage]


def conversation_to_chat_messages(conversation: list[dict]) -> list[ChatMessage]:
    """Filter a raw conversation to storable user/assistant messages."""
    return [
        ChatMessage(role=message["role"], content=message["content"])
        for message in conversation
        if message["role"] in ("user", "assistant") and message.get("content")
    ]
