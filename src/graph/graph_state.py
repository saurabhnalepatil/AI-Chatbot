import uuid
from typing import Annotated, Literal, Optional, Union

from langchain_core.messages import (AnyMessage, MessageLikeRepresentation,
                                     convert_to_messages,
                                     message_chunk_to_message)
from langgraph.graph.message import AnyMessage
from typing_extensions import TypedDict

Messages = Union[list[MessageLikeRepresentation], MessageLikeRepresentation]


def add_messages(left: Messages, right: Messages) -> Messages:
    # coerce to list
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]

    # Take the last 45 messages
    left = left[-45:]

    # coerce to message
    left = [message_chunk_to_message(m) for m in convert_to_messages(left)]
    right = [message_chunk_to_message(m) for m in convert_to_messages(right)]

    # assign missing ids
    for m in left:
        if m.id is None:
            m.id = str(uuid.uuid4())
    for m in right:
        if m.id is None:
            m.id = str(uuid.uuid4())
    # merge
    left_idx_by_id = {m.id: i for i, m in enumerate(left)}
    merged = left.copy()
    for m in right:
        if (existing_idx := left_idx_by_id.get(m.id)) is not None:
            merged[existing_idx] = m
        else:
            merged.append(m)
    return merged


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: dict
    user_lang: str
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
            ]
        ],
        update_dialog_stack,
    ]


class FollowUpState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]