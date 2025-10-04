"""
Chat history management for AgentFrame.

This module provides conversation history management with token-aware
truncation, message filtering, and export capabilities.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in conversation history."""
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"
    TOOL = "tool"
    FUNCTION = "function"


@dataclass
class Message:
    """
    Individual message in conversation history.

    Represents a single message with type, content, and metadata for
    tracking conversation flow and context.

    Attributes:
        type: Type of message (human, ai, system, tool, function)
        content: Message content text
        timestamp: When the message was created
        metadata: Additional message metadata
        token_count: Estimated token count for this message
        tool_calls: Tool calls if this is an AI message with function calls
        tool_call_id: ID of tool call if this is a tool response

    Example:
        >>> message = Message(
        ...     type=MessageType.HUMAN,
        ...     content="Hello, how are you?",
        ...     metadata={"user_id": "user123"}
        ... )
    """
    type: MessageType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert message to dictionary format.

        Returns:
            Dictionary representation of the message

        Example:
            >>> message = Message(MessageType.HUMAN, "Hello")
            >>> msg_dict = message.to_dict()
            >>> assert msg_dict["role"] == "user"
        """
        # Map internal types to standard OpenAI format
        role_mapping = {
            MessageType.HUMAN: "user",
            MessageType.AI: "assistant",
            MessageType.SYSTEM: "system",
            MessageType.TOOL: "tool",
            MessageType.FUNCTION: "function"
        }

        result = {
            "role": role_mapping[self.type],
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }

        # Add optional fields
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.metadata:
            result["metadata"] = self.metadata
        if self.token_count:
            result["token_count"] = self.token_count

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """
        Create message from dictionary format.

        Args:
            data: Dictionary representation

        Returns:
            Message instance

        Example:
            >>> data = {"role": "user", "content": "Hello"}
            >>> message = Message.from_dict(data)
            >>> assert message.type == MessageType.HUMAN
        """
        # Map standard roles to internal types
        type_mapping = {
            "user": MessageType.HUMAN,
            "assistant": MessageType.AI,
            "system": MessageType.SYSTEM,
            "tool": MessageType.TOOL,
            "function": MessageType.FUNCTION
        }

        message_type = type_mapping.get(data.get("role", "user"), MessageType.HUMAN)

        # Parse timestamp
        timestamp = datetime.now()
        if "timestamp" in data:
            try:
                timestamp = datetime.fromisoformat(data["timestamp"])
            except (ValueError, TypeError):
                pass

        return cls(
            type=message_type,
            content=data.get("content", ""),
            timestamp=timestamp,
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id")
        )

    def estimate_tokens(self) -> int:
        """
        Estimate token count for this message.

        Returns:
            Estimated token count

        Example:
            >>> message = Message(MessageType.HUMAN, "Hello world")
            >>> tokens = message.estimate_tokens()
            >>> assert tokens > 0
        """
        if self.token_count is not None:
            return self.token_count

        # Rough estimation: ~4 characters per token for English text
        base_tokens = len(self.content) // 4 + 1

        # Add tokens for metadata
        if self.tool_calls:
            # Tool calls add significant tokens
            base_tokens += len(str(self.tool_calls)) // 4

        self.token_count = base_tokens
        return base_tokens


class ChatHistory:
    """
    Manages conversation history with token-aware truncation.

    Provides efficient storage and retrieval of conversation messages
    with support for windowing, truncation, and export to various formats.

    Attributes:
        messages: List of conversation messages
        max_messages: Maximum number of messages to keep
        max_tokens: Maximum total tokens to keep
        preserve_system: Whether to always preserve system messages

    Example:
        >>> history = ChatHistory(max_messages=100, max_tokens=4000)
        >>> history.add_message(MessageType.HUMAN, "Hello!")
        >>> history.add_message(MessageType.AI, "Hi there!")
        >>> recent = history.get_recent(n=5)
    """

    def __init__(
        self,
        max_messages: int = 1000,
        max_tokens: int = 8000,
        preserve_system: bool = True
    ):
        """
        Initialize chat history manager.

        Args:
            max_messages: Maximum number of messages to store
            max_tokens: Maximum total tokens to store
            preserve_system: Whether to always keep system messages
        """
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.preserve_system = preserve_system

        logger.debug(f"Initialized chat history with max_messages={max_messages}, max_tokens={max_tokens}")

    def add_message(
        self,
        message_type: Union[MessageType, str],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None
    ) -> None:
        """
        Add a message to the conversation history.

        Args:
            message_type: Type of message (MessageType enum or string)
            content: Message content
            metadata: Optional metadata dictionary
            tool_calls: Tool calls if this is an AI message
            tool_call_id: Tool call ID if this is a tool response

        Example:
            >>> history.add_message(MessageType.HUMAN, "What's the weather like?")
            >>> history.add_message("ai", "Let me check that for you.")
        """
        # Convert string to MessageType if needed
        if isinstance(message_type, str):
            type_mapping = {
                "user": MessageType.HUMAN,
                "human": MessageType.HUMAN,
                "assistant": MessageType.AI,
                "ai": MessageType.AI,
                "system": MessageType.SYSTEM,
                "tool": MessageType.TOOL,
                "function": MessageType.FUNCTION
            }
            message_type = type_mapping.get(message_type.lower(), MessageType.HUMAN)

        message = Message(
            type=message_type,
            content=content,
            metadata=metadata or {},
            tool_calls=tool_calls,
            tool_call_id=tool_call_id
        )

        self.messages.append(message)

        # Trigger cleanup if needed
        self._cleanup_if_needed()

        logger.debug(f"Added {message_type.value} message: {content[:50]}...")

    def get_recent(self, n: int = 10) -> List[Message]:
        """
        Get the N most recent messages.

        Args:
            n: Number of recent messages to return

        Returns:
            List of recent messages

        Example:
            >>> recent = history.get_recent(5)
            >>> for msg in recent:
            ...     print(f"{msg.type.value}: {msg.content}")
        """
        return self.messages[-n:] if len(self.messages) > n else self.messages.copy()

    def get_context_window(
        self,
        max_tokens: Optional[int] = None,
        preserve_last_n: int = 1
    ) -> List[Message]:
        """
        Get messages that fit within a token budget.

        Args:
            max_tokens: Maximum tokens (uses instance max_tokens if None)
            preserve_last_n: Always include last N messages

        Returns:
            List of messages within token budget

        Example:
            >>> context = history.get_context_window(max_tokens=2000)
            >>> total_tokens = sum(msg.estimate_tokens() for msg in context)
            >>> assert total_tokens <= 2000
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        if not self.messages:
            return []

        # Always preserve the last few messages
        preserved = self.messages[-preserve_last_n:] if preserve_last_n > 0 else []
        remaining = self.messages[:-preserve_last_n] if preserve_last_n > 0 else self.messages

        # Calculate tokens for preserved messages
        preserved_tokens = sum(msg.estimate_tokens() for msg in preserved)
        available_tokens = max_tokens - preserved_tokens

        if available_tokens <= 0:
            return preserved

        # Add messages from the end until we hit the token limit
        selected = []
        current_tokens = 0

        for message in reversed(remaining):
            msg_tokens = message.estimate_tokens()
            if current_tokens + msg_tokens <= available_tokens:
                selected.insert(0, message)
                current_tokens += msg_tokens
            else:
                break

        # Always include system messages if preserve_system is True
        if self.preserve_system:
            system_messages = [msg for msg in remaining if msg.type == MessageType.SYSTEM and msg not in selected]
            selected = system_messages + selected

        return selected + preserved

    def get_by_type(self, message_type: MessageType) -> List[Message]:
        """
        Get all messages of a specific type.

        Args:
            message_type: Type of messages to retrieve

        Returns:
            List of messages of the specified type

        Example:
            >>> system_msgs = history.get_by_type(MessageType.SYSTEM)
            >>> user_msgs = history.get_by_type(MessageType.HUMAN)
        """
        return [msg for msg in self.messages if msg.type == message_type]

    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation.

        Returns:
            Dictionary with conversation statistics

        Example:
            >>> summary = history.get_conversation_summary()
            >>> print(f"Total messages: {summary['total_messages']}")
            >>> print(f"Total tokens: {summary['total_tokens']}")
        """
        if not self.messages:
            return {
                "total_messages": 0,
                "total_tokens": 0,
                "message_types": {},
                "first_message": None,
                "last_message": None
            }

        message_counts = {}
        total_tokens = 0

        for message in self.messages:
            msg_type = message.type.value
            message_counts[msg_type] = message_counts.get(msg_type, 0) + 1
            total_tokens += message.estimate_tokens()

        return {
            "total_messages": len(self.messages),
            "total_tokens": total_tokens,
            "message_types": message_counts,
            "first_message": self.messages[0].timestamp.isoformat(),
            "last_message": self.messages[-1].timestamp.isoformat(),
            "conversation_duration": (self.messages[-1].timestamp - self.messages[0].timestamp).total_seconds()
        }

    def to_langchain_messages(self, include_metadata: bool = False) -> List[Dict[str, Any]]:
        """
        Export messages in LangChain format.

        Args:
            include_metadata: Whether to include message metadata

        Returns:
            List of messages in LangChain format

        Example:
            >>> lc_messages = history.to_langchain_messages()
            >>> # Can be used directly with LangChain chat models
        """
        result = []
        for message in self.messages:
            msg_dict = message.to_dict()

            # Remove AgentFrame-specific fields for LangChain compatibility
            lc_msg = {
                "role": msg_dict["role"],
                "content": msg_dict["content"]
            }

            # Add tool-related fields if present
            if "tool_calls" in msg_dict:
                lc_msg["tool_calls"] = msg_dict["tool_calls"]
            if "tool_call_id" in msg_dict:
                lc_msg["tool_call_id"] = msg_dict["tool_call_id"]

            # Include metadata if requested
            if include_metadata and "metadata" in msg_dict:
                lc_msg["metadata"] = msg_dict["metadata"]

            result.append(lc_msg)

        return result

    def to_openai_format(self) -> List[Dict[str, str]]:
        """
        Export messages in OpenAI API format.

        Returns:
            List of messages in OpenAI format

        Example:
            >>> openai_messages = history.to_openai_format()
            >>> response = openai.chat.completions.create(
            ...     model="gpt-4",
            ...     messages=openai_messages
            ... )
        """
        result = []
        for message in self.messages:
            msg_dict = message.to_dict()
            openai_msg = {
                "role": msg_dict["role"],
                "content": msg_dict["content"]
            }

            # Add function/tool call fields if present
            if "tool_calls" in msg_dict:
                openai_msg["tool_calls"] = msg_dict["tool_calls"]
            if "tool_call_id" in msg_dict:
                openai_msg["tool_call_id"] = msg_dict["tool_call_id"]

            result.append(openai_msg)

        return result

    def clear(self) -> None:
        """
        Clear all messages from history.

        Example:
            >>> history.clear()
            >>> assert len(history.messages) == 0
        """
        self.messages.clear()
        logger.info("Cleared chat history")

    def save_to_file(self, filepath: str) -> None:
        """
        Save conversation history to a JSON file.

        Args:
            filepath: Path to save the file

        Example:
            >>> history.save_to_file("conversation.json")
        """
        import json

        data = {
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": {
                "max_messages": self.max_messages,
                "max_tokens": self.max_tokens,
                "preserve_system": self.preserve_system,
                "saved_at": datetime.now().isoformat()
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(self.messages)} messages to {filepath}")

    def load_from_file(self, filepath: str) -> None:
        """
        Load conversation history from a JSON file.

        Args:
            filepath: Path to load the file from

        Example:
            >>> history.load_from_file("conversation.json")
        """
        import json

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.messages = [Message.from_dict(msg_data) for msg_data in data["messages"]]

        # Update settings from metadata if available
        if "metadata" in data:
            metadata = data["metadata"]
            self.max_messages = metadata.get("max_messages", self.max_messages)
            self.max_tokens = metadata.get("max_tokens", self.max_tokens)
            self.preserve_system = metadata.get("preserve_system", self.preserve_system)

        logger.info(f"Loaded {len(self.messages)} messages from {filepath}")

    def _cleanup_if_needed(self) -> None:
        """Clean up old messages if limits are exceeded."""
        # Remove excess messages
        if len(self.messages) > self.max_messages:
            # Preserve system messages if configured
            if self.preserve_system:
                system_messages = [msg for msg in self.messages if msg.type == MessageType.SYSTEM]
                other_messages = [msg for msg in self.messages if msg.type != MessageType.SYSTEM]

                # Keep recent other messages
                keep_count = self.max_messages - len(system_messages)
                if keep_count > 0:
                    other_messages = other_messages[-keep_count:]

                self.messages = system_messages + other_messages
            else:
                self.messages = self.messages[-self.max_messages:]

            logger.debug(f"Trimmed messages to {len(self.messages)} (max: {self.max_messages})")

        # Remove excess tokens
        total_tokens = sum(msg.estimate_tokens() for msg in self.messages)
        if total_tokens > self.max_tokens:
            self.messages = self.get_context_window(self.max_tokens)
            logger.debug(f"Trimmed messages for token limit (max: {self.max_tokens})")

    def __len__(self) -> int:
        """Return number of messages in history."""
        return len(self.messages)

    def __iter__(self):
        """Allow iteration over messages."""
        return iter(self.messages)