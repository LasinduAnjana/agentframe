"""
AgentState management for tracking conversation state and execution context.

This module provides the core state management functionality for AgentFrame,
including message tracking, plan storage, and execution context.
"""

import logging
from typing import Any, Dict, List, Optional, TypedDict
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    """
    Complete state representation for an agent session.

    This TypedDict defines the structure of the agent's state throughout
    the conversation and execution lifecycle.

    Attributes:
        messages: Conversation history as list of message dictionaries
        plan: Current execution plan as list of step dictionaries
        tool_results: Results from tool executions
        iteration_count: Number of planning/replanning iterations
        user_intent: Parsed user intention and goals
        context: Additional context and metadata for execution
        metadata: Session metadata (timestamps, model info, etc.)

    Example:
        >>> state: AgentState = {
        ...     "messages": [{"role": "user", "content": "Hello"}],
        ...     "plan": [],
        ...     "tool_results": {},
        ...     "iteration_count": 0,
        ...     "user_intent": None,
        ...     "context": {},
        ...     "metadata": {"session_id": "abc123"}
        ... }
    """
    messages: List[Dict[str, Any]]
    plan: List[Dict[str, Any]]
    tool_results: Dict[str, Any]
    iteration_count: int
    user_intent: Optional[Dict[str, Any]]
    context: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class StateManager:
    """
    Helper class for managing and updating AgentState safely.

    Provides methods for safe state updates and validation to prevent
    corruption of the agent's state during execution.

    Attributes:
        state: The current agent state
        max_iterations: Maximum allowed planning iterations

    Example:
        >>> manager = StateManager()
        >>> manager.initialize_state(session_id="test123")
        >>> manager.add_message({"role": "user", "content": "Hello"})
        >>> manager.increment_iteration()
    """

    state: AgentState = field(default_factory=lambda: AgentState())
    max_iterations: int = 10

    def initialize_state(
        self,
        session_id: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize a fresh agent state.

        Args:
            session_id: Unique identifier for this session
            initial_context: Initial context dictionary

        Example:
            >>> manager = StateManager()
            >>> manager.initialize_state(session_id="abc123",
            ...                          initial_context={"user_id": "user456"})
        """
        timestamp = datetime.now().isoformat()

        self.state = AgentState(
            messages=[],
            plan=[],
            tool_results={},
            iteration_count=0,
            user_intent=None,
            context=initial_context or {},
            metadata={
                "session_id": session_id,
                "created_at": timestamp,
                "last_updated": timestamp
            }
        )

        logger.debug(f"Initialized new agent state with session_id: {session_id}")

    def add_message(self, message: Dict[str, Any]) -> None:
        """
        Add a message to the conversation history.

        Args:
            message: Message dictionary with role and content

        Example:
            >>> manager.add_message({
            ...     "role": "user",
            ...     "content": "Calculate 2+2",
            ...     "timestamp": datetime.now().isoformat()
            ... })
        """
        if "messages" not in self.state:
            self.state["messages"] = []

        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

        self.state["messages"].append(message)
        self._update_timestamp()

        logger.debug(f"Added message with role: {message.get('role', 'unknown')}")

    def update_plan(self, plan: List[Dict[str, Any]]) -> None:
        """
        Update the current execution plan.

        Args:
            plan: List of plan step dictionaries

        Example:
            >>> plan = [
            ...     {"step": 1, "tool": "calculator", "args": {"expr": "2+2"}},
            ...     {"step": 2, "tool": "formatter", "args": {"result": "${step1}"}}
            ... ]
            >>> manager.update_plan(plan)
        """
        self.state["plan"] = plan
        self._update_timestamp()

        logger.debug(f"Updated plan with {len(plan)} steps")

    def add_tool_result(self, tool_name: str, result: Any) -> None:
        """
        Add a tool execution result to the state.

        Args:
            tool_name: Name of the executed tool
            result: Result returned by the tool

        Example:
            >>> manager.add_tool_result("calculator", {"result": 4, "success": True})
        """
        if "tool_results" not in self.state:
            self.state["tool_results"] = {}

        self.state["tool_results"][tool_name] = {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        self._update_timestamp()

        logger.debug(f"Added tool result for: {tool_name}")

    def increment_iteration(self) -> None:
        """
        Increment the iteration counter.

        Raises:
            RuntimeError: If max iterations exceeded

        Example:
            >>> manager.increment_iteration()
            >>> assert manager.state["iteration_count"] == 1
        """
        if "iteration_count" not in self.state:
            self.state["iteration_count"] = 0

        self.state["iteration_count"] += 1
        self._update_timestamp()

        if self.state["iteration_count"] > self.max_iterations:
            raise RuntimeError(
                f"Maximum iterations ({self.max_iterations}) exceeded. "
                "This may indicate an infinite loop in planning."
            )

        logger.debug(f"Incremented iteration count to: {self.state['iteration_count']}")

    def set_user_intent(self, intent: Dict[str, Any]) -> None:
        """
        Set the parsed user intent.

        Args:
            intent: Parsed intent dictionary

        Example:
            >>> intent = {
            ...     "primary_goal": "calculate",
            ...     "entities": ["2", "2"],
            ...     "task_type": "single_step",
            ...     "confidence": 0.95
            ... }
            >>> manager.set_user_intent(intent)
        """
        self.state["user_intent"] = intent
        self._update_timestamp()

        logger.debug(f"Set user intent: {intent.get('primary_goal', 'unknown')}")

    def update_context(self, updates: Dict[str, Any]) -> None:
        """
        Update the execution context.

        Args:
            updates: Dictionary of context updates to merge

        Example:
            >>> manager.update_context({"temperature": 0.7, "model": "gpt-4"})
        """
        if "context" not in self.state:
            self.state["context"] = {}

        self.state["context"].update(updates)
        self._update_timestamp()

        logger.debug(f"Updated context with keys: {list(updates.keys())}")

    def get_latest_messages(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the N most recent messages.

        Args:
            n: Number of recent messages to return

        Returns:
            List of recent message dictionaries

        Example:
            >>> recent = manager.get_latest_messages(5)
            >>> assert len(recent) <= 5
        """
        messages = self.state.get("messages", [])
        return messages[-n:] if len(messages) > n else messages

    def clear_plan(self) -> None:
        """
        Clear the current execution plan.

        Example:
            >>> manager.clear_plan()
            >>> assert manager.state["plan"] == []
        """
        self.state["plan"] = []
        self._update_timestamp()

        logger.debug("Cleared execution plan")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current state.

        Returns:
            Dictionary containing state summary information

        Example:
            >>> summary = manager.get_summary()
            >>> print(f"Messages: {summary['message_count']}")
            >>> print(f"Iterations: {summary['iteration_count']}")
        """
        return {
            "message_count": len(self.state.get("messages", [])),
            "plan_steps": len(self.state.get("plan", [])),
            "tool_results_count": len(self.state.get("tool_results", {})),
            "iteration_count": self.state.get("iteration_count", 0),
            "has_user_intent": self.state.get("user_intent") is not None,
            "session_id": self.state.get("metadata", {}).get("session_id"),
            "last_updated": self.state.get("metadata", {}).get("last_updated")
        }

    def _update_timestamp(self) -> None:
        """Update the last_updated timestamp in metadata."""
        if "metadata" not in self.state:
            self.state["metadata"] = {}
        self.state["metadata"]["last_updated"] = datetime.now().isoformat()

    def validate_state(self) -> bool:
        """
        Validate the current state structure.

        Returns:
            True if state is valid, False otherwise

        Example:
            >>> is_valid = manager.validate_state()
            >>> assert is_valid == True
        """
        required_keys = ["messages", "plan", "tool_results", "iteration_count", "context", "metadata"]

        for key in required_keys:
            if key not in self.state:
                logger.warning(f"Missing required state key: {key}")
                return False

        # Type validation
        if not isinstance(self.state["messages"], list):
            logger.warning("Messages must be a list")
            return False

        if not isinstance(self.state["plan"], list):
            logger.warning("Plan must be a list")
            return False

        if not isinstance(self.state["tool_results"], dict):
            logger.warning("Tool results must be a dictionary")
            return False

        if not isinstance(self.state["iteration_count"], int):
            logger.warning("Iteration count must be an integer")
            return False

        return True