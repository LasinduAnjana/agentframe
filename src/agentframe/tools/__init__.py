"""
Tool system for AgentFrame.

This package provides the tool system including base classes, registry,
and decorators for easy tool creation and management.
"""

from .base import (
    BaseTool,
    ToolResult,
    ToolRegistry,
    get_global_registry
)

from .decorators import (
    tool,
    FunctionTool,
    get_tool_from_function,
    create_tool_from_callable
)

__all__ = [
    # Base classes
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "get_global_registry",

    # Decorators and utilities
    "tool",
    "FunctionTool",
    "get_tool_from_function",
    "create_tool_from_callable"
]