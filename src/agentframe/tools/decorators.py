"""
Tool decorators for AgentFrame.

This module provides decorators for easily creating tools from regular Python
functions with automatic schema generation and registration.
"""

import logging
import inspect
import json
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints
from dataclasses import dataclass

from .base import BaseTool, ToolResult, get_global_registry

logger = logging.getLogger(__name__)


def _python_type_to_json_schema(python_type: Type) -> Dict[str, Any]:
    """Convert Python type hints to JSON Schema type."""
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
        List[str]: {"type": "array", "items": {"type": "string"}},
        List[int]: {"type": "array", "items": {"type": "integer"}},
        List[float]: {"type": "array", "items": {"type": "number"}},
        Dict[str, str]: {"type": "object", "additionalProperties": {"type": "string"}},
        Dict[str, Any]: {"type": "object"}
    }

    # Handle Optional types
    if hasattr(python_type, '__origin__'):
        if python_type.__origin__ is Union:
            # Check if it's Optional (Union with None)
            args = python_type.__args__
            if len(args) == 2 and type(None) in args:
                # It's Optional[T]
                non_none_type = args[0] if args[1] is type(None) else args[1]
                schema = _python_type_to_json_schema(non_none_type)
                return schema
        elif python_type.__origin__ is list:
            # Handle List[T]
            if python_type.__args__:
                item_type = python_type.__args__[0]
                return {
                    "type": "array",
                    "items": _python_type_to_json_schema(item_type)
                }
            return {"type": "array"}
        elif python_type.__origin__ is dict:
            # Handle Dict[K, V]
            return {"type": "object"}

    return type_mapping.get(python_type, {"type": "string"})


def _extract_docstring_info(func: Callable) -> Dict[str, Any]:
    """Extract parameter descriptions and return description from docstring."""
    docstring = inspect.getdoc(func)
    if not docstring:
        return {"description": "", "param_descriptions": {}}

    lines = [line.strip() for line in docstring.split('\n')]
    description = ""
    param_descriptions = {}
    current_section = None

    for line in lines:
        if line.startswith("Args:") or line.startswith("Arguments:"):
            current_section = "args"
            continue
        elif line.startswith("Returns:"):
            current_section = "returns"
            continue
        elif line.startswith("Raises:"):
            current_section = "raises"
            continue
        elif line and not line.startswith("    ") and current_section:
            current_section = None

        if current_section == "args":
            if ":" in line and line.startswith("    "):
                param_line = line.strip()
                if ":" in param_line:
                    param_name = param_line.split(":")[0].strip()
                    param_desc = ":".join(param_line.split(":")[1:]).strip()
                    param_descriptions[param_name] = param_desc
        elif current_section is None and line:
            description += line + " "

    return {
        "description": description.strip(),
        "param_descriptions": param_descriptions
    }


def _generate_schema_from_function(func: Callable) -> Dict[str, Any]:
    """Generate JSON schema from function signature and docstring."""
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)
    docstring_info = _extract_docstring_info(func)

    properties = {}
    required = []

    for param_name, param in signature.parameters.items():
        if param_name in ['self', 'cls']:
            continue

        # Get type information
        param_type = type_hints.get(param_name, str)
        param_schema = _python_type_to_json_schema(param_type)

        # Add description from docstring
        if param_name in docstring_info["param_descriptions"]:
            param_schema["description"] = docstring_info["param_descriptions"][param_name]

        properties[param_name] = param_schema

        # Check if parameter is required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required
    }


class FunctionTool(BaseTool):
    """
    Tool implementation that wraps a Python function.

    This class automatically converts a regular Python function into a tool
    that can be used by the AgentFrame system.

    Attributes:
        func: The wrapped function
        sync: Whether the function is synchronous

    Example:
        >>> def add(a: int, b: int) -> int:
        ...     '''Add two numbers.
        ...
        ...     Args:
        ...         a: First number
        ...         b: Second number
        ...
        ...     Returns:
        ...         Sum of a and b
        ...     '''
        ...     return a + b
        >>>
        >>> tool = FunctionTool.from_function(add)
        >>> result = tool.execute({"a": 2, "b": 3})
        >>> assert result.success == True
        >>> assert result.result == 5
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize function tool.

        Args:
            func: Function to wrap
            name: Tool name (defaults to function name)
            description: Tool description (extracted from docstring if not provided)
            parameters: Parameter schema (auto-generated if not provided)
        """
        self.func = func
        self.sync = not inspect.iscoroutinefunction(func)

        # Use function name if no name provided
        if name is None:
            name = func.__name__

        # Extract description from docstring if not provided
        if description is None:
            docstring_info = _extract_docstring_info(func)
            description = docstring_info["description"] or f"Execute {func.__name__}"

        # Generate schema if not provided
        if parameters is None:
            parameters = _generate_schema_from_function(func)

        super().__init__(name, description, parameters)

    @classmethod
    def from_function(
        cls,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> 'FunctionTool':
        """
        Create a FunctionTool from a function.

        Args:
            func: Function to wrap
            name: Tool name
            description: Tool description
            parameters: Parameter schema

        Returns:
            FunctionTool instance

        Example:
            >>> def multiply(x: float, y: float) -> float:
            ...     return x * y
            >>> tool = FunctionTool.from_function(multiply)
            >>> assert tool.name == "multiply"
        """
        return cls(func, name, description, parameters)

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        """
        Execute the wrapped function.

        Args:
            args: Arguments to pass to the function

        Returns:
            ToolResult with execution outcome

        Example:
            >>> def greet(name: str) -> str:
            ...     return f"Hello, {name}!"
            >>> tool = FunctionTool.from_function(greet)
            >>> result = tool.execute({"name": "Alice"})
            >>> assert result.success == True
            >>> assert result.result == "Hello, Alice!"
        """
        try:
            # Call the function with the provided arguments
            if self.sync:
                result = self.func(**args)
            else:
                # For async functions, we need to handle them properly
                import asyncio
                if asyncio.iscoroutinefunction(self.func):
                    # Create new event loop if none exists
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    result = loop.run_until_complete(self.func(**args))
                else:
                    result = self.func(**args)

            return ToolResult(
                success=True,
                result=result,
                metadata={
                    "function_name": self.func.__name__,
                    "args": args
                }
            )

        except Exception as e:
            logger.error(f"Error executing function {self.func.__name__}: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={
                    "function_name": self.func.__name__,
                    "args": args
                }
            )


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    register: bool = True,
    category: Optional[str] = None
) -> Callable:
    """
    Decorator to convert a function into an AgentFrame tool.

    This decorator automatically generates tool schema from function signature
    and docstring, and optionally registers the tool globally.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (extracted from docstring if not provided)
        parameters: Custom parameter schema (auto-generated if not provided)
        register: Whether to register the tool globally
        category: Category for tool organization

    Returns:
        Decorated function that can be used as a tool

    Example:
        >>> @tool
        ... def calculate_area(width: float, height: float) -> float:
        ...     '''Calculate the area of a rectangle.
        ...
        ...     Args:
        ...         width: Width of the rectangle
        ...         height: Height of the rectangle
        ...
        ...     Returns:
        ...         Area of the rectangle
        ...     '''
        ...     return width * height
        >>>
        >>> # Tool is automatically registered and can be used
        >>> from agentframe.tools import get_global_registry
        >>> registry = get_global_registry()
        >>> tool_instance = registry.get("calculate_area")
        >>> result = tool_instance.execute({"width": 5.0, "height": 3.0})
        >>> assert result.result == 15.0
    """
    def decorator(func: Callable) -> Callable:
        # Create the tool
        tool_instance = FunctionTool.from_function(
            func=func,
            name=name,
            description=description,
            parameters=parameters
        )

        # Register globally if requested
        if register:
            try:
                registry = get_global_registry()
                registry.register(tool_instance, category=category)
                logger.debug(f"Registered tool: {tool_instance.name}")
            except ValueError as e:
                logger.warning(f"Failed to register tool {tool_instance.name}: {e}")

        # Add tool instance as attribute to the function
        func._agentframe_tool = tool_instance

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Add tool-related attributes to wrapper
        wrapper._agentframe_tool = tool_instance
        wrapper.get_tool = lambda: tool_instance
        wrapper.get_schema = lambda: tool_instance.get_schema()

        return wrapper

    return decorator


def get_tool_from_function(func: Callable) -> Optional[BaseTool]:
    """
    Extract the tool instance from a decorated function.

    Args:
        func: Function decorated with @tool

    Returns:
        Tool instance or None if function is not decorated

    Example:
        >>> @tool
        ... def example_function():
        ...     pass
        >>>
        >>> tool_instance = get_tool_from_function(example_function)
        >>> assert tool_instance is not None
        >>> assert tool_instance.name == "example_function"
    """
    return getattr(func, '_agentframe_tool', None)


def create_tool_from_callable(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> BaseTool:
    """
    Create a tool from any callable without decoration.

    Args:
        func: Callable to convert to tool
        name: Tool name
        description: Tool description
        parameters: Parameter schema

    Returns:
        Tool instance

    Example:
        >>> def divide(a: float, b: float) -> float:
        ...     '''Divide two numbers.
        ...
        ...     Args:
        ...         a: Dividend
        ...         b: Divisor
        ...     '''
        ...     return a / b
        >>>
        >>> tool_instance = create_tool_from_callable(divide)
        >>> result = tool_instance.execute({"a": 10.0, "b": 2.0})
        >>> assert result.result == 5.0
    """
    return FunctionTool.from_function(
        func=func,
        name=name,
        description=description,
        parameters=parameters
    )