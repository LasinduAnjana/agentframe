"""
Base tool classes and tool registry for AgentFrame.

This module provides the foundation for tool integration, including the base
tool interface, tool registry for management, and validation utilities.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import json
import inspect
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """
    Result from tool execution.

    This class standardizes the results returned from tool executions,
    providing consistent error handling and metadata tracking.

    Attributes:
        success: Whether the tool execution was successful
        result: The actual result data from the tool
        error: Error message if execution failed
        metadata: Additional metadata about the execution

    Example:
        >>> result = ToolResult(
        ...     success=True,
        ...     result={"answer": 42},
        ...     metadata={"execution_time": 0.1}
        ... )
    """
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    This class defines the interface that all tools must implement to be
    compatible with the AgentFrame system.

    Attributes:
        name: Unique name of the tool
        description: Human-readable description of what the tool does
        parameters: JSON schema describing the tool's parameters

    Example:
        >>> class CalculatorTool(BaseTool):
        ...     def __init__(self):
        ...         super().__init__(
        ...             name="calculator",
        ...             description="Performs basic math calculations",
        ...             parameters={
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "expression": {"type": "string"}
        ...                 },
        ...                 "required": ["expression"]
        ...             }
        ...         )
        ...
        ...     def execute(self, args: Dict[str, Any]) -> ToolResult:
        ...         expr = args["expression"]
        ...         result = eval(expr)  # In production, use safe evaluation
        ...         return ToolResult(success=True, result=result)
    """

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]) -> None:
        """
        Initialize the tool.

        Args:
            name: Unique identifier for the tool
            description: Description of the tool's functionality
            parameters: JSON schema for the tool's parameters

        Example:
            >>> tool = CalculatorTool()
            >>> assert tool.name == "calculator"
        """
        self.name = name
        self.description = description
        self.parameters = parameters

        # Validate the schema
        self._validate_schema()

        logger.debug(f"Initialized tool: {self.name}")

    @abstractmethod
    def execute(self, args: Dict[str, Any]) -> ToolResult:
        """
        Execute the tool with given arguments.

        Args:
            args: Arguments dictionary matching the tool's parameter schema

        Returns:
            ToolResult containing the execution outcome

        Example:
            >>> tool = CalculatorTool()
            >>> result = tool.execute({"expression": "2 + 2"})
            >>> assert result.success == True
            >>> assert result.result == 4
        """
        pass

    def validate_args(self, args: Dict[str, Any]) -> bool:
        """
        Validate arguments against the tool's parameter schema.

        Args:
            args: Arguments to validate

        Returns:
            True if arguments are valid, False otherwise

        Example:
            >>> tool = CalculatorTool()
            >>> assert tool.validate_args({"expression": "2+2"}) == True
            >>> assert tool.validate_args({"wrong_param": "value"}) == False
        """
        try:
            # Basic validation - check required parameters
            required = self.parameters.get("required", [])
            for param in required:
                if param not in args:
                    logger.warning(f"Missing required parameter: {param}")
                    return False

            # Check for unexpected parameters
            expected_params = set(self.parameters.get("properties", {}).keys())
            provided_params = set(args.keys())
            unexpected = provided_params - expected_params

            if unexpected:
                logger.warning(f"Unexpected parameters: {unexpected}")
                return False

            # Type validation (basic)
            properties = self.parameters.get("properties", {})
            for param, value in args.items():
                if param in properties:
                    expected_type = properties[param].get("type")
                    if not self._validate_type(value, expected_type):
                        logger.warning(f"Invalid type for {param}: expected {expected_type}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Error validating arguments: {e}")
            return False

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the complete tool schema for LLM consumption.

        Returns:
            Dictionary containing the tool's complete schema

        Example:
            >>> tool = CalculatorTool()
            >>> schema = tool.get_schema()
            >>> assert schema["name"] == "calculator"
            >>> assert "parameters" in schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

    def _validate_schema(self) -> None:
        """Validate the tool's parameter schema."""
        if not isinstance(self.parameters, dict):
            raise ValueError("Parameters must be a dictionary")

        if "type" not in self.parameters:
            raise ValueError("Parameters schema must have a 'type' field")

        if self.parameters["type"] != "object":
            raise ValueError("Top-level parameters type must be 'object'")

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate a value against an expected JSON Schema type."""
        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }

        if expected_type not in type_mapping:
            return True  # Unknown type, skip validation

        expected_python_type = type_mapping[expected_type]
        return isinstance(value, expected_python_type)

    def __repr__(self) -> str:
        """String representation of the tool."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class ToolRegistry:
    """
    Registry for managing and accessing tools.

    This class provides centralized management of tools, including registration,
    retrieval, validation, and schema generation for LLM consumption.

    Example:
        >>> registry = ToolRegistry()
        >>> calculator = CalculatorTool()
        >>> registry.register(calculator)
        >>> tool = registry.get("calculator")
        >>> assert tool.name == "calculator"
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, Set[str]] = {}

        logger.debug("Initialized empty tool registry")

    def register(self, tool: BaseTool, category: Optional[str] = None) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: Tool instance to register
            category: Optional category for organization

        Raises:
            ValueError: If tool name already exists

        Example:
            >>> registry = ToolRegistry()
            >>> calculator = CalculatorTool()
            >>> registry.register(calculator, category="math")
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")

        self._tools[tool.name] = tool

        # Add to category
        if category:
            if category not in self._categories:
                self._categories[category] = set()
            self._categories[category].add(tool.name)

        logger.info(f"Registered tool: {tool.name}" +
                   (f" in category: {category}" if category else ""))

    def unregister(self, tool_name: str) -> bool:
        """
        Unregister a tool from the registry.

        Args:
            tool_name: Name of the tool to unregister

        Returns:
            True if tool was removed, False if not found

        Example:
            >>> registry = ToolRegistry()
            >>> calculator = CalculatorTool()
            >>> registry.register(calculator)
            >>> assert registry.unregister("calculator") == True
        """
        if tool_name not in self._tools:
            logger.warning(f"Tool '{tool_name}' not found for unregistration")
            return False

        del self._tools[tool_name]

        # Remove from categories
        for category, tools in self._categories.items():
            tools.discard(tool_name)

        logger.info(f"Unregistered tool: {tool_name}")
        return True

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """
        Retrieve a tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            Tool instance or None if not found

        Example:
            >>> registry = ToolRegistry()
            >>> calculator = CalculatorTool()
            >>> registry.register(calculator)
            >>> tool = registry.get("calculator")
            >>> assert tool is not None
        """
        return self._tools.get(tool_name)

    def list_all(self) -> List[str]:
        """
        Get names of all registered tools.

        Returns:
            List of tool names

        Example:
            >>> registry = ToolRegistry()
            >>> registry.register(CalculatorTool())
            >>> tools = registry.list_all()
            >>> assert "calculator" in tools
        """
        return list(self._tools.keys())

    def list_by_category(self, category: str) -> List[str]:
        """
        Get tools in a specific category.

        Args:
            category: Category name

        Returns:
            List of tool names in the category

        Example:
            >>> registry = ToolRegistry()
            >>> registry.register(CalculatorTool(), category="math")
            >>> math_tools = registry.list_by_category("math")
            >>> assert "calculator" in math_tools
        """
        return list(self._categories.get(category, set()))

    def get_schemas(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get schemas for specified tools or all tools.

        Args:
            tool_names: Specific tools to get schemas for, or None for all

        Returns:
            List of tool schema dictionaries

        Example:
            >>> registry = ToolRegistry()
            >>> registry.register(CalculatorTool())
            >>> schemas = registry.get_schemas()
            >>> assert len(schemas) == 1
            >>> assert schemas[0]["name"] == "calculator"
        """
        if tool_names is None:
            tool_names = self.list_all()

        schemas = []
        for name in tool_names:
            tool = self.get(name)
            if tool:
                schemas.append(tool.get_schema())
            else:
                logger.warning(f"Tool '{name}' not found for schema generation")

        return schemas

    def validate_args(self, tool_name: str, args: Dict[str, Any]) -> bool:
        """
        Validate arguments for a specific tool.

        Args:
            tool_name: Name of the tool
            args: Arguments to validate

        Returns:
            True if arguments are valid, False otherwise

        Example:
            >>> registry = ToolRegistry()
            >>> registry.register(CalculatorTool())
            >>> valid = registry.validate_args("calculator", {"expression": "2+2"})
            >>> assert valid == True
        """
        tool = self.get(tool_name)
        if not tool:
            logger.error(f"Tool '{tool_name}' not found for validation")
            return False

        return tool.validate_args(args)

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool with validation.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool

        Returns:
            ToolResult with execution outcome

        Example:
            >>> registry = ToolRegistry()
            >>> registry.register(CalculatorTool())
            >>> result = registry.execute_tool("calculator", {"expression": "2+2"})
            >>> assert result.success == True
        """
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found"
            )

        # Validate arguments
        if not tool.validate_args(args):
            return ToolResult(
                success=False,
                error=f"Invalid arguments for tool '{tool_name}'"
            )

        try:
            result = tool.execute(args)
            logger.debug(f"Successfully executed tool: {tool_name}")
            return result

        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {e}")
            return ToolResult(
                success=False,
                error=str(e)
            )

    def get_tool_count(self) -> int:
        """
        Get the number of registered tools.

        Returns:
            Number of registered tools

        Example:
            >>> registry = ToolRegistry()
            >>> assert registry.get_tool_count() == 0
            >>> registry.register(CalculatorTool())
            >>> assert registry.get_tool_count() == 1
        """
        return len(self._tools)

    def clear(self) -> None:
        """
        Clear all registered tools.

        Example:
            >>> registry = ToolRegistry()
            >>> registry.register(CalculatorTool())
            >>> registry.clear()
            >>> assert registry.get_tool_count() == 0
        """
        self._tools.clear()
        self._categories.clear()
        logger.info("Cleared all tools from registry")


# Global registry instance
_global_registry = ToolRegistry()


def get_global_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.

    Returns:
        Global ToolRegistry instance

    Example:
        >>> registry = get_global_registry()
        >>> registry.register(CalculatorTool())
    """
    return _global_registry