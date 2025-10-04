"""
AgentFrame - A framework for building LLM agents with planning and tool integration.

AgentFrame provides a complete framework for building intelligent agents that can:
- Parse user intents and understand goals
- Plan multi-step execution strategies
- Execute plans using available tools
- Replan when initial approaches fail
- Manage conversation history and context

Example:
    >>> from agentframe import Agent, OpenAIModel, ModelConfig, tool
    >>>
    >>> @tool
    >>> def calculator(expression: str) -> float:
    ...     '''Calculate a mathematical expression.'''
    ...     return eval(expression)
    >>>
    >>> @tool
    >>> def search_web(query: str) -> dict:
    ...     '''Search the web for information.'''
    ...     return {"results": ["Example result"]}
    >>>
    >>> # Initialize model
    >>> config = ModelConfig(api_key="sk-...", model="gpt-4")
    >>> model = OpenAIModel(config)
    >>>
    >>> # Create agent
    >>> agent = Agent(
    ...     model=model,
    ...     tools=[calculator, search_web]
    ... )
    >>>
    >>> # Use the agent
    >>> response = agent.run("Calculate 15% of 200 and search for related info")
    >>> print(response)
"""

# Core agent and configuration
from .core.agent import Agent, AgentConfig

# Prompt system
from .core.prompts import (
    PromptTemplate,
    AgentPrompts,
    DefaultPrompts,
    PromptValidationError
)

# Model providers and configuration
from .models import (
    BaseModel,
    ModelConfig,
    ModelResponse,
    OpenAIModel,
    GeminiModel,
    ClaudeModel,
    ModelError,
    ConfigurationError,
    APIError,
    RateLimitError,
    TokenLimitError
)

# Tool system
from .tools import (
    BaseTool,
    ToolResult,
    ToolRegistry,
    tool,
    FunctionTool,
    get_global_registry,
    get_tool_from_function,
    create_tool_from_callable
)

# Core components
from .core import (
    AgentState,
    StateManager,
    PlanStep,
    PlanStepStatus,
    ExecutionPlan,
    Planner,
    ReplanningStrategy,
    ExecutionStatus,
    ExecutionResult,
    ExecutionContext,
    ExecutionEngine
)

# Memory and conversation
from .memory import (
    Message,
    MessageType,
    ChatHistory
)

# Utilities
from .utils import (
    Intent,
    IntentParser,
    TaskType,
    IntentCategory,
    ValidationError,
    validate_api_key,
    validate_model_name,
    validate_configuration
)

__version__ = "0.1.0"
__author__ = "AgentFrame Contributors"
__email__ = "contributors@agentframe.dev"
__description__ = "A framework for building LLM agents with planning and tool integration"

__all__ = [
    # Core classes
    "Agent",
    "AgentConfig",

    # Prompt system
    "PromptTemplate",
    "AgentPrompts",
    "DefaultPrompts",
    "PromptValidationError",

    # Model providers
    "BaseModel",
    "ModelConfig",
    "ModelResponse",
    "OpenAIModel",
    "GeminiModel",
    "ClaudeModel",

    # Tool system
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "tool",
    "FunctionTool",
    "get_global_registry",
    "get_tool_from_function",
    "create_tool_from_callable",

    # Core components
    "AgentState",
    "StateManager",
    "PlanStep",
    "PlanStepStatus",
    "ExecutionPlan",
    "Planner",
    "ReplanningStrategy",
    "ExecutionStatus",
    "ExecutionResult",
    "ExecutionContext",
    "ExecutionEngine",

    # Memory
    "Message",
    "MessageType",
    "ChatHistory",

    # Utilities
    "Intent",
    "IntentParser",
    "TaskType",
    "IntentCategory",
    "ValidationError",
    "validate_api_key",
    "validate_model_name",
    "validate_configuration",

    # Exceptions
    "ModelError",
    "ConfigurationError",
    "APIError",
    "RateLimitError",
    "TokenLimitError",

    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]