"""
Base model interface for AgentFrame.

This module defines the abstract interface that all model providers must implement,
along with configuration classes and type definitions.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Protocol, Union
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """
    Configuration for model instances.

    This class holds all configuration parameters that can be passed to model
    providers for customizing their behavior.

    Attributes:
        api_key: API key for the model provider
        model: Model name/identifier
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        frequency_penalty: Frequency penalty for repetition
        presence_penalty: Presence penalty for new topics
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        base_url: Custom base URL for API endpoints
        organization: Organization ID (for providers that support it)
        additional_headers: Additional HTTP headers
        custom_params: Provider-specific custom parameters

    Example:
        >>> config = ModelConfig(
        ...     api_key="sk-...",
        ...     model="gpt-4",
        ...     temperature=0.7,
        ...     max_tokens=1000
        ... )
    """

    api_key: str
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30
    max_retries: int = 3
    base_url: Optional[str] = None
    organization: Optional[str] = None
    additional_headers: Dict[str, str] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.

        Returns:
            Dictionary representation of the configuration

        Example:
            >>> config = ModelConfig(api_key="test", model="gpt-4")
            >>> config_dict = config.to_dict()
            >>> assert config_dict["model"] == "gpt-4"
        """
        result = {}
        for key, value in self.__dict__.items():
            if value is not None and value != {} and value != []:
                result[key] = value
        return result


class ToolDefinition(Protocol):
    """
    Protocol for tool definitions used by models.

    This protocol defines the structure that tools must have when passed
    to model providers for function calling.
    """

    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass
class ModelResponse:
    """
    Standardized response from model providers.

    This class provides a uniform interface for model responses across
    different providers, ensuring consistent handling of results.

    Attributes:
        content: The generated text content
        tool_calls: List of tool calls requested by the model
        finish_reason: Reason why generation stopped
        usage: Token usage information
        model: Model used for generation
        metadata: Additional response metadata

    Example:
        >>> response = ModelResponse(
        ...     content="The result is 42",
        ...     tool_calls=[],
        ...     finish_reason="stop",
        ...     usage={"prompt_tokens": 10, "completion_tokens": 5}
        ... )
    """

    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_tool_calls(self) -> bool:
        """
        Check if the response contains tool calls.

        Returns:
            True if tool calls are present, False otherwise

        Example:
            >>> response = ModelResponse(content="Hello", tool_calls=[{"name": "search"}])
            >>> assert response.has_tool_calls() == True
        """
        return bool(self.tool_calls)

    def get_first_tool_call(self) -> Optional[Dict[str, Any]]:
        """
        Get the first tool call if available.

        Returns:
            First tool call dictionary or None

        Example:
            >>> response = ModelResponse(content="", tool_calls=[{"name": "search"}])
            >>> first_call = response.get_first_tool_call()
            >>> assert first_call["name"] == "search"
        """
        return self.tool_calls[0] if self.tool_calls else None


class BaseModel(ABC):
    """
    Abstract base class for all model providers.

    This class defines the interface that all model implementations must follow,
    ensuring consistent behavior across different LLM providers.

    Attributes:
        config: Model configuration
        _client: Internal client instance (provider-specific)

    Example:
        >>> class MyModel(BaseModel):
        ...     def generate(self, messages, tools=None, **kwargs):
        ...         # Implementation here
        ...         return ModelResponse(content="Hello world")
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize the model with configuration.

        Args:
            config: Model configuration object

        Example:
            >>> config = ModelConfig(api_key="test", model="gpt-4")
            >>> model = MyModel(config)
        """
        self.config = config
        self._client: Optional[Any] = None
        self._initialize_client()

        logger.info(f"Initialized {self.__class__.__name__} with model: {config.model}")

    @abstractmethod
    def _initialize_client(self) -> None:
        """
        Initialize the provider-specific client.

        This method should be implemented by each provider to set up
        their specific API client.
        """
        pass

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> ModelResponse:
        """
        Generate a response from the model.

        Args:
            messages: Conversation messages in standard format
            tools: Available tools for function calling
            **kwargs: Additional generation parameters

        Returns:
            Model response object

        Example:
            >>> messages = [{"role": "user", "content": "Hello"}]
            >>> response = model.generate(messages)
            >>> print(response.content)
        """
        pass

    @abstractmethod
    def generate_with_structured_output(
        self,
        messages: List[Dict[str, Any]],
        schema: Union[Dict[str, Any], BaseModel],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Generate structured output conforming to a schema.

        Args:
            messages: Conversation messages
            schema: JSON schema or Pydantic model for output structure
            **kwargs: Additional generation parameters

        Returns:
            Structured output matching the schema

        Example:
            >>> schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
            >>> messages = [{"role": "user", "content": "What is 2+2?"}]
            >>> result = model.generate_with_structured_output(messages, schema)
            >>> assert "answer" in result
        """
        pass

    @abstractmethod
    def stream_generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> Iterator[str]:
        """
        Generate streaming response from the model.

        Args:
            messages: Conversation messages
            tools: Available tools for function calling
            **kwargs: Additional generation parameters

        Yields:
            Chunks of generated text

        Example:
            >>> messages = [{"role": "user", "content": "Tell me a story"}]
            >>> for chunk in model.stream_generate(messages):
            ...     print(chunk, end="")
        """
        pass

    async def agenerate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> ModelResponse:
        """
        Asynchronously generate a response from the model.

        Args:
            messages: Conversation messages
            tools: Available tools for function calling
            **kwargs: Additional generation parameters

        Returns:
            Model response object

        Example:
            >>> messages = [{"role": "user", "content": "Hello"}]
            >>> response = await model.agenerate(messages)
            >>> print(response.content)
        """
        # Default implementation using sync method
        # Providers can override for true async support
        return self.generate(messages, tools, **kwargs)

    async def astream_generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """
        Asynchronously generate streaming response.

        Args:
            messages: Conversation messages
            tools: Available tools for function calling
            **kwargs: Additional generation parameters

        Yields:
            Chunks of generated text

        Example:
            >>> messages = [{"role": "user", "content": "Tell me a story"}]
            >>> async for chunk in model.astream_generate(messages):
            ...     print(chunk, end="")
        """
        # Default implementation using sync method
        # Providers can override for true async support
        for chunk in self.stream_generate(messages, tools, **kwargs):
            yield chunk

    def validate_config(self) -> bool:
        """
        Validate the model configuration.

        Returns:
            True if configuration is valid, False otherwise

        Example:
            >>> config = ModelConfig(api_key="test", model="gpt-4")
            >>> model = MyModel(config)
            >>> assert model.validate_config() == True
        """
        if not self.config.api_key:
            logger.error("API key is required")
            return False

        if not self.config.model:
            logger.error("Model name is required")
            return False

        if not 0.0 <= self.config.temperature <= 2.0:
            logger.error("Temperature must be between 0.0 and 2.0")
            return False

        if self.config.max_tokens <= 0:
            logger.error("Max tokens must be positive")
            return False

        return True

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.

        Returns:
            Dictionary with model information

        Example:
            >>> info = model.get_model_info()
            >>> print(f"Provider: {info['provider']}")
            >>> print(f"Model: {info['model']}")
        """
        return {
            "provider": self.__class__.__name__.replace("Model", "").lower(),
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "supports_tools": hasattr(self, "_supports_tools") and self._supports_tools,
            "supports_streaming": hasattr(self, "_supports_streaming") and self._supports_streaming
        }

    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(model='{self.config.model}')"


class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class ConfigurationError(ModelError):
    """Exception raised for configuration errors."""
    pass


class APIError(ModelError):
    """Exception raised for API-related errors."""
    pass


class RateLimitError(APIError):
    """Exception raised when rate limits are exceeded."""
    pass


class TokenLimitError(ModelError):
    """Exception raised when token limits are exceeded."""
    pass