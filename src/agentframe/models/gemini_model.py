"""
Google Gemini model implementation for AgentFrame.

This module provides integration with Google's Gemini language models,
including Gemini Pro and Gemini Pro Vision, with support for function
calling and structured output.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Union
import json
import time

from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

from .base import (
    BaseModel as AgentBaseModel,
    ModelConfig,
    ModelResponse,
    ToolDefinition,
    APIError,
    RateLimitError,
    TokenLimitError,
    ConfigurationError
)

logger = logging.getLogger(__name__)


class GeminiModel(AgentBaseModel):
    """
    Google Gemini model implementation for AgentFrame.

    This class provides integration with Google's Gemini language models,
    supporting function calling, structured output, and streaming generation.

    Attributes:
        config: Model configuration
        _client: Gemini client instance
        _supports_tools: Whether the model supports tool calling
        _supports_streaming: Whether the model supports streaming

    Example:
        >>> config = ModelConfig(
        ...     api_key="your-google-api-key",
        ...     model="gemini-pro",
        ...     temperature=0.7
        ... )
        >>> model = GeminiModel(config)
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response = model.generate(messages)
        >>> print(response.content)
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize Gemini model.

        Args:
            config: Model configuration with Gemini-specific settings

        Raises:
            ConfigurationError: If configuration is invalid

        Example:
            >>> config = ModelConfig(api_key="your-key", model="gemini-pro")
            >>> model = GeminiModel(config)
        """
        self._supports_tools = True
        self._supports_streaming = True
        super().__init__(config)

        # Validate Gemini-specific configuration
        if not self.config.api_key:
            raise ConfigurationError("Google API key is required for Gemini")

    def _initialize_client(self) -> None:
        """Initialize the Gemini client using LangChain wrapper."""
        try:
            client_kwargs = {
                "model": self.config.model,
                "google_api_key": self.config.api_key,
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries
            }

            # Add custom parameters
            client_kwargs.update(self.config.custom_params)

            self._client = ChatGoogleGenerativeAI(**client_kwargs)

            logger.debug(f"Initialized Gemini client for model: {self.config.model}")

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Gemini client: {e}")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> ModelResponse:
        """
        Generate a response using Gemini model.

        Args:
            messages: Conversation messages
            tools: Available tools for function calling
            **kwargs: Additional generation parameters

        Returns:
            ModelResponse with generated content and tool calls

        Raises:
            APIError: If API request fails
            RateLimitError: If rate limit is exceeded
            TokenLimitError: If token limit is exceeded

        Example:
            >>> messages = [{"role": "user", "content": "What's 2+2?"}]
            >>> tools = [calculator_tool]
            >>> response = model.generate(messages, tools)
            >>> if response.has_tool_calls():
            ...     print("Model wants to use tools")
        """
        try:
            # Convert AgentFrame messages to LangChain format
            langchain_messages = self._convert_messages(messages)

            # Prepare function calling if tools are provided
            invoke_kwargs = {}
            if tools:
                invoke_kwargs["tools"] = self._convert_tools_to_gemini_format(tools)

            # Override config with kwargs
            for key, value in kwargs.items():
                if hasattr(self._client, key):
                    setattr(self._client, key, value)

            start_time = time.time()

            # Make the API call
            result = self._client.invoke(langchain_messages, **invoke_kwargs)

            end_time = time.time()

            # Convert response
            return self._convert_response(result, end_time - start_time)

        except Exception as e:
            return self._handle_exception(e)

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

        Raises:
            APIError: If API request fails
            ValueError: If schema is invalid

        Example:
            >>> messages = [{"role": "user", "content": "Extract info: John is 25"}]
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "name": {"type": "string"},
            ...         "age": {"type": "number"}
            ...     }
            ... }
            >>> result = model.generate_with_structured_output(messages, schema)
            >>> assert "name" in result and "age" in result
        """
        try:
            # Add structured output instruction to messages
            if isinstance(schema, dict):
                schema_str = json.dumps(schema, indent=2)
            else:
                # Pydantic model
                schema_str = schema.model_json_schema()

            structured_prompt = f"""
            Please respond with valid JSON that conforms to this schema:

            {schema_str}

            Only return the JSON, no additional text or formatting.
            """

            # Add system message with schema instruction
            structured_messages = messages.copy()
            structured_messages.insert(0, {
                "role": "system",
                "content": structured_prompt
            })

            # Generate response
            response = self.generate(structured_messages, **kwargs)

            # Parse JSON response
            try:
                # Clean up response content (remove markdown formatting if present)
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                return json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse structured output: {e}")
                raise ValueError(f"Model did not return valid JSON: {response.content}")

        except Exception as e:
            logger.error(f"Error generating structured output: {e}")
            raise

    def stream_generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> Iterator[str]:
        """
        Generate streaming response.

        Args:
            messages: Conversation messages
            tools: Available tools for function calling
            **kwargs: Additional generation parameters

        Yields:
            Chunks of generated text

        Example:
            >>> messages = [{"role": "user", "content": "Tell me a story"}]
            >>> for chunk in model.stream_generate(messages):
            ...     print(chunk, end="", flush=True)
        """
        try:
            # Convert messages
            langchain_messages = self._convert_messages(messages)

            # Prepare streaming
            stream_kwargs = {}
            if tools:
                stream_kwargs["tools"] = self._convert_tools_to_gemini_format(tools)

            # Override config with kwargs
            for key, value in kwargs.items():
                if hasattr(self._client, key):
                    setattr(self._client, key, value)

            # Stream the response
            for chunk in self._client.stream(langchain_messages, **stream_kwargs):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield f"Error: {str(e)}"

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Any]:
        """Convert AgentFrame messages to LangChain format."""
        from langchain_core.messages import (
            HumanMessage,
            AIMessage,
            SystemMessage
        )

        langchain_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                # Handle function calls if present
                if "tool_calls" in msg:
                    ai_msg = AIMessage(
                        content=content,
                        additional_kwargs={"tool_calls": msg["tool_calls"]}
                    )
                    langchain_messages.append(ai_msg)
                else:
                    langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))

        return langchain_messages

    def _convert_tools_to_gemini_format(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert tools to Gemini function calling format."""
        gemini_tools = []

        for tool in tools:
            # Gemini expects a specific format for function declarations
            function_declaration = {
                "function_declarations": [{
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }]
            }
            gemini_tools.append(function_declaration)

        return gemini_tools

    def _convert_response(self, response: Any, execution_time: float) -> ModelResponse:
        """Convert LangChain response to AgentFrame format."""
        content = response.content if hasattr(response, 'content') else str(response)

        # Extract tool calls if present
        tool_calls = []
        if hasattr(response, 'additional_kwargs'):
            additional_kwargs = response.additional_kwargs
            if "tool_calls" in additional_kwargs:
                for call in additional_kwargs["tool_calls"]:
                    if "function" in call:
                        func = call["function"]
                        tool_calls.append({
                            "name": func.get("name"),
                            "arguments": func.get("arguments")
                        })

        # Extract usage information (Gemini might have different format)
        usage = {}
        if hasattr(response, 'usage_metadata'):
            usage_metadata = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(usage_metadata, 'input_tokens', 0),
                "completion_tokens": getattr(usage_metadata, 'output_tokens', 0),
                "total_tokens": getattr(usage_metadata, 'total_tokens', 0)
            }

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=getattr(response, 'finish_reason', None),
            usage=usage,
            model=self.config.model,
            metadata={
                "execution_time": execution_time,
                "provider": "gemini"
            }
        )

    def _handle_exception(self, e: Exception) -> ModelResponse:
        """Handle and convert exceptions to appropriate errors."""
        error_message = str(e)

        # Check for specific Gemini errors
        if "quota exceeded" in error_message.lower() or "rate limit" in error_message.lower():
            raise RateLimitError(f"Gemini rate limit exceeded: {error_message}")
        elif "context length" in error_message.lower() or "token limit" in error_message.lower():
            raise TokenLimitError(f"Token limit exceeded: {error_message}")
        elif "api key" in error_message.lower() or "authentication" in error_message.lower():
            raise ConfigurationError(f"Authentication error: {error_message}")
        else:
            raise APIError(f"Gemini API error: {error_message}")

    def get_supported_models(self) -> List[str]:
        """
        Get list of supported Gemini models.

        Returns:
            List of supported model names

        Example:
            >>> model = GeminiModel(config)
            >>> models = model.get_supported_models()
            >>> assert "gemini-pro" in models
        """
        return [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count

        Example:
            >>> model = GeminiModel(config)
            >>> tokens = model.estimate_tokens("Hello world")
            >>> assert tokens > 0
        """
        # Rough estimation: ~4 characters per token for English text
        return len(text) // 4 + 1

    def supports_vision(self) -> bool:
        """
        Check if current model supports vision capabilities.

        Returns:
            True if model supports vision, False otherwise

        Example:
            >>> model = GeminiModel(ModelConfig(model="gemini-pro-vision"))
            >>> assert model.supports_vision() == True
        """
        return "vision" in self.config.model.lower()