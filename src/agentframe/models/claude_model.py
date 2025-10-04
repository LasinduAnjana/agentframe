"""
Anthropic Claude model implementation for AgentFrame.

This module provides integration with Anthropic's Claude language models,
including Claude 3.5 Sonnet, Claude 3 Opus, and Claude 3 Haiku, with
support for tool use and structured output.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Union
import json
import time

from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic

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


class ClaudeModel(AgentBaseModel):
    """
    Anthropic Claude model implementation for AgentFrame.

    This class provides integration with Anthropic's Claude language models,
    supporting tool use, structured output, and streaming generation.

    Attributes:
        config: Model configuration
        _client: Claude client instance
        _supports_tools: Whether the model supports tool calling
        _supports_streaming: Whether the model supports streaming

    Example:
        >>> config = ModelConfig(
        ...     api_key="sk-ant-...",
        ...     model="claude-3-sonnet-20240229",
        ...     temperature=0.7
        ... )
        >>> model = ClaudeModel(config)
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response = model.generate(messages)
        >>> print(response.content)
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize Claude model.

        Args:
            config: Model configuration with Claude-specific settings

        Raises:
            ConfigurationError: If configuration is invalid

        Example:
            >>> config = ModelConfig(api_key="sk-ant-...", model="claude-3-sonnet-20240229")
            >>> model = ClaudeModel(config)
        """
        self._supports_tools = True
        self._supports_streaming = True
        super().__init__(config)

        # Validate Claude-specific configuration
        if not self.config.api_key.startswith('sk-ant-'):
            logger.warning("Claude API key should start with 'sk-ant-'")

    def _initialize_client(self) -> None:
        """Initialize the Claude client using LangChain wrapper."""
        try:
            client_kwargs = {
                "model": self.config.model,
                "anthropic_api_key": self.config.api_key,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries
            }

            # Add optional parameters
            if self.config.base_url:
                client_kwargs["anthropic_api_url"] = self.config.base_url

            # Add custom parameters
            client_kwargs.update(self.config.custom_params)

            self._client = ChatAnthropic(**client_kwargs)

            logger.debug(f"Initialized Claude client for model: {self.config.model}")

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Claude client: {e}")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> ModelResponse:
        """
        Generate a response using Claude model.

        Args:
            messages: Conversation messages
            tools: Available tools for tool use
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

            # Prepare tool use if tools are provided
            invoke_kwargs = {}
            if tools:
                invoke_kwargs["tools"] = self._convert_tools_to_claude_format(tools)

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

            Important: Only return the JSON object, no additional text, explanations, or formatting.
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
            tools: Available tools for tool use
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
                stream_kwargs["tools"] = self._convert_tools_to_claude_format(tools)

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
            SystemMessage,
            ToolMessage
        )

        langchain_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                # Handle tool calls if present
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
            elif role == "tool":
                langchain_messages.append(
                    ToolMessage(
                        content=content,
                        tool_call_id=msg.get("tool_call_id", "unknown")
                    )
                )

        return langchain_messages

    def _convert_tools_to_claude_format(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert tools to Claude tool use format."""
        claude_tools = []

        for tool in tools:
            tool_def = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters
            }
            claude_tools.append(tool_def)

        return claude_tools

    def _convert_response(self, response: Any, execution_time: float) -> ModelResponse:
        """Convert LangChain response to AgentFrame format."""
        content = response.content if hasattr(response, 'content') else str(response)

        # Extract tool calls if present
        tool_calls = []
        if hasattr(response, 'additional_kwargs'):
            additional_kwargs = response.additional_kwargs
            if "tool_calls" in additional_kwargs:
                for call in additional_kwargs["tool_calls"]:
                    tool_calls.append({
                        "name": call.get("name"),
                        "arguments": call.get("input", {})
                    })

        # Extract usage information
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
            finish_reason=getattr(response, 'stop_reason', None),
            usage=usage,
            model=self.config.model,
            metadata={
                "execution_time": execution_time,
                "provider": "claude"
            }
        )

    def _handle_exception(self, e: Exception) -> ModelResponse:
        """Handle and convert exceptions to appropriate errors."""
        error_message = str(e)

        # Check for specific Claude errors
        if "rate limit" in error_message.lower() or "quota" in error_message.lower():
            raise RateLimitError(f"Claude rate limit exceeded: {error_message}")
        elif "maximum context length" in error_message.lower() or "token limit" in error_message.lower():
            raise TokenLimitError(f"Token limit exceeded: {error_message}")
        elif "api key" in error_message.lower() or "authentication" in error_message.lower():
            raise ConfigurationError(f"Authentication error: {error_message}")
        else:
            raise APIError(f"Claude API error: {error_message}")

    def get_supported_models(self) -> List[str]:
        """
        Get list of supported Claude models.

        Returns:
            List of supported model names

        Example:
            >>> model = ClaudeModel(config)
            >>> models = model.get_supported_models()
            >>> assert "claude-3-sonnet-20240229" in models
        """
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count

        Example:
            >>> model = ClaudeModel(config)
            >>> tokens = model.estimate_tokens("Hello world")
            >>> assert tokens > 0
        """
        # Rough estimation: ~4 characters per token for English text
        return len(text) // 4 + 1

    def get_context_window(self) -> int:
        """
        Get the context window size for the current model.

        Returns:
            Context window size in tokens

        Example:
            >>> model = ClaudeModel(ModelConfig(model="claude-3-sonnet-20240229"))
            >>> window = model.get_context_window()
            >>> assert window > 0
        """
        context_windows = {
            "claude-3-5-sonnet-20241022": 200000,
            "claude-3-5-sonnet-20240620": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000
        }

        return context_windows.get(self.config.model, 200000)