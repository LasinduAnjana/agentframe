"""
OpenAI model implementation for AgentFrame.

This module provides integration with OpenAI's language models, including
GPT-4, GPT-4-turbo, and GPT-3.5-turbo, with support for function calling
and structured output.
"""

import logging
from typing import Any, Dict, Iterator, List, Optional, Union
import json
import time

from pydantic import BaseModel
from langchain_openai import ChatOpenAI

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


class OpenAIModel(AgentBaseModel):
    """
    OpenAI model implementation for AgentFrame.

    This class provides integration with OpenAI's language models, supporting
    function calling, structured output, and streaming generation.

    Attributes:
        config: Model configuration
        _client: OpenAI client instance
        _supports_tools: Whether the model supports tool calling
        _supports_streaming: Whether the model supports streaming

    Example:
        >>> config = ModelConfig(
        ...     api_key="sk-...",
        ...     model="gpt-4",
        ...     temperature=0.7
        ... )
        >>> model = OpenAIModel(config)
        >>> messages = [{"role": "user", "content": "Hello!"}]
        >>> response = model.generate(messages)
        >>> print(response.content)
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        Initialize OpenAI model.

        Args:
            config: Model configuration with OpenAI-specific settings

        Raises:
            ConfigurationError: If configuration is invalid

        Example:
            >>> config = ModelConfig(api_key="sk-...", model="gpt-4")
            >>> model = OpenAIModel(config)
        """
        self._supports_tools = True
        self._supports_streaming = True
        super().__init__(config)

        # Validate OpenAI-specific configuration
        if not self.config.api_key.startswith(('sk-', 'sk-proj-')):
            logger.warning("OpenAI API key should start with 'sk-' or 'sk-proj-'")

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client using LangChain wrapper."""
        try:
            client_kwargs = {
                "model": self.config.model,
                "openai_api_key": self.config.api_key,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries
            }

            # Add optional parameters
            if self.config.base_url:
                client_kwargs["openai_api_base"] = self.config.base_url

            if self.config.organization:
                client_kwargs["openai_organization"] = self.config.organization

            # Add custom parameters
            client_kwargs.update(self.config.custom_params)

            self._client = ChatOpenAI(**client_kwargs)

            logger.debug(f"Initialized OpenAI client for model: {self.config.model}")

        except Exception as e:
            raise ConfigurationError(f"Failed to initialize OpenAI client: {e}")

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs: Any
    ) -> ModelResponse:
        """
        Generate a response using OpenAI model.

        Args:
            messages: Conversation messages in OpenAI format
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
                invoke_kwargs["functions"] = self._convert_tools_to_functions(tools)

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

            Only return the JSON, no additional text.
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
                return json.loads(response.content)
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
                stream_kwargs["functions"] = self._convert_tools_to_functions(tools)

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
            FunctionMessage
        )

        langchain_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                # Handle function calls if present
                if "function_call" in msg:
                    ai_msg = AIMessage(
                        content=content,
                        additional_kwargs={"function_call": msg["function_call"]}
                    )
                    langchain_messages.append(ai_msg)
                else:
                    langchain_messages.append(AIMessage(content=content))
            elif role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "function":
                langchain_messages.append(
                    FunctionMessage(
                        content=content,
                        name=msg.get("name", "unknown")
                    )
                )

        return langchain_messages

    def _convert_tools_to_functions(self, tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert tools to OpenAI function calling format."""
        functions = []

        for tool in tools:
            function_def = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            functions.append(function_def)

        return functions

    def _convert_response(self, response: Any, execution_time: float) -> ModelResponse:
        """Convert LangChain response to AgentFrame format."""
        content = response.content if hasattr(response, 'content') else str(response)

        # Extract tool calls if present
        tool_calls = []
        if hasattr(response, 'additional_kwargs'):
            additional_kwargs = response.additional_kwargs
            if "function_call" in additional_kwargs:
                # Single function call format
                func_call = additional_kwargs["function_call"]
                tool_calls.append({
                    "name": func_call.get("name"),
                    "arguments": func_call.get("arguments")
                })
            elif "tool_calls" in additional_kwargs:
                # Multiple tool calls format
                for call in additional_kwargs["tool_calls"]:
                    if call.get("type") == "function":
                        func = call.get("function", {})
                        tool_calls.append({
                            "name": func.get("name"),
                            "arguments": func.get("arguments")
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
            finish_reason=getattr(response, 'finish_reason', None),
            usage=usage,
            model=self.config.model,
            metadata={
                "execution_time": execution_time,
                "provider": "openai"
            }
        )

    def _handle_exception(self, e: Exception) -> ModelResponse:
        """Handle and convert exceptions to appropriate errors."""
        error_message = str(e)

        # Check for specific OpenAI errors
        if "rate limit" in error_message.lower():
            raise RateLimitError(f"OpenAI rate limit exceeded: {error_message}")
        elif "maximum context length" in error_message.lower():
            raise TokenLimitError(f"Token limit exceeded: {error_message}")
        elif "api key" in error_message.lower():
            raise ConfigurationError(f"API key error: {error_message}")
        else:
            raise APIError(f"OpenAI API error: {error_message}")

    def get_supported_models(self) -> List[str]:
        """
        Get list of supported OpenAI models.

        Returns:
            List of supported model names

        Example:
            >>> model = OpenAIModel(config)
            >>> models = model.get_supported_models()
            >>> assert "gpt-4" in models
        """
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4-turbo-preview",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-16k"
        ]

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count

        Example:
            >>> model = OpenAIModel(config)
            >>> tokens = model.estimate_tokens("Hello world")
            >>> assert tokens > 0
        """
        # Rough estimation: ~4 characters per token for English text
        return len(text) // 4 + 1