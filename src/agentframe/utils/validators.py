"""
Input validation utilities for AgentFrame.

This module provides validation functions for user inputs, model configurations,
tool parameters, and other system inputs to ensure data integrity and security.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exception raised when validation fails."""
    pass


def validate_api_key(api_key: str, provider: str = "unknown") -> bool:
    """
    Validate API key format for different providers.

    Args:
        api_key: API key to validate
        provider: Provider name (openai, anthropic, google)

    Returns:
        True if API key format is valid

    Raises:
        ValidationError: If API key is invalid

    Example:
        >>> validate_api_key("sk-1234567890", "openai")
        True
        >>> validate_api_key("invalid", "openai")
        ValidationError: Invalid OpenAI API key format
    """
    if not api_key or not isinstance(api_key, str):
        raise ValidationError("API key must be a non-empty string")

    provider = provider.lower()

    # OpenAI API key validation
    if provider == "openai":
        if not (api_key.startswith("sk-") or api_key.startswith("sk-proj-")):
            raise ValidationError("Invalid OpenAI API key format")
        if len(api_key) < 10:
            raise ValidationError("OpenAI API key too short")

    # Anthropic Claude API key validation
    elif provider == "anthropic" or provider == "claude":
        if not api_key.startswith("sk-ant-"):
            raise ValidationError("Invalid Anthropic API key format")
        if len(api_key) < 15:
            raise ValidationError("Anthropic API key too short")

    # Google API key validation (basic)
    elif provider == "google" or provider == "gemini":
        if len(api_key) < 10:
            raise ValidationError("Google API key too short")

    # Generic validation for unknown providers
    else:
        if len(api_key) < 8:
            raise ValidationError("API key too short")

    return True


def validate_model_name(model_name: str, provider: str = "unknown") -> bool:
    """
    Validate model name for different providers.

    Args:
        model_name: Model name to validate
        provider: Provider name

    Returns:
        True if model name is valid

    Raises:
        ValidationError: If model name is invalid

    Example:
        >>> validate_model_name("gpt-4", "openai")
        True
        >>> validate_model_name("invalid-model", "openai")
        ValidationError: Invalid OpenAI model name
    """
    if not model_name or not isinstance(model_name, str):
        raise ValidationError("Model name must be a non-empty string")

    provider = provider.lower()

    # OpenAI model validation
    if provider == "openai":
        valid_models = [
            "gpt-4", "gpt-4-turbo", "gpt-4-turbo-preview",
            "gpt-4-0125-preview", "gpt-4-1106-preview",
            "gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106"
        ]
        if model_name not in valid_models:
            logger.warning(f"Unknown OpenAI model: {model_name}")
            # Don't raise error for forward compatibility

    # Anthropic model validation
    elif provider == "anthropic" or provider == "claude":
        if not model_name.startswith("claude-"):
            raise ValidationError("Anthropic models should start with 'claude-'")

    # Google model validation
    elif provider == "google" or provider == "gemini":
        if not model_name.startswith("gemini"):
            raise ValidationError("Google models should start with 'gemini'")

    return True


def validate_temperature(temperature: float) -> bool:
    """
    Validate temperature parameter.

    Args:
        temperature: Temperature value to validate

    Returns:
        True if temperature is valid

    Raises:
        ValidationError: If temperature is invalid

    Example:
        >>> validate_temperature(0.7)
        True
        >>> validate_temperature(3.0)
        ValidationError: Temperature must be between 0.0 and 2.0
    """
    if not isinstance(temperature, (int, float)):
        raise ValidationError("Temperature must be a number")

    if not 0.0 <= temperature <= 2.0:
        raise ValidationError("Temperature must be between 0.0 and 2.0")

    return True


def validate_max_tokens(max_tokens: int) -> bool:
    """
    Validate max_tokens parameter.

    Args:
        max_tokens: Maximum tokens value to validate

    Returns:
        True if max_tokens is valid

    Raises:
        ValidationError: If max_tokens is invalid

    Example:
        >>> validate_max_tokens(1000)
        True
        >>> validate_max_tokens(-1)
        ValidationError: Max tokens must be positive
    """
    if not isinstance(max_tokens, int):
        raise ValidationError("Max tokens must be an integer")

    if max_tokens <= 0:
        raise ValidationError("Max tokens must be positive")

    if max_tokens > 1000000:  # Reasonable upper limit
        raise ValidationError("Max tokens too large (max: 1,000,000)")

    return True


def validate_url(url: str) -> bool:
    """
    Validate URL format.

    Args:
        url: URL to validate

    Returns:
        True if URL is valid

    Raises:
        ValidationError: If URL is invalid

    Example:
        >>> validate_url("https://api.openai.com")
        True
        >>> validate_url("invalid-url")
        ValidationError: Invalid URL format
    """
    if not url or not isinstance(url, str):
        raise ValidationError("URL must be a non-empty string")

    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            raise ValidationError("Invalid URL format")

        if result.scheme not in ["http", "https"]:
            raise ValidationError("URL must use HTTP or HTTPS")

    except Exception as e:
        raise ValidationError(f"Invalid URL: {e}")

    return True


def validate_tool_name(tool_name: str) -> bool:
    """
    Validate tool name format.

    Args:
        tool_name: Tool name to validate

    Returns:
        True if tool name is valid

    Raises:
        ValidationError: If tool name is invalid

    Example:
        >>> validate_tool_name("search_web")
        True
        >>> validate_tool_name("invalid tool name!")
        ValidationError: Tool name contains invalid characters
    """
    if not tool_name or not isinstance(tool_name, str):
        raise ValidationError("Tool name must be a non-empty string")

    # Tool names should be valid Python identifiers
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tool_name):
        raise ValidationError("Tool name contains invalid characters")

    if len(tool_name) > 50:
        raise ValidationError("Tool name too long (max: 50 characters)")

    if tool_name.startswith('_'):
        logger.warning(f"Tool name '{tool_name}' starts with underscore")

    return True


def validate_json_schema(schema: Dict[str, Any]) -> bool:
    """
    Validate JSON schema format.

    Args:
        schema: JSON schema to validate

    Returns:
        True if schema is valid

    Raises:
        ValidationError: If schema is invalid

    Example:
        >>> schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        >>> validate_json_schema(schema)
        True
    """
    if not isinstance(schema, dict):
        raise ValidationError("Schema must be a dictionary")

    # Check required fields
    if "type" not in schema:
        raise ValidationError("Schema must have a 'type' field")

    valid_types = ["object", "array", "string", "number", "integer", "boolean", "null"]
    if schema["type"] not in valid_types:
        raise ValidationError(f"Invalid schema type: {schema['type']}")

    # Validate object schema
    if schema["type"] == "object":
        if "properties" in schema:
            if not isinstance(schema["properties"], dict):
                raise ValidationError("Properties must be a dictionary")

            # Recursively validate property schemas
            for prop_name, prop_schema in schema["properties"].items():
                if isinstance(prop_schema, dict):
                    validate_json_schema(prop_schema)

        if "required" in schema:
            if not isinstance(schema["required"], list):
                raise ValidationError("Required must be a list")

            # Check that required properties exist
            properties = schema.get("properties", {})
            for req_prop in schema["required"]:
                if req_prop not in properties:
                    raise ValidationError(f"Required property '{req_prop}' not in properties")

    # Validate array schema
    elif schema["type"] == "array":
        if "items" in schema:
            if isinstance(schema["items"], dict):
                validate_json_schema(schema["items"])

    return True


def validate_tool_arguments(
    args: Dict[str, Any],
    schema: Dict[str, Any]
) -> bool:
    """
    Validate tool arguments against a schema.

    Args:
        args: Arguments to validate
        schema: JSON schema to validate against

    Returns:
        True if arguments are valid

    Raises:
        ValidationError: If arguments are invalid

    Example:
        >>> args = {"name": "John", "age": 30}
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "integer"}
        ...     },
        ...     "required": ["name"]
        ... }
        >>> validate_tool_arguments(args, schema)
        True
    """
    if not isinstance(args, dict):
        raise ValidationError("Arguments must be a dictionary")

    # Validate schema first
    validate_json_schema(schema)

    if schema.get("type") != "object":
        raise ValidationError("Schema must be for an object type")

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    # Check required properties
    for req_prop in required:
        if req_prop not in args:
            raise ValidationError(f"Missing required argument: {req_prop}")

    # Validate each argument
    for arg_name, arg_value in args.items():
        if arg_name not in properties:
            logger.warning(f"Unexpected argument: {arg_name}")
            continue

        prop_schema = properties[arg_name]
        _validate_value_against_schema(arg_value, prop_schema, arg_name)

    return True


def _validate_value_against_schema(
    value: Any,
    schema: Dict[str, Any],
    field_name: str = "value"
) -> bool:
    """Validate a value against a JSON schema."""
    expected_type = schema.get("type")

    if expected_type == "string":
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string")

    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValidationError(f"{field_name} must be an integer")

    elif expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValidationError(f"{field_name} must be a number")

    elif expected_type == "boolean":
        if not isinstance(value, bool):
            raise ValidationError(f"{field_name} must be a boolean")

    elif expected_type == "array":
        if not isinstance(value, list):
            raise ValidationError(f"{field_name} must be an array")

        # Validate array items if schema is provided
        if "items" in schema:
            items_schema = schema["items"]
            for i, item in enumerate(value):
                _validate_value_against_schema(item, items_schema, f"{field_name}[{i}]")

    elif expected_type == "object":
        if not isinstance(value, dict):
            raise ValidationError(f"{field_name} must be an object")

        # Recursively validate object properties
        if "properties" in schema:
            for prop_name, prop_value in value.items():
                if prop_name in schema["properties"]:
                    prop_schema = schema["properties"][prop_name]
                    _validate_value_against_schema(
                        prop_value, prop_schema, f"{field_name}.{prop_name}"
                    )

    return True


def sanitize_user_input(user_input: str, max_length: int = 10000) -> str:
    """
    Sanitize user input for safety.

    Args:
        user_input: User input to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized user input

    Raises:
        ValidationError: If input is invalid

    Example:
        >>> sanitized = sanitize_user_input("Hello world!")
        >>> assert sanitized == "Hello world!"
    """
    if not isinstance(user_input, str):
        raise ValidationError("User input must be a string")

    if len(user_input) > max_length:
        raise ValidationError(f"Input too long (max: {max_length} characters)")

    # Remove potential security risks
    sanitized = user_input.strip()

    # Remove null bytes
    sanitized = sanitized.replace('\x00', '')

    # Limit consecutive whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized)

    return sanitized


def validate_file_path(file_path: str, allowed_extensions: Optional[List[str]] = None) -> bool:
    """
    Validate file path for safety.

    Args:
        file_path: File path to validate
        allowed_extensions: List of allowed file extensions

    Returns:
        True if file path is valid

    Raises:
        ValidationError: If file path is invalid

    Example:
        >>> validate_file_path("/safe/path/file.json", [".json", ".txt"])
        True
    """
    if not file_path or not isinstance(file_path, str):
        raise ValidationError("File path must be a non-empty string")

    # Check for path traversal attempts
    if ".." in file_path or file_path.startswith("/"):
        raise ValidationError("Unsafe file path detected")

    # Check file extension if provided
    if allowed_extensions:
        import os
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in [e.lower() for e in allowed_extensions]:
            raise ValidationError(f"File extension '{ext}' not allowed")

    return True


def validate_configuration(config: Dict[str, Any]) -> List[str]:
    """
    Validate a configuration dictionary.

    Args:
        config: Configuration to validate

    Returns:
        List of validation warnings (empty if all valid)

    Example:
        >>> config = {"api_key": "sk-test", "model": "gpt-4", "temperature": 0.7}
        >>> warnings = validate_configuration(config)
        >>> if not warnings:
        ...     print("Configuration is valid")
    """
    warnings = []

    # Check for required fields
    required_fields = ["api_key", "model"]
    for field in required_fields:
        if field not in config:
            warnings.append(f"Missing required field: {field}")

    # Validate individual fields
    try:
        if "api_key" in config:
            validate_api_key(config["api_key"])
    except ValidationError as e:
        warnings.append(f"API key validation failed: {e}")

    try:
        if "model" in config:
            validate_model_name(config["model"])
    except ValidationError as e:
        warnings.append(f"Model validation failed: {e}")

    try:
        if "temperature" in config:
            validate_temperature(config["temperature"])
    except ValidationError as e:
        warnings.append(f"Temperature validation failed: {e}")

    try:
        if "max_tokens" in config:
            validate_max_tokens(config["max_tokens"])
    except ValidationError as e:
        warnings.append(f"Max tokens validation failed: {e}")

    return warnings