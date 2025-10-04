# Getting Started with AgentFrame

AgentFrame is a powerful Python framework for building LLM-based chat agents with planning, replanning, and tool integration capabilities. This guide will help you get started quickly.

## Installation

### From PyPI (Recommended)

```bash
pip install agentframe
```

### From Source

```bash
git clone https://github.com/agentframe/agentframe.git
cd agentframe
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/agentframe/agentframe.git
cd agentframe
pip install -e ".[dev,test,docs]"
```

## Quick Start

### 1. Basic Agent Setup

```python
from agentframe import Agent, OpenAIModel, ModelConfig, tool

# Define a simple tool
@tool
def calculator(expression: str) -> float:
    """Calculate a mathematical expression."""
    return eval(expression)  # In production, use safe evaluation

# Configure the model
config = ModelConfig(
    api_key="your-openai-api-key",
    model="gpt-4",
    temperature=0.7
)
model = OpenAIModel(config)

# Create the agent
agent = Agent(model=model, tools=[calculator])

# Use the agent
response = agent.run("What's 15% of 200?")
print(response)
```

### 2. Multiple Tools Example

```python
from agentframe import Agent, OpenAIModel, ModelConfig, tool

@tool
def search_web(query: str) -> dict:
    """Search the web for information."""
    # Your search implementation here
    return {"results": ["Example result 1", "Example result 2"]}

@tool
def weather(city: str) -> dict:
    """Get weather information for a city."""
    # Your weather API implementation here
    return {"city": city, "temperature": 22, "condition": "sunny"}

@tool
def calculator(expression: str) -> float:
    """Calculate mathematical expressions."""
    import math
    # Safe evaluation implementation
    return eval(expression)

# Create agent with multiple tools
agent = Agent(
    model=model,
    tools=[search_web, weather, calculator]
)

# The agent can now plan multi-step tasks
response = agent.run(
    "Search for the weather in Tokyo, then calculate how many degrees "
    "warmer it would be in Fahrenheit"
)
print(response)
```

## Core Concepts

### Agents

The `Agent` class is the main orchestrator that coordinates:
- **Intent Parsing**: Understanding what the user wants
- **Planning**: Breaking down complex tasks into steps
- **Execution**: Running tools to accomplish goals
- **Replanning**: Adjusting strategy when things don't work

### Tools

Tools are functions that agents can use to interact with the world:

```python
@tool
def fetch_user_data(user_id: str) -> dict:
    """Fetch user data from database.

    Args:
        user_id: Unique identifier for the user

    Returns:
        Dictionary containing user information
    """
    # Your implementation here
    return {"id": user_id, "name": "John Doe"}
```

Key features:
- Automatic schema generation from function signatures
- Type validation
- Error handling
- Metadata tracking

### Models

AgentFrame supports multiple LLM providers:

```python
# OpenAI
from agentframe import OpenAIModel, ModelConfig
config = ModelConfig(api_key="sk-...", model="gpt-4")
openai_model = OpenAIModel(config)

# Anthropic Claude
from agentframe import ClaudeModel
claude_config = ModelConfig(api_key="sk-ant-...", model="claude-3-sonnet-20240229")
claude_model = ClaudeModel(claude_config)

# Google Gemini
from agentframe import GeminiModel
gemini_config = ModelConfig(api_key="your-key", model="gemini-pro")
gemini_model = GeminiModel(gemini_config)
```

### Planning and Execution

AgentFrame automatically:
1. **Analyzes** user intent and available tools
2. **Plans** a sequence of tool calls to achieve the goal
3. **Executes** the plan step by step
4. **Replans** if execution fails or new information is needed

## Configuration

### Agent Configuration

```python
from agentframe import AgentConfig

config = AgentConfig(
    max_replanning_iterations=3,  # How many times to retry with new plans
    enable_streaming=True,        # Stream responses as they're generated
    verbose_logging=True,         # Detailed logging for debugging
    timeout=300.0,               # Overall timeout in seconds
    confidence_threshold=0.7     # Minimum confidence for plan acceptance
)

agent = Agent(model=model, tools=tools, config=config)
```

### Model Configuration

```python
from agentframe import ModelConfig

config = ModelConfig(
    api_key="your-api-key",
    model="gpt-4",
    temperature=0.7,      # Creativity level (0.0-2.0)
    max_tokens=1000,      # Maximum response length
    top_p=1.0,           # Nucleus sampling parameter
    timeout=30,          # Request timeout
    max_retries=3        # Retry attempts on failure
)
```

## Error Handling

AgentFrame provides robust error handling:

```python
from agentframe import ModelError, ValidationError

try:
    response = agent.run("Process this request")
except ModelError as e:
    print(f"Model error: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Conversation Management

Access and manage conversation history:

```python
# Get recent messages
recent = agent.chat_history.get_recent(5)
for message in recent:
    print(f"{message.type}: {message.content}")

# Get conversation summary
summary = agent.get_conversation_summary()
print(f"Total messages: {summary['total_messages']}")
print(f"Total tokens: {summary['total_tokens']}")

# Reset conversation
agent.reset_conversation()

# Save/load conversations
agent.save_conversation("conversation.json")
agent.load_conversation("conversation.json")
```

## Best Practices

### 1. Tool Design

- **Single Purpose**: Each tool should do one thing well
- **Clear Documentation**: Use descriptive docstrings
- **Type Hints**: Help with automatic schema generation
- **Error Handling**: Return meaningful error messages

```python
@tool
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email to a recipient.

    Args:
        to: Email address of recipient
        subject: Email subject line
        body: Email body content

    Returns:
        Dictionary with sending status

    Raises:
        ValueError: If email address is invalid
    """
    if "@" not in to:
        raise ValueError("Invalid email address")

    # Send email logic here
    return {"status": "sent", "message_id": "12345"}
```

### 2. Security

- **Validate Inputs**: Always validate user inputs
- **Sanitize Data**: Clean data before processing
- **Limit Permissions**: Only give tools necessary permissions
- **API Key Management**: Use environment variables

```python
import os

# Use environment variables for API keys
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable required")

config = ModelConfig(api_key=api_key, model="gpt-4")
```

### 3. Performance

- **Tool Efficiency**: Make tools fast and reliable
- **Caching**: Cache expensive operations
- **Token Management**: Monitor token usage
- **Timeout Handling**: Set appropriate timeouts

### 4. Testing

```python
# Test your tools independently
def test_calculator_tool():
    result = calculator.execute({"expression": "2+2"})
    assert result.success == True
    assert result.result == 4

# Mock external services in tests
from unittest.mock import Mock
mock_api = Mock()
mock_api.search.return_value = {"results": ["test"]}
```

## Next Steps

- Read the [Architecture Guide](architecture.md) to understand how AgentFrame works
- Check out more [Examples](examples.md) for advanced use cases
- Explore the [API Reference](api_reference.md) for detailed documentation
- Learn about [Publishing](publishing.md) your own AgentFrame applications

## Troubleshooting

### Common Issues

1. **API Key Errors**: Make sure your API keys are correctly set
2. **Tool Registration**: Ensure tools are properly registered
3. **Import Errors**: Check that all dependencies are installed
4. **Token Limits**: Monitor token usage and implement limits

### Getting Help

- Check the [GitHub Issues](https://github.com/agentframe/agentframe/issues)
- Read the [Documentation](https://agentframe.github.io/agentframe)
- Join the [Community Discussions](https://github.com/agentframe/agentframe/discussions)

## Environment Variables

AgentFrame respects these environment variables:

```bash
# API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="your-google-key"

# Logging
export AGENTFRAME_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR

# Development
export AGENTFRAME_ENV="development"  # development, production
```