# AgentFrame

[![PyPI version](https://badge.fury.io/py/agentframe.svg)](https://badge.fury.io/py/agentframe)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful Python framework for building LLM-based chat agents with planning, replanning, and tool integration capabilities.

## 🚀 Key Features

- **Intelligent Planning & Replanning**: Automatically generates and adjusts execution plans based on user goals and available tools
- **Multi-Model Support**: Works with OpenAI GPT, Google Gemini, and Anthropic Claude models
- **Easy Tool Integration**: Simple `@tool` decorator for creating custom tools with automatic schema generation
- **Conversation Memory**: Built-in chat history management with token-aware truncation
- **Intent Understanding**: LLM-powered intent parsing to understand user goals
- **Type Safety**: Comprehensive type hints throughout the codebase
- **Extensible Architecture**: Clean interfaces for adding new models, tools, and planning strategies

## 📦 Installation

```bash
pip install agentframe
```

For development with all optional dependencies:

```bash
pip install agentframe[dev,test,docs]
```

## 🏃 Quick Start

```python
from agentframe import Agent, OpenAIModel, tool

# Define custom tools
@tool
def search_web(query: str) -> dict:
    """Search the web for information.

    Args:
        query: The search query string

    Returns:
        Dictionary containing search results
    """
    # Your search implementation here
    return {"results": ["Example result 1", "Example result 2"]}

@tool
def calculate(expression: str) -> float:
    """Calculate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate

    Returns:
        The calculated result
    """
    # Safe evaluation implementation
    return eval(expression)  # In production, use a safe evaluator

# Initialize the model
model = OpenAIModel(
    api_key="your-openai-api-key",
    model="gpt-4"
)

# Create agent with tools
agent = Agent(
    model=model,
    tools=[search_web, calculate],
    max_replanning_iterations=3,
    verbose=True
)

# Use the agent
response = agent.run("Search for the population of Tokyo and calculate 15% of it")
print(response)

# Access conversation history
history = agent.chat_history.get_recent(n=5)
for message in history:
    print(f"{message.type}: {message.content}")
```

## 🏗️ Architecture

AgentFrame uses a modular architecture built on LangGraph for workflow orchestration:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│   Intent Parser   │───▶│     Planner     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐             ▼
│   Response      │◀───│   Tool Executor   │◀───┌─────────────────┐
└─────────────────┘    └──────────────────┘    │  Plan Execution │
                                                └─────────────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │   Replanning    │
                                               │   (if needed)   │
                                               └─────────────────┘
```

## 🛠️ Supported Models

- **OpenAI**: GPT-4, GPT-4-turbo, GPT-3.5-turbo
- **Google Gemini**: Gemini Pro, Gemini Pro Vision
- **Anthropic Claude**: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku

```python
from agentframe import OpenAIModel, GeminiModel, ClaudeModel

# OpenAI
openai_model = OpenAIModel(api_key="sk-...", model="gpt-4")

# Gemini
gemini_model = GeminiModel(api_key="your-key", model="gemini-pro")

# Claude
claude_model = ClaudeModel(api_key="your-key", model="claude-3-sonnet-20240229")
```

## 🔧 Tool Creation

Create tools easily with the `@tool` decorator:

```python
from agentframe import tool
from typing import List

@tool
def fetch_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: Name of the city
        units: Temperature units ("celsius" or "fahrenheit")

    Returns:
        Dictionary with weather information
    """
    # Your weather API implementation
    return {
        "city": city,
        "temperature": 22,
        "units": units,
        "condition": "sunny"
    }

@tool
def analyze_data(data: List[float]) -> dict:
    """Analyze numerical data and return statistics.

    Args:
        data: List of numerical values to analyze

    Returns:
        Dictionary with statistical analysis
    """
    return {
        "mean": sum(data) / len(data),
        "min": min(data),
        "max": max(data),
        "count": len(data)
    }
```

## 🐳 Docker Support

Run AgentFrame in Docker:

```bash
# Development environment
docker-compose up agentframe-dev

# Run tests
docker-compose up agentframe-test

# Serve documentation
docker-compose up agentframe-docs
```

## 📚 Documentation

- [Getting Started Guide](docs/getting_started.md)
- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Examples](docs/examples.md)
- [Publishing Guide](docs/publishing.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Inspired by the broader LLM agent ecosystem
- Thanks to all contributors and the open-source community

## 📞 Support

- 📚 [Documentation](https://agentframe.github.io/agentframe)
- 🐛 [Issue Tracker](https://github.com/agentframe/agentframe/issues)
- 💬 [Discussions](https://github.com/agentframe/agentframe/discussions)