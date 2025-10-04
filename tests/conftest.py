"""
Pytest configuration and fixtures for AgentFrame tests.

This module provides common fixtures and configuration for testing
all components of the AgentFrame library.
"""

import pytest
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock

from agentframe import (
    ModelConfig, ToolRegistry, BaseTool, ToolResult,
    ChatHistory, MessageType, Intent, TaskType, IntentCategory
)
from agentframe.models.base import BaseModel, ModelResponse


class MockModel(BaseModel):
    """Mock model for testing."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._responses = []
        self._current_response = 0

    def _initialize_client(self):
        self._client = Mock()

    def set_responses(self, responses: List[str]):
        """Set predefined responses for the mock model."""
        self._responses = responses
        self._current_response = 0

    def generate(self, messages, tools=None, **kwargs):
        if self._current_response < len(self._responses):
            response = self._responses[self._current_response]
            self._current_response += 1
        else:
            response = "Mock response"

        return ModelResponse(
            content=response,
            tool_calls=[],
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            model=self.config.model
        )

    def generate_with_structured_output(self, messages, schema, **kwargs):
        return {"mock": "structured_output"}

    def stream_generate(self, messages, tools=None, **kwargs):
        response = "Mock streaming response"
        for word in response.split():
            yield word + " "


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, name: str = "mock_tool", should_fail: bool = False):
        super().__init__(
            name=name,
            description="A mock tool for testing",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": ["input"]
            }
        )
        self.should_fail = should_fail
        self.execution_count = 0

    def execute(self, args: Dict[str, Any]) -> ToolResult:
        self.execution_count += 1

        if self.should_fail:
            return ToolResult(
                success=False,
                error="Mock tool failure"
            )

        return ToolResult(
            success=True,
            result=f"Mock result for: {args.get('input', 'no input')}",
            metadata={"execution_count": self.execution_count}
        )


@pytest.fixture
def mock_model():
    """Fixture providing a mock model."""
    config = ModelConfig(api_key="test-key", model="test-model")
    return MockModel(config)


@pytest.fixture
def mock_tool():
    """Fixture providing a mock tool."""
    return MockTool()


@pytest.fixture
def failing_mock_tool():
    """Fixture providing a mock tool that fails."""
    return MockTool("failing_tool", should_fail=True)


@pytest.fixture
def tool_registry():
    """Fixture providing a tool registry with mock tools."""
    registry = ToolRegistry()
    registry.register(MockTool("calculator"))
    registry.register(MockTool("search"))
    return registry


@pytest.fixture
def sample_chat_history():
    """Fixture providing a chat history with sample messages."""
    history = ChatHistory()
    history.add_message(MessageType.HUMAN, "Hello!")
    history.add_message(MessageType.AI, "Hi there! How can I help you?")
    history.add_message(MessageType.HUMAN, "Calculate 2+2")
    history.add_message(MessageType.AI, "The result is 4.")
    return history


@pytest.fixture
def sample_intent():
    """Fixture providing a sample intent."""
    return Intent(
        primary_goal="Calculate a mathematical expression",
        entities=["calculation", "math", "expression"],
        task_type=TaskType.SINGLE_STEP,
        category=IntentCategory.REQUEST,
        confidence=0.9,
        extracted_parameters={"expression": "2+2"},
        requires_tools=True
    )


@pytest.fixture
def model_config():
    """Fixture providing a model configuration."""
    return ModelConfig(
        api_key="test-api-key",
        model="test-model",
        temperature=0.7,
        max_tokens=1000
    )


# Test data fixtures
@pytest.fixture
def sample_messages():
    """Fixture providing sample conversation messages."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
        {"role": "user", "content": "What's 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ]


@pytest.fixture
def sample_tool_schema():
    """Fixture providing a sample tool schema."""
    return {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Maximum results"}
        },
        "required": ["query"]
    }


@pytest.fixture
def sample_plan_data():
    """Fixture providing sample plan data."""
    return {
        "plan_id": "test_plan_1",
        "goal": "Calculate and search for information",
        "steps": [
            {
                "step_id": "step_1",
                "tool_name": "calculator",
                "arguments": {"expression": "2+2"},
                "expected_output": "4",
                "dependencies": [],
                "description": "Calculate 2+2"
            },
            {
                "step_id": "step_2",
                "tool_name": "search",
                "arguments": {"query": "math calculations"},
                "expected_output": "Search results",
                "dependencies": ["step_1"],
                "description": "Search for math information"
            }
        ],
        "confidence": 0.8
    }


# Mock external dependencies
@pytest.fixture(autouse=True)
def mock_external_apis(monkeypatch):
    """Mock external API calls to prevent real network requests during tests."""
    # Mock OpenAI
    mock_openai = Mock()
    monkeypatch.setattr("langchain_openai.ChatOpenAI", mock_openai)

    # Mock Anthropic
    mock_anthropic = Mock()
    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", mock_anthropic)

    # Mock Google
    mock_google = Mock()
    monkeypatch.setattr("langchain_google_genai.ChatGoogleGenerativeAI", mock_google)


# Configuration for test execution
def pytest_configure(config):
    """Configure pytest settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark tests in integration directory as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark tests that might be slow
        if any(keyword in item.name for keyword in ["test_large", "test_stress", "test_timeout"]):
            item.add_marker(pytest.mark.slow)