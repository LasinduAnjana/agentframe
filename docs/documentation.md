# AgentFrame - Complete Documentation

AgentFrame is a comprehensive Python framework for building intelligent LLM-based agents with advanced planning, tool integration, and conversation management capabilities. This framework enables developers to create sophisticated AI agents that can understand user intents, plan multi-step solutions, execute tasks using available tools, and adapt when initial approaches fail.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Installation](#installation)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Model Providers](#model-providers)
7. [Tool System](#tool-system)
8. [Prompt Customization](#prompt-customization)
9. [Agent Configuration](#agent-configuration)
10. [Planning and Execution](#planning-and-execution)
11. [Memory and Conversation](#memory-and-conversation)
12. [FastAPI Integration](#fastapi-integration)
13. [Examples](#examples)
14. [API Reference](#api-reference)
15. [Best Practices](#best-practices)
16. [Troubleshooting](#troubleshooting)

## Quick Start

```python
from agentframe import Agent, OpenAIModel, ModelConfig, tool

# Define a tool
@tool
def calculator(expression: str) -> float:
    """Calculate mathematical expressions."""
    return eval(expression)

# Configure model
config = ModelConfig(api_key="your-api-key", model="gpt-4")
model = OpenAIModel(config)

# Create agent
agent = Agent(model=model, tools=[calculator])

# Use the agent
response = agent.run("What's 15% of 200?")
print(response)
```

## Core Concepts

### Agents

Agents are the central component of AgentFrame. They combine:
- **Language Models**: For understanding and generation
- **Tools**: For taking actions and accessing information
- **Planning**: For breaking down complex tasks
- **Memory**: For maintaining conversation context
- **Prompts**: For customizing behavior and personality

### Planning and Execution

AgentFrame uses a sophisticated planning system:
1. **Intent Parsing**: Understanding what the user wants to achieve
2. **Plan Generation**: Creating step-by-step execution strategies
3. **Tool Selection**: Choosing appropriate tools for each step
4. **Execution**: Running the plan with error handling
5. **Replanning**: Adapting when plans fail or need adjustment

### Tool Integration

Tools extend agent capabilities:
- **Function Tools**: Python functions wrapped as agent tools
- **API Tools**: Integration with external services
- **Custom Tools**: User-defined tool implementations
- **Tool Registry**: Centralized tool management and discovery

### Conversation Management

Advanced memory and context handling:
- **Message History**: Tracking conversation flow
- **Context Windows**: Managing token limits intelligently
- **Session Management**: Persistent conversations across interactions
- **History Export**: Saving and loading conversation data

## Installation

### From PyPI (Recommended)

```bash
pip install agentframe
```

### From Source

```bash
git clone https://github.com/LasinduAnjana/agentframe.git
cd agentframe
pip install -e .
```

### Dependencies

AgentFrame requires Python 3.8+ and includes:
- `langchain`: LLM integration and orchestration
- `langgraph`: Workflow and state management
- `pydantic`: Data validation and settings
- `openai`: OpenAI API client
- `google-generativeai`: Google Gemini integration
- `anthropic`: Claude API integration

## Basic Usage

### Creating Your First Agent

```python
from agentframe import Agent, OpenAIModel, ModelConfig

# 1. Configure the language model
config = ModelConfig(
    api_key="your-openai-api-key",
    model="gpt-4",
    temperature=0.7,
    max_tokens=1000
)

# 2. Initialize the model
model = OpenAIModel(config)

# 3. Create the agent
agent = Agent(model=model)

# 4. Interact with the agent
response = agent.run("Hello! Can you help me plan a project?")
print(response)
```

### Adding Tools

```python
from agentframe import tool

@tool
def search_web(query: str) -> dict:
    """Search the web for information."""
    # Implementation here
    return {"results": ["result1", "result2"]}

@tool
def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email."""
    # Implementation here
    return True

# Create agent with tools
agent = Agent(
    model=model,
    tools=[search_web, send_email]
)

# The agent can now use these tools
response = agent.run("Search for Python tutorials and email me the best ones")
```

### Custom Agent Configuration

```python
from agentframe import AgentConfig

# Configure agent personality and behavior
config = AgentConfig(
    personality_traits="helpful, analytical, thorough",
    communication_style="clear and professional",
    response_style="detailed",
    max_iterations=5,
    confidence_threshold=0.8,
    custom_guidelines="Always provide sources for factual claims."
)

agent = Agent(model=model, tools=tools, config=config)
```

## Advanced Features

### Custom Prompts

Create specialized agents with custom prompt templates:

```python
from agentframe import PromptTemplate, AgentPrompts

# Define custom system prompt
system_prompt = PromptTemplate(
    name="research_assistant",
    template="""You are a research assistant specializing in {domain}.

Your expertise includes:
- {personality_traits}
- {communication_style}

Always provide evidence-based responses and cite sources.
Guidelines: {custom_guidelines}""",
    required_variables=["domain", "personality_traits", "communication_style"],
    optional_variables={"custom_guidelines": "Focus on peer-reviewed sources."}
)

# Create prompt collection
prompts = AgentPrompts(system_prompt=system_prompt)

# Create specialized agent
research_agent = Agent(
    model=model,
    config=AgentConfig(
        personality_traits="methodical, curious, objective",
        communication_style="academic and rigorous"
    ),
    prompts=prompts
)
```

### Planning and Replanning

AgentFrame automatically handles complex task planning:

```python
# The agent will automatically:
# 1. Parse the user's intent
# 2. Create a step-by-step plan
# 3. Execute each step using available tools
# 4. Replan if any step fails
# 5. Provide a comprehensive response

response = agent.run("""
Help me prepare for a presentation on climate change:
1. Research recent climate data
2. Find compelling visualizations
3. Create an outline
4. Suggest presentation tools
""")
```

### Streaming Responses

For real-time interaction:

```python
# Stream agent responses
for chunk in agent.stream("Tell me about renewable energy"):
    print(chunk, end="", flush=True)
```

### Conversation Persistence

Maintain context across multiple interactions:

```python
# Start a conversation
response1 = agent.run("I'm planning a garden")

# Continue the conversation (context is maintained)
response2 = agent.run("What vegetables grow well in spring?")

# Access conversation history
history = agent.get_chat_history()
print(f"Conversation has {len(history.messages)} messages")

# Export conversation
exported = history.export_to_json()

# Load conversation later
from agentframe import ChatHistory
history = ChatHistory.from_json(exported)
```

## Model Providers

AgentFrame supports multiple language model providers with a unified interface.

### OpenAI Models

```python
from agentframe import OpenAIModel, ModelConfig

# Standard OpenAI
config = ModelConfig(
    api_key="sk-your-openai-key",
    model="gpt-4",
    temperature=0.7
)
model = OpenAIModel(config)

# OpenAI-compatible APIs (Ollama, Groq, etc.)
config = ModelConfig(
    api_key="your-api-key",
    model="llama2:7b",
    base_url="http://localhost:11434/v1",  # Ollama
    temperature=0.7
)
model = OpenAIModel(config)
```

### Google Gemini

```python
from agentframe import GeminiModel, ModelConfig

config = ModelConfig(
    api_key="your-google-api-key",
    model="gemini-pro",
    temperature=0.7
)
model = GeminiModel(config)
```

### Anthropic Claude

```python
from agentframe import ClaudeModel, ModelConfig

config = ModelConfig(
    api_key="your-anthropic-key",
    model="claude-3-sonnet-20240229",
    temperature=0.7
)
model = ClaudeModel(config)
```

### Model Configuration Options

```python
config = ModelConfig(
    api_key="your-api-key",
    model="gpt-4",
    base_url=None,  # Custom API endpoint
    organization=None,  # Organization ID
    temperature=0.7,  # Randomness (0.0-1.0)
    max_tokens=1000,  # Maximum response length
    top_p=1.0,  # Nucleus sampling
    frequency_penalty=0.0,  # Reduce repetition
    presence_penalty=0.0,  # Encourage new topics
    timeout=60,  # Request timeout
    max_retries=3,  # Retry failed requests
    additional_headers={},  # Custom headers
    custom_params={}  # Provider-specific parameters
)
```

## Tool System

The tool system allows agents to interact with external services and perform actions.

### Function Tools

Convert Python functions into agent tools:

```python
from agentframe import tool

@tool
def get_weather(city: str, country: str = "US") -> dict:
    """Get current weather for a city.

    Args:
        city: Name of the city
        country: Country code (default: US)

    Returns:
        Weather information including temperature, description, etc.
    """
    # Implementation
    return {
        "city": city,
        "temperature": 22,
        "description": "Sunny",
        "humidity": 65
    }

# Tool is automatically registered and can be used by agents
agent = Agent(model=model, tools=[get_weather])
```

### Tool Registry

Manage tools centrally:

```python
from agentframe import ToolRegistry, get_global_registry

# Access global registry
registry = get_global_registry()

# Register tools
registry.register_tool(get_weather)
registry.register_tool(calculator)

# Create agent with registered tools
agent = Agent(model=model, tool_registry=registry)

# Or use specific tools
weather_tools = registry.get_tools_by_category("weather")
agent = Agent(model=model, tools=weather_tools)
```

### Custom Tool Classes

For more complex tools:

```python
from agentframe import BaseTool, ToolResult

class DatabaseTool(BaseTool):
    def __init__(self, connection_string: str):
        super().__init__(
            name="database_query",
            description="Execute SQL queries on the database",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"}
                },
                "required": ["query"]
            }
        )
        self.connection_string = connection_string

    def execute(self, query: str) -> ToolResult:
        try:
            # Execute query logic here
            result = {"rows": [], "count": 0}
            return ToolResult(
                success=True,
                data=result,
                message="Query executed successfully"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                message="Query execution failed"
            )

# Use custom tool
db_tool = DatabaseTool("postgresql://localhost/mydb")
agent = Agent(model=model, tools=[db_tool])
```

### Tool Error Handling

Tools should handle errors gracefully:

```python
@tool
def risky_operation(data: str) -> dict:
    """Perform an operation that might fail."""
    try:
        # Potentially failing operation
        result = perform_operation(data)
        return {"success": True, "result": result}
    except ValueError as e:
        return {"success": False, "error": f"Invalid data: {e}"}
    except ConnectionError as e:
        return {"success": False, "error": f"Connection failed: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}
```

## Prompt Customization

Create specialized agents with custom behavior through prompt templates.

### Basic Prompt Templates

```python
from agentframe import PromptTemplate

# Create a custom prompt template
template = PromptTemplate(
    name="code_reviewer",
    template="""You are an expert code reviewer with {experience_years} years of experience.

Your review focus:
- {review_areas}

Communication style: {communication_style}

For the code below, provide:
1. Overall assessment
2. Specific issues found
3. Recommendations for improvement
4. Security considerations

Code to review:
{code}""",
    required_variables=["experience_years", "review_areas", "communication_style", "code"],
    description="Template for code review agents"
)
```

### Agent Prompt Collections

```python
from agentframe import AgentPrompts

# Create a complete prompt set
prompts = AgentPrompts(
    system_prompt=system_template,
    planning_prompt=planning_template,
    response_prompt=response_template,
    custom_prompts={
        "error_handling": error_template,
        "quality_check": quality_template
    }
)

# Use with agent
agent = Agent(model=model, prompts=prompts)

# Use specific custom prompts
response = agent.run_with_prompt(
    "quality_check",
    "Review this code for quality issues",
    code=source_code
)
```

### Specialized Agent Examples

Pre-built prompt templates for common use cases:

```python
# Research Assistant
from agentframe.examples.prompt_templates.research_assistant import (
    create_research_assistant
)
research_agent = create_research_assistant()

# Code Reviewer
from agentframe.examples.prompt_templates.code_reviewer import (
    create_code_reviewer
)
code_agent = create_code_reviewer()

# Customer Support
from agentframe.examples.custom_prompts import (
    create_customer_support_agent
)
support_agent = create_customer_support_agent()
```

## Agent Configuration

Fine-tune agent behavior and personality:

```python
from agentframe import AgentConfig

config = AgentConfig(
    # Personality traits
    personality_traits="analytical, thorough, helpful, patient",

    # Communication style
    communication_style="clear, technical, and educational",

    # Response tone
    response_style="professional",

    # Custom guidelines
    custom_guidelines="""
    1. Always provide evidence for technical claims
    2. Acknowledge uncertainty when appropriate
    3. Suggest next steps for complex problems
    4. Use examples to illustrate concepts
    """,

    # Execution parameters
    max_iterations=5,  # Maximum planning iterations
    confidence_threshold=0.8,  # Minimum confidence for plan execution
    enable_replanning=True,  # Allow plan modifications

    # Memory settings
    max_history_tokens=4000,  # Token limit for conversation history
    history_summarization=True  # Summarize old conversations
)
```

### Dynamic Configuration

Adjust agent behavior based on context:

```python
# Create base agent
agent = Agent(model=model, config=base_config)

# Adjust for different scenarios
if user_expertise == "beginner":
    agent.update_config(
        communication_style="simple and educational",
        response_style="encouraging"
    )
elif user_expertise == "expert":
    agent.update_config(
        communication_style="technical and detailed",
        response_style="concise"
    )
```

## Planning and Execution

AgentFrame's planning system breaks down complex tasks into manageable steps.

### Automatic Planning

```python
# Agent automatically plans and executes
response = agent.run("""
I need to prepare a technical presentation about microservices:
1. Research current microservices patterns
2. Find real-world case studies
3. Create architectural diagrams
4. Prepare speaker notes
5. Suggest Q&A topics
""")

# The agent will:
# - Parse the multi-step request
# - Create a detailed execution plan
# - Use available tools for research
# - Coordinate between different tasks
# - Provide a comprehensive response
```

### Manual Plan Inspection

```python
# Access the planning process
plan = agent.create_plan("Build a REST API with authentication")

print(f"Plan has {len(plan.steps)} steps:")
for i, step in enumerate(plan.steps, 1):
    print(f"{i}. {step.description}")
    print(f"   Tools: {step.required_tools}")
    print(f"   Dependencies: {step.dependencies}")
```

### Custom Planning Strategies

```python
from agentframe import ReplanningStrategy

# Define custom replanning behavior
class ConservativeReplanning(ReplanningStrategy):
    def should_replan(self, execution_context):
        # Only replan on critical failures
        return execution_context.failure_count > 2

    def create_replan(self, failed_plan, context):
        # Conservative approach: modify only failed steps
        return self.modify_failed_steps(failed_plan, context)

# Use with agent
agent = Agent(
    model=model,
    tools=tools,
    replanning_strategy=ConservativeReplanning()
)
```

## Memory and Conversation

Manage conversation context and history effectively.

### Conversation Management

```python
from agentframe import ChatHistory

# Create agent with history tracking
agent = Agent(model=model, tools=tools)

# Start conversation
response1 = agent.run("I'm working on a Python project")

# Continue conversation (context maintained)
response2 = agent.run("Can you help me with error handling?")

# Access conversation history
history = agent.get_chat_history()
print(f"Messages: {len(history.messages)}")

# Get recent context
recent_context = history.get_recent_context(max_tokens=1000)
```

### History Persistence

```python
# Export conversation
history_data = agent.export_conversation()

# Save to file
with open("conversation.json", "w") as f:
    json.dump(history_data, f)

# Load conversation later
with open("conversation.json", "r") as f:
    history_data = json.load(f)

# Create new agent with loaded history
agent = Agent(model=model, tools=tools)
agent.load_conversation(history_data)
```

### Memory Optimization

```python
# Configure memory management
config = AgentConfig(
    max_history_tokens=4000,  # Limit history size
    history_summarization=True,  # Summarize old messages
    context_window_strategy="sliding"  # How to handle token limits
)

agent = Agent(model=model, config=config)

# Manual memory management
agent.clear_history()  # Clear all history
agent.summarize_history()  # Summarize old messages
agent.trim_history(max_tokens=2000)  # Trim to token limit
```

## FastAPI Integration

Build web services with AgentFrame agents.

### Basic FastAPI Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agentframe import Agent, OpenAIModel, ModelConfig

app = FastAPI(title="AgentFrame API")

# Initialize agent
config = ModelConfig(api_key="your-key", model="gpt-4")
model = OpenAIModel(config)
agent = Agent(model=model, tools=tools)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    session_id: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = agent.run(request.message)
        return ChatResponse(
            response=response,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with: uvicorn main:app --reload
```

### Advanced FastAPI Features

```python
from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.responses import StreamingResponse
import asyncio

# Streaming responses
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    def generate_response():
        for chunk in agent.stream(request.message):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/plain"
    )

# WebSocket support
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            response = agent.run(message)
            await websocket.send_text(response)
    except Exception as e:
        await websocket.close()

# Background task processing
@app.post("/chat/async")
async def chat_async(request: ChatRequest, background_tasks: BackgroundTasks):
    task_id = generate_task_id()
    background_tasks.add_task(process_chat, task_id, request.message)
    return {"task_id": task_id, "status": "processing"}

async def process_chat(task_id: str, message: str):
    # Process chat in background
    response = agent.run(message)
    # Store result for later retrieval
    store_result(task_id, response)
```

### Session Management

```python
from typing import Dict
import uuid

# Session storage
sessions: Dict[str, Agent] = {}

def get_or_create_session(session_id: str) -> Agent:
    if session_id not in sessions:
        sessions[session_id] = Agent(model=model, tools=tools)
    return sessions[session_id]

@app.post("/chat")
async def chat_with_session(request: ChatRequest):
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    agent = get_or_create_session(request.session_id)
    response = agent.run(request.message)

    return ChatResponse(
        response=response,
        session_id=request.session_id
    )

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "Session deleted"}
```

## Examples

### Complete Application Examples

#### 1. Research Assistant

```python
from agentframe import Agent, OpenAIModel, ModelConfig, tool
import requests

@tool
def search_papers(query: str, limit: int = 10) -> list:
    """Search for academic papers."""
    # Implementation using arXiv API, PubMed, etc.
    return []

@tool
def summarize_paper(paper_url: str) -> dict:
    """Summarize an academic paper."""
    # Implementation
    return {"title": "", "summary": "", "key_findings": []}

# Create research assistant
config = ModelConfig(api_key="your-key", model="gpt-4")
model = OpenAIModel(config)

research_agent = Agent(
    model=model,
    tools=[search_papers, summarize_paper],
    config=AgentConfig(
        personality_traits="methodical, curious, analytical",
        communication_style="academic and thorough",
        custom_guidelines="Always cite sources and provide evidence"
    )
)

# Use the research assistant
response = research_agent.run(
    "Research the latest developments in quantum computing and provide a summary"
)
```

#### 2. Code Review Assistant

```python
@tool
def analyze_code_quality(code: str, language: str) -> dict:
    """Analyze code quality metrics."""
    return {
        "complexity": 7,
        "maintainability": 8,
        "issues": ["Long function", "Missing docstring"]
    }

@tool
def check_security_issues(code: str) -> list:
    """Check for security vulnerabilities."""
    return [
        {"type": "SQL Injection", "line": 42, "severity": "High"},
        {"type": "XSS", "line": 67, "severity": "Medium"}
    ]

code_reviewer = Agent(
    model=model,
    tools=[analyze_code_quality, check_security_issues],
    config=AgentConfig(
        personality_traits="thorough, constructive, experienced",
        communication_style="technical and specific",
        custom_guidelines="Provide actionable feedback with examples"
    )
)
```

#### 3. Customer Support Bot

```python
@tool
def lookup_order(order_id: str) -> dict:
    """Look up order information."""
    return {
        "status": "shipped",
        "tracking": "1Z999AA1234567890",
        "estimated_delivery": "2024-01-15"
    }

@tool
def create_support_ticket(issue: str, priority: str = "normal") -> str:
    """Create a support ticket."""
    return "TICKET-12345"

support_bot = Agent(
    model=model,
    tools=[lookup_order, create_support_ticket],
    config=AgentConfig(
        personality_traits="empathetic, patient, solution-focused",
        communication_style="friendly and professional",
        custom_guidelines="Always acknowledge customer concerns first"
    )
)
```

### Integration Examples

#### Slack Bot Integration

```python
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

slack_app = App(token="your-bot-token")
agent = Agent(model=model, tools=tools)

@slack_app.event("message")
def handle_message(event, say):
    if event.get("channel_type") == "im":  # Direct message
        user_message = event["text"]
        response = agent.run(user_message)
        say(response)

handler = SocketModeHandler(slack_app, "your-app-token")
handler.start()
```

#### Discord Bot Integration

```python
import discord
from discord.ext import commands

bot = commands.Bot(command_prefix="!")
agent = Agent(model=model, tools=tools)

@bot.event
async def on_message(message):
    if message.author != bot.user:
        response = agent.run(message.content)
        await message.channel.send(response)

bot.run("your-discord-token")
```

## API Reference

### Core Classes

#### Agent

```python
class Agent:
    def __init__(
        self,
        model: BaseModel,
        tools: Optional[List[BaseTool]] = None,
        config: Optional[AgentConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        prompts: Optional[AgentPrompts] = None
    )

    def run(self, user_input: str, **kwargs) -> str
    def stream(self, user_input: str, **kwargs) -> Iterator[str]
    def run_with_prompt(self, prompt_name: str, user_input: str, **kwargs) -> str
    def get_chat_history(self) -> ChatHistory
    def clear_history(self) -> None
    def export_conversation(self) -> dict
    def load_conversation(self, data: dict) -> None
```

#### ModelConfig

```python
class ModelConfig:
    api_key: str
    model: str
    base_url: Optional[str] = None
    organization: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    max_retries: int = 3
    additional_headers: Dict[str, str] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)
```

#### AgentConfig

```python
class AgentConfig:
    personality_traits: str = "helpful, knowledgeable, professional"
    communication_style: str = "clear and informative"
    response_style: str = "helpful"
    custom_guidelines: str = ""
    max_iterations: int = 3
    confidence_threshold: float = 0.7
    enable_replanning: bool = True
    max_history_tokens: int = 4000
    history_summarization: bool = True
```

#### PromptTemplate

```python
class PromptTemplate:
    name: str
    template: str
    required_variables: List[str] = field(default_factory=list)
    optional_variables: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    examples: List[str] = field(default_factory=list)

    def format(self, **kwargs) -> str
    def validate_variables(self, variables: Dict[str, Any]) -> List[str]
```

### Tool System

#### @tool Decorator

```python
@tool
def function_name(param1: type, param2: type = default) -> return_type:
    """Function description for the agent."""
    # Implementation
    return result
```

#### BaseTool

```python
class BaseTool:
    def __init__(self, name: str, description: str, parameters: dict)
    def execute(self, **kwargs) -> ToolResult
    def validate_parameters(self, parameters: dict) -> bool
```

#### ToolResult

```python
class ToolResult:
    success: bool
    data: Any = None
    error: Optional[str] = None
    message: str = ""
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Model Providers

#### OpenAIModel

```python
class OpenAIModel(BaseModel):
    def __init__(self, config: ModelConfig)
    def generate(self, messages: List[Dict], tools: Optional[List] = None) -> ModelResponse
    def stream_generate(self, messages: List[Dict], tools: Optional[List] = None) -> Iterator[str]
    def generate_with_structured_output(self, messages: List[Dict], schema: Union[Dict, BaseModel]) -> Dict
```

#### GeminiModel

```python
class GeminiModel(BaseModel):
    def __init__(self, config: ModelConfig)
    # Same interface as OpenAIModel
```

#### ClaudeModel

```python
class ClaudeModel(BaseModel):
    def __init__(self, config: ModelConfig)
    # Same interface as OpenAIModel
```

## Best Practices

### Agent Design

1. **Start Simple**: Begin with basic functionality and add complexity gradually
2. **Clear Purpose**: Define what your agent should and shouldn't do
3. **Tool Selection**: Choose tools that match your agent's purpose
4. **Error Handling**: Plan for failures and edge cases
5. **Testing**: Test with various inputs and scenarios

### Prompt Engineering

1. **Be Specific**: Clear instructions produce better results
2. **Use Examples**: Include examples when helpful
3. **Set Boundaries**: Define limitations and constraints
4. **Iterate**: Refine prompts based on performance
5. **Version Control**: Track prompt changes

### Performance Optimization

1. **Token Management**: Monitor and optimize token usage
2. **Caching**: Cache expensive operations
3. **Streaming**: Use streaming for long responses
4. **Async Operations**: Use async tools for I/O operations
5. **Monitoring**: Track performance metrics

### Security Considerations

1. **Input Validation**: Validate all user inputs
2. **API Key Management**: Secure API keys properly
3. **Tool Permissions**: Limit tool capabilities
4. **Rate Limiting**: Implement rate limiting
5. **Audit Logging**: Log agent actions

### Error Handling

```python
# Robust error handling example
try:
    response = agent.run(user_input)
except ModelError as e:
    # Handle model-specific errors
    logger.error(f"Model error: {e}")
    response = "I'm experiencing technical difficulties. Please try again."
except ToolError as e:
    # Handle tool execution errors
    logger.error(f"Tool error: {e}")
    response = "I encountered an issue while processing your request."
except ValidationError as e:
    # Handle validation errors
    logger.error(f"Validation error: {e}")
    response = "Please check your input and try again."
except Exception as e:
    # Handle unexpected errors
    logger.error(f"Unexpected error: {e}")
    response = "An unexpected error occurred. Please contact support."
```

## Troubleshooting

### Common Issues

#### Model Connection Issues

```python
# Check API key and configuration
try:
    test_response = model.generate([{"role": "user", "content": "test"}])
    print("Model connection successful")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except APIError as e:
    print(f"API error: {e}")
```

#### Tool Execution Failures

```python
# Debug tool issues
@tool
def debug_tool(input_data: str) -> dict:
    """Debug tool with detailed logging."""
    try:
        logger.info(f"Tool called with: {input_data}")
        result = process_data(input_data)
        logger.info(f"Tool result: {result}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Tool error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
```

#### Memory Issues

```python
# Monitor token usage
history = agent.get_chat_history()
token_count = history.estimate_tokens()
if token_count > 3000:
    agent.summarize_history()
    print(f"History summarized. New token count: {history.estimate_tokens()}")
```

### Performance Issues

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor execution time
import time
start_time = time.time()
response = agent.run(user_input)
execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.2f} seconds")
```

### Debugging Tips

1. **Enable Logging**: Use detailed logging to track execution
2. **Test Tools Separately**: Verify tool functionality independently
3. **Check Token Limits**: Monitor token usage and limits
4. **Validate Configurations**: Verify all configuration parameters
5. **Use Debug Mode**: Enable debug features during development

### Getting Help

- **Documentation**: Check the latest documentation
- **Examples**: Review example implementations
- **GitHub Issues**: Report bugs and request features
- **Community**: Join the AgentFrame community discussions

## Contributing

AgentFrame is open source and welcomes contributions:

1. **Fork the Repository**: Create your own fork
2. **Create Feature Branch**: Work on feature branches
3. **Write Tests**: Include tests for new functionality
4. **Update Documentation**: Keep documentation current
5. **Submit Pull Request**: Follow the contribution guidelines

### Development Setup

```bash
git clone https://github.com/LasinduAnjana/agentframe.git
cd agentframe
pip install -e ".[dev]"
pytest tests/
```

## License

AgentFrame is released under the MIT License. See the LICENSE file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

This documentation provides a comprehensive guide to using AgentFrame. For the latest updates and examples, visit the [GitHub repository](https://github.com/LasinduAnjana/agentframe).