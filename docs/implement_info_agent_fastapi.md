# How to Implement an Info Agent FastAPI Service using AgentFrame

This guide provides complete instructions for implementing a FastAPI-based information-gathering agent service using the AgentFrame framework. The service will provide REST API endpoints for web search, weather checking, and chat history management.

## Project Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Git
- API keys for required services

### Step 1: Create Project Structure

Create a new FastAPI project with the following structure:

```
info_agent_api/
├── .env                      # Environment variables (API keys)
├── .env.example             # Environment template
├── .gitignore               # Git ignore file
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker compose for development
├── config.py                # Configuration management
├── main.py                  # FastAPI application entry point
├── app/                     # Application package
│   ├── __init__.py
│   ├── api/                 # API routes
│   │   ├── __init__.py
│   │   ├── v1/              # API version 1
│   │   │   ├── __init__.py
│   │   │   ├── agent.py     # Agent endpoints
│   │   │   ├── chat.py      # Chat and history endpoints
│   │   │   └── health.py    # Health check endpoints
│   │   └── dependencies.py  # FastAPI dependencies
│   ├── core/                # Core application logic
│   │   ├── __init__.py
│   │   ├── agent_manager.py # Agent instance management
│   │   ├── chat_manager.py  # Chat session management
│   │   └── exceptions.py    # Custom exceptions
│   ├── models/              # Pydantic models
│   │   ├── __init__.py
│   │   ├── requests.py      # Request models
│   │   ├── responses.py     # Response models
│   │   └── chat.py          # Chat-related models
│   ├── tools/               # Custom tools
│   │   ├── __init__.py
│   │   ├── web_search.py    # Web search tool
│   │   └── weather.py       # Weather tool
│   └── utils/               # Utilities
│       ├── __init__.py
│       ├── logging.py       # Logging configuration
│       ├── cache.py         # Caching utilities
│       └── validators.py    # Input validation
├── tests/                   # Test files
│   ├── __init__.py
│   ├── conftest.py          # Test configuration
│   ├── test_api/            # API tests
│   │   ├── __init__.py
│   │   ├── test_agent.py
│   │   ├── test_chat.py
│   │   └── test_health.py
│   ├── test_tools.py        # Tool tests
│   └── test_core.py         # Core logic tests
├── scripts/                 # Utility scripts
│   ├── start_dev.sh         # Development server script
│   └── run_tests.sh         # Test runner script
└── docs/                    # API documentation
    ├── api_examples.md      # Usage examples
    └── deployment.md        # Deployment guide
```

### Step 2: Environment Setup

Create a `.env` file with the following configuration:

```env
# Application Settings
APP_NAME=Info Agent API
APP_VERSION=1.0.0
DEBUG=true
HOST=0.0.0.0
PORT=8000
WORKERS=1

# OpenAI API (for the main agent model)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Alternative model API keys (optional)
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Web Search API (choose one)
SERPER_API_KEY=your-serper-api-key-here
SERPAPI_API_KEY=your-serpapi-key-here
BRAVE_API_KEY=your-brave-search-api-key-here

# Weather API
OPENWEATHERMAP_API_KEY=your-openweathermap-api-key-here
WEATHERAPI_KEY=your-weatherapi-key-here

# Agent Configuration
AGENT_MODEL=gpt-4
AGENT_TEMPERATURE=0.7
AGENT_MAX_TOKENS=1000
AGENT_TIMEOUT=300
MAX_REPLANNING_ITERATIONS=3

# Tool Configuration
WEB_SEARCH_PROVIDER=serper
WEATHER_PROVIDER=openweathermap
MAX_SEARCH_RESULTS=5
SEARCH_CACHE_TTL=3600

# Session Management
SESSION_TIMEOUT=3600
MAX_CHAT_HISTORY=100
CHAT_HISTORY_CLEANUP_INTERVAL=86400

# Optional: Database (for persistent chat history)
DATABASE_URL=sqlite:///./chat_history.db
# DATABASE_URL=postgresql://user:password@localhost/dbname

# Optional: Redis (for caching and session storage)
REDIS_URL=redis://localhost:6379
ENABLE_REDIS_CACHE=false

# API Security
API_KEY_HEADER=X-API-Key
ENABLE_API_KEY_AUTH=false
API_KEYS=api-key-1,api-key-2

# Rate Limiting
ENABLE_RATE_LIMITING=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
ENABLE_ACCESS_LOG=true
LOG_FILE=info_agent_api.log
```

### Step 3: Dependencies

Create `requirements.txt`:

```txt
# FastAPI and ASGI server
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
gunicorn>=21.2.0

# AgentFrame
git+https://github.com/LasinduAnjana/agentframe.git

# Web framework dependencies
pydantic>=2.4.0
pydantic-settings>=2.0.0
python-multipart>=0.0.6

# Database and caching
sqlalchemy>=2.0.0
alembic>=1.12.0
redis>=5.0.0
diskcache>=5.6.0

# HTTP clients and APIs
httpx>=0.25.0
aiohttp>=3.9.0
requests>=2.31.0

# Web search providers
google-search-results>=2.4.2
serper>=0.1.0

# Weather APIs
pyowm>=3.3.0

# Utilities
python-dotenv>=1.0.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-dateutil>=2.8.2
beautifulsoup4>=4.12.0
lxml>=4.9.0
geopy>=2.4.0

# Development and testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
httpx>=0.25.0
black>=23.9.0
ruff>=0.1.0
mypy>=1.6.0

# Monitoring and logging
structlog>=23.2.0
```

## Implementation Prompt

Please implement a FastAPI-based information-gathering agent service using the AgentFrame framework with the following specifications:

### Core API Requirements

1. **FastAPI Application Setup**: Create a production-ready FastAPI application with:
   - Proper project structure and organization
   - Environment-based configuration management
   - Comprehensive error handling and validation
   - API versioning (v1)
   - Health check endpoints
   - Request/response logging
   - CORS configuration
   - Rate limiting (optional)

2. **Agent Management**: Implement agent instance management with:
   - Singleton agent instance with proper initialization
   - Thread-safe agent operations
   - Configuration validation on startup
   - Graceful error handling for missing API keys

3. **Chat Session Management**: Create a chat session system that:
   - Manages multiple concurrent chat sessions
   - Stores chat history per session
   - Provides session cleanup and timeout handling
   - Supports session persistence (optional)

### API Endpoints Specification

#### 1. Health Check Endpoints (`/api/v1/health`)

```python
@router.get("/health")
async def health_check() -> dict:
    """Basic health check endpoint."""

@router.get("/health/detailed")
async def detailed_health_check() -> dict:
    """Detailed health check including agent status and API connectivity."""
```

#### 2. Agent Query Endpoints (`/api/v1/agent`)

```python
@router.post("/agent/query")
async def query_agent(request: AgentQueryRequest) -> AgentQueryResponse:
    """Send a query to the agent and get a response."""

@router.post("/agent/query/stream")
async def query_agent_stream(request: AgentQueryRequest) -> StreamingResponse:
    """Send a query to the agent and stream the response."""
```

#### 3. Chat Management Endpoints (`/api/v1/chat`)

```python
@router.post("/chat/session")
async def create_chat_session() -> ChatSessionResponse:
    """Create a new chat session."""

@router.get("/chat/session/{session_id}")
async def get_chat_session(session_id: str) -> ChatSessionResponse:
    """Get chat session information."""

@router.post("/chat/session/{session_id}/message")
async def send_message(session_id: str, request: ChatMessageRequest) -> ChatMessageResponse:
    """Send a message in a chat session."""

@router.get("/chat/session/{session_id}/history")
async def get_chat_history(session_id: str, limit: int = 50) -> ChatHistoryResponse:
    """Get chat history for a session as JSON."""

@router.delete("/chat/session/{session_id}")
async def delete_chat_session(session_id: str) -> dict:
    """Delete a chat session and its history."""

@router.get("/chat/sessions")
async def list_chat_sessions() -> List[ChatSessionResponse]:
    """List all active chat sessions."""
```

#### 4. Tool-Specific Endpoints (`/api/v1/tools`)

```python
@router.post("/tools/search")
async def web_search(request: WebSearchRequest) -> WebSearchResponse:
    """Direct web search without agent processing."""

@router.post("/tools/weather")
async def get_weather(request: WeatherRequest) -> WeatherResponse:
    """Direct weather query without agent processing."""
```

### Pydantic Models Specification

#### Request Models (`app/models/requests.py`)

```python
class AgentQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    session_id: Optional[str] = Field(None, description="Chat session ID")
    stream: bool = Field(False, description="Enable streaming response")
    model_config: Optional[Dict[str, Any]] = Field(None, description="Model configuration overrides")

class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="Chat message")
    user_id: Optional[str] = Field(None, description="User identifier")

class WebSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200, description="Search query")
    num_results: int = Field(5, ge=1, le=10, description="Number of results")
    region: str = Field("us", description="Search region")

class WeatherRequest(BaseModel):
    location: str = Field(..., min_length=1, max_length=100, description="Location name or coordinates")
    units: str = Field("metric", regex="^(metric|imperial|kelvin)$", description="Temperature units")
    include_forecast: bool = Field(False, description="Include forecast data")
```

#### Response Models (`app/models/responses.py`)

```python
class AgentQueryResponse(BaseModel):
    response: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Chat session ID")
    query_id: str = Field(..., description="Unique query ID")
    timestamp: datetime = Field(..., description="Response timestamp")
    tools_used: List[str] = Field(default_factory=list, description="Tools used in processing")
    execution_time: float = Field(..., description="Processing time in seconds")

class ChatSessionResponse(BaseModel):
    session_id: str = Field(..., description="Session ID")
    created_at: datetime = Field(..., description="Session creation time")
    last_activity: datetime = Field(..., description="Last activity timestamp")
    message_count: int = Field(..., description="Number of messages in session")

class ChatMessageResponse(BaseModel):
    message_id: str = Field(..., description="Message ID")
    session_id: str = Field(..., description="Session ID")
    user_message: str = Field(..., description="User message")
    agent_response: str = Field(..., description="Agent response")
    timestamp: datetime = Field(..., description="Message timestamp")
    tools_used: List[str] = Field(default_factory=list, description="Tools used")

class ChatHistoryResponse(BaseModel):
    session_id: str = Field(..., description="Session ID")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    total_messages: int = Field(..., description="Total message count")
    session_info: ChatSessionResponse = Field(..., description="Session information")

class ChatMessage(BaseModel):
    message_id: str = Field(..., description="Message ID")
    type: str = Field(..., description="Message type (user/agent/system)")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(..., description="Message timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
```

### Implementation Details

#### 1. Configuration Management (`config.py`)

```python
from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    # App settings
    app_name: str = "Info Agent API"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # API keys
    openai_api_key: str
    serper_api_key: Optional[str] = None
    openweathermap_api_key: Optional[str] = None

    # Agent configuration
    agent_model: str = "gpt-4"
    agent_temperature: float = 0.7
    agent_max_tokens: int = 1000

    # Session management
    session_timeout: int = 3600
    max_chat_history: int = 100

    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

#### 2. Agent Manager (`app/core/agent_manager.py`)

Create a singleton agent manager that:
- Initializes AgentFrame agent with tools
- Manages agent lifecycle
- Provides thread-safe access
- Handles agent errors gracefully

#### 3. Chat Manager (`app/core/chat_manager.py`)

Implement chat session management:
- Create and manage chat sessions
- Store chat history in memory/database
- Handle session cleanup and timeouts
- Provide chat history export as JSON

#### 4. Tool Implementations

**Web Search Tool** (`app/tools/web_search.py`):
- Support multiple search providers
- Implement caching and rate limiting
- Return structured search results
- Handle API errors gracefully

**Weather Tool** (`app/tools/weather.py`):
- Support location geocoding
- Multiple weather data providers
- Formatted weather information
- Error handling for invalid locations

#### 5. FastAPI Application (`main.py`)

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn

from app.api.v1 import agent, chat, health
from app.core.exceptions import setup_exception_handlers
from config import settings

def create_application() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Information gathering agent API with web search and weather capabilities",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(health.router, prefix="/api/v1/health", tags=["health"])
    app.include_router(agent.router, prefix="/api/v1/agent", tags=["agent"])
    app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])

    # Exception handlers
    setup_exception_handlers(app)

    return app

app = create_application()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers
    )
```

### Docker Configuration

#### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash agent
USER agent

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  info-agent-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEBUG=true
    env_file:
      - .env
    volumes:
      - .:/app
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### API Usage Examples

#### 1. Simple Query

```bash
curl -X POST "http://localhost:8000/api/v1/agent/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the weather like in Tokyo and search for travel guides about Japan"
  }'
```

#### 2. Chat Session

```bash
# Create session
curl -X POST "http://localhost:8000/api/v1/chat/session"

# Send message
curl -X POST "http://localhost:8000/api/v1/chat/session/{session_id}/message" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Search for the latest AI news"
  }'

# Get chat history as JSON
curl "http://localhost:8000/api/v1/chat/session/{session_id}/history"
```

#### 3. Direct Tool Usage

```bash
# Web search
curl -X POST "http://localhost:8000/api/v1/tools/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "FastAPI best practices",
    "num_results": 5
  }'

# Weather check
curl -X POST "http://localhost:8000/api/v1/tools/weather" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "New York",
    "units": "metric"
  }'
```

### Testing Requirements

Create comprehensive tests for:
- API endpoint functionality
- Request/response validation
- Error handling scenarios
- Chat session management
- Tool integration
- Agent query processing

### Success Criteria

The implementation is successful when:

1. ✅ FastAPI application starts and serves endpoints correctly
2. ✅ Agent can process queries through API endpoints
3. ✅ Chat sessions are created, managed, and provide history as JSON
4. ✅ Web search and weather tools work through direct endpoints
5. ✅ Configuration loads properly from .env file
6. ✅ Error handling works for various failure scenarios
7. ✅ API documentation is generated and accessible
8. ✅ Docker containerization works correctly
9. ✅ Rate limiting and security measures function properly
10. ✅ Comprehensive testing covers all endpoints

### Deployment Considerations

- Configure proper CORS settings for production
- Implement API key authentication if needed
- Set up proper logging and monitoring
- Use production ASGI server (gunicorn + uvicorn)
- Configure database for persistent chat history
- Set up Redis for caching and session storage
- Implement health checks for container orchestration

Please implement this FastAPI-based info agent service following these specifications, ensuring robust error handling, comprehensive API documentation, and thorough testing.