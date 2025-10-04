# How to Implement an Info Agent using AgentFrame

This guide provides complete instructions for implementing an information-gathering agent using the AgentFrame framework. The agent will be capable of searching the web and checking weather information.

## Project Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Git
- API keys for required services

### Step 1: Create Project Structure

Create a new project with the following structure:

```
info_agent_project/
├── .env                    # Environment variables (API keys)
├── .gitignore             # Git ignore file
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── config.py              # Configuration management
├── tools/                 # Custom tools directory
│   ├── __init__.py
│   ├── web_search.py     # Web search tool implementation
│   └── weather.py        # Weather check tool implementation
├── main.py               # Main application entry point
├── agent_runner.py       # Agent execution logic
└── tests/                # Test files
    ├── __init__.py
    ├── test_tools.py
    └── test_agent.py
```

### Step 2: Environment Setup

Create a `.env` file with the following API keys:

```env
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

# Optional: Rate limiting and caching
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600

# Logging
LOG_LEVEL=INFO
LOG_FILE=info_agent.log
```

### Step 3: Dependencies

Create `requirements.txt`:

```txt
# AgentFrame
git+https://github.com/LasinduAnjana/agentframe.git

# Web requests and APIs
requests>=2.31.0
httpx>=0.24.0
aiohttp>=3.8.0

# Environment and configuration
python-dotenv>=1.0.0
pydantic>=2.0.0
pydantic-settings>=2.0.0

# Web search providers (choose based on your preference)
google-search-results>=2.4.2  # SerpAPI
serper>=0.1.0                 # Serper

# Weather APIs
pyowm>=3.3.0                  # OpenWeatherMap

# Utilities
beautifulsoup4>=4.12.0        # HTML parsing
lxml>=4.9.0                   # XML/HTML parser
dateparser>=1.1.8             # Date parsing
geopy>=2.3.0                  # Geocoding

# Development and testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.0.0
ruff>=0.1.0

# Optional: Caching and performance
redis>=4.6.0
diskcache>=5.6.0
```

## Implementation Prompt

Please implement an information-gathering agent using the AgentFrame framework with the following specifications:

### Core Requirements

1. **Agent Setup**: Create an info agent that can handle various information requests including web searches and weather queries.

2. **Web Search Tool**: Implement a comprehensive web search tool that:
   - Supports multiple search providers (Serper, SerpAPI, or Brave Search)
   - Returns formatted search results with titles, URLs, and snippets
   - Handles search result filtering and ranking
   - Implements rate limiting and error handling
   - Caches results to avoid duplicate API calls

3. **Weather Tool**: Implement a weather checking tool that:
   - Accepts city names, coordinates, or addresses
   - Returns current weather conditions
   - Provides weather forecasts (optional)
   - Handles location geocoding
   - Returns formatted weather information

4. **Configuration Management**: Create a robust configuration system that:
   - Loads API keys from .env file
   - Validates required credentials
   - Provides fallback options for different service providers
   - Includes rate limiting and timeout settings

5. **Main Application**: Create a main application that:
   - Initializes the agent with all tools
   - Provides both interactive CLI and single-query modes
   - Handles errors gracefully
   - Logs activities appropriately
   - Demonstrates agent capabilities with example queries

### Implementation Details

#### Web Search Tool (`tools/web_search.py`)

```python
@tool
def web_search(query: str, num_results: int = 5, region: str = "us") -> dict:
    """Search the web for information.

    Args:
        query: Search query string
        num_results: Number of results to return (1-10)
        region: Search region/country code (us, uk, etc.)

    Returns:
        Dictionary containing search results with titles, URLs, and snippets
    """
    # Implementation should:
    # - Support multiple search providers
    # - Handle API errors gracefully
    # - Return structured results
    # - Implement caching
    # - Rate limit requests
```

#### Weather Tool (`tools/weather.py`)

```python
@tool
def get_weather(location: str, units: str = "metric", include_forecast: bool = False) -> dict:
    """Get current weather information for a location.

    Args:
        location: City name, address, or coordinates (lat,lon)
        units: Temperature units (metric, imperial, kelvin)
        include_forecast: Whether to include forecast data

    Returns:
        Dictionary containing weather information
    """
    # Implementation should:
    # - Handle various location formats
    # - Geocode addresses to coordinates
    # - Return comprehensive weather data
    # - Handle API errors
    # - Format data for readability
```

#### Configuration (`config.py`)

Create a configuration class that:
- Uses Pydantic for validation
- Loads from environment variables
- Provides sensible defaults
- Validates API keys on startup
- Supports multiple service providers

#### Main Application (`main.py`)

Create an application that:
- Sets up the agent with all tools
- Provides CLI interface
- Demonstrates various queries
- Handles user input validation
- Provides help and usage information

### Example Usage Scenarios

The implemented agent should handle queries like:

1. **Web Search Queries**:
   - "Search for the latest news about artificial intelligence"
   - "Find information about Python web frameworks"
   - "What are the best practices for API design?"

2. **Weather Queries**:
   - "What's the weather like in Tokyo?"
   - "Check the weather forecast for New York"
   - "Temperature in London right now"

3. **Combined Queries**:
   - "Search for weather apps and check the weather in San Francisco"
   - "Find travel information for Paris and check the weather there"

### Error Handling Requirements

Implement comprehensive error handling for:
- Invalid API keys
- Network connectivity issues
- Rate limiting
- Invalid location names
- Search provider failures
- Malformed user input

### Testing Requirements

Create tests for:
- Tool functionality with mock APIs
- Configuration validation
- Agent integration
- Error scenarios
- CLI interface

### Documentation Requirements

Include:
- Setup instructions
- API key acquisition guides
- Usage examples
- Troubleshooting guide
- Extension guidelines

### Advanced Features (Optional)

Consider implementing:
- Result caching with Redis
- Async tool execution
- Custom search filters
- Weather alerts
- Location-based search optimization
- Result export functionality

## Expected File Structure After Implementation

```
info_agent_project/
├── .env                      # API keys and configuration
├── .gitignore               # Git ignore patterns
├── requirements.txt         # Dependencies
├── README.md                # Project documentation
├── config.py                # Configuration management
├── tools/
│   ├── __init__.py         # Tool exports
│   ├── web_search.py       # Web search implementation
│   └── weather.py          # Weather tool implementation
├── main.py                 # CLI application
├── agent_runner.py         # Agent logic
├── utils/
│   ├── __init__.py
│   ├── cache.py           # Caching utilities
│   ├── validators.py      # Input validation
│   └── formatters.py      # Output formatting
└── tests/
    ├── __init__.py
    ├── test_tools.py      # Tool tests
    ├── test_config.py     # Configuration tests
    └── test_agent.py      # Agent integration tests
```

## Success Criteria

The implementation is successful when:

1. ✅ Agent can perform web searches and return relevant results
2. ✅ Agent can check weather for various locations
3. ✅ Configuration loads properly from .env file
4. ✅ Error handling works for common failure scenarios
5. ✅ CLI interface is user-friendly and functional
6. ✅ Code is well-documented and tested
7. ✅ Agent can handle complex multi-tool queries
8. ✅ Results are formatted clearly and helpfully

## API Key Acquisition Guide

### Web Search APIs

**Serper (Recommended)**:
1. Visit https://serper.dev/
2. Sign up for free account
3. Get API key from dashboard
4. Free tier: 2,500 searches/month

**SerpAPI**:
1. Visit https://serpapi.com/
2. Sign up for account
3. Get API key from dashboard
4. Free tier: 100 searches/month

**Brave Search**:
1. Visit https://api.search.brave.com/
2. Sign up for account
3. Get API key
4. Free tier: 2,000 queries/month

### Weather APIs

**OpenWeatherMap (Recommended)**:
1. Visit https://openweathermap.org/api
2. Sign up for free account
3. Get API key from dashboard
4. Free tier: 1,000 calls/day

**WeatherAPI**:
1. Visit https://www.weatherapi.com/
2. Sign up for free account
3. Get API key
4. Free tier: 1 million calls/month

### OpenAI API
1. Visit https://platform.openai.com/
2. Create account and add payment method
3. Generate API key
4. Monitor usage in dashboard

Please implement this info agent following these specifications, ensuring robust error handling, clear documentation, and comprehensive testing.