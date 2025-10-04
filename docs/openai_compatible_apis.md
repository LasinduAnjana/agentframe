# Using OpenAI-Compatible APIs with AgentFrame

AgentFrame's `OpenAIModel` class supports any OpenAI-compatible API endpoint, allowing you to use local models, alternative providers, or custom deployments while maintaining the same interface.

## Configuration

Use the `ModelConfig` class with custom `base_url` and other parameters:

```python
from agentframe import ModelConfig, OpenAIModel

# Basic configuration with custom base URL
config = ModelConfig(
    api_key="your-api-key",
    model="your-model-name",
    base_url="https://your-api-endpoint.com/v1",
    temperature=0.7
)

model = OpenAIModel(config)
```

## Supported Providers

### 1. Local Ollama Server

[Ollama](https://ollama.ai/) provides local LLM hosting with OpenAI-compatible API.

```python
# Start Ollama server with: ollama serve
# Pull a model with: ollama pull llama2:7b

config = ModelConfig(
    api_key="ollama",  # Can be any string for local models
    model="llama2:7b",  # Available models: llama2, codellama, mistral, etc.
    base_url="http://localhost:11434/v1",
    temperature=0.7,
    max_tokens=2000
)

model = OpenAIModel(config)
```

**Popular Ollama Models:**
- `llama2:7b` - Meta's Llama 2 7B
- `llama2:13b` - Meta's Llama 2 13B
- `codellama:13b` - Code-specialized Llama
- `mistral:7b` - Mistral 7B
- `phi:2.7b` - Microsoft Phi-2
- `neural-chat:7b` - Intel's Neural Chat

### 2. Groq API

[Groq](https://groq.com/) provides ultra-fast inference for open-source models.

```python
config = ModelConfig(
    api_key="gsk_your_groq_api_key_here",
    model="mixtral-8x7b-32768",  # Model with 32k context
    base_url="https://api.groq.com/openai/v1",
    temperature=0.5,
    max_tokens=1000
)

model = OpenAIModel(config)
```

**Available Groq Models:**
- `mixtral-8x7b-32768` - Mixtral 8x7B with 32k context
- `llama2-70b-4096` - Llama 2 70B with 4k context
- `gemma-7b-it` - Google's Gemma 7B Instruct

### 3. Together AI

[Together AI](https://together.ai/) offers various open-source models.

```python
config = ModelConfig(
    api_key="your_together_api_key",
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    base_url="https://api.together.xyz/v1",
    temperature=0.7,
    max_tokens=1000
)

model = OpenAIModel(config)
```

**Popular Together AI Models:**
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-2-13b-chat-hf`
- `codellama/CodeLlama-34b-Instruct-hf`

### 4. Anyscale Endpoints

[Anyscale](https://anyscale.com/) provides serverless inference for open-source models.

```python
config = ModelConfig(
    api_key="esecret_your_anyscale_key",
    model="meta-llama/Llama-2-7b-chat-hf",
    base_url="https://api.endpoints.anyscale.com/v1",
    temperature=0.7,
    max_tokens=1000
)

model = OpenAIModel(config)
```

### 5. Local vLLM Server

[vLLM](https://github.com/vllm-project/vllm) provides high-performance local serving.

```python
# Start vLLM server with:
# python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-2-7b-chat-hf

config = ModelConfig(
    api_key="token-abc123",  # Can be any string
    model="meta-llama/Llama-2-7b-chat-hf",
    base_url="http://localhost:8000/v1",
    temperature=0.7,
    max_tokens=1000
)

model = OpenAIModel(config)
```

### 6. LocalAI

[LocalAI](https://localai.io/) is a drop-in replacement for OpenAI API.

```python
config = ModelConfig(
    api_key="your-localai-key",
    model="your-model-name",
    base_url="http://localhost:8080/v1",
    temperature=0.7,
    max_tokens=1000
)

model = OpenAIModel(config)
```

### 7. OpenRouter

[OpenRouter](https://openrouter.ai/) provides access to multiple models through one API.

```python
config = ModelConfig(
    api_key="sk-or-v1-your-key",
    model="anthropic/claude-3-sonnet",  # Or other available models
    base_url="https://openrouter.ai/api/v1",
    temperature=0.7,
    max_tokens=1000,
    additional_headers={
        "HTTP-Referer": "https://your-app.com",  # Required by OpenRouter
        "X-Title": "Your App Name"
    }
)

model = OpenAIModel(config)
```

## Advanced Configuration

### Custom Headers

For APIs requiring custom authentication or headers:

```python
config = ModelConfig(
    api_key="your-api-key",
    model="your-model",
    base_url="https://api.custom-provider.com/v1",
    additional_headers={
        "X-Custom-Auth": "bearer your-token",
        "X-API-Version": "v1",
        "User-Agent": "AgentFrame/1.0"
    }
)
```

### Custom Parameters

Override any LangChain ChatOpenAI parameters:

```python
config = ModelConfig(
    api_key="your-api-key",
    model="your-model",
    base_url="https://api.provider.com/v1",
    custom_params={
        "streaming": True,
        "tiktoken_model_name": "gpt-3.5-turbo",  # For token counting
        "model_kwargs": {"stop": ["Human:", "AI:"]}
    }
)
```

## Environment Variables

Set up your `.env` file for different providers:

```bash
# Standard OpenAI
OPENAI_API_KEY=sk-your-openai-key
AGENT_MODEL=gpt-4

# Groq
OPENAI_API_KEY=gsk-your-groq-key
OPENAI_BASE_URL=https://api.groq.com/openai/v1
AGENT_MODEL=mixtral-8x7b-32768

# Local Ollama
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
AGENT_MODEL=llama2:7b

# Together AI
OPENAI_API_KEY=your-together-key
OPENAI_BASE_URL=https://api.together.xyz/v1
AGENT_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
```

## Usage in AgentFrame

Once configured, use the model exactly as you would with OpenAI:

```python
from agentframe import Agent, ModelConfig, OpenAIModel, tool

@tool
def calculator(expression: str) -> float:
    """Calculate mathematical expressions."""
    return eval(expression)

# Configure for any OpenAI-compatible API
config = ModelConfig(
    api_key="your-api-key",
    model="your-model",
    base_url="https://your-api.com/v1",  # Custom endpoint
    temperature=0.7
)

model = OpenAIModel(config)
agent = Agent(model=model, tools=[calculator])

# Use exactly the same as with OpenAI
response = agent.run("What's 25% of 400?")
print(response)
```

## FastAPI Configuration

In your FastAPI service, load configuration from environment:

```python
from pydantic_settings import BaseSettings
from agentframe import ModelConfig, OpenAIModel

class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: Optional[str] = None
    agent_model: str = "gpt-3.5-turbo"
    agent_temperature: float = 0.7

    class Config:
        env_file = ".env"

settings = Settings()

# Create model configuration
model_config = ModelConfig(
    api_key=settings.openai_api_key,
    model=settings.agent_model,
    base_url=settings.openai_base_url,  # Will be None for standard OpenAI
    temperature=settings.agent_temperature
)

model = OpenAIModel(model_config)
```

## Troubleshooting

### Common Issues

1. **Connection Errors**: Verify the base URL is correct and accessible
2. **Authentication Errors**: Check API key format and permissions
3. **Model Not Found**: Ensure the model name matches the provider's format
4. **Timeout Issues**: Increase timeout for slower local models

### Debugging

Enable debug logging to see configuration details:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show initialization details
model = OpenAIModel(config)
```

### Testing Connection

Test your configuration before using with agents:

```python
# Simple test
try:
    response = model.generate([{"role": "user", "content": "Hello"}])
    print(f"Success: {response.content}")
except Exception as e:
    print(f"Error: {e}")
```

## Performance Considerations

- **Local Models**: May be slower but provide privacy and cost benefits
- **Cloud APIs**: Faster but require internet and may have usage costs
- **Context Length**: Different models have different context limits
- **Rate Limits**: Each provider has different rate limiting policies

## Security Notes

- Never commit API keys to version control
- Use environment variables for sensitive configuration
- Consider network security for local deployments
- Validate model outputs, especially with local/custom models

This flexibility allows AgentFrame to work with virtually any LLM provider that offers an OpenAI-compatible API, giving you the freedom to choose the best model for your specific needs while maintaining a consistent development experience.