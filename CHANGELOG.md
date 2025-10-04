# Changelog

All notable changes to AgentFrame will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Custom prompt system for agent behavior customization
- PromptTemplate class for creating reusable prompt templates
- AgentPrompts class for managing collections of prompts
- AgentConfig class for comprehensive agent configuration
- DefaultPrompts with built-in templates for common scenarios
- Personality traits and communication style configuration
- Custom guidelines and response style settings
- Prompt validation and variable substitution
- Dynamic prompt updates and context-aware prompt selection
- Specialized prompt templates for different agent types:
  - Research assistant prompts
  - Code reviewer prompts
  - Customer support prompts
  - Creative writing assistant prompts
  - Data analyst prompts
- Comprehensive prompt customization documentation
- Examples and templates for common use cases
- FastAPI integration examples with prompt customization

### Enhanced
- Agent class now supports custom prompts and configuration
- Improved agent initialization with prompt and config parameters
- Better error handling for prompt validation
- Enhanced documentation with prompt system guide
- Updated examples with custom prompt usage

### Changed
- Agent constructor now accepts prompts and config parameters
- Response generation now uses custom prompts when available
- Planning system integrated with custom prompt templates

## [0.1.0] - 2024-01-XX

### Added
- Initial release of AgentFrame
- Core agent framework with planning and execution
- Multi-model support (OpenAI, Gemini, Claude)
- Tool system with @tool decorator
- Function calling and tool integration
- Planning and replanning capabilities
- Conversation history management
- Intent parsing and understanding
- State management with LangGraph
- Comprehensive error handling
- Token-aware conversation management
- Streaming response support
- Structured output generation
- OpenAI-compatible API support for local models
- FastAPI integration examples
- Complete documentation and examples

### Framework Features
- **Agent System**: Core agent implementation with planning and execution
- **Model Providers**: Support for OpenAI, Google Gemini, and Anthropic Claude
- **Tool Integration**: Easy tool creation with function decorators
- **Planning Engine**: Automatic task decomposition and execution planning
- **Memory Management**: Intelligent conversation history handling
- **Error Recovery**: Automatic replanning when execution fails
- **Streaming Support**: Real-time response generation
- **Flexible Configuration**: Comprehensive configuration options

### Model Provider Support
- **OpenAI**: Full GPT-4, GPT-3.5-turbo support with function calling
- **OpenAI-Compatible**: Support for Ollama, Groq, Together AI, Anyscale, vLLM, LocalAI
- **Google Gemini**: Gemini Pro integration with tool support
- **Anthropic Claude**: Claude 3 family support with function calling

### Tool System
- **@tool Decorator**: Simple function-to-tool conversion
- **Tool Registry**: Centralized tool management
- **Custom Tools**: Support for complex tool implementations
- **Tool Categories**: Organized tool discovery and selection
- **Error Handling**: Robust tool execution with fallbacks

### Integration Features
- **FastAPI**: Complete web service integration
- **Session Management**: Persistent conversation handling
- **WebSocket Support**: Real-time communication
- **Streaming Responses**: Server-sent events for live updates
- **Background Tasks**: Asynchronous processing support

### Documentation
- **Complete API Reference**: Comprehensive documentation for all classes
- **Usage Examples**: Extensive examples for common use cases
- **Integration Guides**: FastAPI, Discord, Slack integration examples
- **Best Practices**: Performance, security, and design guidelines
- **Troubleshooting**: Common issues and solutions

### Dependencies
- **langchain**: LLM integration and orchestration
- **langgraph**: Workflow and state management
- **pydantic**: Data validation and settings management
- **openai**: OpenAI API client
- **google-generativeai**: Google Gemini integration
- **anthropic**: Claude API integration

### Installation
- **PyPI Package**: Available via `pip install agentframe`
- **Source Installation**: GitHub repository with development setup
- **Docker Support**: Containerized deployment options

## [0.0.1] - Development

### Added
- Project initialization
- Core architecture design
- Basic implementation framework
- Development environment setup
- Testing infrastructure
- Documentation structure

---

## Release Notes

### Version 0.1.0 - Initial Release

AgentFrame 0.1.0 represents the first stable release of our comprehensive LLM agent framework. This release includes:

**Core Capabilities:**
- Complete agent lifecycle management from intent parsing to response generation
- Multi-step planning with automatic replanning on failures
- Comprehensive tool integration system
- Multi-model provider support with unified interface
- Advanced conversation management with intelligent history handling

**Key Features:**
- **Easy Tool Integration**: Convert any Python function into an agent tool with a simple decorator
- **Intelligent Planning**: Automatic task decomposition and execution strategy
- **Model Flexibility**: Switch between OpenAI, Gemini, Claude, and OpenAI-compatible APIs
- **Production Ready**: Comprehensive error handling, logging, and configuration options
- **Web Integration**: FastAPI examples and integration patterns

**Developer Experience:**
- Simple, intuitive API design
- Comprehensive documentation with examples
- Type hints and IDE support
- Extensible architecture for custom implementations

### Prompt Customization System (Unreleased)

The upcoming prompt customization system adds powerful agent personalization capabilities:

**New Features:**
- **Custom Prompt Templates**: Create reusable prompt templates with variable substitution
- **Agent Personalities**: Define personality traits, communication styles, and response tones
- **Specialized Agents**: Pre-built templates for research assistants, code reviewers, customer support
- **Dynamic Prompts**: Context-aware prompt selection and runtime updates
- **Validation System**: Comprehensive prompt validation and error handling

**Use Cases:**
- Create domain-specific agents (medical, legal, technical, creative)
- Customize agent behavior for different user types (beginner, expert, business)
- Build specialized tools (code review, research, customer support, content creation)
- Implement brand-specific agent personalities and communication styles

This system maintains AgentFrame's ease of use while providing the flexibility needed for production applications requiring specific agent behaviors and personalities.

---

## Migration Guide

### From Development to 0.1.0
- Update import statements to use the new package structure
- Replace any custom planning implementations with the new planning system
- Update tool definitions to use the @tool decorator
- Review configuration settings and update to new ModelConfig format

### Upgrading to Prompt Customization (When Released)
- Existing agents will continue to work with default prompts
- To use custom prompts, create PromptTemplate instances and AgentPrompts collections
- Update agent initialization to include prompts and config parameters
- Review agent behavior and adjust prompts as needed

---

## Support and Community

- **GitHub**: [https://github.com/LasinduAnjana/agentframe](https://github.com/LasinduAnjana/agentframe)
- **Documentation**: Available in the `docs/` directory
- **Examples**: Complete examples in the `examples/` directory
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Community discussions and questions

## Contributors

- **Lasindu Anjana** - Initial development and framework design
- **AgentFrame Contributors** - Community contributions and feedback

Thank you to all contributors who have helped make AgentFrame a powerful and flexible agent development framework!