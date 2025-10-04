# Prompt Customization in AgentFrame

AgentFrame provides a powerful prompt customization system that allows you to create specialized agents with different personalities, communication styles, and behavior patterns. This system enables you to build agents tailored for specific use cases while maintaining the framework's core functionality.

## Overview

The prompt customization system consists of several key components:

- **PromptTemplate**: Individual prompt templates with variables and validation
- **AgentPrompts**: Collection of prompts for different agent functions
- **AgentConfig**: Agent personality and behavior configuration
- **DefaultPrompts**: Built-in prompt templates for common scenarios

## Basic Usage

### Creating Custom Prompts

```python
from agentframe import PromptTemplate, AgentPrompts, AgentConfig, Agent, OpenAIModel, ModelConfig

# Create a custom system prompt
system_prompt = PromptTemplate(
    name="customer_support",
    template="""You are a helpful customer support agent.

Personality: {personality_traits}
Communication Style: {communication_style}
Response Tone: {response_style}

Guidelines:
{custom_guidelines}""",
    required_variables=["personality_traits", "communication_style", "response_style"],
    optional_variables={"custom_guidelines": "Be helpful and professional."},
    description="Customer support system prompt"
)

# Create agent prompts collection
prompts = AgentPrompts(system_prompt=system_prompt)

# Configure agent personality
config = AgentConfig(
    personality_traits="empathetic, patient, knowledgeable",
    communication_style="clear and friendly",
    response_style="helpful",
    custom_guidelines="Always ask for order numbers when needed."
)

# Create the agent
model_config = ModelConfig(api_key="your-key", model="gpt-4")
model = OpenAIModel(model_config)
agent = Agent(model=model, config=config, prompts=prompts)
```

## PromptTemplate Class

The `PromptTemplate` class defines individual prompt templates with validation and variable substitution.

### Properties

- `name`: Unique identifier for the template
- `template`: The prompt text with variable placeholders
- `required_variables`: List of variables that must be provided
- `optional_variables`: Dictionary of optional variables with default values
- `description`: Human-readable description of the template's purpose
- `examples`: List of example use cases

### Example

```python
planning_prompt = PromptTemplate(
    name="task_planning",
    template="""Plan how to accomplish: {user_input}

Available tools: {available_tools}
Success criteria: {success_criteria}

Create a step-by-step plan considering:
1. Task complexity and requirements
2. Available resources and constraints
3. Potential risks and mitigation strategies
4. Expected outcomes and deliverables""",
    required_variables=["user_input", "available_tools"],
    optional_variables={"success_criteria": "Task completion"},
    description="Template for planning task execution",
    examples=[
        "Plan a research project",
        "Organize a multi-step workflow",
        "Coordinate team activities"
    ]
)
```

## AgentPrompts Class

The `AgentPrompts` class manages a collection of prompt templates for different agent functions.

### Built-in Prompt Types

- `system_prompt`: Defines agent personality and role
- `planning_prompt`: Used for task planning and strategy
- `response_prompt`: Controls response generation
- `replanning_prompt`: Handles plan adjustments and recovery
- `custom_prompts`: Dictionary for additional specialized prompts

### Example

```python
# Create comprehensive prompt set
prompts = AgentPrompts(
    system_prompt=system_prompt,
    planning_prompt=planning_prompt,
    custom_prompts={
        "error_handling": error_prompt,
        "quality_check": quality_prompt
    }
)

# Update specific prompts
prompts.update_prompt("planning_prompt", new_planning_template)

# Validate all prompts
prompts.validate_all_prompts(variables_dict)
```

## AgentConfig Class

The `AgentConfig` class defines agent personality and behavior parameters.

### Configuration Options

```python
config = AgentConfig(
    personality_traits="analytical, thorough, objective",
    communication_style="technical and precise",
    response_style="informative",
    custom_guidelines="""
    1. Always provide evidence for claims
    2. Acknowledge uncertainty when appropriate
    3. Suggest next steps for complex problems
    4. Use technical terminology appropriately
    """,
    max_iterations=5,
    confidence_threshold=0.8
)
```

### Personality Traits

Define the agent's core characteristics:
- **Analytical**: Data-driven, logical approach
- **Empathetic**: Understanding and supportive
- **Creative**: Innovative and imaginative
- **Professional**: Formal and business-focused
- **Casual**: Friendly and approachable

### Communication Styles

Control how the agent expresses itself:
- **Technical**: Precise, detailed explanations
- **Conversational**: Natural, flowing dialogue
- **Educational**: Teaching-focused with examples
- **Concise**: Brief, to-the-point responses
- **Narrative**: Story-like, engaging explanations

### Response Styles

Set the emotional tone:
- **Helpful**: Supportive and solution-oriented
- **Professional**: Formal and business-appropriate
- **Encouraging**: Positive and motivating
- **Analytical**: Objective and fact-based
- **Creative**: Imaginative and inspiring

## Specialized Agent Examples

### Research Assistant

```python
from agentframe.examples.prompt_templates.research_assistant import (
    research_prompts, research_config, create_research_assistant
)

# Use pre-built research assistant configuration
research_agent = create_research_assistant()

# Or customize further
custom_research_config = AgentConfig(
    personality_traits="methodical, curious, skeptical",
    communication_style="academic and rigorous",
    response_style="informative",
    custom_guidelines="""
    1. Always cite sources
    2. Distinguish facts from opinions
    3. Acknowledge limitations
    4. Suggest additional research when needed
    """
)
```

### Code Reviewer

```python
from agentframe.examples.prompt_templates.code_reviewer import (
    code_reviewer_prompts, code_reviewer_config, create_code_reviewer
)

# Create specialized code reviewer
code_agent = create_code_reviewer()

# Use specific review types
security_review = code_agent.run_with_prompt(
    "security_review",
    "Review this authentication function for security issues",
    code_context=auth_code
)
```

### Customer Support

```python
# Customer support agent with escalation handling
support_config = AgentConfig(
    personality_traits="patient, empathetic, solution-focused",
    communication_style="clear, friendly, professional",
    response_style="helpful",
    custom_guidelines="""
    1. Acknowledge customer frustration
    2. Provide step-by-step solutions
    3. Escalate complex technical issues
    4. Follow up to ensure satisfaction
    """
)

support_prompts = AgentPrompts(
    system_prompt=support_system_template,
    custom_prompts={
        "escalation": escalation_template,
        "resolution": resolution_template
    }
)
```

## Advanced Features

### Dynamic Prompt Updates

```python
# Update prompts based on context
if user_expertise_level == "beginner":
    agent.update_prompt_variable("communication_style", "simple and educational")
elif user_expertise_level == "expert":
    agent.update_prompt_variable("communication_style", "technical and detailed")

# Context-aware prompt selection
if task_type == "debugging":
    response = agent.run_with_prompt("debugging_prompt", user_input)
elif task_type == "feature_request":
    response = agent.run_with_prompt("feature_planning", user_input)
```

### Prompt Validation

```python
# Validate prompt templates
try:
    prompts.validate_prompt("system_prompt", config.__dict__)
except PromptValidationError as e:
    print(f"Validation error: {e}")

# Check required variables
missing_vars = prompts.get_missing_variables("planning_prompt", provided_vars)
if missing_vars:
    print(f"Missing required variables: {missing_vars}")
```

### Custom Prompt Functions

```python
# Create domain-specific prompt generators
def create_medical_agent_prompts():
    """Create prompts for medical information agent."""
    system_prompt = PromptTemplate(
        name="medical_system",
        template="""You are a medical information assistant.

IMPORTANT DISCLAIMERS:
- You provide general information only
- Not a substitute for professional medical advice
- Always recommend consulting healthcare providers
- Emergency situations require immediate medical attention

Your approach: {personality_traits}
Communication: {communication_style}

Medical information guidelines:
{custom_guidelines}""",
        required_variables=["personality_traits", "communication_style"],
        optional_variables={"custom_guidelines": "Focus on evidence-based information."}
    )

    return AgentPrompts(system_prompt=system_prompt)
```

## Best Practices

### 1. Prompt Design

- **Be Specific**: Clear, detailed instructions produce better results
- **Use Examples**: Include example inputs and outputs when helpful
- **Set Boundaries**: Define what the agent should and shouldn't do
- **Consider Context**: Design prompts for your specific use case

### 2. Variable Management

- **Required vs Optional**: Clearly distinguish necessary from optional variables
- **Default Values**: Provide sensible defaults for optional variables
- **Validation**: Always validate variables before prompt generation
- **Documentation**: Document variable purposes and formats

### 3. Agent Personality

- **Consistency**: Ensure personality traits align with communication style
- **Appropriateness**: Match personality to use case and audience
- **Flexibility**: Allow for personality adjustments based on context
- **Testing**: Test different personality combinations

### 4. Prompt Evolution

- **Iterative Improvement**: Refine prompts based on agent performance
- **A/B Testing**: Compare different prompt versions
- **User Feedback**: Incorporate feedback into prompt updates
- **Version Control**: Track prompt changes over time

## Integration with Tools

Prompts work seamlessly with AgentFrame's tool system:

```python
@tool
def analyze_code(code: str, language: str) -> dict:
    """Analyze code for issues and improvements."""
    # Tool implementation
    pass

# Custom prompt for code analysis
analysis_prompt = PromptTemplate(
    name="code_analysis",
    template="""Analyze the provided {language} code using available tools.

Code to analyze:
{code}

Focus areas:
1. Syntax and logic errors
2. Performance optimization opportunities
3. Security vulnerabilities
4. Code style and best practices

Provide specific, actionable recommendations.""",
    required_variables=["language", "code"]
)

# Agent uses custom prompt with tools
agent = Agent(
    model=model,
    tools=[analyze_code],
    prompts=AgentPrompts(custom_prompts={"analysis": analysis_prompt})
)
```

## Error Handling

The prompt system includes comprehensive error handling:

```python
try:
    # Attempt to use custom prompt
    response = agent.run_with_prompt("custom_prompt", user_input)
except PromptValidationError as e:
    # Handle validation errors
    logger.error(f"Prompt validation failed: {e}")
    response = agent.run(user_input)  # Fall back to default prompts
except KeyError as e:
    # Handle missing prompts
    logger.warning(f"Prompt not found: {e}")
    response = agent.run(user_input)
```

## Conclusion

AgentFrame's prompt customization system provides the flexibility to create specialized agents for any domain while maintaining consistent, high-quality interactions. By combining custom prompts with agent configuration and tool integration, you can build powerful, tailored AI assistants that meet specific requirements and user expectations.

For more examples and templates, see the `examples/prompt_templates/` directory in the AgentFrame repository.