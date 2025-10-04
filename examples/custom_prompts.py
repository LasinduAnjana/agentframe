"""
Example: Custom Prompt Templates for AgentFrame

This example demonstrates how to create custom prompt templates for different
types of agents using the AgentFrame prompt system.
"""

from agentframe import (
    Agent,
    OpenAIModel,
    ModelConfig,
    PromptTemplate,
    AgentPrompts,
    AgentConfig,
    tool
)

@tool
def calculator(expression: str) -> float:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return float(result)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def search_web(query: str) -> dict:
    """Search the web for information (mock implementation)."""
    return {
        "query": query,
        "results": [
            f"Mock search result for: {query}",
            f"Another result about {query}"
        ],
        "total_results": 2
    }

# Example 1: Customer Support Agent
def create_customer_support_agent():
    """Create a customer support agent with custom prompts."""

    # Custom system prompt for customer support
    system_prompt = PromptTemplate(
        name="customer_support_system",
        template="""You are a helpful customer support agent for TechCorp.

Your personality traits:
- {personality_traits}

Your communication style:
- {communication_style}

Company policies:
- Always be polite and professional
- Escalate complex technical issues to specialists
- Offer step-by-step solutions when possible
- Follow up to ensure customer satisfaction

Custom guidelines:
{custom_guidelines}

Remember to maintain a {response_style} tone throughout the conversation.""",
        required_variables=["personality_traits", "communication_style", "response_style"],
        optional_variables={"custom_guidelines": "No additional guidelines."},
        description="System prompt for customer support agents",
        examples=[
            "Handle customer inquiries professionally",
            "Provide step-by-step technical guidance",
            "Escalate when necessary"
        ]
    )

    # Custom planning prompt
    planning_prompt = PromptTemplate(
        name="customer_support_planning",
        template="""Plan how to resolve this customer issue: {user_input}

Available tools: {available_tools}

Consider:
1. What type of issue is this?
2. Can it be resolved with available tools?
3. Does it need escalation?
4. What information do you need from the customer?

Create a step-by-step plan to help the customer effectively.""",
        required_variables=["user_input", "available_tools"],
        description="Planning prompt for customer support scenarios"
    )

    # Create custom prompts
    prompts = AgentPrompts(
        system_prompt=system_prompt,
        planning_prompt=planning_prompt
    )

    # Configure agent personality
    config = AgentConfig(
        personality_traits="empathetic, patient, knowledgeable, solution-oriented",
        communication_style="clear, friendly, and professional",
        response_style="helpful",
        custom_guidelines="Always ask for order numbers when dealing with purchase issues. Offer alternative solutions when the first option doesn't work."
    )

    # Setup model
    model_config = ModelConfig(
        api_key="your-api-key",
        model="gpt-4",
        temperature=0.3  # Lower temperature for consistent support responses
    )
    model = OpenAIModel(model_config)

    # Create agent
    agent = Agent(
        model=model,
        tools=[calculator, search_web],
        config=config,
        prompts=prompts
    )

    return agent

# Example 2: Creative Writing Assistant
def create_creative_writing_agent():
    """Create a creative writing assistant with custom prompts."""

    # Creative system prompt
    system_prompt = PromptTemplate(
        name="creative_writing_system",
        template="""You are a creative writing assistant specializing in storytelling and content creation.

Your personality: {personality_traits}
Your style: {communication_style}

Expertise areas:
- Character development
- Plot structure and pacing
- Dialogue writing
- World building
- Genre conventions

Approach:
- Encourage creativity and experimentation
- Provide constructive feedback
- Offer multiple perspectives
- Use {response_style} communication

Additional focus areas:
{custom_guidelines}""",
        required_variables=["personality_traits", "communication_style", "response_style"],
        optional_variables={"custom_guidelines": "Focus on helping writers overcome creative blocks."},
        description="System prompt for creative writing assistance"
    )

    # Creative planning prompt
    planning_prompt = PromptTemplate(
        name="creative_planning",
        template="""Help with this creative writing request: {user_input}

Available tools: {available_tools}

Consider:
1. What type of creative assistance is needed?
2. What genre or style applies?
3. What research might be helpful?
4. What creative techniques could enhance the work?

Plan a creative approach to help the writer achieve their goals.""",
        required_variables=["user_input", "available_tools"],
        description="Planning prompt for creative writing tasks"
    )

    # Response generation with creative flair
    response_prompt = PromptTemplate(
        name="creative_response",
        template="""Based on the plan and context, provide creative writing assistance.

User request: {user_input}
Current plan: {plan}
Available tools: {available_tools}

Provide your response in a {response_style} manner, incorporating:
- Specific examples and techniques
- Encouraging and inspiring language
- Practical actionable advice
- Creative alternatives when relevant

Remember your personality: {personality_traits}""",
        required_variables=["user_input", "plan", "available_tools", "response_style", "personality_traits"],
        description="Response generation for creative writing assistance"
    )

    prompts = AgentPrompts(
        system_prompt=system_prompt,
        planning_prompt=planning_prompt,
        response_prompt=response_prompt
    )

    config = AgentConfig(
        personality_traits="imaginative, encouraging, insightful, passionate about storytelling",
        communication_style="inspiring and supportive with vivid examples",
        response_style="enthusiastic",
        custom_guidelines="Always provide specific examples. Encourage experimentation with different narrative techniques. Help writers find their unique voice."
    )

    model_config = ModelConfig(
        api_key="your-api-key",
        model="gpt-4",
        temperature=0.8  # Higher temperature for more creative responses
    )
    model = OpenAIModel(model_config)

    agent = Agent(
        model=model,
        tools=[search_web],  # For research
        config=config,
        prompts=prompts
    )

    return agent

# Example 3: Data Analyst Agent
def create_data_analyst_agent():
    """Create a data analyst agent with custom prompts."""

    system_prompt = PromptTemplate(
        name="data_analyst_system",
        template="""You are a senior data analyst with expertise in statistical analysis and data interpretation.

Your approach:
- {personality_traits}
- Communication: {communication_style}
- Tone: {response_style}

Specializations:
- Statistical analysis and hypothesis testing
- Data visualization recommendations
- Pattern recognition and insights
- Business intelligence and reporting

Methodology:
- Always validate data quality first
- Use appropriate statistical methods
- Explain assumptions and limitations
- Provide actionable insights

{custom_guidelines}""",
        required_variables=["personality_traits", "communication_style", "response_style"],
        optional_variables={"custom_guidelines": "Focus on practical business applications of analysis."},
        description="System prompt for data analysis tasks"
    )

    prompts = AgentPrompts(system_prompt=system_prompt)

    config = AgentConfig(
        personality_traits="analytical, methodical, detail-oriented, evidence-based",
        communication_style="clear and precise with data-driven explanations",
        response_style="professional",
        custom_guidelines="Always ask about data sources and quality. Recommend visualization types. Explain statistical significance in business terms."
    )

    model_config = ModelConfig(
        api_key="your-api-key",
        model="gpt-4",
        temperature=0.2  # Low temperature for analytical consistency
    )
    model = OpenAIModel(model_config)

    agent = Agent(
        model=model,
        tools=[calculator],
        config=config,
        prompts=prompts
    )

    return agent

# Example usage
def main():
    """Demonstrate different agent types with custom prompts."""

    print("=== Customer Support Agent ===")
    support_agent = create_customer_support_agent()

    # Example customer support query
    support_response = support_agent.run(
        "I'm having trouble with my order #12345. It was supposed to arrive yesterday but I haven't received it yet."
    )
    print(f"Support Response: {support_response}")

    print("\n=== Creative Writing Agent ===")
    creative_agent = create_creative_writing_agent()

    # Example creative writing query
    creative_response = creative_agent.run(
        "I'm writing a sci-fi story about time travel but I'm stuck on how to handle the paradoxes. Any suggestions?"
    )
    print(f"Creative Response: {creative_response}")

    print("\n=== Data Analyst Agent ===")
    analyst_agent = create_data_analyst_agent()

    # Example data analysis query
    analyst_response = analyst_agent.run(
        "I have sales data showing a 15% decrease last quarter. What analysis should I perform to understand the causes?"
    )
    print(f"Analyst Response: {analyst_response}")

if __name__ == "__main__":
    main()