"""
Basic AgentFrame Example

This example demonstrates how to create a simple agent with custom tools
and use it to process user requests with planning and execution.
"""

import os
from agentframe import Agent, OpenAIModel, ModelConfig, tool, AgentConfig


# Define custom tools using the @tool decorator
@tool
def calculator(expression: str) -> float:
    """
    Calculate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., "2+2", "sqrt(16)")

    Returns:
        The calculated result
    """
    import math
    import re

    # Simple safety check - only allow basic math operations
    allowed_chars = set('0123456789+-*/().sqrt() ')
    if not all(c in allowed_chars for c in expression.replace(' ', '')):
        raise ValueError("Invalid characters in expression")

    # Replace sqrt with math.sqrt
    expression = re.sub(r'sqrt\(([^)]+)\)', r'math.sqrt(\1)', expression)

    try:
        # Evaluate safely
        result = eval(expression, {"__builtins__": {}, "math": math})
        return float(result)
    except Exception as e:
        raise ValueError(f"Cannot evaluate expression: {e}")


@tool
def text_analyzer(text: str) -> dict:
    """
    Analyze text and return statistics.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with text statistics
    """
    words = text.split()
    characters = len(text)
    sentences = len([s for s in text.split('.') if s.strip()])

    return {
        "word_count": len(words),
        "character_count": characters,
        "sentence_count": sentences,
        "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> dict:
    """
    Convert between different units.

    Args:
        value: Numeric value to convert
        from_unit: Source unit (e.g., "celsius", "fahrenheit", "meters", "feet")
        to_unit: Target unit

    Returns:
        Dictionary with conversion result
    """
    # Temperature conversions
    if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
        result = (value * 9/5) + 32
    elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
        result = (value - 32) * 5/9

    # Length conversions
    elif from_unit.lower() == "meters" and to_unit.lower() == "feet":
        result = value * 3.28084
    elif from_unit.lower() == "feet" and to_unit.lower() == "meters":
        result = value / 3.28084

    # Weight conversions
    elif from_unit.lower() == "kilograms" and to_unit.lower() == "pounds":
        result = value * 2.20462
    elif from_unit.lower() == "pounds" and to_unit.lower() == "kilograms":
        result = value / 2.20462

    else:
        raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported")

    return {
        "original_value": value,
        "original_unit": from_unit,
        "converted_value": round(result, 4),
        "converted_unit": to_unit,
        "conversion_formula": f"{value} {from_unit} = {round(result, 4)} {to_unit}"
    }


def create_agent():
    """Create and configure the agent."""
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("Using placeholder key for demonstration.")
        api_key = "your-openai-api-key-here"

    # Configure the model
    model_config = ModelConfig(
        api_key=api_key,
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000
    )

    # Create the model
    model = OpenAIModel(model_config)

    # Configure the agent
    agent_config = AgentConfig(
        max_replanning_iterations=3,
        enable_streaming=False,
        verbose_logging=True,
        confidence_threshold=0.6
    )

    # Create agent with tools
    agent = Agent(
        model=model,
        tools=[calculator, text_analyzer, unit_converter],
        config=agent_config
    )

    return agent


def demo_agent():
    """Demonstrate the agent with various tasks."""
    print("🤖 AgentFrame Basic Example")
    print("=" * 50)

    # Create the agent
    agent = create_agent()

    # Example conversations
    examples = [
        "Calculate the area of a circle with radius 5 (use π ≈ 3.14159)",
        "Analyze this text: 'The quick brown fox jumps over the lazy dog. This is a test sentence.'",
        "Convert 25 degrees Celsius to Fahrenheit",
        "What's the square root of 144 plus 8?",
        "Convert 6 feet to meters and then calculate the area of a square with that side length"
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n📝 Example {i}: {example}")
        print("-" * 60)

        try:
            response = agent.run(example)
            print(f"🤖 Agent: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")

        print()

    # Show conversation summary
    print("\n📊 Conversation Summary:")
    print("-" * 30)
    summary = agent.get_conversation_summary()
    print(f"Total messages: {summary['total_messages']}")
    print(f"Total tokens: {summary['total_tokens']}")
    print(f"Available tools: {summary['available_tools']}")
    print(f"Session ID: {summary['session_id']}")


def interactive_mode():
    """Run the agent in interactive mode."""
    print("🤖 AgentFrame Interactive Mode")
    print("Type 'quit' to exit, 'reset' to reset conversation")
    print("=" * 50)

    agent = create_agent()

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'reset':
                agent.reset_conversation()
                print("🔄 Conversation reset!")
                continue
            elif user_input.lower() == 'summary':
                summary = agent.get_conversation_summary()
                print(f"📊 Total messages: {summary['total_messages']}")
                print(f"📊 Available tools: {summary['available_tools']}")
                continue
            elif not user_input:
                continue

            # Process user input
            response = agent.run(user_input)
            print(f"🤖 Agent: {response}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys

    print("AgentFrame - Basic Example")
    print("Choose mode:")
    print("1. Demo mode (predefined examples)")
    print("2. Interactive mode")

    try:
        choice = input("Enter choice (1 or 2): ").strip()

        if choice == "1":
            demo_agent()
        elif choice == "2":
            interactive_mode()
        else:
            print("Invalid choice. Running demo mode.")
            demo_agent()

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error running example: {e}")
        print("\nMake sure you have:")
        print("1. Installed the dependencies: pip install agentframe")
        print("2. Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")