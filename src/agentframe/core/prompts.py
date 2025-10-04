"""
Customizable prompt system for AgentFrame.

This module provides a flexible prompt system that allows customization of
agent behavior, instructions, and response generation patterns.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Types of prompts used in the agent system."""
    SYSTEM_INSTRUCTION = "system_instruction"
    INTENT_PARSING = "intent_parsing"
    PLANNING = "planning"
    REPLANNING = "replanning"
    RESPONSE_GENERATION = "response_generation"
    TOOL_SELECTION = "tool_selection"
    ERROR_HANDLING = "error_handling"


@dataclass
class PromptTemplate:
    """
    A customizable prompt template with variables.

    Attributes:
        name: Unique name for the template
        template: Template string with {variable} placeholders
        required_variables: List of required variable names
        optional_variables: List of optional variable names with defaults
        description: Description of what this prompt does
        examples: Usage examples for the prompt

    Example:
        >>> template = PromptTemplate(
        ...     name="custom_greeting",
        ...     template="Hello! I'm {agent_name}, {agent_role}. How can I help you?",
        ...     required_variables=["agent_name", "agent_role"],
        ...     description="Custom greeting for the agent"
        ... )
    """
    name: str
    template: str
    required_variables: List[str] = field(default_factory=list)
    optional_variables: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    examples: List[str] = field(default_factory=list)

    def format(self, **kwargs) -> str:
        """
        Format the template with provided variables.

        Args:
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If required variables are missing

        Example:
            >>> prompt = template.format(agent_name="Assistant", agent_role="helpful AI")
            >>> print(prompt)
            "Hello! I'm Assistant, helpful AI. How can I help you?"
        """
        # Check for required variables
        missing = set(self.required_variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Add optional variables with defaults
        variables = {**self.optional_variables, **kwargs}

        try:
            return self.template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Template variable {e} not provided")

    def validate(self) -> bool:
        """
        Validate the template structure.

        Returns:
            True if template is valid

        Example:
            >>> assert template.validate() == True
        """
        try:
            # Try formatting with dummy values
            dummy_vars = {var: f"dummy_{var}" for var in self.required_variables}
            dummy_vars.update(self.optional_variables)
            self.template.format(**dummy_vars)
            return True
        except Exception as e:
            logger.error(f"Template validation failed: {e}")
            return False


@dataclass
class AgentPrompts:
    """
    Collection of prompts that define agent behavior.

    This class contains all the prompts used by an agent to customize its
    behavior, responses, and interaction patterns.

    Attributes:
        system_instruction: Core system prompt defining agent personality and rules
        intent_parsing: Prompt for understanding user intent
        planning: Prompt for generating execution plans
        replanning: Prompt for creating new plans when original fails
        response_generation: Prompt for generating final responses
        tool_selection: Prompt for choosing appropriate tools
        error_handling: Prompt for handling errors gracefully
        custom_prompts: Additional custom prompts for specific use cases

    Example:
        >>> prompts = AgentPrompts(
        ...     system_instruction=PromptTemplate(
        ...         name="system",
        ...         template="You are {agent_name}, a {agent_type} assistant..."
        ...     )
        ... )
    """
    system_instruction: Optional[PromptTemplate] = None
    intent_parsing: Optional[PromptTemplate] = None
    planning: Optional[PromptTemplate] = None
    replanning: Optional[PromptTemplate] = None
    response_generation: Optional[PromptTemplate] = None
    tool_selection: Optional[PromptTemplate] = None
    error_handling: Optional[PromptTemplate] = None
    custom_prompts: Dict[str, PromptTemplate] = field(default_factory=dict)

    def get_prompt(self, prompt_type: Union[PromptType, str]) -> Optional[PromptTemplate]:
        """
        Get a prompt by type.

        Args:
            prompt_type: Type of prompt to retrieve

        Returns:
            PromptTemplate if found, None otherwise

        Example:
            >>> template = prompts.get_prompt(PromptType.SYSTEM_INSTRUCTION)
        """
        if isinstance(prompt_type, PromptType):
            prompt_type = prompt_type.value

        # Map prompt types to attributes
        mapping = {
            "system_instruction": self.system_instruction,
            "intent_parsing": self.intent_parsing,
            "planning": self.planning,
            "replanning": self.replanning,
            "response_generation": self.response_generation,
            "tool_selection": self.tool_selection,
            "error_handling": self.error_handling
        }

        # Check standard prompts first
        if prompt_type in mapping:
            return mapping[prompt_type]

        # Check custom prompts
        return self.custom_prompts.get(prompt_type)

    def set_prompt(self, prompt_type: Union[PromptType, str], template: PromptTemplate) -> None:
        """
        Set a prompt template.

        Args:
            prompt_type: Type of prompt to set
            template: PromptTemplate to use

        Example:
            >>> prompts.set_prompt(PromptType.SYSTEM_INSTRUCTION, new_template)
        """
        if isinstance(prompt_type, PromptType):
            prompt_type = prompt_type.value

        # Validate template
        if not template.validate():
            raise ValueError(f"Invalid template for {prompt_type}")

        # Set standard prompts
        if prompt_type == "system_instruction":
            self.system_instruction = template
        elif prompt_type == "intent_parsing":
            self.intent_parsing = template
        elif prompt_type == "planning":
            self.planning = template
        elif prompt_type == "replanning":
            self.replanning = template
        elif prompt_type == "response_generation":
            self.response_generation = template
        elif prompt_type == "tool_selection":
            self.tool_selection = template
        elif prompt_type == "error_handling":
            self.error_handling = template
        else:
            # Custom prompt
            self.custom_prompts[prompt_type] = template

        logger.debug(f"Set prompt template for: {prompt_type}")

    def add_custom_prompt(self, name: str, template: PromptTemplate) -> None:
        """
        Add a custom prompt template.

        Args:
            name: Name for the custom prompt
            template: PromptTemplate to add

        Example:
            >>> prompts.add_custom_prompt("greeting", greeting_template)
        """
        self.custom_prompts[name] = template
        logger.debug(f"Added custom prompt: {name}")

    def validate_all(self) -> Dict[str, bool]:
        """
        Validate all prompt templates.

        Returns:
            Dictionary mapping prompt names to validation results

        Example:
            >>> results = prompts.validate_all()
            >>> if all(results.values()):
            ...     print("All prompts are valid")
        """
        results = {}

        # Validate standard prompts
        for attr_name in ["system_instruction", "intent_parsing", "planning",
                         "replanning", "response_generation", "tool_selection", "error_handling"]:
            template = getattr(self, attr_name)
            if template:
                results[attr_name] = template.validate()

        # Validate custom prompts
        for name, template in self.custom_prompts.items():
            results[f"custom_{name}"] = template.validate()

        return results


class DefaultPrompts:
    """
    Factory class for creating default prompt templates.

    Provides standard prompt templates that can be used as-is or
    customized for specific agent types.
    """

    @staticmethod
    def get_system_instruction() -> PromptTemplate:
        """Get default system instruction prompt."""
        return PromptTemplate(
            name="default_system",
            template="""You are {agent_name}, {agent_description}.

Core Principles:
- Be helpful, accurate, and honest in all interactions
- Use available tools when needed to gather information or perform tasks
- If you're unsure about something, say so rather than guessing
- Break down complex tasks into manageable steps
- Always explain your reasoning when using tools or making decisions

Capabilities:
- I have access to various tools that I can use to help you
- I can plan multi-step tasks and adjust my approach if needed
- I can search for information, perform calculations, and more
- I maintain conversation context throughout our interaction

Guidelines:
- {guidelines}
- Ask for clarification if requests are ambiguous
- Provide step-by-step explanations for complex processes
- Be concise but thorough in responses
- Respect user preferences and adapt communication style accordingly

How can I assist you today?""",
            required_variables=["agent_name", "agent_description"],
            optional_variables={"guidelines": "Follow user instructions carefully"},
            description="Default system instruction for general-purpose agents"
        )

    @staticmethod
    def get_intent_parsing() -> PromptTemplate:
        """Get default intent parsing prompt."""
        return PromptTemplate(
            name="default_intent_parsing",
            template="""Analyze the user's message and extract their intent.

User Message: "{user_message}"

Available Tools: {available_tools}
Conversation Context: {conversation_context}

Extract the following information and return as JSON:
{{
    "primary_goal": "main objective the user wants to achieve",
    "entities": ["list", "of", "important", "entities"],
    "task_type": "single_step|multi_step|conversational|informational|creative|analytical",
    "category": "question|request|command|clarification|greeting|goodbye",
    "confidence": 0.0-1.0,
    "extracted_parameters": {{"key": "value pairs from the message"}},
    "requires_tools": true/false,
    "urgency": "low|medium|high",
    "context_needed": true/false,
    "ambiguity_score": 0.0-1.0
}}

Guidelines:
- {intent_guidelines}
- Be specific about the primary goal
- Extract all relevant entities and parameters
- Assess tool requirements accurately
- Consider conversation context when determining intent

Return only the JSON, no additional text.""",
            required_variables=["user_message", "available_tools"],
            optional_variables={
                "conversation_context": "No previous context",
                "intent_guidelines": "Analyze intent thoroughly and accurately"
            },
            description="Default prompt for parsing user intent"
        )

    @staticmethod
    def get_planning() -> PromptTemplate:
        """Get default planning prompt."""
        return PromptTemplate(
            name="default_planning",
            template="""You are an intelligent task planner. Create a detailed execution plan.

Goal: {goal}
Available Tools: {available_tools}
Context: {context}

Planning Guidelines:
- {planning_guidelines}
- Break down complex goals into simple, atomic steps
- Each step should use exactly one tool
- Ensure dependencies are properly ordered
- Be specific with tool arguments
- Aim for high confidence plans

Create a JSON plan with this structure:
{{
    "plan_id": "unique_plan_identifier",
    "goal": "restated goal",
    "confidence": 0.0-1.0,
    "estimated_duration": seconds_to_complete,
    "steps": [
        {{
            "step_id": "step_1",
            "tool_name": "tool_to_use",
            "arguments": {{"param": "value"}},
            "expected_output": "what this step should produce",
            "dependencies": [],
            "description": "human readable description",
            "reasoning": "why this step is needed"
        }}
    ],
    "alternative_approaches": ["other ways to achieve the goal"]
}}

Maximum {max_steps} steps. Return only the JSON, no additional text.""",
            required_variables=["goal", "available_tools"],
            optional_variables={
                "context": "No additional context",
                "planning_guidelines": "Create efficient and reliable plans",
                "max_steps": "10"
            },
            description="Default prompt for generating execution plans"
        )

    @staticmethod
    def get_response_generation() -> PromptTemplate:
        """Get default response generation prompt."""
        return PromptTemplate(
            name="default_response",
            template="""Generate a helpful response based on the user's request and execution results.

User Request: {user_request}
Execution Results: {execution_results}
Agent Role: {agent_role}

Response Guidelines:
- {response_guidelines}
- Be natural and conversational
- Directly address the user's original request
- Explain what was done and why
- Present results clearly and concisely
- Acknowledge any limitations or uncertainties
- Offer additional help if appropriate

Response Style: {response_style}

Generate a helpful, accurate response that addresses the user's needs.""",
            required_variables=["user_request", "execution_results"],
            optional_variables={
                "agent_role": "helpful assistant",
                "response_guidelines": "Be helpful, accurate, and clear",
                "response_style": "professional and friendly"
            },
            description="Default prompt for generating final responses"
        )

    @staticmethod
    def get_error_handling() -> PromptTemplate:
        """Get default error handling prompt."""
        return PromptTemplate(
            name="default_error_handling",
            template="""Handle the following error gracefully and provide a helpful response.

Original Request: {original_request}
Error Details: {error_details}
Failed Step: {failed_step}
Agent Role: {agent_role}

Error Handling Guidelines:
- {error_guidelines}
- Acknowledge the issue without overwhelming technical details
- Explain what went wrong in simple terms
- Suggest alternative approaches if possible
- Maintain a helpful and professional tone
- Offer to try a different approach

Generate a response that:
1. Acknowledges the issue
2. Explains what happened (simply)
3. Suggests next steps or alternatives
4. Maintains user confidence in the system

Keep the response helpful and solution-oriented.""",
            required_variables=["original_request", "error_details"],
            optional_variables={
                "failed_step": "Unknown step",
                "agent_role": "helpful assistant",
                "error_guidelines": "Handle errors gracefully and helpfully"
            },
            description="Default prompt for handling errors"
        )

    @staticmethod
    def create_custom_agent_prompts(
        agent_name: str,
        agent_description: str,
        personality_traits: List[str],
        specific_guidelines: List[str],
        response_style: str = "professional and friendly"
    ) -> AgentPrompts:
        """
        Create a complete set of customized prompts for a specific agent type.

        Args:
            agent_name: Name of the agent
            agent_description: Description of the agent's role
            personality_traits: List of personality characteristics
            specific_guidelines: List of specific behavior guidelines
            response_style: Preferred communication style

        Returns:
            AgentPrompts with customized templates

        Example:
            >>> prompts = DefaultPrompts.create_custom_agent_prompts(
            ...     agent_name="DataBot",
            ...     agent_description="a data analysis specialist",
            ...     personality_traits=["analytical", "precise", "thorough"],
            ...     specific_guidelines=["Always cite data sources", "Explain statistical methods"],
            ...     response_style="technical but accessible"
            ... )
        """
        # Create customized system instruction
        personality_str = ", ".join(personality_traits)
        guidelines_str = "\n- ".join(specific_guidelines)

        system_template = PromptTemplate(
            name=f"{agent_name.lower()}_system",
            template=f"""You are {agent_name}, {agent_description}.

Personality: You are {personality_str}.

Core Principles:
- Be helpful, accurate, and honest in all interactions
- Use available tools when needed to gather information or perform tasks
- If you're unsure about something, say so rather than guessing
- Break down complex tasks into manageable steps
- Always explain your reasoning when using tools or making decisions

Specific Guidelines:
- {guidelines_str}
- {{additional_guidelines}}

Capabilities:
- I have access to various tools that I can use to help you
- I can plan multi-step tasks and adjust my approach if needed
- I maintain conversation context throughout our interaction

Communication Style: {response_style}

How can I assist you today?""",
            required_variables=[],
            optional_variables={"additional_guidelines": "Follow user instructions carefully"},
            description=f"Custom system instruction for {agent_name}"
        )

        # Create customized response generation
        response_template = PromptTemplate(
            name=f"{agent_name.lower()}_response",
            template=f"""Generate a helpful response based on the user's request and execution results.

User Request: {{user_request}}
Execution Results: {{execution_results}}

As {agent_name} ({agent_description}), respond in a way that is:
- {personality_str}
- {response_style}

Guidelines:
- {guidelines_str}
- {{response_guidelines}}
- Present information clearly and accurately
- Explain your reasoning and methods
- Acknowledge any limitations or uncertainties

Generate a response that reflects your role as {agent_name}.""",
            required_variables=["user_request", "execution_results"],
            optional_variables={"response_guidelines": "Be thorough and helpful"},
            description=f"Custom response generation for {agent_name}"
        )

        # Create the AgentPrompts object
        prompts = AgentPrompts()
        prompts.set_prompt(PromptType.SYSTEM_INSTRUCTION, system_template)
        prompts.set_prompt(PromptType.RESPONSE_GENERATION, response_template)

        # Use default prompts for other types
        prompts.set_prompt(PromptType.INTENT_PARSING, DefaultPrompts.get_intent_parsing())
        prompts.set_prompt(PromptType.PLANNING, DefaultPrompts.get_planning())
        prompts.set_prompt(PromptType.ERROR_HANDLING, DefaultPrompts.get_error_handling())

        return prompts


def create_agent_prompts_from_config(config: Dict[str, Any]) -> AgentPrompts:
    """
    Create AgentPrompts from a configuration dictionary.

    Args:
        config: Configuration dictionary with prompt definitions

    Returns:
        AgentPrompts object

    Example:
        >>> config = {
        ...     "agent_name": "Helper",
        ...     "agent_description": "a general assistant",
        ...     "system_prompt": "You are a helpful assistant...",
        ...     "custom_prompts": {
        ...         "greeting": {
        ...             "template": "Hello! I'm {name}",
        ...             "required_variables": ["name"]
        ...         }
        ...     }
        ... }
        >>> prompts = create_agent_prompts_from_config(config)
    """
    prompts = AgentPrompts()

    # Create system instruction if provided
    if "system_prompt" in config:
        system_template = PromptTemplate(
            name="config_system",
            template=config["system_prompt"],
            required_variables=config.get("system_required_vars", []),
            optional_variables=config.get("system_optional_vars", {}),
            description="System prompt from configuration"
        )
        prompts.set_prompt(PromptType.SYSTEM_INSTRUCTION, system_template)

    # Add custom prompts
    for name, prompt_config in config.get("custom_prompts", {}).items():
        template = PromptTemplate(
            name=name,
            template=prompt_config["template"],
            required_variables=prompt_config.get("required_variables", []),
            optional_variables=prompt_config.get("optional_variables", {}),
            description=prompt_config.get("description", f"Custom prompt: {name}")
        )
        prompts.add_custom_prompt(name, template)

    return prompts