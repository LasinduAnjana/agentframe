"""
LLM-based intent parsing for AgentFrame.

This module provides intelligent intent extraction from user messages,
identifying goals, entities, task complexity, and confidence scoring.
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from ..models.base import BaseModel

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks based on complexity."""
    SINGLE_STEP = "single_step"
    MULTI_STEP = "multi_step"
    CONVERSATIONAL = "conversational"
    INFORMATIONAL = "informational"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"


class IntentCategory(Enum):
    """Categories of user intents."""
    QUESTION = "question"
    REQUEST = "request"
    COMMAND = "command"
    CLARIFICATION = "clarification"
    GREETING = "greeting"
    GOODBYE = "goodbye"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"


@dataclass
class Intent:
    """
    Parsed user intent with extracted information.

    Represents the user's goal and associated information extracted
    from their message using LLM-based analysis.

    Attributes:
        primary_goal: Main objective the user wants to achieve
        entities: Named entities and important terms extracted
        task_type: Type of task based on complexity
        category: Category of the intent
        confidence: Confidence score (0.0 to 1.0)
        extracted_parameters: Key-value pairs extracted from the message
        requires_tools: Whether achieving this intent requires tool usage
        urgency: Urgency level (low, medium, high)
        context_needed: Whether additional context is needed
        ambiguity_score: How ambiguous the intent is (0.0 to 1.0)

    Example:
        >>> intent = Intent(
        ...     primary_goal="Calculate the area of a circle",
        ...     entities=["circle", "area", "radius"],
        ...     task_type=TaskType.SINGLE_STEP,
        ...     confidence=0.95
        ... )
    """
    primary_goal: str
    entities: List[str] = field(default_factory=list)
    task_type: TaskType = TaskType.CONVERSATIONAL
    category: IntentCategory = IntentCategory.REQUEST
    confidence: float = 0.0
    extracted_parameters: Dict[str, Any] = field(default_factory=dict)
    requires_tools: bool = False
    urgency: str = "medium"
    context_needed: bool = False
    ambiguity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert intent to dictionary format.

        Returns:
            Dictionary representation of the intent

        Example:
            >>> intent = Intent("search for information", entities=["search"])
            >>> intent_dict = intent.to_dict()
            >>> assert "primary_goal" in intent_dict
        """
        return {
            "primary_goal": self.primary_goal,
            "entities": self.entities,
            "task_type": self.task_type.value,
            "category": self.category.value,
            "confidence": self.confidence,
            "extracted_parameters": self.extracted_parameters,
            "requires_tools": self.requires_tools,
            "urgency": self.urgency,
            "context_needed": self.context_needed,
            "ambiguity_score": self.ambiguity_score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Intent':
        """
        Create intent from dictionary format.

        Args:
            data: Dictionary representation

        Returns:
            Intent instance

        Example:
            >>> data = {"primary_goal": "search", "task_type": "single_step"}
            >>> intent = Intent.from_dict(data)
            >>> assert intent.task_type == TaskType.SINGLE_STEP
        """
        intent = cls(primary_goal=data["primary_goal"])

        # Set optional fields
        intent.entities = data.get("entities", [])
        intent.confidence = data.get("confidence", 0.0)
        intent.extracted_parameters = data.get("extracted_parameters", {})
        intent.requires_tools = data.get("requires_tools", False)
        intent.urgency = data.get("urgency", "medium")
        intent.context_needed = data.get("context_needed", False)
        intent.ambiguity_score = data.get("ambiguity_score", 0.0)

        # Parse enums
        if "task_type" in data:
            try:
                intent.task_type = TaskType(data["task_type"])
            except ValueError:
                intent.task_type = TaskType.CONVERSATIONAL

        if "category" in data:
            try:
                intent.category = IntentCategory(data["category"])
            except ValueError:
                intent.category = IntentCategory.REQUEST

        return intent

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """
        Check if intent has high confidence.

        Args:
            threshold: Confidence threshold

        Returns:
            True if confidence is above threshold

        Example:
            >>> intent = Intent("test", confidence=0.9)
            >>> assert intent.is_high_confidence() == True
        """
        return self.confidence >= threshold

    def needs_clarification(self, ambiguity_threshold: float = 0.7) -> bool:
        """
        Check if intent needs clarification.

        Args:
            ambiguity_threshold: Ambiguity threshold

        Returns:
            True if clarification is needed

        Example:
            >>> intent = Intent("test", ambiguity_score=0.8)
            >>> assert intent.needs_clarification() == True
        """
        return self.ambiguity_score >= ambiguity_threshold or self.context_needed


class IntentParser:
    """
    LLM-based intent parser for extracting user goals and context.

    Uses a language model to analyze user messages and extract structured
    intent information including goals, entities, task types, and metadata.

    Attributes:
        model: Language model for intent analysis
        available_tools: List of available tool names for context
        custom_entities: Custom entity types to look for
        confidence_threshold: Minimum confidence for intent acceptance

    Example:
        >>> from agentframe import OpenAIModel, ModelConfig
        >>> config = ModelConfig(api_key="sk-...", model="gpt-4")
        >>> model = OpenAIModel(config)
        >>> parser = IntentParser(model)
        >>> intent = parser.parse("Calculate the area of a circle with radius 5")
        >>> print(f"Goal: {intent.primary_goal}")
        >>> print(f"Confidence: {intent.confidence}")
    """

    def __init__(
        self,
        model: BaseModel,
        available_tools: Optional[List[str]] = None,
        custom_entities: Optional[Set[str]] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the intent parser.

        Args:
            model: Language model for intent analysis
            available_tools: List of available tool names
            custom_entities: Custom entity types to extract
            confidence_threshold: Minimum confidence threshold
        """
        self.model = model
        self.available_tools = available_tools or []
        self.custom_entities = custom_entities or set()
        self.confidence_threshold = confidence_threshold

        logger.debug(f"Initialized intent parser with {len(self.available_tools)} tools")

    def parse(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Intent:
        """
        Parse intent from a user message.

        Args:
            message: User message to analyze
            context: Additional context information
            conversation_history: Recent conversation for context

        Returns:
            Parsed intent with extracted information

        Raises:
            ValueError: If intent parsing fails

        Example:
            >>> intent = parser.parse("Find me restaurants near downtown")
            >>> assert intent.primary_goal.lower().find("restaurant") != -1
            >>> assert "restaurants" in intent.entities
        """
        try:
            # Create intent parsing prompt
            parsing_prompt = self._create_parsing_prompt(message, context, conversation_history)

            # Generate intent analysis
            messages = [{"role": "user", "content": parsing_prompt}]
            response = self.model.generate(messages)

            # Parse intent from response
            intent = self._parse_intent_response(response.content, message)

            # Validate intent
            if intent.confidence < self.confidence_threshold:
                logger.warning(f"Intent confidence {intent.confidence} below threshold {self.confidence_threshold}")

            logger.debug(f"Parsed intent: {intent.primary_goal} (confidence: {intent.confidence})")
            return intent

        except Exception as e:
            logger.error(f"Failed to parse intent: {e}")
            # Return fallback intent
            return Intent(
                primary_goal=message,
                task_type=TaskType.CONVERSATIONAL,
                category=IntentCategory.REQUEST,
                confidence=0.0,
                context_needed=True
            )

    def parse_batch(self, messages: List[str]) -> List[Intent]:
        """
        Parse intents for multiple messages.

        Args:
            messages: List of messages to analyze

        Returns:
            List of parsed intents

        Example:
            >>> messages = ["Hello", "Calculate 2+2", "What's the weather?"]
            >>> intents = parser.parse_batch(messages)
            >>> assert len(intents) == 3
        """
        intents = []
        for message in messages:
            try:
                intent = self.parse(message)
                intents.append(intent)
            except Exception as e:
                logger.error(f"Failed to parse message '{message[:50]}...': {e}")
                # Add fallback intent
                intents.append(Intent(
                    primary_goal=message,
                    confidence=0.0
                ))

        return intents

    def _create_parsing_prompt(
        self,
        message: str,
        context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, Any]]]
    ) -> str:
        """Create the intent parsing prompt for the LLM."""
        tools_context = ""
        if self.available_tools:
            tools_context = f"\nAvailable tools: {', '.join(self.available_tools)}\n"

        entities_context = ""
        if self.custom_entities:
            entities_context = f"\nCustom entities to look for: {', '.join(self.custom_entities)}\n"

        history_context = ""
        if conversation_history:
            history_context = "\nRecent conversation:\n"
            for msg in conversation_history[-3:]:  # Last 3 messages for context
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:100]  # Truncate for brevity
                history_context += f"{role}: {content}\n"

        additional_context = ""
        if context:
            additional_context = f"\nAdditional context: {json.dumps(context, indent=2)}\n"

        return f"""
You are an expert intent parser. Analyze the user message and extract structured intent information.

User Message: "{message}"
{tools_context}{entities_context}{history_context}{additional_context}

Extract the following information and return as JSON:

{{
    "primary_goal": "main objective the user wants to achieve",
    "entities": ["list", "of", "important", "entities", "and", "terms"],
    "task_type": "single_step|multi_step|conversational|informational|creative|analytical",
    "category": "question|request|command|clarification|greeting|goodbye|complaint|compliment",
    "confidence": 0.0-1.0,
    "extracted_parameters": {{"key": "value pairs from the message"}},
    "requires_tools": true/false,
    "urgency": "low|medium|high",
    "context_needed": true/false,
    "ambiguity_score": 0.0-1.0
}}

Guidelines:
1. primary_goal: Clear, actionable description of what the user wants
2. entities: Extract nouns, proper nouns, numbers, and key terms
3. task_type:
   - single_step: Can be completed in one action
   - multi_step: Requires multiple coordinated actions
   - conversational: Casual conversation or clarification
   - informational: Seeking information or explanation
   - creative: Creative writing, brainstorming, etc.
   - analytical: Analysis, comparison, calculation
4. category: Primary intent category
5. confidence: How confident you are in the analysis (0.0-1.0)
6. extracted_parameters: Key-value pairs that could be tool parameters
7. requires_tools: Whether tools/external actions are needed
8. urgency: Time sensitivity of the request
9. context_needed: Whether more information is needed to proceed
10. ambiguity_score: How ambiguous or unclear the message is

Return only the JSON, no additional text.
"""

    def _parse_intent_response(self, response: str, original_message: str) -> Intent:
        """Parse the LLM response into an Intent object."""
        try:
            # Clean up response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            # Parse JSON
            intent_data = json.loads(response)

            # Create intent from parsed data
            return Intent.from_dict(intent_data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse intent JSON: {e}")
            logger.debug(f"Raw response: {response}")
            raise ValueError(f"Invalid intent format: {response}")
        except Exception as e:
            logger.error(f"Error parsing intent: {e}")
            raise ValueError(f"Intent parsing failed: {e}")

    def analyze_conversation_flow(
        self,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze the flow and patterns in a conversation.

        Args:
            conversation_history: List of messages in the conversation

        Returns:
            Dictionary with conversation flow analysis

        Example:
            >>> analysis = parser.analyze_conversation_flow(history)
            >>> print(f"Conversation theme: {analysis['theme']}")
            >>> print(f"User satisfaction: {analysis['satisfaction']}")
        """
        if not conversation_history:
            return {"error": "No conversation history provided"}

        try:
            # Create conversation analysis prompt
            history_text = "\n".join([
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in conversation_history
            ])

            analysis_prompt = f"""
Analyze this conversation and provide insights:

{history_text}

Provide analysis as JSON:
{{
    "theme": "main topic or theme of conversation",
    "user_satisfaction": "low|medium|high",
    "conversation_stage": "opening|information_gathering|problem_solving|closing",
    "unresolved_issues": ["list of unresolved topics"],
    "next_likely_intent": "what the user might ask next",
    "complexity_trend": "increasing|decreasing|stable",
    "tool_usage_pattern": "description of how tools were used"
}}

Return only the JSON.
"""

            messages = [{"role": "user", "content": analysis_prompt}]
            response = self.model.generate(messages)

            # Parse response
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            return json.loads(response_text)

        except Exception as e:
            logger.error(f"Failed to analyze conversation flow: {e}")
            return {"error": str(e)}

    def suggest_follow_up_questions(
        self,
        intent: Intent,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Suggest follow-up questions based on parsed intent.

        Args:
            intent: Parsed intent
            context: Additional context

        Returns:
            List of suggested follow-up questions

        Example:
            >>> intent = parser.parse("I want to book a flight")
            >>> questions = parser.suggest_follow_up_questions(intent)
            >>> # Might return: ["Where would you like to fly to?", "When do you want to travel?"]
        """
        try:
            context_str = ""
            if context:
                context_str = f"\nContext: {json.dumps(context)}"

            prompt = f"""
Based on this user intent, suggest 3-5 helpful follow-up questions to gather more information:

Intent: {intent.primary_goal}
Task Type: {intent.task_type.value}
Entities: {intent.entities}
Requires Tools: {intent.requires_tools}
Context Needed: {intent.context_needed}
{context_str}

Return a JSON list of follow-up questions:
["question 1", "question 2", "question 3"]

Make questions specific, helpful, and relevant to achieving the user's goal.
"""

            messages = [{"role": "user", "content": prompt}]
            response = self.model.generate(messages)

            # Parse response
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            return json.loads(response_text)

        except Exception as e:
            logger.error(f"Failed to suggest follow-up questions: {e}")
            return []

    def update_available_tools(self, tools: List[str]) -> None:
        """
        Update the list of available tools.

        Args:
            tools: List of tool names

        Example:
            >>> parser.update_available_tools(["search", "calculator", "weather"])
        """
        self.available_tools = tools
        logger.debug(f"Updated available tools: {len(tools)} tools")

    def add_custom_entities(self, entities: Set[str]) -> None:
        """
        Add custom entity types to look for.

        Args:
            entities: Set of entity types

        Example:
            >>> parser.add_custom_entities({"product_name", "customer_id", "order_number"})
        """
        self.custom_entities.update(entities)
        logger.debug(f"Added custom entities: {entities}")

    def get_intent_statistics(self, intents: List[Intent]) -> Dict[str, Any]:
        """
        Get statistics about a collection of intents.

        Args:
            intents: List of parsed intents

        Returns:
            Dictionary with statistics

        Example:
            >>> stats = parser.get_intent_statistics(parsed_intents)
            >>> print(f"Average confidence: {stats['avg_confidence']}")
        """
        if not intents:
            return {"error": "No intents provided"}

        total_confidence = sum(intent.confidence for intent in intents)
        avg_confidence = total_confidence / len(intents)

        task_type_counts = {}
        category_counts = {}
        tool_requiring_count = 0

        for intent in intents:
            task_type = intent.task_type.value
            category = intent.category.value

            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1

            if intent.requires_tools:
                tool_requiring_count += 1

        return {
            "total_intents": len(intents),
            "avg_confidence": avg_confidence,
            "high_confidence_count": len([i for i in intents if i.is_high_confidence()]),
            "tool_requiring_count": tool_requiring_count,
            "task_type_distribution": task_type_counts,
            "category_distribution": category_counts,
            "needs_clarification_count": len([i for i in intents if i.needs_clarification()])
        }