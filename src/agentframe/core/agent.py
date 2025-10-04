"""
Main Agent class for AgentFrame.

This module provides the main Agent orchestrator that coordinates all components
including planning, execution, tool usage, and conversation management using
LangGraph for workflow orchestration.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Iterator
from dataclasses import dataclass, field

from langgraph.graph import Graph, END
from langchain_core.messages import BaseMessage

from ..models.base import BaseModel
from ..tools.base import ToolRegistry, BaseTool
from ..memory.chat_history import ChatHistory, MessageType
from ..utils.intent_parser import IntentParser, Intent
from .state import AgentState, StateManager
from .planner import Planner, ReplanningStrategy, ExecutionPlan
from .executor import ExecutionEngine, ExecutionContext, ExecutionResult
from .prompts import AgentPrompts, DefaultPrompts, PromptType

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """
    Configuration for Agent behavior.

    Controls various aspects of agent operation including planning,
    execution, conversation management, and prompt customization.

    Attributes:
        max_replanning_iterations: Maximum number of replanning attempts
        enable_streaming: Whether to support streaming responses
        verbose_logging: Enable detailed logging
        timeout: Overall timeout for agent operations (seconds)
        auto_save_history: Whether to automatically save conversation history
        require_confirmation: Whether to require user confirmation for actions
        max_plan_steps: Maximum steps allowed in a plan
        confidence_threshold: Minimum confidence for plan acceptance
        agent_name: Name of the agent (used in prompts)
        agent_description: Description of the agent's role
        personality_traits: List of personality characteristics
        response_style: Preferred communication style
        custom_guidelines: Additional behavior guidelines

    Example:
        >>> config = AgentConfig(
        ...     max_replanning_iterations=3,
        ...     enable_streaming=True,
        ...     verbose_logging=True,
        ...     agent_name="DataBot",
        ...     agent_description="a data analysis specialist",
        ...     personality_traits=["analytical", "precise"],
        ...     response_style="technical but accessible"
        ... )
    """
    max_replanning_iterations: int = 3
    enable_streaming: bool = False
    verbose_logging: bool = False
    timeout: float = 300.0
    auto_save_history: bool = False
    require_confirmation: bool = False
    max_plan_steps: int = 10
    confidence_threshold: float = 0.7
    agent_name: str = "Assistant"
    agent_description: str = "a helpful AI assistant"
    personality_traits: List[str] = field(default_factory=lambda: ["helpful", "accurate", "friendly"])
    response_style: str = "professional and friendly"
    custom_guidelines: List[str] = field(default_factory=list)


class Agent:
    """
    Main Agent orchestrator for AgentFrame.

    The Agent class coordinates all components of the framework including
    intent parsing, planning, execution, and conversation management using
    a LangGraph workflow for robust orchestration.

    Attributes:
        model: Language model for text generation
        tools: List of available tools
        config: Agent configuration
        chat_history: Conversation history manager
        state_manager: Agent state manager
        intent_parser: Intent parsing component
        planner: Plan generation component
        executor: Plan execution component
        workflow: LangGraph workflow

    Example:
        >>> from agentframe import Agent, OpenAIModel, ModelConfig, tool
        >>>
        >>> @tool
        >>> def calculator(expression: str) -> float:
        ...     return eval(expression)
        >>>
        >>> config = ModelConfig(api_key="sk-...", model="gpt-4")
        >>> model = OpenAIModel(config)
        >>> agent = Agent(model=model, tools=[calculator])
        >>>
        >>> response = agent.run("What's 15% of 200?")
        >>> print(response)
    """

    def __init__(
        self,
        model: BaseModel,
        tools: Optional[List[BaseTool]] = None,
        config: Optional[AgentConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        prompts: Optional[AgentPrompts] = None
    ):
        """
        Initialize the Agent.

        Args:
            model: Language model for text generation
            tools: List of tools available to the agent
            config: Agent configuration
            tool_registry: Custom tool registry (creates new if None)
            prompts: Custom prompts for agent behavior (creates default if None)

        Example:
            >>> agent = Agent(
            ...     model=my_model,
            ...     tools=[search_tool, calculator_tool],
            ...     config=AgentConfig(verbose_logging=True),
            ...     prompts=custom_prompts
            ... )
        """
        self.model = model
        self.config = config or AgentConfig()

        # Initialize prompts
        if prompts is None:
            # Create prompts based on config
            self.prompts = DefaultPrompts.create_custom_agent_prompts(
                agent_name=self.config.agent_name,
                agent_description=self.config.agent_description,
                personality_traits=self.config.personality_traits,
                specific_guidelines=self.config.custom_guidelines,
                response_style=self.config.response_style
            )
        else:
            self.prompts = prompts

        # Initialize tool registry
        self.tool_registry = tool_registry or ToolRegistry()
        if tools:
            for tool in tools:
                try:
                    self.tool_registry.register(tool)
                except ValueError as e:
                    logger.warning(f"Failed to register tool {tool.name}: {e}")

        # Initialize components
        self.chat_history = ChatHistory()
        self.state_manager = StateManager()
        self.intent_parser = IntentParser(
            model=model,
            available_tools=self.tool_registry.list_all()
        )
        self.planner = Planner(
            model=model,
            tool_registry=self.tool_registry,
            max_steps=self.config.max_plan_steps,
            min_confidence=self.config.confidence_threshold
        )
        self.replanning_strategy = ReplanningStrategy(self.planner)
        self.executor = ExecutionEngine(
            tool_registry=self.tool_registry,
            context=ExecutionContext(
                total_timeout=self.config.timeout,
                fail_fast=not self.config.require_confirmation
            )
        )

        # Initialize workflow
        self.workflow = self._create_workflow()

        # Session management
        self.session_id = str(uuid.uuid4())
        self.state_manager.initialize_state(
            session_id=self.session_id,
            initial_context={
                "agent_config": self.config.__dict__,
                "available_tools": self.tool_registry.list_all()
            }
        )

        logger.info(f"Initialized Agent with {len(self.tool_registry.list_all())} tools")

    def run(self, user_input: str, stream: bool = False) -> str:
        """
        Process user input and return response.

        Args:
            user_input: User message to process
            stream: Whether to stream the response

        Returns:
            Agent response as string

        Example:
            >>> response = agent.run("Calculate the area of a circle with radius 5")
            >>> print(response)
            "The area of a circle with radius 5 is approximately 78.54 square units."
        """
        try:
            # Add user message to history
            self.chat_history.add_message(MessageType.HUMAN, user_input)
            self.state_manager.add_message({
                "role": "user",
                "content": user_input
            })

            # Process through workflow
            if stream and self.config.enable_streaming:
                return self._run_streaming(user_input)
            else:
                return self._run_synchronous(user_input)

        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            error_response = f"I apologize, but I encountered an error: {str(e)}"
            self.chat_history.add_message(MessageType.AI, error_response)
            return error_response

    def _run_synchronous(self, user_input: str) -> str:
        """Run the agent workflow synchronously."""
        # Execute workflow
        workflow_input = {
            "user_input": user_input,
            "session_id": self.session_id
        }

        result = self.workflow.invoke(workflow_input)
        response = result.get("response", "I'm not sure how to help with that.")

        # Add response to history
        self.chat_history.add_message(MessageType.AI, response)
        self.state_manager.add_message({
            "role": "assistant",
            "content": response
        })

        return response

    def _run_streaming(self, user_input: str) -> Iterator[str]:
        """Run the agent workflow with streaming."""
        # Note: This is a simplified streaming implementation
        # In a full implementation, you'd use LangGraph's streaming capabilities
        response = self._run_synchronous(user_input)

        # Simulate streaming by yielding chunks
        words = response.split()
        for i, word in enumerate(words):
            if i == len(words) - 1:
                yield word
            else:
                yield word + " "

    def _create_workflow(self) -> Graph:
        """Create the LangGraph workflow."""
        # Define workflow nodes
        def intent_parsing_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Parse user intent from input."""
            user_input = state["user_input"]

            try:
                # Get conversation context
                context = {
                    "session_id": state.get("session_id"),
                    "available_tools": self.tool_registry.list_all()
                }

                # Use custom intent parsing prompt if available
                custom_intent_prompt = self.prompts.get_prompt(PromptType.INTENT_PARSING)
                if custom_intent_prompt:
                    # Create custom intent parser with custom prompt
                    custom_parser = IntentParser(
                        model=self.model,
                        available_tools=self.tool_registry.list_all()
                    )
                    # Override the default prompt (this would need IntentParser modification)
                    intent = custom_parser.parse(
                        user_input,
                        context=context,
                        conversation_history=self.chat_history.to_langchain_messages()[-5:]
                    )
                else:
                    # Use default intent parser
                    intent = self.intent_parser.parse(
                        user_input,
                        context=context,
                        conversation_history=self.chat_history.to_langchain_messages()[-5:]
                    )

                # Update state
                self.state_manager.set_user_intent(intent.to_dict())

                state["intent"] = intent
                state["needs_planning"] = intent.requires_tools or intent.task_type.value in ["multi_step", "analytical"]

                logger.debug(f"Parsed intent: {intent.primary_goal} (confidence: {intent.confidence})")

            except Exception as e:
                logger.error(f"Intent parsing failed: {e}")
                # Fallback intent
                state["intent"] = Intent(
                    primary_goal=user_input,
                    confidence=0.0,
                    context_needed=True
                )
                state["needs_planning"] = False

            return state

        def planning_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Generate execution plan."""
            intent = state["intent"]

            try:
                # Create plan
                plan = self.planner.create_plan(
                    goal=intent.primary_goal,
                    context={
                        "user_intent": intent.to_dict(),
                        "session_context": self.state_manager.state.get("context", {})
                    }
                )

                self.state_manager.update_plan([step.to_dict() for step in plan.steps])
                state["plan"] = plan
                state["plan_created"] = True

                logger.info(f"Created plan with {len(plan.steps)} steps")

            except Exception as e:
                logger.error(f"Planning failed: {e}")
                state["plan"] = None
                state["plan_created"] = False
                state["error"] = str(e)

            return state

        def execution_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Execute the plan."""
            plan = state.get("plan")

            if not plan:
                state["execution_result"] = None
                state["needs_replanning"] = False
                return state

            try:
                # Execute plan
                execution_result = self.executor.execute_plan(plan)

                # Update state with results
                for step_id, result in execution_result.results.items():
                    self.state_manager.add_tool_result(step_id, result)

                state["execution_result"] = execution_result
                state["needs_replanning"] = execution_result.needs_replanning

                logger.info(f"Plan execution completed: {execution_result.status.value}")

            except Exception as e:
                logger.error(f"Execution failed: {e}")
                state["execution_result"] = None
                state["needs_replanning"] = True
                state["error"] = str(e)

            return state

        def replanning_decision_node(state: Dict[str, Any]) -> str:
            """Decide whether replanning is needed."""
            execution_result = state.get("execution_result")
            current_iteration = state.get("replanning_iteration", 0)

            # Check if replanning is needed and we haven't exceeded max iterations
            if (state.get("needs_replanning", False) and
                current_iteration < self.config.max_replanning_iterations):

                state["replanning_iteration"] = current_iteration + 1
                logger.info(f"Replanning iteration {state['replanning_iteration']}")
                return "replan"
            else:
                return "respond"

        def replanning_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Generate a new plan based on execution results."""
            original_plan = state.get("plan")
            execution_result = state.get("execution_result")

            if not original_plan:
                return state

            try:
                # Determine failed step
                failed_step = None
                if execution_result and execution_result.errors:
                    failed_step_id = list(execution_result.errors.keys())[0]
                    failed_step = next(
                        (step for step in original_plan.steps if step.step_id == failed_step_id),
                        None
                    )

                # Create replan context
                context = {
                    "original_goal": original_plan.goal,
                    "execution_results": execution_result.results if execution_result else {},
                    "errors": execution_result.errors if execution_result else {}
                }

                # Generate new plan
                new_plan = self.replanning_strategy.replan(
                    original_plan=original_plan,
                    failed_step=failed_step,
                    context=context,
                    completed_work=execution_result.results if execution_result else {}
                )

                self.state_manager.update_plan([step.to_dict() for step in new_plan.steps])
                self.state_manager.increment_iteration()

                state["plan"] = new_plan
                logger.info(f"Generated replan with {len(new_plan.steps)} steps")

            except Exception as e:
                logger.error(f"Replanning failed: {e}")
                state["needs_replanning"] = False
                state["error"] = str(e)

            return state

        def response_generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Generate final response to user."""
            intent = state["intent"]
            execution_result = state.get("execution_result")
            error = state.get("error")

            try:
                # Build response based on execution results
                if error:
                    response = self._handle_error_response(intent, error)
                elif execution_result and execution_result.success:
                    # Successful execution - format results using custom prompt
                    response = self._format_successful_response(intent, execution_result)
                elif execution_result and execution_result.needs_replanning:
                    response = "I need to reconsider my approach to help you with this request."
                elif not state.get("needs_planning", False):
                    # Simple conversational response
                    response = self._generate_conversational_response(intent)
                else:
                    response = "I'm not sure how to help with that request."

                state["response"] = response

            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                state["response"] = "I apologize, but I'm having trouble generating a response."

            return state

        # Create the workflow graph
        workflow = Graph()

        # Add nodes
        workflow.add_node("intent_parsing", intent_parsing_node)
        workflow.add_node("planning", planning_node)
        workflow.add_node("execution", execution_node)
        workflow.add_node("replanning_decision", replanning_decision_node)
        workflow.add_node("replanning", replanning_node)
        workflow.add_node("response_generation", response_generation_node)

        # Add edges
        workflow.add_edge("intent_parsing", "planning")

        # Conditional edge from planning
        workflow.add_conditional_edges(
            "planning",
            lambda state: "execution" if state.get("plan_created", False) else "response_generation"
        )

        workflow.add_edge("execution", "replanning_decision")

        # Conditional edges from replanning decision
        workflow.add_conditional_edges(
            "replanning_decision",
            replanning_decision_node,
            {
                "replan": "replanning",
                "respond": "response_generation"
            }
        )

        workflow.add_edge("replanning", "execution")
        workflow.add_edge("response_generation", END)

        # Set entry point
        workflow.set_entry_point("intent_parsing")

        return workflow.compile()

    def _format_successful_response(self, intent: Intent, execution_result: ExecutionResult) -> str:
        """Format a response when execution was successful using custom prompts."""
        if not execution_result.results:
            return "I completed the task successfully."

        try:
            # Use custom response generation prompt if available
            response_template = self.prompts.get_prompt(PromptType.RESPONSE_GENERATION)

            if response_template:
                # Format results for the prompt
                results_summary = []
                for step_id, result in execution_result.results.items():
                    results_summary.append(f"{step_id}: {result}")

                # Format the custom prompt
                prompt_text = response_template.format(
                    user_request=intent.primary_goal,
                    execution_results="\n".join(results_summary),
                    agent_role=f"{self.config.agent_name} ({self.config.agent_description})"
                )
            else:
                # Fallback to default prompt
                results_summary = []
                for step_id, result in execution_result.results.items():
                    results_summary.append(f"{step_id}: {result}")

                prompt_text = f"""
                The user asked: "{intent.primary_goal}"

                I executed the following steps successfully:
                {chr(10).join(results_summary)}

                Please provide a natural, helpful response to the user based on these results.
                Be concise and focus on answering their original question.
                """

            messages = [{"role": "user", "content": prompt_text}]
            response = self.model.generate(messages)
            return response.content

        except Exception as e:
            logger.error(f"Failed to generate response using prompts: {e}")
            # Fallback to simple response
            if len(execution_result.results) == 1:
                result = list(execution_result.results.values())[0]
                return f"Here's the result: {result}"
            else:
                return f"I completed {len(execution_result.results)} steps successfully."

    def _generate_conversational_response(self, intent: Intent) -> str:
        """Generate a conversational response for non-task intents using custom prompts."""
        try:
            # Use system instruction prompt if available
            system_template = self.prompts.get_prompt(PromptType.SYSTEM_INSTRUCTION)

            # Get recent conversation context
            recent_messages = self.chat_history.get_recent(5)
            context_messages = []

            # Add system instruction if available
            if system_template:
                try:
                    system_prompt = system_template.format(
                        agent_name=self.config.agent_name,
                        agent_description=self.config.agent_description
                    )
                    context_messages.append({
                        "role": "system",
                        "content": system_prompt
                    })
                except Exception as e:
                    logger.warning(f"Failed to format system prompt: {e}")

            # Add conversation history
            context_messages.extend([msg.to_dict() for msg in recent_messages])

            # Add the current intent
            context_messages.append({
                "role": "user",
                "content": intent.primary_goal
            })

            # Generate response
            response = self.model.generate(context_messages)
            return response.content

        except Exception as e:
            logger.error(f"Failed to generate conversational response: {e}")
            return "I understand. How can I help you further?"

    def _handle_error_response(self, intent: Intent, error: str) -> str:
        """Generate an error response using custom error handling prompts."""
        try:
            # Use custom error handling prompt if available
            error_template = self.prompts.get_prompt(PromptType.ERROR_HANDLING)

            if error_template:
                # Format the error prompt
                prompt_text = error_template.format(
                    original_request=intent.primary_goal,
                    error_details=error,
                    agent_role=f"{self.config.agent_name} ({self.config.agent_description})"
                )

                messages = [{"role": "user", "content": prompt_text}]
                response = self.model.generate(messages)
                return response.content
            else:
                # Fallback error response
                return f"I encountered an issue while processing your request: {error}. Let me try a different approach."

        except Exception as e:
            logger.error(f"Failed to generate error response: {e}")
            return f"I'm sorry, but I encountered an error: {error}"

    def add_tool(self, tool: BaseTool) -> None:
        """
        Add a tool to the agent's toolkit.

        Args:
            tool: Tool to add

        Example:
            >>> @tool
            >>> def new_tool(param: str) -> str:
            ...     return f"Processed: {param}"
            >>>
            >>> agent.add_tool(new_tool._agentframe_tool)
        """
        try:
            self.tool_registry.register(tool)
            # Update intent parser with new tool list
            self.intent_parser.update_available_tools(self.tool_registry.list_all())
            logger.info(f"Added tool: {tool.name}")
        except ValueError as e:
            logger.error(f"Failed to add tool {tool.name}: {e}")
            raise

    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the agent's toolkit.

        Args:
            tool_name: Name of tool to remove

        Returns:
            True if tool was removed

        Example:
            >>> success = agent.remove_tool("old_tool")
            >>> if success:
            ...     print("Tool removed successfully")
        """
        success = self.tool_registry.unregister(tool_name)
        if success:
            # Update intent parser
            self.intent_parser.update_available_tools(self.tool_registry.list_all())
            logger.info(f"Removed tool: {tool_name}")
        return success

    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current conversation.

        Returns:
            Dictionary with conversation statistics and summary

        Example:
            >>> summary = agent.get_conversation_summary()
            >>> print(f"Messages: {summary['total_messages']}")
            >>> print(f"Tools used: {summary['tools_used']}")
        """
        chat_summary = self.chat_history.get_conversation_summary()
        state_summary = self.state_manager.get_summary()

        return {
            **chat_summary,
            **state_summary,
            "available_tools": len(self.tool_registry.list_all()),
            "session_id": self.session_id
        }

    def reset_conversation(self) -> None:
        """
        Reset the conversation history and state.

        Example:
            >>> agent.reset_conversation()
            >>> # Fresh start with clean history
        """
        self.chat_history.clear()
        self.session_id = str(uuid.uuid4())
        self.state_manager.initialize_state(
            session_id=self.session_id,
            initial_context={
                "agent_config": self.config.__dict__,
                "available_tools": self.tool_registry.list_all()
            }
        )
        logger.info("Reset conversation and state")

    def save_conversation(self, filepath: str) -> None:
        """
        Save conversation history to file.

        Args:
            filepath: Path to save the conversation

        Example:
            >>> agent.save_conversation("my_conversation.json")
        """
        self.chat_history.save_to_file(filepath)

    def load_conversation(self, filepath: str) -> None:
        """
        Load conversation history from file.

        Args:
            filepath: Path to load the conversation from

        Example:
            >>> agent.load_conversation("my_conversation.json")
        """
        self.chat_history.load_from_file(filepath)

    def update_prompts(self, prompts: AgentPrompts) -> None:
        """
        Update the agent's prompt templates.

        Args:
            prompts: New AgentPrompts object

        Example:
            >>> new_prompts = DefaultPrompts.create_custom_agent_prompts(
            ...     agent_name="SpecialistBot",
            ...     agent_description="an expert in data analysis"
            ... )
            >>> agent.update_prompts(new_prompts)
        """
        self.prompts = prompts
        logger.info("Updated agent prompts")

    def set_prompt(self, prompt_type: Union[PromptType, str], template: 'PromptTemplate') -> None:
        """
        Set a specific prompt template.

        Args:
            prompt_type: Type of prompt to set
            template: PromptTemplate to use

        Example:
            >>> from agentframe.core.prompts import PromptTemplate, PromptType
            >>> custom_template = PromptTemplate(
            ...     name="custom_system",
            ...     template="You are {agent_name}, specialized in {domain}.",
            ...     required_variables=["agent_name", "domain"]
            ... )
            >>> agent.set_prompt(PromptType.SYSTEM_INSTRUCTION, custom_template)
        """
        self.prompts.set_prompt(prompt_type, template)
        logger.info(f"Updated prompt: {prompt_type}")

    def get_prompt(self, prompt_type: Union[PromptType, str]) -> Optional['PromptTemplate']:
        """
        Get a specific prompt template.

        Args:
            prompt_type: Type of prompt to retrieve

        Returns:
            PromptTemplate if found, None otherwise

        Example:
            >>> system_prompt = agent.get_prompt(PromptType.SYSTEM_INSTRUCTION)
            >>> if system_prompt:
            ...     print(system_prompt.template)
        """
        return self.prompts.get_prompt(prompt_type)

    def validate_prompts(self) -> Dict[str, bool]:
        """
        Validate all prompt templates.

        Returns:
            Dictionary mapping prompt names to validation results

        Example:
            >>> results = agent.validate_prompts()
            >>> invalid_prompts = [name for name, valid in results.items() if not valid]
            >>> if invalid_prompts:
            ...     print(f"Invalid prompts: {invalid_prompts}")
        """
        return self.prompts.validate_all()

    def __repr__(self) -> str:
        """String representation of the agent."""
        return (f"Agent(name={self.config.agent_name}, "
                f"model={self.model.__class__.__name__}, "
                f"tools={len(self.tool_registry.list_all())}, "
                f"session={self.session_id[:8]}...)")
