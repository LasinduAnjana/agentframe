"""
Planning and replanning system for AgentFrame.

This module provides intelligent planning capabilities that analyze user goals,
available tools, and execution context to generate step-by-step execution plans.
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from ..models.base import BaseModel, ModelResponse
from ..tools.base import BaseTool, ToolRegistry

logger = logging.getLogger(__name__)


class PlanStepStatus(Enum):
    """Status of a plan step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """
    Individual step in an execution plan.

    Represents a single action that needs to be taken as part of achieving
    the user's goal, including tool selection and arguments.

    Attributes:
        step_id: Unique identifier for this step
        tool_name: Name of the tool to execute
        arguments: Arguments to pass to the tool
        expected_output: Description of expected output
        dependencies: List of step IDs this step depends on
        status: Current status of the step
        description: Human-readable description of the step
        reasoning: Explanation of why this step is needed
        retry_count: Number of times this step has been retried
        result: Actual result from execution (if completed)

    Example:
        >>> step = PlanStep(
        ...     step_id="step_1",
        ...     tool_name="search_web",
        ...     arguments={"query": "Python tutorial"},
        ...     expected_output="List of Python tutorial links",
        ...     description="Search for Python tutorials online"
        ... )
    """
    step_id: str
    tool_name: str
    arguments: Dict[str, Any]
    expected_output: str
    dependencies: List[str] = field(default_factory=list)
    status: PlanStepStatus = PlanStepStatus.PENDING
    description: str = ""
    reasoning: str = ""
    retry_count: int = 0
    result: Optional[Any] = None

    def can_execute(self, completed_steps: Set[str]) -> bool:
        """
        Check if this step can be executed based on dependencies.

        Args:
            completed_steps: Set of completed step IDs

        Returns:
            True if all dependencies are satisfied

        Example:
            >>> step = PlanStep("step_2", "calculator", {}, "", dependencies=["step_1"])
            >>> assert step.can_execute({"step_1"}) == True
            >>> assert step.can_execute(set()) == False
        """
        return all(dep in completed_steps for dep in self.dependencies)

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary format."""
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "expected_output": self.expected_output,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "description": self.description,
            "reasoning": self.reasoning,
            "retry_count": self.retry_count,
            "result": self.result
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanStep':
        """Create step from dictionary format."""
        step = cls(
            step_id=data["step_id"],
            tool_name=data["tool_name"],
            arguments=data["arguments"],
            expected_output=data["expected_output"],
            dependencies=data.get("dependencies", []),
            description=data.get("description", ""),
            reasoning=data.get("reasoning", ""),
            retry_count=data.get("retry_count", 0),
            result=data.get("result")
        )
        step.status = PlanStepStatus(data.get("status", "pending"))
        return step


@dataclass
class ExecutionPlan:
    """
    Complete execution plan for achieving a user goal.

    Contains a sequence of steps that should be executed to accomplish
    the user's objective, along with metadata about the plan.

    Attributes:
        plan_id: Unique identifier for this plan
        goal: Description of the goal this plan achieves
        steps: List of plan steps in execution order
        estimated_duration: Estimated time to complete (in seconds)
        confidence: Confidence score (0.0 to 1.0) in plan success
        alternative_approaches: List of alternative plan descriptions
        metadata: Additional metadata about plan creation

    Example:
        >>> plan = ExecutionPlan(
        ...     plan_id="plan_001",
        ...     goal="Calculate compound interest",
        ...     steps=[step1, step2, step3],
        ...     confidence=0.9
        ... )
    """
    plan_id: str
    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    estimated_duration: float = 0.0
    confidence: float = 0.0
    alternative_approaches: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_executable_steps(self, completed_steps: Set[str]) -> List[PlanStep]:
        """
        Get steps that can be executed now.

        Args:
            completed_steps: Set of completed step IDs

        Returns:
            List of steps ready for execution

        Example:
            >>> plan = ExecutionPlan("test", "test goal", steps=[step1, step2])
            >>> executable = plan.get_executable_steps(set())
            >>> # Returns steps with no dependencies
        """
        return [
            step for step in self.steps
            if step.status == PlanStepStatus.PENDING and step.can_execute(completed_steps)
        ]

    def get_next_step(self, completed_steps: Set[str]) -> Optional[PlanStep]:
        """
        Get the next step to execute.

        Args:
            completed_steps: Set of completed step IDs

        Returns:
            Next step to execute or None if no steps available

        Example:
            >>> next_step = plan.get_next_step({"step_1"})
            >>> if next_step:
            ...     print(f"Execute: {next_step.tool_name}")
        """
        executable = self.get_executable_steps(completed_steps)
        return executable[0] if executable else None

    def is_complete(self) -> bool:
        """
        Check if all steps in the plan are completed.

        Returns:
            True if plan is fully executed

        Example:
            >>> if plan.is_complete():
            ...     print("Plan execution finished!")
        """
        return all(step.status == PlanStepStatus.COMPLETED for step in self.steps)

    def has_failed_steps(self) -> bool:
        """
        Check if any steps have failed.

        Returns:
            True if any steps have failed

        Example:
            >>> if plan.has_failed_steps():
            ...     print("Plan needs replanning")
        """
        return any(step.status == PlanStepStatus.FAILED for step in self.steps)

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary format."""
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [step.to_dict() for step in self.steps],
            "estimated_duration": self.estimated_duration,
            "confidence": self.confidence,
            "alternative_approaches": self.alternative_approaches,
            "metadata": self.metadata
        }


class Planner:
    """
    Intelligent planner that generates execution plans.

    The planner analyzes user goals, available tools, and context to create
    detailed step-by-step execution plans that achieve the desired outcome.

    Attributes:
        model: Language model for plan generation
        tool_registry: Registry of available tools
        max_steps: Maximum number of steps per plan
        min_confidence: Minimum confidence threshold for plans

    Example:
        >>> from agentframe import OpenAIModel, ModelConfig
        >>> config = ModelConfig(api_key="sk-...", model="gpt-4")
        >>> model = OpenAIModel(config)
        >>> planner = Planner(model, tool_registry)
        >>> plan = planner.create_plan("Calculate the area of a circle with radius 5")
    """

    def __init__(
        self,
        model: BaseModel,
        tool_registry: ToolRegistry,
        max_steps: int = 10,
        min_confidence: float = 0.7
    ):
        """
        Initialize the planner.

        Args:
            model: Language model for plan generation
            tool_registry: Registry of available tools
            max_steps: Maximum steps allowed in a plan
            min_confidence: Minimum confidence for plan acceptance
        """
        self.model = model
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.min_confidence = min_confidence

        logger.debug(f"Initialized planner with {tool_registry.get_tool_count()} tools")

    def create_plan(
        self,
        goal: str,
        context: Optional[Dict[str, Any]] = None,
        available_tools: Optional[List[str]] = None
    ) -> ExecutionPlan:
        """
        Create an execution plan for achieving a goal.

        Args:
            goal: Description of what needs to be accomplished
            context: Additional context information
            available_tools: Specific tools to use (None for all available)

        Returns:
            Generated execution plan

        Raises:
            ValueError: If no viable plan can be generated

        Example:
            >>> plan = planner.create_plan(
            ...     goal="Find the current weather in Tokyo",
            ...     context={"user_location": "Japan"}
            ... )
            >>> print(f"Plan has {len(plan.steps)} steps")
        """
        try:
            # Get available tools
            if available_tools is None:
                available_tools = self.tool_registry.list_all()

            tools_info = self._get_tools_description(available_tools)

            # Create planning prompt
            planning_prompt = self._create_planning_prompt(goal, tools_info, context)

            # Generate plan using LLM
            messages = [{"role": "user", "content": planning_prompt}]
            response = self.model.generate(messages)

            # Parse plan from response
            plan = self._parse_plan_response(response.content, goal)

            # Validate plan
            if not self._validate_plan(plan):
                raise ValueError("Generated plan failed validation")

            logger.info(f"Created plan '{plan.plan_id}' with {len(plan.steps)} steps")
            return plan

        except Exception as e:
            logger.error(f"Failed to create plan: {e}")
            raise ValueError(f"Plan generation failed: {e}")

    def _get_tools_description(self, tool_names: List[str]) -> str:
        """Get formatted description of available tools."""
        descriptions = []

        for name in tool_names:
            tool = self.tool_registry.get(name)
            if tool:
                schema = tool.get_schema()
                descriptions.append(f"""
Tool: {schema['name']}
Description: {schema['description']}
Parameters: {json.dumps(schema['parameters'], indent=2)}
""")

        return "\n".join(descriptions)

    def _create_planning_prompt(
        self,
        goal: str,
        tools_info: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Create the planning prompt for the LLM."""
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, indent=2)}\n"

        return f"""
You are an intelligent task planner. Given a goal and available tools, create a detailed execution plan.

Goal: {goal}
{context_str}
Available Tools:
{tools_info}

Create a JSON plan with the following structure:
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
            "dependencies": ["step_id1", "step_id2"],
            "description": "human readable description",
            "reasoning": "why this step is needed"
        }}
    ],
    "alternative_approaches": ["other ways to achieve the goal"]
}}

Guidelines:
1. Break down complex goals into simple, atomic steps
2. Each step should use exactly one tool
3. Ensure dependencies are properly ordered
4. Be specific with tool arguments
5. Keep steps focused and achievable
6. Aim for high confidence plans
7. Maximum {self.max_steps} steps

Return only the JSON, no additional text.
"""

    def _parse_plan_response(self, response: str, goal: str) -> ExecutionPlan:
        """Parse the LLM response into an ExecutionPlan."""
        try:
            # Clean up response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            # Parse JSON
            plan_data = json.loads(response)

            # Create plan steps
            steps = []
            for step_data in plan_data.get("steps", []):
                step = PlanStep(
                    step_id=step_data["step_id"],
                    tool_name=step_data["tool_name"],
                    arguments=step_data["arguments"],
                    expected_output=step_data["expected_output"],
                    dependencies=step_data.get("dependencies", []),
                    description=step_data.get("description", ""),
                    reasoning=step_data.get("reasoning", "")
                )
                steps.append(step)

            # Create execution plan
            plan = ExecutionPlan(
                plan_id=plan_data.get("plan_id", f"plan_{hash(goal)}"),
                goal=plan_data.get("goal", goal),
                steps=steps,
                confidence=plan_data.get("confidence", 0.0),
                estimated_duration=plan_data.get("estimated_duration", 0.0),
                alternative_approaches=plan_data.get("alternative_approaches", [])
            )

            return plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            raise ValueError(f"Invalid plan format: {response}")
        except Exception as e:
            logger.error(f"Error parsing plan: {e}")
            raise ValueError(f"Plan parsing failed: {e}")

    def _validate_plan(self, plan: ExecutionPlan) -> bool:
        """Validate that a plan is executable."""
        # Check confidence threshold
        if plan.confidence < self.min_confidence:
            logger.warning(f"Plan confidence {plan.confidence} below threshold {self.min_confidence}")
            return False

        # Check step count
        if len(plan.steps) > self.max_steps:
            logger.warning(f"Plan has {len(plan.steps)} steps, max is {self.max_steps}")
            return False

        # Check that all tools exist
        for step in plan.steps:
            if not self.tool_registry.get(step.tool_name):
                logger.warning(f"Tool '{step.tool_name}' not found in registry")
                return False

        # Check dependency cycles
        if self._has_circular_dependencies(plan.steps):
            logger.warning("Plan has circular dependencies")
            return False

        # Validate tool arguments
        for step in plan.steps:
            if not self.tool_registry.validate_args(step.tool_name, step.arguments):
                logger.warning(f"Invalid arguments for tool '{step.tool_name}': {step.arguments}")
                return False

        return True

    def _has_circular_dependencies(self, steps: List[PlanStep]) -> bool:
        """Check for circular dependencies in plan steps."""
        # Build dependency graph
        graph = {step.step_id: step.dependencies for step in steps}

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for step_id in graph:
            if step_id not in visited:
                if has_cycle(step_id):
                    return True

        return False


class ReplanningStrategy:
    """
    Strategy for determining when and how to replan.

    Analyzes execution results and context to determine if replanning
    is necessary and generates updated plans when needed.

    Example:
        >>> strategy = ReplanningStrategy(planner)
        >>> if strategy.should_replan(plan, failed_step, context):
        ...     new_plan = strategy.replan(plan, failed_step, context)
    """

    def __init__(self, planner: Planner):
        """
        Initialize replanning strategy.

        Args:
            planner: Planner instance for generating new plans
        """
        self.planner = planner

    def should_replan(
        self,
        plan: ExecutionPlan,
        failed_step: Optional[PlanStep] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Determine if replanning is necessary.

        Args:
            plan: Current execution plan
            failed_step: Step that failed (if any)
            context: Current execution context

        Returns:
            True if replanning is recommended

        Example:
            >>> if strategy.should_replan(current_plan, failed_step):
            ...     print("Replanning needed")
        """
        # Always replan if there are failed steps
        if plan.has_failed_steps():
            logger.info("Replanning due to failed steps")
            return True

        # Check if too many steps have been retried
        high_retry_steps = [s for s in plan.steps if s.retry_count > 2]
        if high_retry_steps:
            logger.info("Replanning due to high retry count")
            return True

        # Check if context has changed significantly
        if context and context.get("force_replan", False):
            logger.info("Replanning forced by context")
            return True

        return False

    def replan(
        self,
        original_plan: ExecutionPlan,
        failed_step: Optional[PlanStep] = None,
        context: Optional[Dict[str, Any]] = None,
        completed_work: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Generate a new plan based on current situation.

        Args:
            original_plan: Original execution plan
            failed_step: Step that failed (if any)
            context: Current execution context
            completed_work: Results from completed steps

        Returns:
            New execution plan

        Example:
            >>> new_plan = strategy.replan(old_plan, failed_step, context)
            >>> print(f"New plan: {new_plan.plan_id}")
        """
        # Build replanning context
        replan_context = context.copy() if context else {}

        # Add information about what failed
        if failed_step:
            replan_context["failed_step"] = {
                "tool": failed_step.tool_name,
                "arguments": failed_step.arguments,
                "error": failed_step.result
            }

        # Add completed work information
        if completed_work:
            replan_context["completed_work"] = completed_work

        # Add original plan information
        replan_context["original_plan"] = {
            "goal": original_plan.goal,
            "attempted_steps": [step.to_dict() for step in original_plan.steps]
        }

        # Generate new plan with enhanced context
        enhanced_goal = f"""
        REPLANNING REQUIRED

        Original Goal: {original_plan.goal}

        Please create a new plan that:
        1. Achieves the original goal
        2. Takes into account what has already been completed
        3. Avoids the approaches that failed
        4. Uses alternative methods where necessary
        """

        return self.planner.create_plan(
            goal=enhanced_goal,
            context=replan_context
        )