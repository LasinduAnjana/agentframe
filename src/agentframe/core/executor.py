"""
Execution engine for AgentFrame.

This module provides the execution engine that runs plans step-by-step,
handles tool invocation, manages execution state, and determines when
replanning is needed.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from .planner import ExecutionPlan, PlanStep, PlanStepStatus
from ..tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Overall execution status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REPLANNING = "requires_replanning"
    PAUSED = "paused"


@dataclass
class ExecutionResult:
    """
    Result from executing a plan or step.

    Contains comprehensive information about execution outcomes,
    including success status, results, errors, and timing.

    Attributes:
        success: Whether execution was successful
        status: Overall execution status
        results: Dictionary of results keyed by step ID
        errors: Dictionary of errors keyed by step ID
        execution_time: Total execution time in seconds
        steps_completed: Number of steps completed
        steps_failed: Number of steps that failed
        needs_replanning: Whether replanning is recommended
        metadata: Additional execution metadata

    Example:
        >>> result = ExecutionResult(
        ...     success=True,
        ...     status=ExecutionStatus.COMPLETED,
        ...     results={"step_1": {"output": "Hello World"}},
        ...     execution_time=1.5
        ... )
    """
    success: bool
    status: ExecutionStatus
    results: Dict[str, Any] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    execution_time: float = 0.0
    steps_completed: int = 0
    steps_failed: int = 0
    needs_replanning: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step_result(self, step_id: str, result: Any) -> None:
        """Add result for a completed step."""
        self.results[step_id] = result
        self.steps_completed += 1

    def add_step_error(self, step_id: str, error: str) -> None:
        """Add error for a failed step."""
        self.errors[step_id] = error
        self.steps_failed += 1


@dataclass
class ExecutionContext:
    """
    Context information for plan execution.

    Maintains state and configuration throughout the execution process,
    including retry policies, timeout settings, and intermediate results.

    Attributes:
        max_retries: Maximum retry attempts per step
        step_timeout: Timeout per step in seconds
        total_timeout: Total execution timeout in seconds
        retry_delay: Delay between retries in seconds
        fail_fast: Whether to stop on first failure
        parallel_execution: Whether to execute steps in parallel when possible
        intermediate_results: Storage for intermediate step results
        execution_metadata: Additional execution metadata

    Example:
        >>> context = ExecutionContext(
        ...     max_retries=3,
        ...     step_timeout=30.0,
        ...     fail_fast=False
        ... )
    """
    max_retries: int = 3
    step_timeout: float = 30.0
    total_timeout: float = 300.0
    retry_delay: float = 1.0
    fail_fast: bool = True
    parallel_execution: bool = False
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionEngine:
    """
    Engine for executing plans step-by-step.

    The execution engine manages the complete lifecycle of plan execution,
    including step scheduling, tool invocation, error handling, and result
    aggregation.

    Attributes:
        tool_registry: Registry of available tools
        context: Execution context and configuration
        _execution_state: Internal execution state tracking

    Example:
        >>> engine = ExecutionEngine(tool_registry)
        >>> result = engine.execute_plan(plan)
        >>> if result.success:
        ...     print("Plan executed successfully!")
        >>> else:
        ...     print(f"Execution failed: {result.errors}")
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        context: Optional[ExecutionContext] = None
    ):
        """
        Initialize the execution engine.

        Args:
            tool_registry: Registry of available tools
            context: Execution context (default context if None)
        """
        self.tool_registry = tool_registry
        self.context = context or ExecutionContext()
        self._execution_state: Dict[str, Any] = {}

        logger.debug("Initialized execution engine")

    def execute_plan(
        self,
        plan: ExecutionPlan,
        start_from_step: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a complete plan.

        Args:
            plan: Execution plan to run
            start_from_step: Step ID to start from (for resuming execution)

        Returns:
            Execution result with outcomes and metadata

        Example:
            >>> result = engine.execute_plan(my_plan)
            >>> print(f"Completed {result.steps_completed} steps")
            >>> for step_id, output in result.results.items():
            ...     print(f"{step_id}: {output}")
        """
        start_time = time.time()
        result = ExecutionResult(
            success=False,
            status=ExecutionStatus.IN_PROGRESS
        )

        try:
            logger.info(f"Starting execution of plan '{plan.plan_id}'")

            # Initialize execution state
            self._initialize_execution_state(plan, start_from_step)

            # Execute steps
            completed_steps: Set[str] = set()
            if start_from_step:
                # Find already completed steps when resuming
                completed_steps = self._get_completed_steps_before(plan, start_from_step)

            while not plan.is_complete() and not self._should_stop_execution(result):
                # Check timeout
                if time.time() - start_time > self.context.total_timeout:
                    logger.warning("Execution timeout reached")
                    result.status = ExecutionStatus.FAILED
                    result.metadata["timeout"] = True
                    break

                # Get next executable step
                next_step = plan.get_next_step(completed_steps)
                if not next_step:
                    if plan.has_failed_steps():
                        logger.warning("No executable steps and some steps failed")
                        result.status = ExecutionStatus.REQUIRES_REPLANNING
                        result.needs_replanning = True
                    else:
                        logger.info("No more executable steps - checking for completion")
                        if plan.is_complete():
                            result.status = ExecutionStatus.COMPLETED
                            result.success = True
                        else:
                            logger.warning("Deadlock detected - no executable steps remaining")
                            result.status = ExecutionStatus.FAILED
                    break

                # Execute the step
                step_result = self._execute_step(next_step)

                # Process step result
                if step_result.success:
                    next_step.status = PlanStepStatus.COMPLETED
                    next_step.result = step_result.result
                    completed_steps.add(next_step.step_id)
                    result.add_step_result(next_step.step_id, step_result.result)
                    logger.info(f"Step '{next_step.step_id}' completed successfully")
                else:
                    # Handle step failure
                    if self._should_retry_step(next_step):
                        next_step.retry_count += 1
                        next_step.status = PlanStepStatus.PENDING
                        logger.warning(f"Step '{next_step.step_id}' failed, retrying ({next_step.retry_count}/{self.context.max_retries})")
                        time.sleep(self.context.retry_delay)
                    else:
                        next_step.status = PlanStepStatus.FAILED
                        next_step.result = step_result.error
                        result.add_step_error(next_step.step_id, step_result.error or "Unknown error")
                        logger.error(f"Step '{next_step.step_id}' failed permanently: {step_result.error}")

                        if self.context.fail_fast:
                            logger.info("Fail-fast enabled, stopping execution")
                            result.status = ExecutionStatus.FAILED
                            break

            # Finalize result
            if result.status == ExecutionStatus.IN_PROGRESS:
                if plan.is_complete():
                    result.status = ExecutionStatus.COMPLETED
                    result.success = True
                    logger.info(f"Plan '{plan.plan_id}' completed successfully")
                elif plan.has_failed_steps():
                    result.status = ExecutionStatus.REQUIRES_REPLANNING
                    result.needs_replanning = True
                    logger.warning(f"Plan '{plan.plan_id}' requires replanning")
                else:
                    result.status = ExecutionStatus.FAILED
                    logger.error(f"Plan '{plan.plan_id}' failed")

        except Exception as e:
            logger.error(f"Unexpected error during plan execution: {e}")
            result.status = ExecutionStatus.FAILED
            result.metadata["unexpected_error"] = str(e)

        finally:
            result.execution_time = time.time() - start_time
            result.metadata.update({
                "plan_id": plan.plan_id,
                "total_steps": len(plan.steps),
                "execution_engine": "AgentFrame"
            })

        logger.info(f"Plan execution completed: {result.status.value} in {result.execution_time:.2f}s")
        return result

    def execute_step(self, step: PlanStep) -> ToolResult:
        """
        Execute a single plan step.

        Args:
            step: Plan step to execute

        Returns:
            Tool execution result

        Example:
            >>> step = PlanStep("step_1", "calculator", {"expression": "2+2"}, "result")
            >>> result = engine.execute_step(step)
            >>> if result.success:
            ...     print(f"Result: {result.result}")
        """
        return self._execute_step(step)

    def _execute_step(self, step: PlanStep) -> ToolResult:
        """Internal method to execute a single step."""
        logger.debug(f"Executing step '{step.step_id}' with tool '{step.tool_name}'")

        try:
            # Mark step as in progress
            step.status = PlanStepStatus.IN_PROGRESS

            # Get the tool
            tool = self.tool_registry.get(step.tool_name)
            if not tool:
                error = f"Tool '{step.tool_name}' not found"
                logger.error(error)
                return ToolResult(success=False, error=error)

            # Validate arguments
            if not tool.validate_args(step.arguments):
                error = f"Invalid arguments for tool '{step.tool_name}': {step.arguments}"
                logger.error(error)
                return ToolResult(success=False, error=error)

            # Execute with timeout
            start_time = time.time()
            result = self._execute_with_timeout(tool, step.arguments, self.context.step_timeout)

            execution_time = time.time() - start_time
            logger.debug(f"Step '{step.step_id}' executed in {execution_time:.2f}s")

            # Add execution metadata
            if result.metadata is None:
                result.metadata = {}
            result.metadata.update({
                "step_id": step.step_id,
                "tool_name": step.tool_name,
                "execution_time": execution_time,
                "retry_count": step.retry_count
            })

            return result

        except Exception as e:
            error = f"Unexpected error executing step '{step.step_id}': {e}"
            logger.error(error)
            return ToolResult(success=False, error=error)

    def _execute_with_timeout(
        self,
        tool,
        arguments: Dict[str, Any],
        timeout: float
    ) -> ToolResult:
        """Execute a tool with timeout (simplified implementation)."""
        # Note: In a production implementation, you would use proper timeout mechanisms
        # This is a simplified version for demonstration
        try:
            return tool.execute(arguments)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _should_retry_step(self, step: PlanStep) -> bool:
        """Determine if a failed step should be retried."""
        return step.retry_count < self.context.max_retries

    def _should_stop_execution(self, result: ExecutionResult) -> bool:
        """Determine if execution should stop based on current result."""
        return (
            result.status in [ExecutionStatus.FAILED, ExecutionStatus.COMPLETED] or
            (self.context.fail_fast and result.steps_failed > 0)
        )

    def _initialize_execution_state(
        self,
        plan: ExecutionPlan,
        start_from_step: Optional[str]
    ) -> None:
        """Initialize execution state for a plan."""
        self._execution_state = {
            "plan_id": plan.plan_id,
            "start_time": time.time(),
            "start_from_step": start_from_step,
            "step_history": []
        }

    def _get_completed_steps_before(
        self,
        plan: ExecutionPlan,
        start_step_id: str
    ) -> Set[str]:
        """Get steps that should be considered completed when resuming."""
        # This is a simplified implementation
        # In practice, you'd load this from persistent storage
        completed = set()
        for step in plan.steps:
            if step.step_id == start_step_id:
                break
            if step.status == PlanStepStatus.COMPLETED:
                completed.add(step.step_id)
        return completed

    def get_execution_summary(self, result: ExecutionResult) -> Dict[str, Any]:
        """
        Get a summary of execution results.

        Args:
            result: Execution result to summarize

        Returns:
            Dictionary with execution summary

        Example:
            >>> summary = engine.get_execution_summary(result)
            >>> print(f"Success rate: {summary['success_rate']}")
        """
        total_steps = result.steps_completed + result.steps_failed
        success_rate = result.steps_completed / total_steps if total_steps > 0 else 0.0

        return {
            "status": result.status.value,
            "success": result.success,
            "total_steps": total_steps,
            "steps_completed": result.steps_completed,
            "steps_failed": result.steps_failed,
            "success_rate": success_rate,
            "execution_time": result.execution_time,
            "needs_replanning": result.needs_replanning,
            "errors": result.errors,
            "results_count": len(result.results)
        }

    def pause_execution(self) -> None:
        """
        Pause the current execution.

        Example:
            >>> engine.pause_execution()
            >>> # Execution will pause at the next step boundary
        """
        self._execution_state["paused"] = True
        logger.info("Execution pause requested")

    def resume_execution(self) -> None:
        """
        Resume paused execution.

        Example:
            >>> engine.resume_execution()
            >>> # Execution will continue from where it was paused
        """
        self._execution_state["paused"] = False
        logger.info("Execution resumed")

    def is_paused(self) -> bool:
        """
        Check if execution is currently paused.

        Returns:
            True if execution is paused

        Example:
            >>> if engine.is_paused():
            ...     print("Execution is paused")
        """
        return self._execution_state.get("paused", False)