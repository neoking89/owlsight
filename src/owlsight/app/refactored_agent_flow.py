"""
Agentic Flow Example for OwlSight.

This script demonstrates a more structured agentic flow using dedicated data
classes (`Result`, `PlanStep`, `ExecutionPlan`) and an `AgentOrchestrator`
to manage the process.

**Core Flow:**
1.  A user question is received by `AgentOrchestrator.process_user_question`.
2.  The `PlannerAgent` is invoked to analyze the question and create an
    `ExecutionPlan`, which is a list of `PlanStep` objects. Each step
    specifies an agent to run and a description of the task.
3.  The `AgentOrchestrator` iterates through the `PlanStep`s in the
    `ExecutionPlan`.
4.  For each step, the orchestrator retrieves the corresponding agent
    (e.g., `ContextAgent`, `ToolCreationAgent`) and calls its `execute` method.
5.  The executed agent performs its task and updates the `result` attribute
    (a `Result` object) within its assigned `PlanStep` in the shared context.
6.  If any step fails (either by raising an exception caught by the
    orchestrator or by setting `step.result.success = False` and
    `stop_on_step_failure=True`), the orchestrator can optionally attempt to
    re-plan by calling the `PlannerAgent` again with error context.
7.  After all steps in the plan are executed (or execution stops), the
    `ValidationAgent` is called to check the overall success based on the
    results in the `ExecutionPlan`.
8.  Finally, the `ResponseSynthesisAgent` is called to compile the results
    from the executed steps and the validation check into a final, coherent
    response for the user.
"""

import logging
from typing import TypedDict, Any, List, Optional, ClassVar, Dict
from dataclasses import dataclass, field

from owlsight.processors.text_generation_manager import TextGenerationManager
from owlsight.utils.code_execution import CodeExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Prompt Templates ---
PLANNER_PROMPT = """
You are an expert planner and router. Analyze the user request:
1. Break it into several subtasks if needed. Try to make the steps as atomic as possible.
2. Assign each subtask to the most suitable agent:
   - ToolSelectionAgent for external data retrieval, API calls, or using existing tools
   - ToolCreationAgent ONLY for creating new dynamic tool functions (do NOT use it for direct computations)
   - ContextAgent for analysis, summarization, or strategy generation
3. For tasks requiring computation or custom functionality, ALWAYS use ToolCreationAgent first to create a dynamic tool
4. ONLY use ToolCreationAgent with the plan of creating a new dynamic tool that can be used in a later step by the ToolSelectionAgent
5. Return a structured plan.

KEY WORKFLOW PATTERN:
- If the task requires computation or custom functionality, first use ToolCreationAgent to create a dynamic tool
- Then use ToolSelectionAgent in the next step to execute/use that tool
- NEVER use ToolCreationAgent for direct computation - it ONLY creates reusable tools (functions)

AVAILABLE AGENTS:
{agent_list}

User Question:
{user_question}

Your task: Create a plan (several steps) with an agent for each subtask.

AVAILABLE TOOLS:
{available_tools}

Response Format:
<plan>
Step 1: ...
Agent: [ToolSelectionAgent | ToolCreationAgent | ContextAgent]
Reason: ...
Step 2: ...
Agent: ...
Reason: ...
</plan>

<reasoning>
...
</reasoning>
""".strip()

CONTEXT_AGENT_PROMPT = """
You answer questions based on the provided context. Analyze the user query and relevant context to provide a direct answer.

User Request:
{user_question}

Step {current_step}/{max_steps}

Previous Results: {previous_results}
Final Results from Previous Steps: {final_results}
Additional Info: {additional_info}

Instructions:
Provide a direct response to the user's request based on all available context.
Be concise but thorough.
""".strip()

TOOL_CREATION_PROMPT = """
You are an expert Python developer specialized in creating tool functions. Your job is to ONLY create reusable Python tool functions based on the user's request.

Guidelines for creating tools:
1. Create functions starting with "dynamic_tool_" followed by a descriptive name
2. All tools must have helpful docstrings (created in Numpy style) explaining what they do and their parameters
3. Tools should perform a single, specific task.
4. Use proper error handling and input validation.
5. Tools should be designed for reuse by the ToolSelectionAgent
6. Never execute tasks directly - ONLY create tools that can be executed later

These functions will be registered in the global namespace for the ToolSelectionAgent to use.

Create a dynamic tool function to help with this request: 
{user_question}

Additional context from previous steps:
{previous_results}

REQUIREMENTS:
1. Function name must start with "dynamic_tool_"
2. Include comprehensive docstrings (created in Numpy style)
3. Implement proper error handling
4. Return results in a structured format (dict, list, etc.)
5. DO NOT execute tasks directly - ONLY create functions that can be called later

Example structure:
```python
def dynamic_tool_name(param1, param2=None):
    \"\"\"
    Description of what this tool does.
    
    Parameters:
    ----------
        param1: Description of parameter
        param2: Description of optional parameter
    
    Returns:
    --------
        Description of return value
    \"\"\"
    # Implementation
    return result
```

Create ONLY the tool function(s) required. Do not provide examples of usage or explanations outside the function definition.
""".strip()

TOOL_SELECTION_PROMPT = """
You are an expert in tool selection. If you need a tool, respond ONLY with a JSON object:
{"name": "tool_name", "arguments": {...}}
No extra text.

User Request:
{user_question}

Step {current_step}/{max_steps}

Previous Results: {previous_results}
Final Results from Previous Steps: {final_results}
Additional Info: {additional_info}

Instructions:
1. Check if previous tool calls gave needed info.
2. Decide next steps carefully. If you must use another tool, return only valid JSON.
3. Do NOT repeat the same tool call with same arguments.

Available Built-in Tools:
- owl_search, owl_scrape, owl_read, owl_write, owl_import

Available Dynamic Tools (created in previous steps):
{dynamic_tools}

Return Format:
{"name": "tool_name", "arguments": {...}}
""".strip()

VALIDATION_PROMPT = """
You are a meticulous quality assurance agent. Your task is to validate the results of the previous execution steps and report any failures.

User Request:
{user_question}

Execution Plan & Results:
{execution_plan}

Validation Instructions:
1. Review all step results for completeness and success
2. Check if the user's original request has been fulfilled
3. Identify any missing information or failures
4. Provide a clear validation result

Return a detailed validation report including:
- Overall pass/fail status
- Specific issues found (if any)
- Suggestions for improvement
""".strip()

RESPONSE_SYNTHESIS_PROMPT = """
You are a response synthesis agent. Your task is to compile the results from the execution plan and validation into a final, coherent answer for the user.

User Request:
{user_question}

Execution Results:
{execution_results}

Validation Result:
{validation_result}

Instructions:
1. Synthesize all the information into a well-structured, concise response
2. Focus on directly answering the user's original question 
3. Include relevant details from the execution steps
4. Acknowledge any limitations or issues found during validation
5. Format your response in a clear, user-friendly manner
""".strip()


# Datastructures used in the agentic flow
@dataclass
class Result:
    """Container for the outcome of an agent's execution step."""

    success: bool
    details: Any = None  # Can store success messages, data, or error details/tracebacks


@dataclass
class PlanStep:
    """Represents a single step in the execution plan."""

    agent_name: str
    description: str
    result: Optional[Result] = None  # To be filled in by the agent


@dataclass
class ExecutionPlan:
    """Container for the sequence of steps to be executed."""

    steps: List[PlanStep] = field(default_factory=list)

    def get_step(self, index: int) -> Optional[PlanStep]:
        """Safely get a step by index."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def __getitem__(self, index: int) -> PlanStep:
        """
        Make ExecutionPlan indexable (e.g., plan[0] instead of plan.get_step(0)).

        Args:
            index: The index of the step to retrieve

        Returns:
            The PlanStep at the specified index

        Raises:
            IndexError: If the index is out of range
        """
        if 0 <= index < len(self.steps):
            return self.steps[index]

        raise IndexError(f"ExecutionPlan index {index} out of range")

    def __len__(self) -> int:
        """
        Return the number of steps in the plan.

        This allows using len(plan) to get the step count.
        """
        return len(self.steps)

    def __str__(self) -> str:
        """Return the plan as a string."""
        return "\n".join(
            [f"Step {i + 1}: {step.description} (Agent: {step.agent_name})" for i, step in enumerate(self.steps)]
        )


@dataclass
class AgentPrompt:
    """
    A flexible prompt template that can be formatted with various parameters.

    This class allows for dynamic prompt construction by accepting a template with
    placeholders and formatting it with provided parameters. Parameters can be
    provided during initialization or when calling format().

    Attributes:
        template: The prompt template with formatting placeholders
        params: Pre-defined parameters to use when formatting the template
    """

    template: str
    params: dict[str, Any] = field(default_factory=dict)

    def format(self, **kwargs) -> str:
        """
        Format the template with given parameters.

        Args:
            **kwargs: Format parameters to be merged with any pre-defined parameters.
                     Overlapping keys will be overridden by these parameters.

        Returns:
            The formatted prompt string
        """
        # Combine pre-defined params with new params, with new params taking precedence
        format_params = {**self.params, **kwargs}

        try:
            return self.template.format(**format_params)
        except KeyError as e:
            raise KeyError(f"Missing required parameter in prompt template: {e}") from None

    def __str__(self) -> str:
        """Return the template string in its unformatted form."""
        return self.template


# --- Mock AgentContext (Refactored) ---
# Represents the shared state passed between agents
class AgentContext(TypedDict, total=False):
    user_question: str
    execution_plan: Optional[ExecutionPlan]  # Contains PlanSteps with Results
    current_plan_index: int
    validation_result: str  # Kept separate for now as per flow diagram
    final_response: str  # Kept separate for now
    error_context: str  # Context from any *orchestration* error that occurred


# --- Mock Agent Classes (Adapting to new structures) ---


class BaseAgent:
    # Class variables shared across all instances
    manager: ClassVar[Optional["TextGenerationManager"]] = None
    code_executor: ClassVar[Optional["CodeExecutor"]] = None

    def __init__(self, name: str, system_prompt: AgentPrompt):
        self.name = name
        self.system_prompt = system_prompt

    def llm_call(self, formatted_prompt: str) -> str:
        """
        Makes a call to the LLM using the TextGenerationManager.

        Args:
            formatted_prompt: The formatted prompt to send to the LLM

        Returns:
            The response from the LLM or a mock response if manager is not available
        """
        if BaseAgent.manager:
            # Make the actual LLM call
            return BaseAgent.manager.generate(formatted_prompt)
        else:
            # For testing without a real LLM
            logging.warning(f"{self.name}: No TextGenerationManager available, returning mock response")
            return f"Mock LLM response for: {formatted_prompt[:50]}..."

    def execute(self, context: AgentContext) -> AgentContext:
        """Executes the agent's logic.

        Modifies the context, specifically by updating the result of the
        current plan step within the execution_plan.
        Should ideally not raise exceptions for plan execution errors,
        but capture them in the Result object. Unforeseen errors might still raise.
        """
        logging.info(f"Executing {self.name}...")
        plan = context.get("execution_plan")
        current_index = context.get("current_plan_index", -1)

        if plan and 0 <= current_index < len(plan.steps):
            step = plan.steps[current_index]
            # Subclasses will override this part to create the actual result
            # Ensure a default result is NOT set here, let subclasses handle it or leave None
            # step.result = Result(success=True, details=f"{self.name} base execution (should be overridden).")
        else:
            # This condition might occur for agents like Router/Validator/Synthesizer
            # called outside the main plan execution loop, or if context is invalid.
            logging.debug(f"{self.name}: Not operating on a specific plan step or plan missing.")

        return context


class PlannerAgent(BaseAgent):
    def __init__(self):
        planner_prompt = AgentPrompt(
            template=PLANNER_PROMPT,
            params={
                "agent_list": "- ToolSelectionAgent: Use when external data retrieval, API calls, or specialized tool usage is required.\n"
                "- ToolCreationAgent: Use ONLY to create dynamic tool functions that can later be used by ToolSelectionAgent.\n"
                "- ContextAgent: Use for analyzing text, summarizing, extracting info, or generating strategies."
            },
        )
        super().__init__("PlannerAgent", planner_prompt)

    def execute(self, context: AgentContext) -> AgentContext:
        # This agent *creates* the plan, doesn't execute a step within it.
        logging.info(f"Executing {self.name}...")
        user_question = context.get("user_question", "")
        error_context = context.get("error_context")

        # TODO: In a real implementation, this would use the system_prompt with LLM
        # formatted_prompt = self.system_prompt.format(
        #    user_question=user_question,
        #    available_tools="owl_search, owl_scrape, owl_read, owl_write, owl_import"
        # )

        # Mock planning logic - Create PlanStep objects
        plan_steps = []
        if error_context:
            logging.warning(f"Planner received error context: {error_context}. Generating fallback plan.")
            plan_steps = [PlanStep(agent_name="ContextAgent", description="Analyze fallback options")]
        elif "analyze" in user_question.lower():
            plan_steps = [PlanStep(agent_name="ContextAgent", description="Analyze the core request")]
        elif "python" in user_question.lower() or "tool" in user_question.lower():
            plan_steps = [
                PlanStep(agent_name="ToolCreationAgent", description="Generate Python code snippet for a tool"),
                PlanStep(
                    agent_name="ToolSelectionAgent",
                    description="Select tool based on Python output (mandatory follow-up)",
                ),
            ]
        else:
            plan_steps = [PlanStep(agent_name="ContextAgent", description="Default analysis for general query")]

        execution_plan = ExecutionPlan(steps=plan_steps)
        logging.info(f"Planner generated plan: {execution_plan}")
        context["execution_plan"] = execution_plan
        context["current_plan_index"] = 0  # Reset index for new plan
        context["error_context"] = None  # Clear error context after planning
        return context


class ContextAgent(BaseAgent):
    def __init__(self):
        context_prompt = AgentPrompt(template=CONTEXT_AGENT_PROMPT)
        super().__init__("ContextAgent", context_prompt)

    def execute(self, context: AgentContext) -> AgentContext:
        super().execute(context)
        plan = context.get("execution_plan")
        current_index = context.get("current_plan_index", -1)
        step = plan.get_step(current_index) if plan else None

        if step:
            try:
                # Format the prompt with actual values
                formatted_prompt = self.system_prompt.format(
                    user_question=context.get("user_question", ""),
                    current_step=current_index + 1,
                    max_steps=len(plan.steps) if plan else 1,
                    previous_results="None",
                    final_results="None",
                    additional_info="",
                )

                # Make the LLM call using the BaseAgent's llm_call method
                response = self.llm_call(formatted_prompt)

                # In a real implementation, process the response as needed
                analysis_details = response

                # Fallback to simple mock for testing when no actual LLM is available
                if not response or "Mock LLM response" in response:
                    analysis_details = f"Contextual Answer based on '{context.get('user_question', '')}'."

                logging.info(f"ContextAgent result: {analysis_details[:50]}...")
                step.result = Result(success=True, details=analysis_details)
            except Exception as e:
                logging.error(f"{self.name} failed: {e}", exc_info=True)
                step.result = Result(success=False, details=f"Error during context analysis: {e}")
        # Error logging for missing step handled by BaseAgent or orchestrator

        return context


class ToolCreationAgent(BaseAgent):
    def __init__(self):
        tool_creation_prompt = AgentPrompt(template=TOOL_CREATION_PROMPT)
        super().__init__("ToolCreationAgent", tool_creation_prompt)

    def execute(self, context: AgentContext) -> AgentContext:
        super().execute(context)
        plan = context.get("execution_plan")
        current_index = context.get("current_plan_index", -1)
        step = plan.get_step(current_index) if plan else None

        if step:
            try:
                # In a real implementation, this would format the prompt with actual values
                # formatted_prompt = self.system_prompt.format(
                #    user_question=context.get('user_question', ''),
                #    previous_results=""  # Would contain info from previous steps
                # )

                # Mock Python code generation or execution
                python_code = "def new_tool():\n    print('Hello from dynamic ToolCreationAgent!')"  # Example tool code
                logging.info(f"ToolCreationAgent result: {python_code}")

                # Simulate an error sometimes for testing the error loop
                # import random
                # if random.random() < 0.3: # 30% chance of error
                #     error_msg = "Simulated error in ToolCreationAgent"
                #     logging.error(f"ToolCreationAgent encountered a simulated error!")
                #     step.result = Result(success=False, details=error_msg)
                # else:
                # Only register if no error
                step.result = Result(success=True, details=python_code)
            except Exception as e:
                logging.error(f"{self.name} failed: {e}", exc_info=True)
                step.result = Result(success=False, details=f"Error during tool creation: {e}")

        return context


class ToolSelectionAgent(BaseAgent):
    def __init__(self):
        tool_selection_prompt = AgentPrompt(template=TOOL_SELECTION_PROMPT)
        super().__init__("ToolSelectionAgent", tool_selection_prompt)

    def execute(self, context: AgentContext) -> AgentContext:
        super().execute(context)
        plan = context.get("execution_plan")
        current_index = context.get("current_plan_index", -1)
        step = plan.get_step(current_index) if plan else None

        if step:
            try:
                # In a real implementation, we would format the prompt with actual values
                # formatted_prompt = self.system_prompt.format(
                #    user_question=context.get('user_question', ''),
                #    current_step=current_index + 1,
                #    max_steps=len(plan.steps),
                #    previous_results="None",
                #    final_results="None",
                #    additional_info="",
                #    dynamic_tools=""  # Would populate with dynamic tools from previous steps
                # )

                # Default success status
                success_status = True

                # Logic for checking previous step
                if current_index > 0:
                    # For this example, let's assume ToolSelectionAgent depends on ToolCreationAgent
                    prev_index = current_index - 1
                    prev_step = plan.get_step(prev_index)

                    if prev_step and prev_step.agent_name == "ToolCreationAgent" and prev_step.result:
                        if prev_step.result.success:
                            python_result = prev_step.result.details
                            tool_selection_details = (
                                f"Selected 'MockTool' based on ToolCreationAgent result: {python_result}"
                            )
                        else:
                            # Previous step failed, select fallback but maybe this step still 'succeeds' in selecting fallback
                            tool_selection_details = f"Selected 'FallbackTool' as previous ToolCreationAgent step failed ({prev_step.result.details})."
                            logging.warning(f"{self.name}: Previous step failed, selecting fallback.")
                            # Decide if ToolSelection *itself* failed due to dependency failure.
                            # Let's say it still succeeds in selecting *something* (the fallback).
                            success_status = True
                    else:
                        # Previous step missing or no result (shouldn't happen with ToolCreation->ToolSelection plan)
                        logging.warning(f"{self.name}: Previous step or its result missing.")
                        tool_selection_details = "Selected 'DefaultTool' due to missing previous step info."

                    logging.info(f"ToolSelectionAgent result: {tool_selection_details}")
                    step.result = Result(success=success_status, details=tool_selection_details)

                # Handle no dependency scenario
                else:
                    tool_selection_details = "Selected 'IndependentTool' as there are no previous steps to depend on."
                    logging.info(f"{self.name} result: {tool_selection_details}")
                    step.result = Result(success=True, details=tool_selection_details)

            except Exception as e:
                logging.error(f"{self.name} failed: {e}", exc_info=True)
                step.result = Result(success=False, details=f"Error during tool selection: {e}")

        return context


class ValidationAgent(BaseAgent):
    def __init__(self):
        validation_prompt = AgentPrompt(template=VALIDATION_PROMPT)
        super().__init__("ValidationAgent", validation_prompt)

    def execute(self, context: AgentContext) -> AgentContext:
        # This agent is called *after* the plan execution loop.
        # It inspects the results in the plan and updates context["validation_result"].
        super().execute(context)
        plan = context.get("execution_plan")
        validation_passed = True
        details = "Validation passed."

        # In a real implementation, we would format the prompt with actual values
        # formatted_prompt = self.system_prompt.format(
        #    user_question=context.get('user_question', ''),
        #    execution_plan=str(plan),  # Would need proper formatting for plan representation
        # )

        if plan and plan.steps:
            for i, step in enumerate(plan.steps):
                if not step.result:
                    validation_passed = False
                    details = f"Validation failed: Step {i + 1} '{step.description}' ({step.agent_name}) seems not to have run or produced a result."
                    logging.warning(details)
                    break
                elif not step.result.success:
                    validation_passed = False
                    details = f"Validation failed: Step {i + 1} '{step.description}' ({step.agent_name}) did not succeed. Details: {step.result.details if step.result else 'N/A'}"
                    logging.warning(details)
                    break  # Stop on first failure
        else:
            validation_passed = False
            details = "Validation failed: No execution plan found or plan is empty."
            logging.error(details)

        logging.info(f"ValidationAgent result: {details}")
        context["validation_result"] = details
        # We could also return a Result object from this agent if needed elsewhere
        # return Result(success=validation_passed, details=details) # If we wanted to standardize
        return context


class ResponseSynthesisAgent(BaseAgent):
    def __init__(self):
        response_synthesis_prompt = AgentPrompt(template=RESPONSE_SYNTHESIS_PROMPT)
        super().__init__("ResponseSynthesisAgent", response_synthesis_prompt)

    def execute(self, context: AgentContext) -> AgentContext:
        # This agent is called *after* the ValidationAgent.
        # It examines the execution results and validation to create a final response.
        super().execute(context)

        # Extract relevant information from context
        user_question = context.get("user_question", "")
        validation_result = context.get("validation_result", "Validation not performed.")
        plan = context.get("execution_plan")

        # In a real implementation, we would format the prompt with actual values
        # formatted_prompt = self.system_prompt.format(
        #    user_question=user_question,
        #    execution_results=self._format_execution_results(plan) if plan else "No execution results available.",
        #    validation_result=validation_result
        # )

        # Default response if there's an issue with the plan
        if not plan or not plan.steps:
            final_response = f"I couldn't process your request: '{user_question}'. No valid execution plan was created."
            logging.warning("ResponseSynthesisAgent: No execution plan to synthesize.")
        else:
            # Custom response based on validation result
            if "failed" in validation_result.lower():
                final_response = f"I encountered an issue while processing your request: '{validation_result}'. Would you like to try again or modify your request?"
            else:
                # Success path: Combine results from all steps
                successful_steps = []
                for i, step in enumerate(plan.steps):
                    if step.result and step.result.success:
                        successful_steps.append(f"Step {i + 1}: {step.result.details}")

                # Format a nice response
                if successful_steps:
                    steps_summary = "\n".join(successful_steps)
                    final_response = (
                        f"I've processed your request: '{user_question}'.\n\nHere's what I found:\n{steps_summary}"
                    )
                else:
                    final_response = f"I processed your request, but there were no successful steps to report. The validation says: {validation_result}"

        logging.info(f"ResponseSynthesisAgent produced final response")
        context["final_response"] = final_response
        return context


# --- Agent Orchestrator (Adapting to new structures) ---


class AgentOrchestrator:
    def __init__(self):
        # Agent instances are still stored here for lookup by name
        self.agents: dict[str, BaseAgent] = {
            "PlannerAgent": PlannerAgent(),
            "ContextAgent": ContextAgent(),
            "ToolCreationAgent": ToolCreationAgent(),
            "ToolSelectionAgent": ToolSelectionAgent(),
            "ValidationAgent": ValidationAgent(),
            "ResponseSynthesisAgent": ResponseSynthesisAgent(),
        }
        self.max_error_loops = 2  # Prevent infinite loops on persistent errors
        self.stop_on_step_failure = (
            False  # Configuration: Should orchestrator stop if a step returns Result(success=False)?
        )

    def _get_agent(self, agent_name: str) -> BaseAgent:
        agent = self.agents.get(agent_name)
        if not agent:
            # This indicates a mismatch between planner and available agents
            raise ValueError(f"Orchestrator Error: Unknown agent '{agent_name}' specified in plan.")
        return agent

    def process_user_question(self, user_question: str) -> str:
        context: AgentContext = {"user_question": user_question}
        error_loops = 0

        while error_loops <= self.max_error_loops:
            plan_failed_or_incomplete = False
            try:
                # 1. Planning Phase (or Re-planning on error)
                router = self._get_agent("PlannerAgent")
                context = router.execute(context)

                plan = context.get("execution_plan")
                if not plan or not plan.steps:
                    logging.error("Planner failed to generate a valid plan.")
                    context["validation_result"] = "Planning Failed: Planner did not produce a valid plan."
                    context = self._get_agent("ResponseSynthesisAgent").execute(context)  # Synthesize error response
                    return context.get("final_response", "Error: Planning failed.")

                # 2. Execution Phase (Process plan steps)
                while context.get("current_plan_index", 0) < len(plan.steps):
                    current_index = context["current_plan_index"]
                    current_step = plan.steps[current_index]

                    # Ensure result is None before execution (important for re-runs)
                    current_step.result = None

                    agent_name = current_step.agent_name
                    logging.info(
                        f"--- Executing Plan Step {current_index + 1}/{len(plan.steps)}: Agent '{agent_name}' ---"
                    )
                    logging.info(f"    Description: {current_step.description}")

                    agent_to_execute = self._get_agent(agent_name)  # Raises ValueError if agent not found

                    # Execute the agent - it should update current_step.result internally
                    context = agent_to_execute.execute(context)

                    # --- Check step result AFTER execution ---
                    if not current_step.result:
                        # Agent didn't produce a result - treat as failure
                        current_step.result = Result(
                            success=False, details=f"Agent '{agent_name}' did not produce a result object."
                        )
                        logging.error(current_step.result.details)
                        plan_failed_or_incomplete = True
                    elif not current_step.result.success:
                        # Agent reported failure
                        error_msg = f"Agent '{agent_name}' reported failure for step {current_index + 1}. Details: {current_step.result.details}"
                        logging.error(error_msg)
                        plan_failed_or_incomplete = True

                    # --- Decide whether to continue based on failure ---
                    if plan_failed_or_incomplete and self.stop_on_step_failure:
                        logging.warning(
                            f"Stopping execution due to failure in step {current_index + 1} and stop_on_step_failure=True."
                        )
                        break  # Exit the execution loop

                    context["current_plan_index"] += 1
                # --- End of Execution Loop ---

                # 3. Validation Phase (Executed after all plan steps attempted or stop_on_failure)
                logging.info("--- Executing Validation Phase ---")
                validator = self._get_agent("ValidationAgent")
                context = validator.execute(context)

                # 4. Response Synthesis Phase
                logging.info("--- Executing Response Synthesis Phase ---")
                synthesizer = self._get_agent("ResponseSynthesisAgent")
                context = synthesizer.execute(context)

                # If orchestration successful (no fatal exceptions), break the error loop
                return context.get("final_response", "Error: No final response generated.")

            except Exception as e:  # Catch orchestration errors (e.g., unknown agent) or unexpected agent errors
                logging.error(f"Orchestration Error during execution: {e}", exc_info=True)
                error_loops += 1
                if error_loops > self.max_error_loops:
                    logging.critical(f"Maximum error re-planning loops ({self.max_error_loops}) exceeded.")
                    return f"Error: Processing failed after multiple attempts. Last error: {e}"

                # Prepare context for re-planning
                plan = context.get("execution_plan")
                step_info = "N/A"
                if plan:
                    current_index = context.get("current_plan_index", -1)
                    step = plan.get_step(current_index)
                    if step:
                        step_info = f"Agent '{step.agent_name}', Step Index {current_index}"

                context["error_context"] = f"Orchestration error: {e}. Occurred near: {step_info}"
                context["execution_plan"] = None  # Clear potentially inconsistent plan
                context["current_plan_index"] = 0
                logging.warning(f"Attempting to re-plan (Attempt {error_loops}/{self.max_error_loops})...")
                # Loop continues, starting with PlannerAgent again

        # Fallback if loop finishes unexpectedly
        return "Error: Unexpected exit from processing loop."

# Example usage
if __name__ == "__main__":
    orchestrator = AgentOrchestrator()
    # orchestrator.stop_on_step_failure = True # Uncomment to test early stopping

    print("\n--- Example 1: Analyze Question ---")
    question1 = "Analyze the main components of the system."
    response1 = orchestrator.process_user_question(question1)
    print(f"\nUser Question: {question1}")
    print(f"Final Response:\n{response1}")

    print("\n\n--- Example 2: Python/Tool Question ---")
    question2 = "Create a tool using python for a simple calculator and select it."
    response2 = orchestrator.process_user_question(question2)
    print(f"\nUser Question: {question2}")
    print(f"Final Response:\n{response2}")

    print("\n\n--- Example 3: General Question ---")
    question3 = "What is OwlSight?"
    response3 = orchestrator.process_user_question(question3)
    print(f"\nUser Question: {question3}")
    print(f"Final Response:\n{response3}")

    # To test error handling, uncomment the error simulation in ToolCreationAgent
    # and run the Python/Tool question again.
    # Note how failure is now handled via Result(success=False)
    print("\n\n--- Example 4: (Potential Error Test if uncommented in ToolCreationAgent) ---")
    question4 = "Give me python code to create a file reading tool."
    response4 = orchestrator.process_user_question(question4)
    print(f"\nUser Question: {question4}")
    print(f"Final Response:\n{response4}")
