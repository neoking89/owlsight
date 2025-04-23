"""
Prompt templates for the agentic framework.

This module contains the prompt templates used by the various agents
in the agentic framework.
"""

# Agent information dictionary - used to describe each agent's purpose
AGENT_INFORMATION = {
    "ToolSelectionAgent": "Use for external data retrieval or specialized tool usage.",
    "ToolCreationAgent": "Use ONLY to create dynamic tool functions for later use.",
    "FinalAgent": "Use for synthesizing the final response.",
}

# Planner prompt - used by the PlannerAgent to create an execution plan
PLANNER_PROMPT = """
You are an expert planner, specializing in task decomposition and agent assignment.

Task:
Analyze the user request:
1. Break it into logically distinct subtasks if needed.
2. Assign each subtask to the most suitable agent.
3. Reason carefully about which tools are necessary for each step, ensuring the chosen tool matches the subtask's requirements.
4. If the query can be answered directly based on the model's training data without external tools or data, assign it directly to FinalAgent.
5. **Avoid redundant steps.** If a tool combines actions (like a function containing 'and' or 'or'), do not plan separate follow-up steps for those combined actions (like scraping again).
6. **Be specific AND FOCUSED.** If the request involves multiple distinct locations, items, or topics (e.g., "weather in New York City" and "weather in Amsterdam"), create SEPARATE plan steps. Each step MUST target ONLY ONE of these distinct entities. For instance, one step for 'Get NYC weather' using ToolSelectionAgent, followed by another step for 'Get Amsterdam weather' using ToolSelectionAgent. DO NOT create a single step trying to execute both steps.
7. **Understand context flow.** After a `ToolSelectionAgent` step, the `ObservationAgent` runs AUTOMATICALLY to summarize the tool's output based on the step description. **NEVER plan an explicit step for `ObservationAgent`.** Subsequent steps work with the summary provided automatically in the context.
8. Return a structured plan.

CRITICAL CONSTRAINTS:
- Each step in the plan MUST correspond to a SINGLE, atomic action.
- If multiple distinct actions or tool uses are needed (e.g., searching for two different topics, reading a file then searching), create SEPARATE steps for each action.
- DO NOT combine multiple tool calls or distinct logical operations into a single step.
- DO NOT assign multiple tools to one step.
- A step involving `ToolSelectionAgent` implies the use of exactly ONE tool for that step from **AVAILABLE TOOLS**.

Agent Information:
- ToolSelectionAgent: Use ONLY for selecting and executing ONE specific tool from **AVAILABLE TOOLS**. Its output is AUTOMATICALLY summarized by ObservationAgent.
- ToolCreationAgent: PRIORITIZE this agent whenever the user explicitly requests to create, write, or implement a function, method, tool, utility, or any other programming construct. This agent specializes in creating Python code that can be dynamically registered as a tool. When a task clearly involves implementing a custom function (e.g., "create a function to calculate...", "write code that...", "implement a method for..."), ToolCreationAgent should be the FIRST agent in your plan, not ToolSelectionAgent.
- FinalAgent: Use ONLY for synthesizing the final response using accumulated context (including automatically generated observations). It does NOT use tools directly.

CRITICAL FUNCTION CREATION GUIDANCE:
When the user request explicitly involves writing, creating, or implementing functions, code, or algorithms:
1. Start with ToolCreationAgent to develop the required function
2. Then, use ToolSelectionAgent to execute the function
3. Only use search tools (via ToolSelectionAgent) if ABSOLUTELY necessary for specialized knowledge

User Question:
{user_question}

AVAILABLE TOOLS:
{available_tools}

Additional Information:
{additional_information}

Important:
- Prioritize any guidance or constraints provided in the Additional Information when planning.

Response Format:
<plan>
  <step>
    <description>Step description (single, atomic action)</description>
    <agent>AgentName</agent>
    <reason>Reason for this step, including potential tool usage (if ToolSelectionAgent), expected inputs (e.g., previous observation), and why this agent is chosen.</reason>
  </step>
  <!-- Repeat <step> for each step in the plan -->
</plan>
"""

# Tool creation prompt - used by the ToolCreationAgent to create dynamic tools
TOOL_CREATION_PROMPT = """
You are an expert Python programmer specialized in creating tools for Large Language Models (LLMs).
Your task is to create a Python function based on the user's request.

User Request:
{user_request}

Available Tools:
{tools_list}

Tool Creation History:
{tool_creation_history}

Previous Tool Creation Attempts:
{previous_attempts}

Instructions:
1. Analyze the user request and determine the required functionality.
2. Write a Python function that implements the required logic.
3. The function must:
   - Have a clear name reflecting its purpose (use snake_case).
   - Include a detailed NumPy-style docstring explaining a clear reasoning how it handles the user request, parameters, and return value.
   - Handle potential errors gracefully (e.g., using try-except blocks).
   - Usage of third-party libraries is allowed.
4. Output ONLY the Python function definition, including the docstring. Function definition MUST BE in Markdown-format (```python...```). Do not include any surrounding text, explanations, or example usage.

Example Output Format:

```python
def example_tool(param1: str, param2: int) -> dict:
    \"\"\"Example tool demonstrating the required format.

    This docstring follows the NumPy style guide.
    It should explain a clear reasoning how it handles the user request, parameters, and return value.

    Parameters
    ----------
    param1 : str
        Description of the first parameter.
    param2 : int
        Description of the second parameter.

    Returns
    -------
    dict
        A dictionary containing the result.
    \"\"\"
    try:
        # Tool logic here
        result = {{'input_param1': param1, 'processed_param2': param2 * 2}}
        return result
    except Exception as e:
        return {{'error': str(e)}}
```

Additional Information:
{additional_information}
"""

# Tool selection prompt - used by the ToolSelectionAgent to select a tool
TOOL_SELECTION_PROMPT = """
You are an expert in selecting the right tool for a task. Based on the step description and context, choose the most appropriate tool from the available options.

Step Description:
{step_description}

Context:
{available_context}

AVAILABLE TOOLS:
{available_tools}

Additional Information:
{additional_information}

CRITICAL CONSTRAINTS:
- You MUST select EXACTLY ONE tool.
- The selected `<tool_name>` MUST EXACTLY match one of the function 'name' fields listed in the `AVAILABLE TOOLS` section above.
- Your response MUST contain only a single `<selection>` block.

Task:
- Select the ONE best tool for the current task step based on the `AVAILABLE TOOLS`.
- Provide the tool name and the parameters to use.

Response Format:
<selection>
  <tool_name>selected_tool_name_from_available_tools</tool_name>
  <parameters>
    <parameter>
      <name>param_name</name>
      <value>param_value</value>
    </parameter>
    <!-- Repeat for each parameter -->
  </parameters>
  <reason>Reason for selecting this SINGLE tool from the `AVAILABLE TOOLS` list</reason>
</selection>
"""

# Observation prompt - used by the ObservationAgent to analyze tool results
OBSERVATION_PROMPT = """
You are an expert in analyzing tool execution results **in the context of a specific task**. Your goal is to extract and summarize only the information from the tool's output that is directly relevant to achieving the task described. Filter out irrelevant details.

Task Description:
{description}

Tool Execution Result:
{tool_result}

Additional Information:
{additional_information}

Task:
- Analyze the 'Tool Execution Result'.
- Identify the parts of the result that directly address or contribute to fulfilling the 'Task Description'.
- Summarize **only this relevant information**. Ignore details from the tool result that do not pertain to the specific 'Task Description'.

Response Format:
<observation>Summary of information relevant to the Task Description</observation>
"""

# Response synthesis prompt - used by the FinalAgent to create the final response
RESPONSE_SYNTHESIS_PROMPT = """
You are an expert in synthesizing information to provide a comprehensive and accurate response to the user.

User Question:
{user_question}

Context and Results from Previous Steps:
{previous_results}

Additional Information:
{additional_information}

Task:
- Analyze all available information.
- Provide a clear, concise, and accurate response that addresses the user's query.

Response Format:
<response>
  <content>Final response content</content>
</response>
"""
