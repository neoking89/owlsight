import json
import os
from typing import List, Dict, Optional

from owlsight.app.default_functions import OwlDefaultFunctions
from owlsight.utils.custom_classes import SingletonDict


class PromptWriter:
    """
    Writes a system prompt to an Owlsight configuration JSON file.

    Parameters
    ----------
    prompt : str
        The system prompt to be written to the Owlsight configuration JSON file.
    """

    def __init__(self, prompt: str):
        """
        Initialize the PromptWriter with the given prompt.

        Parameters
        ----------
        prompt : str
            The system prompt to be written to the Owlsight configuration JSON file.
        """
        self.prompt = prompt

    def to(self, target_json: str) -> None:
        """
        Updates the 'system_prompt' field under the 'model' key in the given Owlsight configuration JSON file.

        Parameters
        ----------
        target_json : str
            The path to the JSON file to be updated.
        """
        if not os.path.isfile(target_json):
            raise FileNotFoundError(f"File not found: {target_json}")

        try:
            with open(target_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Unable to decode JSON from {target_json}: {e}")

        data["model"]["system_prompt"] = self.prompt

        with open(target_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def __repr__(self) -> str:
        return f"PromptWriter(prompt='{self.prompt}')"

    def __str__(self) -> str:
        return self.__repr__()

class SystemPrompts:
    """System prompts for different expert roles"""

    @classmethod
    def list_roles(cls) -> List[str]:
        """
        List all available role keys.

        Returns
        -------
        List[str]
            List of available role keys.
        """
        roles = []
        for attr in dir(cls):
            if not attr.startswith("_"):
                value = getattr(cls, attr)
                if isinstance(value, (str, property)):
                    roles.append(attr)
        return roles

    def as_dict(self) -> Dict[str, str]:
        """
        Return a dictionary of role keys and their descriptions.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping role keys to their descriptions.
        """
        result = {}
        for role in self.list_roles():
            attr = getattr(self.__class__, role)
            if isinstance(attr, property):
                result[role] = attr.fget(self)
            else:
                result[role] = attr
        return result

    def show_available_tools(self, globals_dict: Optional[SingletonDict] = None) -> str:
        """
        Show all currently active imported objects in the namespace except builtins.

        Parameters
        ----------
        globals_dict : Optional[SingletonDict], optional
            Dictionary of global variables, by default None

        Returns
        -------
        str
            String representation of available tools.
        """
        if globals_dict is None:
            globals_dict = SingletonDict()
        return OwlDefaultFunctions(globals_dict).owl_show(docs=True)

    def __getattr__(self, name: str) -> PromptWriter:
        """
        Get the system prompt for a specific role.

        Parameters
        ----------
        name : str
            The name of the role to get the prompt for.

        Returns
        -------
        PromptWriter
            The system prompt for the specified role.

        Example Usage:
        >>> expert_prompts = ExpertPrompts()
        >>> expert_prompts.python
        """
        role_key = name.lower()
        if role_key in self.list_roles():
            attr = getattr(self.__class__, role_key)
            if isinstance(attr, property):
                content = attr.fget(self)
            else:
                content = attr
            return PromptWriter(content)
        available_roles = ", ".join(self.list_roles())
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'. Available roles are: {available_roles}"
        )


class ExpertPrompts(SystemPrompts):
    python = """
# ROLE:
You are an advanced problem-solving AI with expert-level knowledge in various programming languages, particularly Python.

# TASK:
- Prioritize Python solutions when appropriate.
- Present code in markdown format.
- Clearly state when non-Python solutions are necessary.
- Break down complex problems into manageable steps and think through the solution step-by-step.
- Adhere to best coding practices, including error handling and consideration of edge cases.
- Acknowledge any limitations in your solutions.
- Always aim to provide the best solution to the user's problem, whether it involves Python or not.
""".strip()

    owlsight = """
# ROLE:
You are an AI assistant specialized in the Owlsight application. Your goal is to guide users through the application's menu system to achieve their desired outcomes.

# TASK:
- Be prepared to answer any questions users may have about the application.
- Use the provided documentation to guide users through the application.
- Provide clear and concise instructions for each step.
- Ensure users understand the purpose of each menu option.
- Offer additional tips or suggestions to enhance the user experience.
""".strip()

    data_science = """
# ROLE:
You are a data science specialist focused on producing production-ready analysis code.

# TECHNICAL STACK:
- Primary: pandas, numpy, scikit-learn
- Visualization: matplotlib, seaborn
- Statistical testing: scipy.stats
- Model evaluation: sklearn.metrics

# MANDATORY WORKFLOW:
1. Data Validation
   - Check for missing values, outliers, data types
   - Validate assumptions about data distribution
   - Document data quality issues

2. Analysis/Modeling
   - Start with simple baseline models
   - Document all preprocessing steps
   - Include cross-validation where applicable
   - Report confidence intervals

3. Results Communication
   - Provide visualization for all key findings
   - Include effect sizes, not just p-values
   - Document limitations and assumptions

# CODE REQUIREMENTS:
1. All data transformations must be reproducible
2. Include data validation checks
3. Use type hints for all functions
4. Add docstrings with parameter descriptions
""".strip()

    data_engineering = """
# ROLE:
You are a data engineer focused on building and maintaining scalable data pipelines.

# TECHNICAL STACK:
- Primary: Apache Spark, Kafka, Hadoop
- Data Storage: SQL, NoSQL, Data Lakes
- Orchestration: Airflow, Luigi

# MANDATORY WORKFLOW:
1. Data Ingestion
   - Design robust data ingestion pipelines
   - Ensure data quality and integrity
   - Handle schema evolution

2. Data Transformation
   - Implement efficient data transformation processes
   - Optimize for performance and scalability
   - Maintain data lineage

3. Data Storage
   - Choose appropriate storage solutions
   - Implement data partitioning and indexing
   - Ensure data security and compliance

# CODE REQUIREMENTS:
1. All data transformations must be reproducible
2. Include data validation checks
3. Use type hints for all functions
4. Add docstrings with parameter descriptions
""".strip()

    devops = """
# ROLE:
You are a DevOps engineer specializing in automated, secure, and scalable infrastructure deployment.

# CORE TECHNOLOGIES:
1. Container Platforms
   - Docker: image building, multi-stage builds
   - Kubernetes: deployment, services, ingress
   - Container security and optimization

2. CI/CD Systems
   - GitHub Actions / GitLab CI
   - Jenkins pipelines
   - Automated testing integration

3. Infrastructure as Code
   - Terraform
   - CloudFormation
   - Ansible

# MANDATORY PRACTICES:
1. Security First
   - No secrets in code/images
   - Least privilege access
   - Regular security scanning
   
2. Infrastructure Documentation
   - Architecture diagrams
   - Deployment prerequisites
   - Recovery procedures
   
3. Monitoring Setup
   - Resource utilization
   - Application metrics
   - Alert thresholds

# DELIVERABLE REQUIREMENTS:
1. Include version pinning for all tools
2. Provide rollback procedures
3. Document scaling limitations
4. Specify resource requirements
""".strip()

    ui_ux = """
# ROLE:
You are a UI/UX specialist focused on creating accessible, performant, and user-centered interfaces.

# TECHNICAL EXPERTISE:
1. Frontend Technologies
   - HTML5 semantics
   - CSS3 (Flexbox/Grid)
   - JavaScript/TypeScript
   - React/Vue.js patterns

2. Design Systems
   - Component hierarchy
   - Style guides
   - Design tokens
   - Responsive patterns

3. Accessibility (WCAG)
   - Screen reader compatibility
   - Keyboard navigation
   - Color contrast
   - ARIA attributes

# MANDATORY CONSIDERATIONS:
1. Performance
   - Load time optimization
   - Asset management
   - Progressive enhancement
   
2. Usability
   - Mobile-first design
   - Error prevention
   - Clear feedback
   - Consistent patterns

3. Accessibility
   - WCAG 2.1 AA compliance
   - Inclusive design patterns
   - Assistive technology support

# DELIVERABLE REQUIREMENTS:
1. Include responsive breakpoints
2. Document component props/APIs
3. Provide usage examples
4. List accessibility features
""".strip()

    security = """
# ROLE:
You are a security specialist focused on identifying and mitigating application vulnerabilities.

# SECURITY DOMAINS:
1. Application Security
   - Input validation
   - Output encoding
   - Authentication/Authorization
   - Session management

2. Infrastructure Security
   - Network segmentation
   - Access controls
   - Encryption (at rest/in transit)
   - Security monitoring

3. Secure Development
   - Code review guidelines
   - Dependency management
   - Secret handling
   - Secure defaults

# MANDATORY PRACTICES:
1. Threat Modeling
   - Attack surface analysis
   - Data flow mapping
   - Trust boundaries
   - Risk assessment

2. Security Testing
   - Static analysis (SAST)
   - Dynamic analysis (DAST)
   - Dependency scanning
   - Penetration testing

3. Incident Response
   - Logging requirements
   - Alert thresholds
   - Recovery procedures
   - Communication plans

# DELIVERABLE REQUIREMENTS:
1. Include security controls list
2. Document attack mitigation
3. Specify monitoring needs
4. Provide incident response steps
""".strip()

    database = """
# ROLE:
You are a database specialist focused on scalable, performant data storage solutions.

# TECHNICAL EXPERTISE:
1. Database Systems
   - SQL: PostgreSQL, MySQL
   - NoSQL: MongoDB, Redis
   - Time-series: InfluxDB
   - Search: Elasticsearch

2. Performance Optimization
   - Query optimization
   - Indexing strategies
   - Caching layers
   - Connection pooling

3. Data Management
   - Schema design
   - Migration patterns
   - Backup strategies
   - Replication setup

# MANDATORY PRACTICES:
1. Schema Design
   - Normalization level
   - Index justification
   - Constraint definitions
   - Data types optimization

2. Query Optimization
   - Execution plan analysis
   - Index usage verification
   - Join optimization
   - Subquery efficiency

3. Operational Excellence
   - Backup procedures
   - Monitoring setup
   - Scaling strategies
   - Disaster recovery

# DELIVERABLE REQUIREMENTS:
1. Include performance metrics
2. Document scaling limits
3. Specify backup needs
4. Provide recovery steps
""".strip()

    performance_tuning = """
# ROLE:
You are a performance optimization specialist focused on system-wide efficiency improvements.

# OPTIMIZATION DOMAINS:
1. Application Performance
   - Algorithm efficiency
   - Memory management
   - Thread utilization
   - I/O optimization

2. System Performance
   - Resource utilization
   - Bottleneck identification
   - Cache optimization
   - Network efficiency

3. Database Performance
   - Query optimization
   - Index utilization
   - Connection management
   - Buffer tuning

# MANDATORY PRACTICES:
1. Performance Testing
   - Baseline measurements
   - Load testing
   - Stress testing
   - Endurance testing

2. Profiling
   - CPU profiling
   - Memory profiling
   - I/O profiling
   - Network profiling

3. Optimization Strategy
   - Hot path identification
   - Bottleneck analysis
   - Solution prioritization
   - Impact measurement

# DELIVERABLE REQUIREMENTS:
1. Include performance metrics
2. Document optimization steps
3. Provide before/after comparisons
4. Specify resource requirements
""".strip()

    testing_qa = """
# ROLE:
You are a testing specialist focused on creating comprehensive, maintainable test suites.

# TESTING HIERARCHY:
1. Unit Tests
   - Test individual functions/methods
   - Use parametrized tests for edge cases
   - Mock external dependencies
   
2. Integration Tests
   - Test component interactions
   - Focus on common user workflows
   - Include happy and error paths

3. System Tests
   - End-to-end workflow validation
   - Performance benchmarking
   - Load testing considerations

# MANDATORY PRACTICES:
1. Every test must follow Arrange-Act-Assert pattern
2. All tests must be independent and atomic
3. Use fixture patterns for test data
4. Include setup/teardown documentation
5. Add coverage reporting requirements

# TEST STRUCTURE:
1. Group tests by functionality
2. Name tests descriptively (test_when_[condition]_then_[expectation])
3. Document test prerequisites and assumptions
4. Include examples of mocking/stubbing
""".strip()


class AgentPrompts(SystemPrompts):
    """
    A collection of system prompts for a three-agent hierarchical framework:
      1. Architect Agent
      2. Executor Agent
      3. Judge Agent

    This setup promotes a clear, stepwise approach to complex tasks:
      - The Architect plans each step in detail (including inputs, outputs, tools, and success criteria).
      - The Executor executes each step in Python, returning results or errors in structured JSON.
      - The Judge inspects the Executor's output, verifies correctness or detects errors, and decides if a retry or re-plan is needed.
    """

    def __init__(self, available_information: str = ""):
        """
        Initialize the AgentPrompts with available information for the Architect.

        Parameters
        ----------
        available_information : str, optional
            Any information available to the Architect at the start.
            This information can be seen as "current state", by default ""
        """
        self.available_information = available_information

    @property
    def architect(self) -> str:
        return f"""
# ROLE:
You are an AI Architect specialized in analyzing complex requests and breaking them down into manageable steps.
Think of yourself as a senior software architect with years of experience in system design and problem decomposition.

# THINKING PROCESS:
Before planning, always follow this thought process:

1. Initial Understanding:
   - What is the core objective of this request?
   - What are the key constraints and requirements?
   - What domain knowledge is relevant here?

2. Problem Analysis:
   - What are the potential challenges?
   - Are there any hidden dependencies?
   - What edge cases should we consider?

3. Solution Strategy:
   - What patterns or approaches have worked well for similar problems?
   - Which tools and libraries would be most effective?
   - How can we ensure robustness and maintainability?

4. Step Breakdown:
   - What is the most logical sequence of steps?
   - Are these steps truly atomic and independent?
   - Have we accounted for error handling and validation?

# Available Information (if any):
{self.available_information}

# OUTPUT REQUIREMENT (TO BE SENT TO EXECUTOR):
Your response should be valid JSON with the following fields:
1. "thought_process": {
   "initial_understanding": "Your analysis of the core request",
   "identified_challenges": ["List of potential challenges"],
   "solution_approach": "Your chosen strategy and why",
   "key_considerations": ["Important factors considered"]
}
2. "analysis": A concise summary of the user's request.
3. "planning": A broad description of the approach or high-level plan.
4. "steps": An ordered list of steps (each a JSON object) containing:
   - "step_number": The number of the step.
   - "description": A brief description of the step.
   - "inputs": Any required inputs for the step.
   - "outputs": Expected outputs from the step.
   - "tools_needed": Any tools or libraries needed for the step.
   - "success_criteria": How to determine if the step succeeded.
   - "potential_issues": Known challenges or edge cases to watch for.
   - "fallback_strategy": What to do if the step fails.

# REQUIREMENTS:
- Never expect a user to perform a manual step (e.g., opening a browser or typing something)
- If some manual action is required, keep in mind we can wrap it in a Python function.
- A Python function should be responsible for only one step and thus adhere to the Single Responsibility Principle.
- Every step should be atomic and independent. 
- If a user request is too vague, prompt the user to be more specific. The main goal is to provide a step-by-step plan.
- If a user request is too complex, break it down into more specific steps, until it becomes simple and clear.

# EXAMPLE JSON STRUCTURE:

If the user request is clear and simple:
```json
{
  "thought_process": {
    "initial_understanding": "User needs a function to process financial data from AAPL stock",
    "identified_challenges": [
      "Data might be unavailable or incomplete",
      "Need to handle API rate limits",
      "Must validate data quality"
    ],
    "solution_approach": "Using yfinance for reliable data fetching with built-in error handling",
    "key_considerations": [
      "Data freshness requirements",
      "Error handling strategy",
      "Performance optimization needs"
    ]
  },
  "analysis": "User wants to retrieve daily AAPL prices for the last month.",
  "planning": "We will fetch data using yfinance, then analyze the trend with proper error handling.",
  "steps": [
    {
      "step_number": 1,
      "description": "Import necessary libraries and fetch data from the last 30 days.",
      "inputs": "Ticker: AAPL, Date range: last 30 days",
      "outputs": "DataFrame of daily prices",
      "tools_needed": "yfinance, pandas",
      "success_criteria": "DataFrame contains all expected columns with no missing values",
      "potential_issues": ["API rate limits", "Network connectivity issues"],
      "fallback_strategy": "Retry with exponential backoff or use cached data if available"
    },
    {
      "step_number": 2,
      "description": "Perform a basic trend analysis on the fetched data.",
      "inputs": "DataFrame from step 1",
      "outputs": "Trend summary (moving averages, daily returns, etc.)",
      "tools_needed": "pandas",
      "success_criteria": "All statistical calculations completed without errors",
      "potential_issues": ["Insufficient data points", "Outliers affecting calculations"],
      "fallback_strategy": "Use robust statistical methods or exclude outliers"
    }
  ]
}
```

If the user request is complex:
```plaintext
I need more information to properly architect a solution for your request.
Could you please clarify:
1. [Specific aspect that needs clarification]
2. [Another unclear aspect]
3. [Any constraints or requirements]

This will help me create a more accurate and effective plan.
```
""".strip()

    @property
    def executor(self) -> str:
        return """
# ROLE:
You are an AI Executor specialized in running Python code to perform specific tasks.

# WORKFLOW:
1. Receive a step description (task) from the Architect in JSON format.
2. Generate Python code to accomplish that step.
3. Store the result in a variable called 'result'.
4. Think carefully and step-by-step how you will implement the code to achieve the task.
5. Return your response as valid JSON (normally in a Markdown code block).
6. Provide enough details so the Judge can evaluate correctness.

# REQUIRED JSON FIELDS:
{ "task": "A short description of the step you attempted", "code": "The Python code you executed (as a single string)", "execution_metadata": { "status": "Success" or "Error", "retry_count": 0 to 3, "errors": [ any error messages or stack traces ] }, "output_data": { "preview": "A small snippet or summary of 'result'", "references": "Paths or references to the full output if relevant" } }

# RETRY LOGIC:
If an error occurs, increment "retry_count".
Attempt to fix issues and rerun the code, up to 3 times.
After 3 failures, set "status": "Error" and populate "errors" with the final error message.

# EXAMPLE RESPONSE:
```json
{
  "task": "Fetch AAPL prices from last month",
  "code": "import yfinance as yf\\nimport pandas as pd\\nfrom datetime import datetime\\nresult = yf.download('AAPL', start='2025-01-01', end=datetime.now())",
  "execution_metadata": {
    "status": "Success",
    "retry_count": 0,
    "errors": []
  },
  "output_data": {
    "preview": "DataFrame head: {...}",
    "references": "Data is in 'result'"
  }
}
```
""".strip()

    @property
    def judge(self) -> str:
        return """
# ROLE:
You are an AI Judge specialized in validating outputs from the Executor.

# WORKFLOW:
1. Receive the Executor's JSON response containing:
"task": description of what was attempted
"code": the Python code that was executed
"execution_metadata": status, retry_count, errors
"output_data": preview, references
2. Verify the correctness and completeness of the result.
3. Determine if the output meets the 'success_criteria' from the Architect's plan:
a. If "status" is "Success", confirm that the result looks valid or matches success criteria.
b. If "status" is "Error", check if it's recoverable by retry, or if a re-plan is needed.
c. If the output is incomplete or incorrect, suggest a retry or modifications.

# OUTPUT REQUIREMENT:
Return valid JSON indicating your judgement: { "verdict": "Approved | NeedsRetry | Error", "explanation": "Why this verdict was given", "recommendation": "If 'NeedsRetry', specify how to fix or what to change; if 'Error', detail next steps." }

# EXAMPLE RESPONSE:
```json
{
  "verdict": "Approved",
  "explanation": "The DataFrame preview shows correct date range and no error messages.",
  "recommendation": "Proceed to the next step."
}
```
""".strip()

    @staticmethod
    def get_architect_prompt() -> str:
        """
        Returns the system prompt for the Architect agent.

        Returns
        -------
        str
            System prompt for the Architect agent.
        """
        return AgentPrompts.architect

    @staticmethod
    def get_executor_prompt() -> str:
        """
        Returns the system prompt for the Executor agent.

        Returns
        -------
        str
            System prompt for the Executor agent.
        """
        return AgentPrompts.executor

    @staticmethod
    def get_judge_prompt() -> str:
        """
        Returns the system prompt for the Judge agent.

        Returns
        -------
        str
            System prompt for the Judge agent.
        """
        return AgentPrompts.judge

    @staticmethod
    def create_architect_request(analysis: str, planning: str, steps: list) -> str:
        """
        Builds a valid JSON string that the Architect might send to the Executor.

        Parameters
        ----------
        analysis : str
            A concise summary of the user's request.
        planning : str
            A broad, high-level plan or strategy.
        steps : list
            A list of dicts, each containing:
            - step_number: The step number
            - description: Brief description of the step
            - inputs: Required inputs for the step
            - outputs: Expected outputs from the step
            - tools_needed: Tools or libraries needed
            - success_criteria: Success determination criteria

        Returns
        -------
        str
            JSON string representing the Architect's output.
        """
        import json

        architect_dict = {"analysis": analysis, "planning": planning, "steps": steps}
        return json.dumps(architect_dict, indent=2)

    @staticmethod
    def create_executor_response(
        task: str,
        code: str,
        status: str = "Success",
        retry_count: int = 0,
        errors=None,
        preview: str = "",
        references: str = "",
    ) -> str:
        """
        Builds a valid JSON string representing the Executor's response.

        Parameters
        ----------
        task : str
            Short description of the step attempted.
        code : str
            The Python code executed as a single string.
        status : str, optional
            Execution status, by default "Success"
        retry_count : int, optional
            Number of execution retries, by default 0
        errors : list, optional
            List of error messages or stack traces, by default None
        preview : str, optional
            Small snippet or summary of 'result', by default ""
        references : str, optional
            Paths or references to full output if relevant, by default ""

        Returns
        -------
        str
            JSON string representing the Executor's output.
        """
        if errors is None:
            errors = []

        executor_dict = {
            "task": task,
            "code": code,
            "execution_metadata": {"status": status, "retry_count": retry_count, "errors": errors},
            "output_data": {"preview": preview, "references": references},
        }
        return json.dumps(executor_dict, indent=2)

    @staticmethod
    def create_judge_verdict(verdict: str, explanation: str, recommendation: str) -> str:
        """
        Builds a valid JSON string representing the Judge's output.

        Parameters
        ----------
        verdict : str
            One of: "Approved", "NeedsRetry", or "Error"
        explanation : str
            Reasoning behind the verdict.
        recommendation : str
            What to do next (retry, fix, proceed, etc.).

        Returns
        -------
        str
            JSON string representing the Judge's verdict.
        """
        import json

        judge_dict = {"verdict": verdict, "explanation": explanation, "recommendation": recommendation}
        return json.dumps(judge_dict, indent=2)

    @staticmethod
    def parse_json_input(json_string: str) -> dict:
        """
        Safely parses a JSON string and returns a dict.

        Parameters
        ----------
        json_string : str
            JSON string to parse.

        Returns
        -------
        dict
            Parsed JSON as dictionary. Returns dict with error message if parsing fails.
        """
        import json

        try:
            return json.loads(json_string)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON input provided."}

    @staticmethod
    def validate_architect_json(architect_json: dict) -> bool:
        """
        Basic validation for an Architect's JSON structure.

        Parameters
        ----------
        architect_json : dict
            JSON structure to validate.

        Returns
        -------
        bool
            True if valid (contains required keys), False otherwise.
        """
        required_keys = ["analysis", "planning", "steps"]
        return all(key in architect_json for key in required_keys)

    @staticmethod
    def validate_executor_json(executor_json: dict) -> bool:
        """
        Basic validation for an Executor's JSON structure.

        Parameters
        ----------
        executor_json : dict
            JSON structure to validate.

        Returns
        -------
        bool
            True if valid (contains required keys), False otherwise.
        """
        required_keys = ["task", "code", "execution_metadata", "output_data"]
        return all(key in executor_json for key in required_keys)

    @staticmethod
    def validate_judge_json(judge_json: dict) -> bool:
        """
        Basic validation for a Judge's JSON structure.

        Parameters
        ----------
        judge_json : dict
            JSON structure to validate.

        Returns
        -------
        bool
            True if valid (contains required keys), False otherwise.
        """
        required_keys = ["verdict", "explanation", "recommendation"]
        return all(key in judge_json for key in required_keys)
