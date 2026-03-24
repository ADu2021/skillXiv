---
name: coact-1-coding-agents
title: CoAct-1 - Computer-using Agents with Coding as Actions
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.03923
keywords: [computer-agents, code-execution, multi-agent, orchestration]
description: "Hybrid multi-agent architecture where orchestrator delegates tasks to GUI Operator or Programmer agent. Coding enables efficiency on computational tasks, achieving 60.76% on OSWorld with 33% fewer steps."
---

# CoAct-1: Computer-using Agents with Coding as Actions

## Core Concept

CoAct-1 overcomes the fundamental limitation of GUI-only computer agents by enabling agents to write and execute code for computational tasks. A central Orchestrator analyzes each subtask and delegates to either a GUI Operator for visual interaction or a Programmer agent that writes executable scripts. This hybrid approach dramatically improves efficiency on complex computer automation tasks.

## Architecture Overview

- **Orchestrator Agent**: Analyzes subtasks and routes to GUI Operator or Programmer
- **GUI Operator Agent**: Interacts with desktop through visual recognition and clicking
- **Programmer Agent**: Writes and executes Python/Bash scripts for computational tasks
- **Task Decomposition**: Breaks complex tasks into actionable subtasks
- **Multi-agent Coordination**: Seamless handoff between interaction modes

## Implementation Steps

### Step 1: Build Task Analyzer and Orchestrator

Create the central coordinator that decides execution strategy for each subtask.

```python
from enum import Enum
from typing import Dict, List, Tuple

class ExecutionMode(Enum):
    GUI = "gui"
    CODE = "code"
    HYBRID = "hybrid"

class TaskOrchestrator:
    """
    Analyzes subtasks and routes to GUI or Programmer agent.
    """

    def __init__(self, model_name="gpt-4"):
        self.model = model_name

    def analyze_subtask(self, subtask: str, context: Dict) -> Tuple[ExecutionMode, str]:
        """
        Determine optimal execution mode for subtask.

        Args:
            subtask: Subtask description
            context: Current context (files, applications, etc.)

        Returns:
            (execution_mode, routing_reason)
        """
        analysis_prompt = f"""
        Subtask: {subtask}
        Current context: {context}

        For this subtask, should we use:
        1. GUI: Click buttons, fill forms, visual interaction
        2. CODE: Write Python/Bash script for automation
        3. HYBRID: Use both approaches

        Consider:
        - Is this a computational/file operation? (favor CODE)
        - Does this need visual interaction? (favor GUI)
        - Can this be automated programmatically? (favor CODE)

        Respond with JSON:
        {{
            "mode": "GUI|CODE|HYBRID",
            "reasoning": "brief explanation",
            "complexity": 1-5
        }}
        """

        response = self.model.generate(analysis_prompt)
        result = self._parse_json_response(response)

        mode_map = {"GUI": ExecutionMode.GUI, "CODE": ExecutionMode.CODE, "HYBRID": ExecutionMode.HYBRID}
        mode = mode_map.get(result["mode"], ExecutionMode.HYBRID)

        return mode, result["reasoning"]

    def decompose_task(self, task: str) -> List[Dict]:
        """
        Break task into subtasks.

        Args:
            task: Main task description

        Returns:
            List of subtasks with context
        """
        decomposition_prompt = f"""
        Main task: {task}

        Break this into 3-8 concrete subtasks that can be executed sequentially.
        For each subtask, identify:
        1. What needs to be done
        2. Prerequisites
        3. Success criteria

        Return as JSON array of subtasks.
        """

        response = self.model.generate(decomposition_prompt)
        subtasks = self._parse_json_response(response)

        return subtasks

    def coordinate_execution(self, task: str, gui_agent, code_agent) -> Dict:
        """
        Orchestrate full task execution.

        Args:
            task: Main task to execute
            gui_agent: GUI interaction agent
            code_agent: Code execution agent

        Returns:
            Execution results
        """
        # Decompose task
        subtasks = self.decompose_task(task)

        execution_log = []
        context = {}

        for idx, subtask_desc in enumerate(subtasks):
            print(f"Subtask {idx + 1}: {subtask_desc['task']}")

            # Analyze how to execute
            mode, reasoning = self.analyze_subtask(subtask_desc["task"], context)

            print(f"  Mode: {mode.value} ({reasoning})")

            # Execute based on mode
            if mode == ExecutionMode.GUI:
                result = gui_agent.execute(subtask_desc["task"], context)

            elif mode == ExecutionMode.CODE:
                result = code_agent.execute(subtask_desc["task"], context)

            elif mode == ExecutionMode.HYBRID:
                # Try code first, fallback to GUI
                try:
                    result = code_agent.execute(subtask_desc["task"], context)
                except Exception as e:
                    print(f"  Code execution failed: {e}. Falling back to GUI.")
                    result = gui_agent.execute(subtask_desc["task"], context)

            # Update context
            context.update(result.get("context_updates", {}))

            execution_log.append({
                "subtask": subtask_desc["task"],
                "mode": mode.value,
                "success": result["success"],
                "result": result
            })

            if not result["success"]:
                print(f"  Failed: {result.get('error', 'Unknown error')}")
                break

        return {
            "task": task,
            "completed": all(log["success"] for log in execution_log),
            "steps": len(execution_log),
            "log": execution_log
        }

    def _parse_json_response(self, response: str) -> Dict:
        """Extract JSON from response."""
        import json
        import re
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}
```

### Step 2: Implement Programmer Agent

Create agent that writes and executes code.

```python
class ProgrammerAgent:
    """
    Executes subtasks by writing and executing code.
    """

    def __init__(self, model_name="gpt-4", sandbox=True):
        self.model = model_name
        self.sandbox = sandbox
        self.execution_history = []

    def execute(self, subtask: str, context: Dict) -> Dict:
        """
        Execute subtask via code generation and execution.

        Args:
            subtask: Task description
            context: Current context (files, variables, etc.)

        Returns:
            Execution result with success flag and outputs
        """
        # Generate code for subtask
        code = self.generate_code(subtask, context)

        if not code:
            return {"success": False, "error": "Could not generate code"}

        # Execute code
        try:
            result = self.execute_code(code, context)
            return {
                "success": True,
                "code": code,
                "output": result,
                "context_updates": self._extract_context_updates(result)
            }
        except Exception as e:
            return {
                "success": False,
                "code": code,
                "error": str(e),
                "context_updates": {}
            }

    def generate_code(self, subtask: str, context: Dict) -> str:
        """
        Generate Python/Bash code for subtask.

        Args:
            subtask: Task description
            context: Available context

        Returns:
            Code string
        """
        code_prompt = f"""
        Task: {subtask}
        Context: {context}

        Generate Python code to accomplish this task.
        The code should:
        1. Be self-contained and executable
        2. Handle errors gracefully
        3. Return results in a structured format

        Code:
        ```python
        # Solution
        ```

        Important: Only return the code block, nothing else.
        """

        response = self.model.generate(code_prompt)

        # Extract code from markdown
        import re
        code_match = re.search(r'```(?:python)?\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)

        return response

    def execute_code(self, code: str, context: Dict) -> Dict:
        """
        Execute generated code safely.

        Args:
            code: Python code to execute
            context: Execution context

        Returns:
            Execution result
        """
        if self.sandbox:
            result = self._execute_sandboxed(code, context)
        else:
            result = self._execute_unsafe(code, context)

        self.execution_history.append({
            "code": code,
            "result": result
        })

        return result

    def _execute_sandboxed(self, code: str, context: Dict) -> Dict:
        """Execute code in sandbox environment."""
        import tempfile
        import subprocess
        import json

        # Create temporary script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # Add context setup
            f.write("import os\nimport json\n")

            # Write user code
            f.write(code)

            script_path = f.name

        try:
            # Execute with timeout
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                timeout=30,
                text=True
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0
            }

        except subprocess.TimeoutExpired:
            return {
                "error": "Code execution timed out",
                "success": False
            }

    def _execute_unsafe(self, code: str, context: Dict) -> Dict:
        """Execute code directly (unsafe, for development)."""
        try:
            exec_globals = {"__builtins__": {}}
            exec_globals.update(context)

            exec(code, exec_globals)

            return {
                "globals": exec_globals,
                "success": True
            }

        except Exception as e:
            return {
                "error": str(e),
                "success": False
            }

    def _extract_context_updates(self, result: Dict) -> Dict:
        """Extract new context from execution result."""
        # Return relevant outputs
        return {
            "last_output": result.get("stdout", ""),
            "last_error": result.get("stderr", "")
        }
```

### Step 3: Implement GUI Operator Agent

Create agent for visual interaction.

```python
class GUIOperatorAgent:
    """
    Executes subtasks via GUI interaction.
    """

    def __init__(self, browser_controller=None):
        self.browser = browser_controller
        self.interaction_log = []

    def execute(self, subtask: str, context: Dict) -> Dict:
        """
        Execute subtask via GUI interaction.

        Args:
            subtask: Task description
            context: Current GUI context

        Returns:
            Execution result
        """
        # Take screenshot
        screenshot = self.browser.take_screenshot()

        # Analyze what to do
        action_plan = self.plan_actions(subtask, screenshot, context)

        # Execute actions
        try:
            for action in action_plan:
                self.execute_action(action, screenshot)
                screenshot = self.browser.take_screenshot()

            return {
                "success": True,
                "final_screenshot": screenshot,
                "actions": action_plan,
                "context_updates": {}
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "context_updates": {}
            }

    def plan_actions(self, subtask: str, screenshot_bytes: bytes, context: Dict) -> List[Dict]:
        """
        Plan GUI actions needed for subtask.

        Args:
            subtask: Task description
            screenshot_bytes: Current screenshot
            context: Current context

        Returns:
            List of actions to execute
        """
        # Analyze screenshot (use vision model)
        visual_analysis = self._analyze_screenshot(screenshot_bytes)

        planning_prompt = f"""
        Subtask: {subtask}
        Current screen elements: {visual_analysis}

        What GUI actions should we take? List specific actions:
        1. find_element(description)
        2. click(element_id)
        3. type(element_id, text)
        4. scroll(direction)

        Return as JSON list of actions.
        """

        # Generate action plan
        response = self.model.generate(planning_prompt)

        actions = self._parse_action_list(response)

        return actions

    def execute_action(self, action: Dict, screenshot: bytes):
        """Execute single GUI action."""
        action_type = action.get("type")

        if action_type == "click":
            self.browser.click_element(action["element_id"])

        elif action_type == "type":
            self.browser.type_text(action["element_id"], action["text"])

        elif action_type == "scroll":
            self.browser.scroll(action["direction"])

    def _analyze_screenshot(self, screenshot_bytes: bytes) -> Dict:
        """Analyze screenshot to identify elements."""
        # Use vision model to detect UI elements
        pass

    def _parse_action_list(self, response: str) -> List[Dict]:
        """Parse action list from model response."""
        import json
        import re
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            return json.loads(match.group())
        return []
```

### Step 4: Integrate Components

Create end-to-end orchestration system.

```python
def run_coact_agent(task: str, browser_controller) -> Dict:
    """
    Run CoAct-1 agent on a task.

    Args:
        task: Task description
        browser_controller: Browser control interface

    Returns:
        Execution results
    """
    # Initialize agents
    orchestrator = TaskOrchestrator()
    programmer = ProgrammerAgent()
    gui_operator = GUIOperatorAgent(browser_controller)

    # Orchestrate execution
    result = orchestrator.coordinate_execution(
        task,
        gui_operator,
        programmer
    )

    # Print summary
    print(f"\nTask: {task}")
    print(f"Result: {'SUCCESS' if result['completed'] else 'FAILED'}")
    print(f"Steps: {result['steps']}")

    return result
```

## Practical Guidance

### When to Use CoAct-1

- **Complex computer automation**: Multi-step tasks requiring both GUI and computation
- **File/data processing**: Scripts handle these more efficiently than GUI clicks
- **Mixed-mode interactions**: Tasks needing visual components and computation
- **Efficiency-critical workflows**: Code execution 10x faster than GUI simulation

### When NOT to Use CoAct-1

- **High visual complexity**: UI changes frequently or requires deep understanding
- **Security restrictions**: Code execution may be restricted
- **Real-time responsiveness**: GUI-only simpler for predictable delays
- **Fully visual tasks**: No computational component to automate

### Hyperparameter Recommendations

- **Code execution timeout**: 30-60 seconds per script
- **Max subtasks**: 10-20 per main task
- **Sandbox restrictions**: Disable file system access if untrusted
- **Vision model**: GPT-4V or Claude Vision for screenshot analysis

### Key Insights

The critical insight is recognizing that many computer automation tasks have computational components that GUI interaction handles inefficiently. By enabling code execution alongside GUI interaction, CoAct-1 exploits this asymmetry. The Orchestrator's routing decision is key: it must identify when coding is faster and safer than GUI simulation.

## Reference

**CoAct-1: Computer-using Agents with Coding as Actions** (arXiv:2508.03923)

Introduces hybrid multi-agent architecture where Programmer writes code for computational tasks while GUI Operator handles visual interaction. Achieves 60.76% on OSWorld with 33% reduction in execution steps through intelligent task routing.
