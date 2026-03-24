---
name: computer-use-hybrid-actions
title: "UltraCUA: A Foundation Model for Computer Use Agents with Hybrid Action"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.17790"
keywords: [computer use, hybrid actions, GUI automation, tool calling, agent framework]
description: "Enable computer-use agents to flexibly choose between GUI primitives (click, type) and high-level tool calls, reducing cascading errors by 22% and improving execution speed by 11%."
---

# Technique: Hybrid Actions for Computer-Use Agents

Traditional computer-use agents rely exclusively on low-level GUI primitives (click, type, scroll), which creates fragile execution chains vulnerable to UI detection errors. One wrong click location cascades into multiple failures. UltraCUA solves this by enabling agents to **dynamically choose between GUI primitives and direct API tool calls**, bypassing brittle UI interactions when more reliable tool access is available.

The key insight is that agents should have options: when GUI interaction is unreliable or inefficient, fall back to tool calls; when direct tool access is unavailable, use GUI. This flexibility reduces error cascades while maintaining generality.

## Core Concept

Hybrid actions operate on three principles:
- **Flexible Action Space**: Agents can choose to invoke primitive GUI actions OR high-level tool APIs
- **Error Recovery**: Failed GUI actions → fallback to tool calls instead of cascading errors
- **Efficiency**: Some tasks complete faster via tool calls (API call < 5 UI steps)
- **Integration**: Unified decision-making: "should I click this button or call the API directly?"

The result is 22% relative gains on OSWorld and 11% faster execution by intelligently routing to the right action type.

## Architecture Overview

- **GUI Environment Simulator**: Provide screenshots and UI element coordinates
- **Tool Registry**: Catalog of available APIs (from documentation/GitHub)
- **Action Router**: LLM decides action type (GUI primitive vs tool call)
- **GUI Executor**: Handle click, type, scroll actions
- **Tool Executor**: Invoke APIs with parameter binding
- **State Monitor**: Track execution history, detect failures
- **Feedback Loop**: Learn which action types work best for which subtasks

## Implementation Steps

The core decision point is the action router: when should the agent invoke a tool vs interact with the GUI? This example shows the hybrid action framework.

```python
from typing import Union, List, Dict, Literal
from dataclasses import dataclass

@dataclass
class GUIPrimitive:
    """Low-level GUI action."""
    action_type: Literal["click", "type", "scroll"]
    coordinates: tuple = None  # (x, y) for click
    text: str = None           # for type
    direction: str = None      # "up" or "down" for scroll
    times: int = 1             # repeat count

@dataclass
class ToolCall:
    """High-level tool API invocation."""
    tool_name: str
    function_name: str
    parameters: Dict[str, any]
    description: str

HybridAction = Union[GUIPrimitive, ToolCall]


class HybridActionRouter:
    """
    Decide whether to use GUI primitive or tool call for each action.
    """

    def __init__(self, model, tool_registry: Dict[str, Dict]):
        self.model = model
        self.tools = tool_registry  # {tool_name: {functions: {...}}}
        self.execution_history = []

    def build_action_prompt(
        self,
        current_screenshot: str,
        current_state: str,
        next_goal: str,
        available_tools: List[str]
    ) -> str:
        """
        Format prompt for LLM to decide action type.
        """
        prompt = f"""
You are a computer-use agent. You can perform two types of actions:

1. GUI PRIMITIVES (if GUI interaction is necessary):
   - click(x, y): Click at coordinates
   - type(text): Type text
   - scroll(direction, times): Scroll up/down

2. TOOL CALLS (if direct API access is available):
   - Call functions from: {', '.join(available_tools)}

Current Screenshot:
{current_screenshot}

Current State: {current_state}

Next Goal: {next_goal}

Available Tools and their functions:
{self.format_tool_catalog(available_tools)}

Decide which action to take. For TOOL CALLS, provide the exact function name and parameters.
For GUI PRIMITIVES, provide the exact coordinates or text.

Respond in JSON format:
{{
  "action_type": "gui_primitive" or "tool_call",
  "decision_reasoning": "Why this action?",
  "action": {{...action details...}}
}}
"""
        return prompt

    def decide_action(
        self,
        screenshot: str,
        state: str,
        goal: str,
        available_tools: List[str]
    ) -> HybridAction:
        """
        Use LLM to decide between GUI primitive and tool call.
        """
        prompt = self.build_action_prompt(screenshot, state, goal, available_tools)

        response = self.model.generate(prompt)
        decision = parse_json_response(response)

        if decision["action_type"] == "gui_primitive":
            action = GUIPrimitive(
                action_type=decision["action"]["action_type"],
                coordinates=decision["action"].get("coordinates"),
                text=decision["action"].get("text"),
                direction=decision["action"].get("direction")
            )
        else:
            action = ToolCall(
                tool_name=decision["action"]["tool_name"],
                function_name=decision["action"]["function_name"],
                parameters=decision["action"]["parameters"],
                description=decision["decision_reasoning"]
            )

        self.execution_history.append({
            "goal": goal,
            "action": action,
            "reasoning": decision["decision_reasoning"]
        })

        return action

    def format_tool_catalog(self, available_tools: List[str]) -> str:
        """
        Format available tools and their functions as readable text.
        """
        catalog = []
        for tool_name in available_tools:
            if tool_name in self.tools:
                funcs = self.tools[tool_name].get("functions", {})
                for func_name, func_spec in funcs.items():
                    catalog.append(
                        f"  {tool_name}.{func_name}: {func_spec.get('description', 'No description')}"
                    )
        return "\n".join(catalog)


class ExecutorWithFallback:
    """
    Execute hybrid actions with fallback: if GUI action fails, try tool call.
    """

    def __init__(self, gui_executor, tool_executor, router: HybridActionRouter):
        self.gui = gui_executor
        self.tools = tool_executor
        self.router = router

    def execute_action(self, action: HybridAction) -> Dict:
        """
        Execute action with fallback logic.
        """
        if isinstance(action, GUIPrimitive):
            # Try GUI action
            try:
                result = self.gui.execute(action)
                if result["success"]:
                    return result
            except Exception as e:
                print(f"GUI action failed: {e}. Attempting fallback to tool...")

                # Fallback: suggest tool-based alternative
                if hasattr(self.router, 'execution_history'):
                    history = self.router.execution_history[-1]
                    # Trigger re-routing to tool call
                    pass

        elif isinstance(action, ToolCall):
            # Execute tool call
            try:
                result = self.tools.invoke(
                    tool_name=action.tool_name,
                    function_name=action.function_name,
                    parameters=action.parameters
                )
                return {
                    "success": True,
                    "output": result,
                    "action_type": "tool_call"
                }
            except Exception as e:
                print(f"Tool call failed: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "fallback_to_gui": True
                }

        return {"success": False}


def computer_use_agent_with_hybrid_actions(
    initial_task: str,
    router: HybridActionRouter,
    executor: ExecutorWithFallback,
    max_steps: int = 20
):
    """
    Execute task using hybrid action routing.
    """
    current_state = get_initial_state()
    remaining_steps = max_steps

    for step in range(max_steps):
        # Get current screenshot
        screenshot = get_screenshot(current_state)

        # Decide next action
        action = router.decide_action(
            screenshot=screenshot,
            state=current_state,
            goal=initial_task,
            available_tools=["browser", "api_client", "file_system"]
        )

        # Execute with fallback
        result = executor.execute_action(action)

        if not result["success"]:
            print(f"Step {step}: Action failed")
            remaining_steps -= 1
            if remaining_steps <= 0:
                break
        else:
            print(f"Step {step}: {action} -> {result['output']}")
            current_state = update_state(current_state, result)

    return current_state
```

The key insight is teaching agents to **compare action types**: is this task easier via GUI or tool? For example, "open a file" might be easier via click→navigate dialog or directly via file_system.read_file(). Let the agent decide.

## Practical Guidance

| Task Type | GUI Primitives | Tool Calls | Hybrid Win |
|-----------|---|---|---|
| Fill form field | 3-5 steps | 1 API call | +40% |
| Search and click | 2-3 steps | 1 API call | +30% |
| Navigate pages | 5-8 steps | Direct access | +50% |

**When to Use:**
- Complex computer-use tasks where GUI can fail
- Mix of UI-dependent and API-available functionality
- Error recovery is important (cascading UI failures)
- You have tool/API documentation available

**When NOT to Use:**
- GUI-only interfaces (no tools/APIs available)
- Real-time interactive tasks requiring UI feedback
- Tasks where tool invocation requires manual setup

**Common Pitfalls:**
- Router indecisive → falls back too often
- Tool documentation incomplete → function calls fail
- GUI executor unreliable → defeats hybrid advantage
- Not tracking which action type succeeds (lose learning signal)
- Overly complex tool registry → router confused by too many options

## Reference

[UltraCUA: A Foundation Model for Computer Use Agents with Hybrid Action](https://arxiv.org/abs/2510.17790)
