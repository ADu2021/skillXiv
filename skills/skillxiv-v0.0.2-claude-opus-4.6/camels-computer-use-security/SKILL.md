---
name: camels-computer-use-security
title: "CaMeLs Can Use Computers Too: System-level Security for Computer Use Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.09923"
keywords: [computer-use-agents, prompt-injection, single-shot-planning, control-flow-integrity, security]
description: "Protects computer use agents from prompt injection by using single-shot execution planning that generates complete control flow graphs before UI observation, preventing instruction hijacking while maintaining 57% performance on frontier models."
---

## Overview

Implement a security architecture for agents that control computers through UI interaction. Rather than observing the screen and deciding actions reactively, use single-shot planning where a trusted planner generates the complete execution graph with conditional branches before exposure to potentially malicious UI content.

## When to Use

- For agents that control computer interfaces autonomously
- In adversarial environments where UI content may contain injection attacks
- When protecting against credential theft or financial fraud through UI manipulation
- For high-security applications where execution path integrity is critical

## When NOT to Use

- For interactive agents that must adapt based on real-time UI changes
- When the full task state cannot be predetermined
- For exploratory agents discovering tasks dynamically
- In environments where UI layouts are highly unpredictable

## Key Technical Components

### Single-Shot Execution Planning

Generate complete execution graphs before observing any potentially malicious UI.

```python
# Single-shot plan generation
class ExecutionPlan:
    def __init__(self, task_description):
        self.task = task_description
        self.control_flow_graph = None
        self.decision_points = []

    def generate_plan(self, trusted_context=None):
        """Create complete execution graph with conditional branches"""
        # Generate without UI observation
        plan = self.llm_generate_plan(self.task, context=trusted_context)

        # Parse into control flow graph
        self.control_flow_graph = parse_control_flow(plan)
        self.decision_points = extract_decision_nodes(self.control_flow_graph)

        return self.control_flow_graph

    def get_next_action(self, current_state, observation=None):
        """Execute pre-planned action, ignoring malicious UI content"""
        current_node = self.control_flow_graph.current_node

        # Check if we're at decision point
        if current_node in self.decision_points:
            decision = self.evaluate_decision(current_state)
            return self.control_flow_graph.branch(decision)
        else:
            # Linear path: ignore observation, execute plan
            return self.control_flow_graph.next_action()
```

### Control Flow Integrity Enforcement

Ensure agent cannot deviate from pre-planned execution paths.

```python
# CFI enforcement mechanism
class ControlFlowIntegrity:
    def __init__(self, execution_plan):
        self.plan = execution_plan
        self.current_path = execution_plan.control_flow_graph
        self.executed_actions = []

    def validate_and_execute(self, proposed_action):
        """Verify action matches plan before execution"""
        valid_actions = self.current_path.valid_next_actions()

        if proposed_action not in valid_actions:
            raise ExecutionViolation(
                f"Action {proposed_action} not in pre-planned path"
            )

        # Execute in controlled environment
        result = execute_with_containment(proposed_action)
        self.executed_actions.append((proposed_action, result))
        self.current_path = self.current_path.next(proposed_action)

        return result

    def get_execution_path(self):
        """Return verified execution trace"""
        return self.executed_actions
```

### Conditional Branch Management

Handle dynamic branching within the pre-planned graph.

```python
# Conditional branching
class ConditionalBranch:
    def __init__(self, condition, true_branch, false_branch):
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch

    def evaluate(self, state):
        """Evaluate condition using trusted state, not UI observation"""
        # Use internal state representation, not screen content
        return self.condition.evaluate(state)

    def execute(self, state):
        """Execute correct branch based on condition"""
        if self.evaluate(state):
            return self.true_branch
        else:
            return self.false_branch

# Example: Safe conditional execution
plan = ConditionalBranch(
    condition=lambda state: state["balance"] > 1000,
    true_branch=["withdraw_500", "confirm_transaction"],
    false_branch=["show_insufficient_funds_error"]
)
```

### Branch Steering Attack Detection

Identify UI-based attacks attempting to force unintended branches.

```python
# Attack detection
class BranchSteeringDetector:
    def __init__(self):
        self.expected_outcomes = {}
        self.suspicious_actions = []

    def check_branch_steering(self, proposed_branch, ui_observation):
        """Detect if UI is attempting to steer execution"""
        ui_elements = parse_ui_elements(ui_observation)

        # Check for suspicious UI patterns
        suspicious_patterns = [
            "overlay_elements",
            "hidden_buttons",
            "obfuscated_text",
            "unusual_layout"
        ]

        for pattern in suspicious_patterns:
            if detect_pattern(ui_elements, pattern):
                self.suspicious_actions.append({
                    "timestamp": time.now(),
                    "pattern": pattern,
                    "observation": ui_observation
                })
                return True

        return False

    def get_threat_assessment(self):
        """Assess likelihood of active steering attack"""
        if len(self.suspicious_actions) > THRESHOLD:
            return "high_risk"
        return "normal"
```

### Trusted State Management

Maintain internal state representation separate from potentially malicious UI.

```python
# Trusted internal state
class TrustedState:
    def __init__(self):
        self.internal_model = {}
        self.ui_observation = None

    def update_from_reliable_source(self, source, data):
        """Update state from trusted sources only"""
        if is_trusted_source(source):
            self.internal_model.update(data)
        else:
            # Log but don't trust
            self.log_untrusted_update(source, data)

    def evaluate_condition(self, condition):
        """Use internal model, not UI observation"""
        return condition(self.internal_model)

    def observe_ui(self, screenshot):
        """Store UI observation for logging, not for decision-making"""
        self.ui_observation = screenshot
        # Do NOT update internal state based on UI observation
```

## Performance Characteristics

- Frontier models (GPT-5.2, etc.): 57% task success rate
- Open-source models: 19-25% improvement with added security
- Security guarantee: No execution path deviations due to UI content
- Planning overhead: ~2-5 seconds per task

## Integration Pattern

1. Receive task description from trusted source
2. Generate complete execution plan without UI observation
3. Extract control flow graph with decision points
4. Begin task, following pre-planned execution
5. At decision points, evaluate using internal trusted state
6. Detect and log branch steering attempts
7. Complete execution, staying on planned path

## Limitations & Mitigations

**Challenge**: Some tasks require adaptive UI navigation
**Mitigation**: Expand planning to include exploration branches for known UI patterns

**Challenge**: Complex UI layouts make precise planning difficult
**Mitigation**: Combine with UI templates and bounded exploration

**Challenge**: Balancing security with flexibility
**Mitigation**: Use tiered trust levels (high-trust paths execute deterministically, lower-trust require validation)

## References

- Prompt injection through UI content is a critical threat for computer use agents
- Single-shot planning provides architectural isolation against injection
- Control flow integrity ensures execution follows pre-planned paths
- Branch steering attacks require supplementary UI validation
