---
name: toolsafe-agent-safety
title: "ToolSafe: Enhancing Tool Invocation Safety of LLM-based Agents via Proactive Step-level Guardrails"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.10156"
keywords: [agent-safety, tool-invocation, guardrails, prompt-injection, risk-detection]
description: "Detects and prevents unsafe tool invocations in LLM agents through proactive step-level guardrails, reducing harmful tool calls by 65% while improving task success rates by 10% against prompt injection attacks."
---

## Overview

Implement a safety layer that monitors LLM agent tool invocations in real-time and prevents potentially harmful actions before execution. The system analyzes interaction history to identify risky requests and tool-usage patterns that could lead to data exfiltration, privilege escalation, or prompt injection attacks.

## When to Use

- For agents that invoke external tools (file systems, APIs, databases)
- When operating in untrusted or adversarial environments
- For systems where tool misuse could cause financial, privacy, or security harm
- When agents interact with user-supplied prompts or content

## When NOT to Use

- For sandboxed environments with no external tool access
- When tool invocation is fully controlled by trusted operators
- For read-only tool operations with no side effects
- In low-stakes applications where errors are harmless

## Key Technical Components

### TS-Guard: Multi-Task Risk Detection Model

Train a smaller guardrail model that analyzes agent interaction history to identify risky tool invocations.

```python
# Multi-task risk detection
class TSGuard:
    def __init__(self, model_name="ts-guard-base"):
        self.model = load_model(model_name)  # Smaller, faster model

    def assess_tool_risk(self, interaction_history, tool_name, tool_args):
        """Assess risk of proposed tool invocation"""
        context = format_interaction_history(interaction_history)
        risk_assessment = self.model.predict({
            "context": context,
            "tool": tool_name,
            "args": tool_args
        })
        return {
            "harmfulness_score": risk_assessment["harm"],
            "attack_likelihood": risk_assessment["attack_likelihood"],
            "risk_category": risk_assessment["category"]
        }
```

### Risk Category Classification

Identify specific types of risks to enable targeted mitigation.

```python
# Risk categories
RISK_CATEGORIES = {
    "data_exfiltration": {
        "severity": "high",
        "indicators": ["write_to_external", "network_transmission", "log_access"]
    },
    "privilege_escalation": {
        "severity": "high",
        "indicators": ["permission_change", "sudo_execution", "admin_access"]
    },
    "prompt_injection": {
        "severity": "medium",
        "indicators": ["unescaped_input", "dynamic_query_construction"]
    },
    "resource_exhaustion": {
        "severity": "medium",
        "indicators": ["infinite_loop", "memory_allocation", "cpu_intensive"]
    }
}
```

### TS-Flow: Integrated Reasoning with Guardrails

Incorporate guardrail feedback directly into agent behavior.

```python
# Integrated safety reasoning
class TSFlow:
    def __init__(self, agent, guard):
        self.agent = agent
        self.guard = guard

    def step_with_safety(self, observation):
        """Execute agent step with safety monitoring"""
        # Get agent's proposed tool invocation
        action = self.agent.decide(observation)

        # Assess risk
        risk = self.guard.assess_tool_risk(
            self.agent.history,
            action["tool"],
            action["args"]
        )

        # If risky, modify action or reject
        if risk["harmfulness_score"] > THRESHOLD:
            action = self.mitigate_risk(action, risk)
            if action is None:
                return self.agent.handle_blocked_action(risk)

        return action
```

### Interaction History Analysis

Maintain and analyze agent history to detect attack patterns.

```python
# Pattern-based attack detection
def detect_attack_patterns(history):
    """Identify sequences indicating prompt injection or social engineering"""
    patterns = []

    # Check for escalating privileges
    if has_increasing_permissions(history):
        patterns.append("privilege_escalation_attempt")

    # Check for credential extraction attempts
    if contains_credential_requests(history):
        patterns.append("credential_harvesting")

    # Check for repetitive denied attempts
    if has_repeated_blocked_actions(history):
        patterns.append("persistent_attack_pattern")

    return patterns
```

### Adaptive Risk Thresholding

Adjust risk tolerance based on context and task requirements.

```python
# Context-aware thresholding
class AdaptiveThreshold:
    def get_threshold(self, context):
        """Determine risk threshold based on task and environment"""
        base_threshold = 0.5

        if context["environment"] == "untrusted":
            return base_threshold * 0.7  # More strict
        elif context["task_criticality"] == "high":
            return base_threshold * 0.8
        else:
            return base_threshold
```

## Performance Characteristics

- Harmful tool invocation reduction: 65% average across ReAct-style agents
- Task success rate improvement: ~10% on prompt injection scenarios
- Runtime overhead: <50ms per tool invocation assessment
- Detection precision: 86.7%, Recall: 82.5%

## Integration Pattern

1. Initialize TS-Guard model and load interaction history
2. Before each tool invocation, call `assess_tool_risk()`
3. If risk score exceeds threshold, apply mitigation
4. Log all assessments for audit trail
5. Update guardrail model periodically with new attack patterns

## Mitigation Strategies

- **Rejection**: Block invocation entirely
- **Modification**: Rewrite unsafe arguments (e.g., sanitize file paths)
- **Confirmation**: Request human approval for medium-risk actions
- **Sandboxing**: Execute in restricted environment with reduced permissions

## References

- LLM agents are vulnerable to prompt injection through tool invocations
- Step-level monitoring is more effective than outcome-only verification
- Real-time guardrails improve both safety and task success rates
