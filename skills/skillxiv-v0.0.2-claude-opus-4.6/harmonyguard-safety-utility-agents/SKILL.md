---
name: harmonyguard-safety-utility-agents
title: HarmonyGuard - Safety and Utility Optimization for Web Agents
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.04010
keywords: [web-agents, safety-alignment, multi-agent-systems, policy-optimization]
description: "Multi-agent framework that balances safety compliance with task completion through adaptive policy extraction and dual-objective optimization. Achieves 38% improvement in policy compliance while maintaining 20% higher task completion."
---

# HarmonyGuard: Safety and Utility Optimization for Web Agents

## Core Concept

HarmonyGuard addresses the tension between safety and utility in autonomous web agents through collaborative multi-agent optimization. Rather than treating safety and utility as competing objectives, the framework coordinates specialized agents that jointly optimize both dimensions. A Policy Agent maintains security rules from unstructured documents, while a Utility Agent reasons about task completion while respecting safety constraints.

## Architecture Overview

- **Policy Agent**: Automatically extracts and maintains security policies from unstructured text, adapting to emerging threats
- **Utility Agent**: Performs real-time reasoning to balance safety compliance with task completion objectives
- **Dual-Objective Optimization**: Metacognitive reasoning about trade-offs between competing goals
- **Policy Extraction Module**: NLP-based extraction of rules from diverse document formats
- **Real-time Feedback Loop**: Both agents observe outcomes and adjust strategies

## Implementation Steps

### Step 1: Build Policy Extraction Module

Create an NLP system to extract structured security policies from unstructured documents.

```python
class PolicyExtractor:
    """
    Extract security policies from unstructured documents.
    """

    def __init__(self, model_name="gpt-4"):
        self.model = model_name

    def extract_policies(self, document_text):
        """
        Extract structured policies from unstructured documents.

        Args:
            document_text: Raw text containing policy information

        Returns:
            List of structured policy objects
        """
        extraction_prompt = f"""
        Extract all security policies from the following document.
        For each policy, identify:
        1. Policy statement (what is restricted)
        2. Scope (where it applies)
        3. Severity level (critical/high/medium/low)
        4. Exceptions (if any)

        Document:
        {document_text}

        Return as structured JSON list.
        """

        response = self.model.generate(extraction_prompt)
        policies = self._parse_policy_json(response)

        return policies

    def _parse_policy_json(self, json_text):
        """Parse and validate extracted policies."""
        import json
        policies = json.loads(json_text)

        # Normalize policy format
        structured = []
        for policy in policies:
            structured.append({
                "statement": policy.get("statement"),
                "scope": policy.get("scope", "global"),
                "severity": policy.get("severity", "medium"),
                "exceptions": policy.get("exceptions", []),
                "extracted_at": datetime.now().isoformat()
            })

        return structured

    def update_policy_set(self, existing_policies, new_document):
        """
        Update existing policies with new information.

        Args:
            existing_policies: Current policy set
            new_document: New document to extract from

        Returns:
            Updated policy set
        """
        new_policies = self.extract_policies(new_document)

        # Merge, handling conflicts through severity
        merged = self._merge_policies(existing_policies, new_policies)

        return merged

    def _merge_policies(self, existing, new):
        """Merge policy sets with conflict resolution."""
        policy_dict = {p["statement"]: p for p in existing}

        for new_policy in new:
            stmt = new_policy["statement"]
            if stmt in policy_dict:
                # Keep stricter version
                if new_policy["severity"] > policy_dict[stmt]["severity"]:
                    policy_dict[stmt] = new_policy
            else:
                policy_dict[stmt] = new_policy

        return list(policy_dict.values())
```

### Step 2: Implement Policy Agent

Create an agent that maintains policies and evaluates task safety.

```python
class PolicyAgent:
    """
    Maintains security policies and evaluates action safety.
    """

    def __init__(self, policy_extractor):
        self.policy_extractor = policy_extractor
        self.current_policies = []
        self.policy_violations = {}

    def update_policies(self, document):
        """
        Update policy set from new document.

        Args:
            document: Text document with policy information
        """
        self.current_policies = self.policy_extractor.update_policy_set(
            self.current_policies,
            document
        )

    def evaluate_safety(self, proposed_action, context):
        """
        Evaluate whether an action violates any policies.

        Args:
            proposed_action: Action description
            context: Current task and execution context

        Returns:
            (is_safe, violations, severity_score)
        """
        violations = []
        max_severity = 0

        for policy in self.current_policies:
            # Check if action violates this policy
            violates = self._check_violation(proposed_action, policy, context)

            if violates:
                violations.append({
                    "policy": policy["statement"],
                    "severity": policy["severity"],
                    "scope": policy["scope"]
                })
                max_severity = max(max_severity, self._severity_score(policy["severity"]))

        is_safe = len(violations) == 0

        return is_safe, violations, max_severity

    def _check_violation(self, action, policy, context):
        """Check if action violates specific policy."""
        # Rule-based or learned classifier
        policy_keywords = self._extract_keywords(policy["statement"])
        action_keywords = set(action.lower().split())

        # Simple keyword overlap check (in practice, use semantic similarity)
        overlap = len(policy_keywords & action_keywords)
        return overlap > 0

    def _extract_keywords(self, text):
        """Extract keywords from policy statement."""
        import re
        # Simple implementation; use NLP in practice
        words = re.findall(r'\w+', text.lower())
        return set(words)

    def _severity_score(self, severity_str):
        """Convert severity string to numeric score."""
        severity_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        return severity_map.get(severity_str.lower(), 1)

    def suggest_alternative(self, unsafe_action, policy_violations):
        """
        Suggest safer alternative actions.

        Args:
            unsafe_action: Unsafe action description
            policy_violations: Violated policies

        Returns:
            List of suggested safe alternatives
        """
        prompt = f"""
        The following action violates safety policies:
        Action: {unsafe_action}

        Violated policies:
        {chr(10).join(v['policy'] for v in policy_violations)}

        Suggest 3 alternative approaches that accomplish the intent while respecting policies.
        """

        suggestions = self.model.generate(prompt)
        return suggestions
```

### Step 3: Implement Utility Agent with Dual-Objective Reasoning

Create an agent that reasons about task completion while respecting safety constraints.

```python
class UtilityAgent:
    """
    Optimizes task completion while respecting safety constraints.
    """

    def __init__(self, policy_agent, model_name="gpt-4"):
        self.policy_agent = policy_agent
        self.model = model_name

    def plan_task(self, task_description):
        """
        Create task plan balancing utility and safety.

        Args:
            task_description: Description of task to accomplish

        Returns:
            List of safe steps to complete task
        """
        reasoning_prompt = f"""
        Task: {task_description}

        Constraints: All actions must comply with current security policies.

        Generate a step-by-step plan that:
        1. Accomplishes the task objective
        2. Respects all security constraints
        3. Maximizes utility while maintaining 100% policy compliance

        For each step, explain how it advances the task and why it's safe.
        """

        plan = self.model.generate(reasoning_prompt)
        return self._parse_plan(plan)

    def optimize_action(self, task_goal, proposed_action, context):
        """
        Optimize an action for both utility and safety.

        Args:
            task_goal: Overall task objective
            proposed_action: Initial action proposal
            context: Current execution context

        Returns:
            (optimized_action, utility_score, safety_score)
        """
        # Check safety
        is_safe, violations, severity = self.policy_agent.evaluate_safety(
            proposed_action,
            context
        )

        if is_safe:
            utility_score = self._estimate_utility(proposed_action, task_goal)
            return proposed_action, utility_score, 1.0

        # If unsafe, find optimal alternative
        alternatives = self.policy_agent.suggest_alternative(proposed_action, violations)

        best_action = None
        best_utility = -float('inf')

        for alt_action in alternatives:
            is_safe, _, _ = self.policy_agent.evaluate_safety(alt_action, context)

            if is_safe:
                utility = self._estimate_utility(alt_action, task_goal)
                if utility > best_utility:
                    best_utility = utility
                    best_action = alt_action

        safety_score = 1.0 if best_action else 0.0

        return best_action, best_utility, safety_score

    def _estimate_utility(self, action, goal):
        """Estimate how much action contributes to goal."""
        prompt = f"""
        Goal: {goal}
        Action: {action}

        On a scale of 0-1, how much does this action advance the goal?
        Respond with only a single number.
        """

        response = self.model.generate(prompt)
        return float(response.strip())

    def _parse_plan(self, plan_text):
        """Parse generated plan into structured steps."""
        steps = []
        # Parse plan_text into structured steps
        return steps
```

### Step 4: Integrate into Web Agent Loop

Embed both agents into the agent's action selection loop.

```python
def run_web_agent_with_harmonyguard(task, policy_document, max_steps=50):
    """
    Run web agent with HarmonyGuard safety-utility optimization.

    Args:
        task: Task description
        policy_document: Document containing security policies
        max_steps: Maximum steps before timeout

    Returns:
        Task completion status and execution trace
    """
    # Initialize agents
    extractor = PolicyExtractor()
    policy_agent = PolicyAgent(extractor)
    utility_agent = UtilityAgent(policy_agent)

    # Load policies
    policy_agent.update_policies(policy_document)

    # Get initial plan
    plan = utility_agent.plan_task(task)

    for step_num in range(max_steps):
        # Get next action from agent
        proposed_action = agent.propose_action(task, step_num)

        # Optimize with HarmonyGuard
        optimized_action, utility_score, safety_score = utility_agent.optimize_action(
            task,
            proposed_action,
            context={"step": step_num, "previous_actions": []}
        )

        if optimized_action is None:
            print(f"Step {step_num}: Could not find safe action")
            break

        # Execute optimized action
        result = execute_action(optimized_action)

        # Update policies based on new information
        if "policy_update" in result:
            policy_agent.update_policies(result["policy_update"])

        # Check task completion
        if task_completed(task, result):
            return {"status": "success", "steps": step_num, "safety_score": safety_score}

    return {"status": "incomplete", "steps": max_steps}
```

## Practical Guidance

### When to Use HarmonyGuard

- **Regulated environments**: Healthcare, finance where compliance is mandatory
- **Multi-stakeholder systems**: Multiple conflicting objectives must be balanced
- **Dynamic policy contexts**: Security requirements change over time
- **High-stakes automation**: Autonomous systems where failure is costly

### When NOT to Use HarmonyGuard

- **Simple tasks with clear safety**: Single security constraint systems
- **Non-adversarial environments**: Cooperative settings without safety threats
- **Real-time systems**: Multi-agent coordination adds latency
- **Unclear policies**: Extracting policies from ambiguous documents is unreliable

### Hyperparameter Recommendations

- **Policy extraction confidence threshold**: 0.8+ for policy acceptance
- **Severity weight in optimization**: Critical=4x, High=2x, Medium=1x, Low=0.5x
- **Alternative generation count**: 3-5 alternatives per unsafe action
- **Policy update frequency**: After every 10-20 steps or when new documents appear

### Key Insights

The critical innovation is explicit separation between policy maintenance and utility optimization. Rather than hard-coding safety constraints, the system learns policies and optimizes against them. This enables graceful handling of conflicting requirements and makes the system adaptable to new policies.

## Reference

**HarmonyGuard: Safety and Utility in Web Agents** (arXiv:2508.04010)

Introduces multi-agent framework with adaptive policy extraction and dual-objective optimization. Demonstrates 38% improvement in policy compliance while maintaining 20% higher task completion rates in web-based autonomous agents.
