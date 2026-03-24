---
name: os-sentinel-safety-mobile-gui-agents
title: "OS-Sentinel: Safety-Enhanced Mobile GUI Agents via Hybrid Validation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.24411"
keywords: [Mobile Agents, Safety Detection, VLM Validation, System Security, Risk Detection]
description: "Detect safety violations in mobile GUI agents using dual mechanisms: formal verification rules for system-level risks (privilege escalation, credential exposure) and VLM-based contextual judges evaluating action appropriateness, achieving 10-30% safety improvement over baselines."
---

# Title: Dual-Mechanism Safety Detection for Mobile Autonomous Agents

Autonomous mobile agents powered by vision-language models can accidentally compromise system security or leak sensitive data. OS-Sentinel adds a safety layer that prevents harmful actions through **formal verification** (detecting explicit rule violations) and **contextual judgment** (evaluating whether actions make sense given task goals).

The approach operates at two levels: formal rules catch absolute violations, while VLM judges catch subtle contextual risks that rules might miss.

## Core Concept

**Hybrid Safety Validation**:
- **Formal Verifier**: Rule-based system detecting explicit violations (privilege escalation, credential patterns, sensitive keywords)
- **VLM-Based Judge**: Evaluates step-level or trajectory-level action appropriateness given task context
- **Dual-Pass Detection**: Action must pass both checks to proceed
- **Real-World Risk Taxonomy**: 10-category risk framework from 204 annotated mobile trajectories

This combines the transparency and certainty of formal rules with the semantic understanding of VLMs.

## Architecture Overview

- **Formal Verifier**: Filesystem hash monitoring, keyword/regex pattern matching for sensitive data
- **VLM Judge**: Vision-language model fine-tuned on MobileRisk trajectories to recognize contextual risks
- **Risk Taxonomy**: 10 categories spanning financial, personal, and system security risks
- **Evaluation Scope**: Step-level (individual actions) and trajectory-level (aggregate risk)
- **MobileRisk Benchmark**: 204 trajectories (102 safe, 102 unsafe) across 48 applications

## Implementation Steps

**1. Design Formal Verifier with Three Components**

Implement a rule-based verifier combining system state monitoring, sensitive keyword detection, and pattern matching.

```python
class FormalVerifier:
    def __init__(self):
        # System state: track filesystem metadata hashes
        self.filesystem_baseline = self.compute_baseline_hashes()

        # Sensitive keywords: financial, PII, credentials
        self.sensitive_keywords = {
            'credit_card': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'password': r'(?i)(password|passwd|pwd)\s*[:=]',
            'api_key': r'(?i)(api[_-]?key|apikey)\s*[:=]'
        }

    def check_privilege_escalation(self, action, current_permissions):
        # Detect attempts to elevate privileges
        if action.type == 'permission_grant':
            return action.target_permission not in current_permissions
        return False

    def check_sensitive_data_exposure(self, text_output):
        # Scan output for credential patterns
        for keyword, pattern in self.sensitive_keywords.items():
            if re.search(pattern, text_output):
                return keyword
        return None

    def verify_action(self, observation, action):
        # Check all three components
        has_escalation = self.check_privilege_escalation(action, self.current_perms)
        exposed_data = self.check_sensitive_data_exposure(action.text)
        filesystem_changed = self.check_filesystem_integrity()

        return not (has_escalation or exposed_data or filesystem_changed)
```

**2. Train VLM-Based Contextual Judge**

Fine-tune a vision-language model on annotated trajectory data to recognize contextual risks. Use the MobileRisk risk taxonomy as supervision signal.

```python
def prepare_training_data(trajectories, risk_taxonomy):
    # Trajectories contain: screenshots, action sequences, risk labels
    training_data = []

    for trajectory in trajectories:
        for i, (observation, action) in enumerate(trajectory.steps):
            # Pair observation-action with risk label
            training_data.append({
                'screenshot': observation.image,
                'action': action.description,
                'risk_category': trajectory.risks[i],  # 10-category taxonomy
                'is_risky': trajectory.risks[i] != 'SAFE'
            })

    # Fine-tune VLM on binary classification: risky vs safe
    return training_data
```

**3. Implement Step-Level and Trajectory-Level Evaluation**

Evaluate safety at both granularities: individual actions and aggregate sequences.

```python
class VLMSafetyJudge(nn.Module):
    def __init__(self, vlm_model):
        self.vlm = vlm_model

    def evaluate_step(self, observation, action):
        # Single step evaluation
        prompt = f"Is this action safe? Screenshot: {observation}, Action: {action.description}"
        risk_score = self.vlm.classify(prompt)
        return risk_score > 0.5  # Safe if score > threshold

    def evaluate_trajectory(self, trajectory, window_size=5):
        # Trajectory-level: either window-based or uniform sampling
        trajectory_risk = False

        for i in range(0, len(trajectory.steps), window_size):
            window = trajectory.steps[i:i+window_size]
            # Combine window observations
            aggregated_prompt = self.aggregate_observations(window)
            window_risk = self.vlm.classify(aggregated_prompt)
            trajectory_risk = trajectory_risk or (window_risk > 0.5)

        return trajectory_risk
```

**4. Combine Verifier and Judge with Delay Penalties**

For safety evaluation, apply delay-penalized metrics that account for when risks are caught.

```python
def evaluate_agent_safety(formal_verifier, vlm_judge, trajectories):
    step_level_results = []
    trajectory_level_results = []

    for trajectory in trajectories:
        formal_violations = []
        contextual_risks = []

        for step_idx, (obs, action) in enumerate(trajectory.steps):
            # Formal check
            if not formal_verifier.verify_action(obs, action):
                formal_violations.append(step_idx)

            # Contextual check
            if vlm_judge.evaluate_step(obs, action):
                contextual_risks.append(step_idx)

        # Step-level metrics
        accuracy = len(formal_violations) / len(trajectory.steps)

        # Delay-penalized metric: earlier detection is better
        if formal_violations:
            first_violation = min(formal_violations)
            delay_penalty = max(0, 1 - first_violation / len(trajectory.steps))
            step_level_results.append(1 - delay_penalty)

    return np.mean(step_level_results)
```

## Practical Guidance

**When to Use**:
- Mobile agent deployment where security/privacy risks are high
- Autonomous agents with internet access or filesystem permissions
- Financial applications, healthcare systems, credential-handling workflows

**Hyperparameters**:
- Judge threshold: 0.5 for binary risk classification (calibrate on validation set)
- Window size: 5 steps for trajectory-level evaluation (balance between granularity and efficiency)
- Pattern regex: Customize sensitive_keywords to your domain

**When NOT to Use**:
- Offline agents without system access
- Trusted environments where all data is non-sensitive
- High-latency scenarios (verifier adds per-step overhead)

**Pitfalls**:
- **Overfitting formal rules**: Rules are domain-specific; false positives if rules too strict
- **VLM judge confidence calibration**: Probability scores may not reflect true risk; validate on held-out trajectories
- **Incomplete taxonomy**: If risk categories don't cover your threat model, contextual judge may miss threats

**Integration Strategy**: Deploy as middleware between agent planning and action execution. Catch violations before they propagate.

## Reference

arXiv: https://arxiv.org/abs/2510.24411
