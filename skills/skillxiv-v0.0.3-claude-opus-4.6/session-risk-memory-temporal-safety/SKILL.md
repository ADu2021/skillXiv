---
name: session-risk-memory-temporal-safety
title: "Session Risk Memory: Temporal Authorization in Multi-Turn Agent Safety Gates"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.22350"
keywords: [Agent Safety, Temporal Authorization, Risk Memory, Exponential Moving Average, Trajectory Evaluation]
description: "Add trajectory-level temporal authorization to stateless execution gates by maintaining a semantic centroid of agent behavioral profiles with exponential moving average risk accumulation. Improves F1 from 0.9756 to 1.0 and reduces false positives from 5% to 0% on slow-burn security violations. Use when deploying multi-turn agents and need to detect gradual privilege escalation and data exfiltration."
category: "Component Innovation"
---

## What This Skill Does

Extend stateless per-action execution gates with a lightweight Session Risk Memory module that maintains behavioral context across turns. Uses exponential moving average to accumulate risk signals from slow-burn attacks (gradual privilege escalation, incremental data exfiltration) that evade per-turn thresholds.

## The Component Swap

The old ILION approach evaluates authorization per-action without trajectory context, missing slow-burn attacks that individually appear benign:

```python
# Old: stateless per-action evaluation
for action in agent_actions:
    risk_score = evaluate_action(action)  # Binary: safe or unsafe
    if risk_score > threshold:
        block_action(action)
    # No memory of previous actions; slow escalation undetected
```

The new approach adds a Session Risk Memory module that maintains semantic behavioral context and exponential moving average (EMA) risk accumulation:

```python
# New: Session Risk Memory for trajectory-level authorization
class SessionRiskMemory:
    def __init__(self, embedding_dim=768):
        self.semantic_centroid = None  # Behavioral profile
        self.ema_risk = 0.0  # Accumulated risk
        self.ema_alpha = 0.2  # Decay factor

    def update(self, action_embedding, per_action_risk):
        """
        Update centroid and risk based on current action.
        Args:
            action_embedding: semantic vector of action (768-dim)
            per_action_risk: scalar risk from per-action evaluation [0,1]
        """
        # Update semantic centroid incrementally
        if self.semantic_centroid is None:
            self.semantic_centroid = action_embedding
        else:
            self.semantic_centroid = (
                0.9 * self.semantic_centroid +
                0.1 * action_embedding
            )

        # Exponential moving average of risk
        self.ema_risk = (
            (1 - self.ema_alpha) * self.ema_risk +
            self.ema_alpha * per_action_risk
        )

    def get_authorization(self, action_embedding, per_action_risk,
                          trajectory_threshold=0.5):
        """
        Spatial (per-action) + temporal (trajectory) authorization.
        Returns: (is_authorized, confidence)
        """
        self.update(action_embedding, per_action_risk)

        # Temporal risk: accumulated slow-burn behavior
        temporal_risk = self.ema_risk

        # Spatial risk: current action evaluation
        spatial_risk = per_action_risk

        # Combined authorization
        combined_risk = 0.7 * spatial_risk + 0.3 * temporal_risk

        is_safe = combined_risk < trajectory_threshold
        return is_safe, combined_risk
```

Key design: uses deterministic EMA (no probabilistic inference) on existing semantic embeddings (no new model training required). The semantic centroid tracks behavioral drift over a session:

```python
# Detecting deviation from established behavior pattern
def detect_behavioral_shift(action_embedding, centroid):
    """Measure how far this action deviates from session norm."""
    deviation = torch.norm(action_embedding - centroid)
    return deviation
```

## Performance Impact

**Security metrics (ILION multi-turn agent safety):**
- F1 Score: 0.9756 (stateless) → 1.0000 (with SRM) = **+0.0244 (perfect detection)**
- False Positive Rate: 5% → 0% = **-5 pp** (no false alarms on benign actions)
- Detection Rate: 100% maintained for both (all attacks caught)

**Computational efficiency:**
- Overhead: <250 microseconds per turn (negligible)
- Memory: O(embedding_dim) = ~3KB for 768-dim embedding

**Threat scenarios detected:**
- Slow-burn data exfiltration (gradual file access escalation)
- Incremental privilege escalation (step-by-step permission increases)
- Compliance drift (gradual deviation from expected behavior)

## When to Use

- Multi-turn agent deployments with long session durations
- Scenarios where attackers use gradual escalation rather than one-shot exploits
- When per-action gates miss temporal patterns (stateless baseline fails)
- Safety-critical systems requiring trajectory-level authorization

## When NOT to Use

- Single-action or very short sessions (EMA accumulation insufficient)
- Agents with naturally high-variance behavior where centroid is uninformative
- Systems requiring immediate response to per-turn risks
- Scenarios without pre-computed semantic embeddings for actions

## Implementation Checklist

To adopt this component swap:

1. **Ensure action embeddings exist:**
   ```python
   # Actions must be converted to semantic vectors (768-dim standard)
   action_embedding = action_encoder.encode(action)  # Must exist in system
   assert action_embedding.shape == (768,)
   ```

2. **Initialize Session Risk Memory:**
   ```python
   # Create one instance per agent session
   session_memory = SessionRiskMemory(embedding_dim=768)

   # Store across turns (persist in session state)
   session_state['risk_memory'] = session_memory
   ```

3. **Integrate into authorization pipeline:**
   ```python
   # At each turn:
   action = get_agent_action()
   action_embedding = encode_action(action)
   per_action_risk = evaluate_action(action)  # Existing per-action gate

   # Add temporal authorization
   is_authorized, combined_risk = session_memory.get_authorization(
       action_embedding,
       per_action_risk,
       trajectory_threshold=0.5  # Tunable
   )

   if not is_authorized:
       block_action(action)
   ```

4. **Verify security improvements:**
   - Test on slow-burn scenarios (e.g., file access escalation over 10 turns)
   - Measure F1 on multi-turn attack dataset
   - Verify false positive rate on benign sessions

5. **Hyperparameter tuning:**
   - `ema_alpha` (decay factor): 0.1-0.3 (lower = longer memory)
   - `trajectory_threshold`: 0.4-0.6 (lower = stricter)
   - `spatial_weight` (0.7 default): 0.5-0.9 (higher = trust per-action more)
   - `temporal_weight` (0.3 default): 0.1-0.5 (complement to spatial)

6. **Known issues:**
   - Initial actions have weak temporal signal until centroid stabilizes (~5 turns)
   - Sudden legitimate behavior changes (e.g., task switching) can falsely elevate temporal risk
   - Requires meaningful semantic embeddings; random embeddings won't work

## Related Work

This builds on temporal modeling in security (anomaly detection) and extends stateless gates (ILION) with lightweight memory. Relates to exponential smoothing in monitoring and trajectory-level reasoning in agent evaluation.
