---
name: tower-mind-agent-benchmark
title: "TowerMind: A Tower Defence Game Learning Environment and Benchmark for LLM as Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.05899"
keywords: [agent-evaluation, real-time-strategy, planning-benchmark, hallucination-measurement, game-environment]
description: "Evaluate LLM agent capabilities using tower defense game environment with multimodal observations (pixel, text, structured state). Benchmark reveals critical agent limitations: inadequate planning validation, inflexible decision-making, and inefficient action use. Demonstrates significant performance gap between current LLMs and human experts, providing structured framework for measuring agent planning, adaptation, and hallucination tendencies."
---

## Problem

Current LLM agent evaluation relies on narrow benchmarks:

1. **Limited Domains**: Most benchmarks test single skills (QA, tool use, reasoning)
2. **Unrealistic Complexity**: Real-world agents face simultaneous constraints and dynamic environments
3. **Missing Capability Assessment**: No systematic measurement of planning, adaptation, and strategic reasoning
4. **Hallucination Blindness**: Benchmarks don't explicitly measure when agents fabricate information
5. **Human-Agent Gap Unknown**: Unclear how far agents lag behind human performance on complex tasks

Agents need evaluation frameworks that measure real-world reasoning capabilities beyond text understanding.

## Solution

**TowerMind** introduces a **Tower Defense Game Benchmark** for LLM agents:

1. **Multimodal Observations**: Agents receive three observation types:
   - **Pixel-Based**: Raw game visuals (tower positions, enemy locations)
   - **Textual**: Natural-language game state descriptions
   - **Structured**: JSON game state with numeric values
2. **Strategic Complexity**: Tower defense requires simultaneous optimization:
   - **Planning**: Predict enemy paths and plan tower placement
   - **Resource Management**: Budget constraints on tower construction
   - **Adaptation**: Adjust strategies as enemy waves change
3. **Explicit Hallucination Measurement**: Benchmark includes scenarios designed to expose fabricated knowledge
   - Out-of-place game elements
   - Contradictions between observation modes
   - Impossible win conditions

## When to Use

- **Agent Capability Assessment**: Measure planning, adaptation, and strategic reasoning
- **Multimodal Evaluation**: Test agents on multiple observation formats
- **Hallucination Testing**: Diagnose tendency to fabricate information
- **Comparative Benchmarking**: Track agent improvements across versions
- **Planning Validation**: Verify that agents think ahead vs. reacting locally
- **Academic Research**: Study how LLMs approach real-time strategy

## When NOT to Use

- For production deployment decisions (game performance doesn't predict real-world capability)
- For single-domain specialists (game tests general reasoning, not domain expertise)
- In latency-critical applications (game simulation adds evaluation overhead)
- For systems already operating well on established benchmarks

## Core Concepts

The framework operates on the principle that **complex games reveal agent reasoning**:

1. **Multifinality Challenge**: Multiple ways to solve tower defense force strategic thinking rather than pattern matching
2. **Dynamic Environment**: Enemies change paths → agents must adapt plans, not execute pre-learned scripts
3. **Quantifiable Performance**: Win/loss, score, survival time provide clear metrics
4. **Hallucination Visibility**: Game rules are learnable; invented rules become obvious

## Key Implementation Pattern

Evaluating agents on TowerMind:

```python
# Conceptual: agent evaluation on TowerMind
class TowerMindBenchmark:
    def evaluate_agent(self, agent, num_episodes=10):
        results = {
            'win_rate': 0,
            'planning_validity': 0,
            'adaptation_score': 0,
            'hallucination_count': 0
        }

        for episode in range(num_episodes):
            # Step 1: Provide multimodal observation
            obs_pixels, obs_text, obs_structured = self.get_observation()
            agent.receive_observation(obs_pixels, obs_text, obs_structured)

            # Step 2: Get agent plan
            plan = agent.generate_plan()
            results['planning_validity'] += self.validate_plan(plan)

            # Step 3: Execute and track adaptation
            for step in range(max_steps):
                action = agent.choose_action()

                # Check for hallucination: action requires non-existent tower
                if self.is_hallucinated_action(action):
                    results['hallucination_count'] += 1

                obs_pixels, obs_text, obs_structured = self.step(action)
                agent.update(obs_pixels, obs_text, obs_structured)

            # Step 4: Score episode
            if self.won(episode):
                results['win_rate'] += 1
            results['adaptation_score'] += self.measure_adaptation()

        return {k: v / num_episodes for k, v in results.items()}
```

Key evaluation metrics:
- **Win Rate**: Percentage of games successfully completed
- **Planning Validation**: Are proposed plans executable?
- **Adaptation Score**: Do agents adjust strategy when environment changes?
- **Hallucination Count**: How often does agent reference non-existent game elements?

## Expected Outcomes

- **Baseline Performance Gap**: Current LLMs achieve 20-30% win rate vs. 80%+ for human experts
- **Capability Diagnostics**: Identify specific weaknesses (inadequate planning, rigid strategy)
- **Multimodal Insights**: Measure how agents perform with pixel vs. text vs. structured observations
- **Hallucination Quantification**: Explicit counts of fabricated game state claims

## Limitations and Considerations

- Tower defense performance doesn't directly predict capability in other domains
- Simulation overhead makes TowerMind slower than text-based benchmarks
- Some agents may perform poorly due to interaction interface limitations rather than reasoning
- Game difficulty varies; may need level selection for fair agent comparison

## Benchmark Structure

Includes:
- **Level Progression**: Increasing difficulty from basic to expert
- **Multimodal Variants**: Test agents with different observation types
- **Hallucination Scenarios**: Explicit tests for fabrication tendencies
- **Human Baseline**: Expert player scores for reference
- **Adaptation Tests**: Dynamic enemy waves force strategy revision

Use for comprehensive agent evaluation.

## Integration in Agent Development

For an agent development pipeline:

1. **Train Agent**: On standard benchmarks
2. **Evaluate on TowerMind**: Measure planning, adaptation, hallucination
3. **Diagnose Weaknesses**: Identify specific capability gaps
4. **Iterate**: Target improvements based on benchmark feedback
5. **Track Progress**: Monitor improvement across versions

This systematic evaluation catches capabilities standard benchmarks miss.

## Related Work Context

TowerMind addresses the evaluation gap by providing a complex, real-time environment that demands planning, adaptation, and strategic reasoning. Unlike narrow task-specific benchmarks, tower defense captures the multifaceted nature of agent reasoning.
