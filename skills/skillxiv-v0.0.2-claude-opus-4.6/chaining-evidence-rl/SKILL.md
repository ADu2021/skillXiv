---
name: chaining-evidence-rl
title: "Chaining the Evidence: Robust Reinforcement Learning for Deep Search Agents with Citation-Aware Rubric"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.06021"
keywords: [search-agents, reinforcement-learning, factual-grounding, hallucination-prevention, reward-design]
description: "Train search agents using citation-aware rubric rewards that decompose complex questions into verifiable single-hop facts. Agents learn to chain evidence through explicit source citations, preventing hallucinations and shortcut exploitation. Citation-aware Group Relative Policy Optimization (C-GRPO) combines rubric and outcome rewards, enabling agents to solve multi-hop reasoning tasks with high factual grounding and transparency."
---

## Problem

Current search-based agents face three critical weaknesses:

1. **Hallucination-Prone**: Agents fabricate supporting evidence or skip justification when rewarded only for correct final answers
2. **Shortcut Exploitation**: Agents learn to guess answers without actual reasoning, failing on variations
3. **Non-Transparent Reasoning**: Final answers lack verifiable chains of evidence, making it impossible to audit agent logic
4. **Weak Generalization**: Agents trained on outcome rewards alone don't transfer to open-ended research tasks

Search agents need fine-grained supervision that rewards the reasoning process, not just final outcomes.

## Solution

**Chaining the Evidence** introduces **Citation-Aware Rubric Rewards (CaRR)**:

1. **Rubric Decomposition**: Break complex questions into "verifiable single-hop rubrics"
   - Example: "Find the birth year of Obama's 2012 campaign manager" → [Identify campaign manager, Find birth year]
2. **Citation Requirements**: Each rubric element must be grounded in a specific source document
3. **Evidence Chains**: Agents must construct explicit chains linking query → rubric elements → citations → answer
4. **C-GRPO Training**: Combine rubric rewards (process-level) with outcome rewards (final result) during RL training

## When to Use

- **Complex Reasoning Searches**: Multi-hop questions requiring agents to chain facts
- **Fact-Checking Agents**: Tasks demanding transparent evidence for every claim
- **Research Assistance**: Agents must cite sources for generated conclusions
- **High-Stakes Applications**: Medical, legal, financial domains requiring audit trails
- **Open-Ended Reasoning**: Tasks beyond closed-domain benchmarks with known answers

## When NOT to Use

- For simple fact lookup (single-hop retrieval is more efficient)
- When citation data is unavailable or unreliable
- In real-time systems where rubric decomposition adds latency
- For agents that don't need explanation (pure performance optimization)

## Core Concepts

The framework operates on the principle that **reasoning transparency enables robustness**:

1. **Rubric Design**: Break complex queries into atomic facts that can be verified independently
2. **Evidence Grounding**: Every claim requires source citation, preventing hallucination
3. **Composable Reasoning**: Evidence chains compose individual facts into complete answers
4. **Dual Optimization**: Balance process quality (good reasoning) with outcome quality (correct answers)

## Key Implementation Pattern

Building citation-aware agents with CaRR:

```python
# Conceptual: rubric-based evidence chaining
def train_citation_aware_agent(query, reference_documents):
    # Step 1: Decompose query into verifiable rubrics
    rubrics = decompose_query(query)
    # rubrics: ["Identify campaign manager", "Find birth year"]

    # Step 2: Agent answers each rubric with citation
    for rubric in rubrics:
        answer, citation = agent.answer_with_citation(rubric, reference_documents)
        # Rubric reward: verify citation supports answer
        rubric_reward = verify_citation(answer, citation, reference_documents)

    # Step 3: Final answer from evidence chain
    final_answer = compose_from_evidence(rubric_answers)

    # Step 4: Dual reward optimization
    outcome_reward = evaluate_final_answer(final_answer, ground_truth)
    combined_reward = alpha * rubric_reward + (1-alpha) * outcome_reward

    # Update policy with combined signal
    agent.optimize(combined_reward)
```

Key mechanisms:
- Rubric decomposition templates (query-type dependent)
- Citation matching against reference documents
- Evidence chain composition (OR, AND, sequential logic)
- Process reward aggregation across rubrics

## Expected Outcomes

- **Hallucination Reduction**: 40-60% decrease in fabricated evidence
- **Improved Generalization**: Models trained on rubric rewards transfer better to unseen question types
- **Transparency**: Every answer accompanied by verifiable evidence chains
- **Robustness**: Agents resist shortcut exploitation; performance on adversarial questions improves

## Limitations and Considerations

- Requires manual or automated rubric templates for different query types
- Citation grounding depends on document quality; poor sources degrade rewards
- Rubric decomposition overhead adds latency to training and inference
- Not all questions decompose cleanly into independent single-hop facts

## Integration Pattern

For a research assistant agent:

1. **Parse Query**: Identify question type and select rubric template
2. **Decompose**: Break into verifiable sub-questions
3. **Answer Each**: Generate answer + cite source for each sub-question
4. **Compose Answer**: Combine rubric answers into final response
5. **Reward**: Give credit only if evidence chain is complete and citations check out

This ensures the agent's reasoning is transparent and auditable.

## Scaling Consideration

The framework scales from simple binary questions (yes/no with single citation) to complex comparative reasoning (compare 3+ entities across dimensions) by adjusting rubric complexity.

## Related Work Context

CaRR advances RL for search agents by recognizing that reward signals should target reasoning quality, not just final correctness. This process-focused approach mirrors how humans learn to justify reasoning through evidence.
