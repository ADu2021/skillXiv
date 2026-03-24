---
name: enterprise-tool-calling-finetuning
title: "Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.03336"
keywords: [Tool Calling, Enterprise APIs, Disambiguation, Fine-Tuning, Interactive Agents]
description: "Train LLMs to disambiguate tool calls in enterprise settings where multiple similar APIs exist and parameters are incomplete. Generates synthetic multi-turn dialogues with realistic ambiguity to improve tool selection accuracy by 27+ percentage points."
---

# DiaFORGE: Disambiguation-Centric Fine-Tuning for Reliable Tool-Calling

In enterprise environments, a single business query often maps to multiple near-duplicate APIs, and most real calls arrive missing required parameters. Current LLMs trained on clean, fully-specified tool-calling benchmarks fail dramatically when deployed against production APIs. DiaFORGE solves this mismatch by generating realistic, ambiguous multi-turn dialogues where the agent must disambiguate between competing tools and ask clarifying questions—turning a brittle single-turn problem into a robust interactive one.

The gap between benchmark performance and production reality is massive: static benchmarks use fully-specified queries with one obvious tool, while real APIs have overlapping functionality, required parameters, and user confusion. DiaFORGE bridges this gap through synthetic dialogue generation that mimics real deployment dynamics.

## Core Concept

The framework operates on the insight that realistic tool-calling is fundamentally interactive. Rather than expecting agents to infer all details from a single query, train them to:

1. **Recognize ambiguity**: Detect when multiple tools could satisfy the request
2. **Ask clarifying questions**: Engage users to disambiguate and collect missing parameters
3. **Validate completeness**: Ensure all required parameters are present before calling the tool
4. **Recover from errors**: Gracefully handle ambiguous or impossible requests

This interactive approach reduces both false positives (wrong tool calls) and false negatives (refusing to act).

## Architecture Overview

- **UTC-Gen engine**: Multi-agent system generating realistic multi-turn dialogues with ground-truth tools and semantic distractors
- **Dialogue structure**: Simulated user gradually reveals information while assistant asks clarifying questions
- **Validation layer**: Rule-based and LLM-based checks ensure generated data is correct
- **Dynamic evaluation**: Interactive rollout where trained agent engages with user simulator (not just static metrics)
- **Parameter handling**: Explicit tracking of required vs. provided parameters across turns

## Implementation

Generate synthetic dialogues using the UTC-Gen multi-agent system. The engine seeds conversations with a ground-truth tool and includes semantic "distractor" tools:

```python
from diaforge.generator import UTCGen

gen = UTCGen(api_catalog="enterprise_apis.json")

# Seed dialogue with ground-truth tool and distractors
dialogue = gen.generate_dialogue(
    ground_truth_tool="CRM.CreateLead",
    # Similar tools to create disambiguation need
    distractors=[
        "CRM.UpdateLead",
        "CRM.CreateContact",
        "Sales.CreateOpportunity"
    ],
    # Define what information is revealed per turn
    information_stages=[
        {"turn": 1, "revealed": ["customer_name"]},
        {"turn": 2, "revealed": ["email", "phone"]},
        {"turn": 3, "revealed": ["industry", "company_size"]}
    ]
)

# Output: multi-turn conversation where assistant must ask
# clarifying questions to determine correct tool
print(dialogue)
# User: "Add John to our system"
# Assistant: "Is John a prospect or an existing customer?"
# User: "He's a new prospect we want to track"
# Assistant: "I'll create him as a new lead. What's his email?"
```

Fine-tune an open-source model (like Llama) on the generated dialogues. Use supervised learning with loss masking to focus on assistant responses:

```python
from diaforge.training import ToolCallingTrainer

trainer = ToolCallingTrainer(model="meta-llama/Llama-2-7b-chat")

# Load generated dialogues
dialogues = load_synthetic_data("generated_dialogues.jsonl")

# Fine-tune focusing only on assistant turns
trainer.train(
    dialogues=dialogues,
    loss_mask="assistant_only",  # Only optimize assistant responses
    epochs=3,
    batch_size=32,
    learning_rate=2e-5
)
```

Evaluate models using dynamic evaluation where they interact with a user simulator:

```python
from diaforge.evaluation import DynamicEvaluator

evaluator = DynamicEvaluator(
    user_simulator="interactive",  # Simulated user responds to agent questions
    api_catalog="enterprise_apis.json"
)

# Full interactive rollout: agent picks tool, gets feedback
results = evaluator.evaluate(
    model=trained_model,
    test_scenarios=test_dialogues,
    metrics={
        "tool_accuracy": "Did agent pick the right tool?",
        "false_positives": "Any incorrect tool calls?",
        "abstention_rate": "When does agent refuse to act?",
        "success_rate": "Did tool call complete the user's goal?"
    }
)

print(f"Tool accuracy: {results['tool_accuracy']:.1%}")
print(f"Success rate: {results['success_rate']:.1%}")
```

## Practical Guidance

### When to Use DiaFORGE

Use this approach when:
- Deploying agents against enterprise API catalogs with overlapping functionality
- Real-world tool calls arrive incomplete or ambiguous
- False positives (wrong tools) are dangerous or costly
- Users expect the agent to ask clarifying questions
- You need to reduce training data by generating realistic scenarios

### When NOT to Use

Avoid this for:
- Single well-defined tool domains (calculator, weather API)
- Scenarios where user interaction is not possible
- Tasks with perfectly unambiguous specifications
- Environments where asking clarifying questions creates friction

### Training Data Strategy

| Scenario | Distractors | Difficulty |
|----------|------------|-----------|
| Simple disambiguation | 2-3 similar tools | Baseline |
| Semantic overlap | 4-5 tools with overlapping capabilities | Medium |
| Parameter interdependence | Tools where required params depend on previous choices | Hard |
| Multi-step resolution | Requires 3+ clarifying questions to fully specify | Expert |

Generate ~5,000 dialogues mixing all difficulty levels for robust coverage.

### Critical Metrics

1. **Tool accuracy**: Correct tool selected at least 90% of the time
2. **False positive rate**: Wrong tools called <5% of time
3. **Abstention rate**: Agent refuses to act when truly ambiguous
4. **Success rate**: Final tool call achieves user's goal
5. **Interaction cost**: Average turns to complete vs. human baseline

The paper reports DiaFORGE-trained models achieve 27 percentage points improvement over GPT-4o and 49 points over Claude-3.5-Sonnet on dynamic evaluation.

### Common Pitfalls

1. **Generating perfect dialogues**: Real users are messy. Add typos, false starts, and unclear requests.
2. **Over-using static metrics**: Agent performance on isolated queries ≠ performance in interactive settings.
3. **Missing parameter validation**: Track required vs. optional parameters explicitly; don't assume models infer this.
4. **Ignoring failure modes**: Test what happens when the agent genuinely cannot disambiguate.
5. **Forgetting context length**: Enterprise APIs are verbose; ensure model can track state across 5+ turns.

## Reference

"Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky" - [arXiv:2507.03336](https://arxiv.org/abs/2507.03336)
