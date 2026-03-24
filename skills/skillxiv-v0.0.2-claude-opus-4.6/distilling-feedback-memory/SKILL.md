---
name: distilling-feedback-memory
title: "Distilling Feedback into Memory-as-a-Tool"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.05960"
keywords: [agent-learning, feedback-integration, memory-systems, cost-efficiency, iterative-refinement]
description: "Convert inference-time feedback into persistent, retrievable guidelines stored as agent memory. Framework enables LLMs to improve performance over time by systematically accumulating and applying learned critiques. Augmented models rapidly match test-time refinement performance while drastically reducing inference cost. Memory-as-tool pattern enables agents to learn from feedback without expensive retraining."
---

## Problem

Standard LLM refinement approaches face limitations:

1. **High Inference Cost**: In-context test-time refinement requires multiple forward passes, multiplying inference cost
2. **Knowledge Loss**: Feedback provided during inference is not retained for future tasks
3. **Repeated Mistakes**: Agents don't systematically avoid past errors across separate problems
4. **No Explicit Learning**: Models can't accumulate guidelines or rubrics learned from feedback
5. **Inefficiency**: Each task gets similar refinement budget even if pattern is clear

Agents need ways to learn from feedback that are cheaper than test-time refinement and don't require retraining.

## Solution

**Distilling Feedback into Memory** introduces **Memory-as-a-Tool**:

1. **Feedback Capture**: During inference, capture structured feedback from external evaluators or self-reflection
   - "Don't assume the user meant X when they said Y"
   - "Always verify numerical claims before asserting them"
2. **Persistent Guidelines**: Convert feedback into reusable rubrics stored in file-based memory
3. **Tool Integration**: Agents can *call* memory to retrieve relevant learned guidelines during task execution
4. **Rubric-Feedback Bench**: New benchmark with structured rubric feedback for measuring guideline internalization

## When to Use

- **Iterative Agent Improvement**: Agents that process multiple similar tasks and should improve over time
- **Cost-Sensitive Deployment**: Systems where refinement cost must be minimized
- **Learning Agents**: Systems where feedback is available but retraining is infeasible
- **Safety Enhancement**: Systems that must learn and apply safety guidelines from feedback
- **Domain Adaptation**: Agents adapting to new domains with user-provided feedback

## When NOT to Use

- For one-shot tasks where no feedback is available
- When sufficient retraining capacity exists (direct fine-tuning is more effective)
- In systems with no feedback mechanism
- For tasks where learned guidelines might become stale

## Core Concepts

The framework operates on the principle that **feedback is learnable as stored guidelines**:

1. **Feedback Abstraction**: Convert specific corrections into generalizable rubrics
2. **Memory as Tool**: Store learned guidelines in retrievable, agent-accessible memory
3. **Progressive Learning**: Accumulate guidelines across tasks without model retraining
4. **Cost-Benefit Balance**: Memory lookups are cheaper than full refinement loops

## Key Implementation Pattern

Building feedback-learning agents with memory-as-tool:

```python
# Conceptual: distilling feedback into agent memory
class FeedbackLearningAgent:
    def __init__(self):
        self.learned_guidelines = {}  # rubric_id -> guideline_text

    def process_task(self, task, evaluator):
        # Step 1: Generate initial response
        response = self.generate_response(task)

        # Step 2: Get feedback
        feedback = evaluator.evaluate(response)

        # Step 3: Distill feedback into guideline
        if feedback.is_corrective():
            guideline = self.distill_to_rubric(feedback)
            rubric_id = self.store_guideline(guideline)
            self.learned_guidelines[rubric_id] = guideline

        return response

    def process_new_task(self, new_task):
        # Step 1: Retrieve relevant guidelines from memory
        relevant_guidelines = self.retrieve_guidelines(new_task)

        # Step 2: Incorporate as system context
        context = self.build_context_from_guidelines(relevant_guidelines)

        # Step 3: Generate improved response using learned guidelines
        response = self.generate_response(new_task, context=context)

        return response
```

Key mechanisms:
- Feedback abstraction: specific corrections → general rubrics
- Guideline storage: file-based memory indexed by topic/domain
- Retrieval: matching new tasks to relevant stored guidelines
- Tool integration: guidelines provided as agent context, not model weights

## Expected Outcomes

- **50% Cost Reduction**: Match test-time refinement quality with single forward pass
- **Rapid Improvement**: Guidelines help with new tasks immediately after learning
- **Persistent Learning**: Guidelines accumulate and compound over time
- **Transparency**: Explicit rubrics show what agent has learned
- **Adaptability**: Agents learn domain-specific guidelines without retraining

## Limitations and Considerations

- Guideline quality depends on feedback quality and abstraction process
- Over-specificity in guidelines can hurt generalization
- Guidelines may become stale if task distribution shifts
- Feedback mechanism must be available (human or automated)

## Integration Pattern

For a document editing assistant:

1. **User Edits**: Provides feedback on AI suggestions
2. **Feedback Abstraction**: "Use active voice" → stored as guideline
3. **Memory Storage**: Guideline indexed under "writing-style"
4. **New Document**: Agent retrieves "writing-style" guidelines for consistency
5. **Apply Guidelines**: Generates suggestions informed by past feedback

This enables the agent to improve without model retraining.

## Rubric Feedback Bench

The benchmark includes:
- **Rubric-Annotated Tasks**: QA problems with feedback rubrics
- **Correction Patterns**: Common error types and corrections
- **Guideline Evaluation**: Measure how well agents internalize rubrics
- **Transfer Tasks**: Test generalization of learned guidelines

Use to train and evaluate feedback-learning agents.

## Guideline Types

Effective guidelines cover:
- **Constraint-Based**: "Never assume context not explicitly provided"
- **Strategy-Based**: "For numerical tasks, show intermediate steps"
- **Domain-Specific**: "In medical context, prioritize safety over optimization"
- **Process-Based**: "When uncertain, indicate confidence level"

Mix types for comprehensive agent improvement.

## Related Work Context

Distilling Feedback into Memory recognizes that learning from feedback doesn't require model retraining. By treating guidelines as tools agents can query, it enables efficient, systematic improvement without the cost of full refinement loops or model updates.
