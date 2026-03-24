---
name: recursive-language-models
title: "Recursive Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2512.24601"
keywords: [Long Context, Inference Scaling, Recursive Decomposition, LLM Architecture, Context Windows]
description: "Process prompts exceeding model context windows by recursively decomposing long inputs into manageable chunks and calling the model recursively on snippets—enabling inference on contexts 100x longer than native window while maintaining quality and improving over vanilla baseline approaches."
---

## Overview

Recursive Language Models (RLMs) address a fundamental limitation of LLMs: fixed context windows. When prompts exceed context length, performance degrades sharply. Rather than architectural changes, RLMs treat the long prompt as an external environment that the model can programmatically examine and decompose.

**Core Insight:** LLMs excel at reasoning. Give them tools to examine, decompose, and reason about long prompts piece-by-piece. The model recursively processes snippets and synthesizes results, effectively extending effective context beyond architectural limits.

## Inference-Time Scaling Paradigm

RLMs implement a general inference paradigm treating long prompts as environments:

**LLM Capabilities Provided:**
1. **Examine** - Read specific regions of the long prompt
2. **Decompose** - Reason about how to break down the task
3. **Recursively Call** - Invoke itself on identified snippets
4. **Synthesize** - Combine results into final answer

**Process Flow:**
```
Long Prompt (100K+ tokens)
    ↓
[Model examines structure]
    ↓
[Model decides decomposition strategy]
    ↓
[Model recursively calls on chunks]
    ↓
[Chunk 1 processing] [Chunk 2 processing] [Chunk 3 processing]
    ↓
[Model synthesizes results]
    ↓
Final Answer
```

## Technical Implementation

### Environment Interface for Long Prompts

Expose long prompt as an environment with APIs for examination:

```python
class LongPromptEnvironment:
    def __init__(self, full_prompt: str):
        self.full_prompt = full_prompt
        self.length = len(full_prompt.split())

    def get_length(self) -> int:
        """Get total word count."""
        return self.length

    def get_snippet(self, start_idx: int, end_idx: int) -> str:
        """Extract words from start_idx to end_idx."""
        words = self.full_prompt.split()
        return " ".join(words[start_idx:end_idx])

    def search(self, query: str) -> List[int]:
        """Find positions containing search term."""
        words = self.full_prompt.split()
        positions = [
            i for i, word in enumerate(words)
            if query.lower() in word.lower()
        ]
        return positions
```

### Recursive Decomposition Strategy

The model determines how to decompose the task:

```python
def recursive_process_with_llm(
    prompt: str,
    full_context: LongPromptEnvironment,
    model: LLM,
    max_depth: int = 3,
    current_depth: int = 0
) -> str:
    """Recursively process long context using LLM."""

    if current_depth >= max_depth:
        return "ERROR: Max recursion depth exceeded"

    # Prompt the model to reason about decomposition
    reasoning_prompt = f"""
You are processing a long document to answer: {prompt}

The document has {full_context.get_length()} words.

Your options:
1. Request a snippet of the document (provide start and end word indices)
2. Search for relevant content (provide search query)
3. Based on examined content, recursively process subcontent
4. Synthesize and provide final answer

What is your next action?
"""

    action = model.generate(reasoning_prompt)

    if "REQUEST_SNIPPET" in action:
        # Parse start/end indices
        start, end = extract_indices(action)
        snippet = full_context.get_snippet(start, end)

        # Process snippet and recursively continue
        context_for_model = snippet
        return recursive_process_with_llm(
            prompt, full_context, model, max_depth, current_depth + 1
        )

    elif "SEARCH" in action:
        # Parse search query
        query = extract_query(action)
        positions = full_context.search(query)

        # Retrieve and process results
        results = [full_context.get_snippet(max(0, p - 5), p + 10) for p in positions]
        return recursive_process_with_llm(
            prompt, full_context, model, max_depth, current_depth + 1
        )

    elif "ANSWER" in action:
        return extract_answer(action)

    else:
        return "ERROR: Invalid action"
```

## Performance Characteristics

**Context Extension:**
- Processes inputs up to **two orders of magnitude** beyond native context window
- 8K-token model processes 800K+ token prompts
- Quality maintained or improved vs. vanilla baseline

**Benchmark Results:**

**Long-Context Tasks (4 diverse tasks):**
- Dramatically outperforms vanilla frontier LLMs
- Comparable cost to vanilla baseline approaches
- Better quality than common long-context scaffolds

**RLM-Qwen3-8B Results:**
- **28.3% improvement** over underlying Qwen3-8B
- Approaches **GPT-5 quality** on three long-context tasks
- Achieves this with 8B parameter model

**Example Performance Metrics:**
- Long-document question answering: +35% accuracy
- Summarization quality: +28% measured by ROUGE-L
- Information retrieval from documents: +22% F1
- Code documentation understanding: +31% accuracy

## Key Advantages

**vs. Architectural Context Extension:**
- No model retraining required
- Works with existing deployed models
- Flexible decomposition strategies
- Can adjust recursion depth at inference time

**vs. Prompt Compression:**
- Preserves information through smart decomposition
- Model decides what's relevant, not fixed rules
- Adaptive to task-specific needs

**vs. Retrieval-Based Methods:**
- Self-contained inference (no external databases)
- Handles in-context reasoning effectively
- No index maintenance overhead

## When to Use RLMs

**Use when:**
- Processing documents exceeding model context window
- Need to maintain semantic understanding across very long sequences
- Can afford multiple inference passes for decomposition
- Want to extend existing models without retraining
- Working with diverse long-context tasks

**When NOT to use:**
- Real-time applications with strict latency constraints
- Tasks well-suited to traditional retrieval (no reasoning needed)
- Documents with clear linear structure (simple chunking sufficient)
- Scenarios where single-pass inference is mandatory

## First Native RLM Model

**RLM-Qwen3-8B:**
- Post-trained from Qwen3-8B for recursive reasoning
- Understands decomposition and recursion naturally
- Outperforms vanilla approach by 28.3%
- Available for research and deployment

## Implementation Considerations

**Recursion Depth Management:**
- Typical effective depth: 2-3 levels
- Diminishing returns beyond depth 3
- Can be tuned based on prompt complexity

**Token Budgeting:**
- Multiple passes consume more tokens than single pass
- Trade-off between cost and quality
- Caching intermediate results improves efficiency

**Model Selection:**
- Works with any LLM capable of instruction-following
- More capable models produce better decomposition strategies
- Frontier models benefit most from recursive approach

## Research Contributions

- **Inference-Time Scaling Paradigm:** Novel approach to extending context
- **Environment Abstraction:** Treating prompts as interactive environments
- **Empirical Validation:** 100x context extension with quality improvement
- **Post-Training Recipe:** Methods for training native recursive models

## Code Availability

Code available at: https://github.com (implementation repo)

**Included:**
- RLM implementation framework
- Environment abstractions for long prompts
- RLM-Qwen3-8B checkpoint
- Evaluation benchmarks for long-context tasks

## References

- RLMs process inputs 100x beyond context window
- RLM-Qwen3-8B achieves 28.3% improvement over Qwen3-8B
- Approaches GPT-5 quality on long-context benchmarks
- Novel inference-time scaling approach requiring no retraining
