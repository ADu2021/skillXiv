---
name: sculptor-active-context-management
title: Sculptor - Active Context Management for LLMs
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.04664
keywords: [context-management, working-memory, attention-control, llm-capabilities]
description: "Framework enabling LLMs to actively manage internal working memory and context through tools for fragmentation, summarization, and semantic search. Mitigates proactive interference and improves reasoning at extended context lengths."
---

# Sculptor: Active Context Management for LLMs

## Core Concept

Sculptor empowers large language models to self-manage their attention and working memory through active context control. Rather than passively processing entire contexts, models gain explicit tools to focus on relevant information, hide distractions, and reconstruct hidden content as needed. This mimics human cognitive strategies of selective attention and information filtering.

## Architecture Overview

- **Context Fragmentation Tools**: Break extended contexts into manageable chunks that can be independently accessed and processed
- **Summary/Hide/Restore Operations**: Allow models to compress irrelevant information while preserving critical context for later retrieval
- **Semantic Search Capabilities**: Enable precise extraction of relevant passages from fragmented context without linear scanning
- **Proactive Interference Mitigation**: Reduce performance degradation from irrelevant early-context information through explicit hiding mechanisms
- **Adaptive Focus**: Models learn to prioritize reasoning on pertinent information rather than all available context

## Implementation Steps

### Step 1: Instrument Context with Fragmentation Markers

Augment input contexts with structural markers that enable efficient partitioning and selective access.

```python
def fragment_context(text, chunk_size=500, overlap=50):
    """
    Fragment a long context into overlapping chunks with unique identifiers.

    Args:
        text: Input context to fragment
        chunk_size: Target characters per fragment
        overlap: Characters to overlap between consecutive fragments

    Returns:
        List of (fragment_id, content) tuples with metadata
    """
    fragments = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        fragment_id = f"FRAG_{len(fragments):04d}"

        fragments.append({
            "id": fragment_id,
            "content": text[start:end],
            "start_idx": start,
            "end_idx": end
        })

        start = end - overlap

    return fragments


def annotate_context_with_fragments(context, fragments):
    """
    Create a context reference system with fragment identifiers.

    Returns:
        Annotated context with access hints
    """
    annotated = "Fragment Access Guide:\n"
    for frag in fragments:
        annotated += f"- {frag['id']}: characters {frag['start_idx']}-{frag['end_idx']}\n"

    return annotated + "\n" + context
```

This enables the model to reference specific fragments rather than re-scanning the entire context.

### Step 2: Implement Summary and Hide/Restore Tools

Create tools that allow models to compress and selectively suppress context elements.

```python
class ContextManager:
    """
    Manages context state including summary generation and hiding.
    """

    def __init__(self):
        self.hidden_content = {}
        self.summaries = {}

    def hide_fragment(self, fragment_id, reason="low relevance"):
        """
        Hide a fragment from active context.

        Args:
            fragment_id: ID of fragment to hide
            reason: Why this fragment is being hidden

        Returns:
            Confirmation of hide operation
        """
        self.hidden_content[fragment_id] = {
            "hide_reason": reason,
            "hidden_at_step": self.current_step
        }
        return f"Fragment {fragment_id} hidden ({reason})"

    def summarize_fragment(self, fragment_id, summary_text):
        """
        Replace fragment with a concise summary.

        Args:
            fragment_id: Fragment to summarize
            summary_text: Compressed representation of content

        Returns:
            Summary registration
        """
        self.summaries[fragment_id] = {
            "summary": summary_text,
            "original_length": len(self.fragments[fragment_id]),
            "compression_ratio": len(summary_text) / len(self.fragments[fragment_id])
        }
        return f"Fragment {fragment_id} summarized"

    def restore_fragment(self, fragment_id):
        """
        Retrieve and restore a previously hidden fragment.

        Args:
            fragment_id: Fragment to restore

        Returns:
            Original fragment content
        """
        if fragment_id in self.hidden_content:
            del self.hidden_content[fragment_id]
        return self.fragments[fragment_id]

    def get_active_context(self):
        """
        Generate context excluding hidden fragments.

        Returns:
            Current active context for reasoning
        """
        active = []
        for frag_id, content in self.fragments.items():
            if frag_id not in self.hidden_content:
                if frag_id in self.summaries:
                    active.append(self.summaries[frag_id]["summary"])
                else:
                    active.append(content)
        return "\n".join(active)
```

### Step 3: Implement Semantic Search Tool

Enable precise retrieval of relevant passages without full-context scanning.

```python
def semantic_search_fragments(query, fragments, embedding_model):
    """
    Search fragments by semantic similarity to a query.

    Args:
        query: Search query
        fragments: List of fragment dictionaries
        embedding_model: Pre-trained embedding model

    Returns:
        Ranked list of relevant fragments
    """
    query_embedding = embedding_model.encode(query)
    scores = []

    for fragment in fragments:
        fragment_embedding = embedding_model.encode(fragment["content"])
        similarity = cosine_similarity(query_embedding, fragment_embedding)
        scores.append((fragment["id"], similarity, fragment))

    # Sort by relevance
    scores.sort(key=lambda x: x[1], reverse=True)

    return [{"id": s[0], "score": s[1], "content": s[2]["content"]}
            for s in scores]
```

### Step 4: Integrate into Model Inference

Embed fragment management into the inference loop.

```python
def inference_with_active_context(prompt, context, model, context_manager):
    """
    Run inference with active context management.

    Args:
        prompt: User query
        context: Full document or context
        model: Language model
        context_manager: ContextManager instance

    Returns:
        Model response
    """
    # Fragment the context
    fragments = fragment_context(context)
    context_manager.load_fragments(fragments)

    # Initial reasoning with full context
    response = model.generate(
        prompt=prompt,
        context=context_manager.get_active_context(),
        tools=["hide_fragment", "summarize_fragment", "restore_fragment", "search"]
    )

    # Model can call tools during generation to manage context
    while response.uses_tools():
        for tool_call in response.tool_calls:
            if tool_call.name == "hide_fragment":
                context_manager.hide_fragment(tool_call.args["fragment_id"])
            elif tool_call.name == "summarize_fragment":
                context_manager.summarize_fragment(
                    tool_call.args["fragment_id"],
                    tool_call.args["summary"]
                )
            # Continue generation with updated context
        response = model.continue_generation(
            context=context_manager.get_active_context()
        )

    return response.text
```

## Practical Guidance

### When to Use Sculptor

- **Long-context reasoning**: Documents >10K tokens where not all information is equally relevant
- **Noisy contexts**: Sources containing irrelevant or distracting information mixed with critical details
- **Multi-step reasoning**: Tasks where intermediate reasoning steps need to suppress and restore information
- **Heterogeneous documents**: Contexts mixing multiple topics that require selective focus

### When NOT to Use Sculptor

- **Short contexts**: <2K tokens where fragmentation overhead exceeds benefit
- **Linear narrative tasks**: Reading comprehension where all context is equally important in order
- **Real-time latency-critical**: Tool calling adds overhead unsuitable for high-throughput applications
- **Adversarial settings**: Information hiding could be exploited if prompts can override instructions

### Hyperparameter Recommendations

- **Fragment size**: 400-800 characters balances accessibility with structural clarity
- **Fragment overlap**: 50-100 characters captures transition information between chunks
- **Embedding model**: Use dense retrievers (e.g., BAAI/bge-base) for semantic search
- **Hide threshold**: Configure to trigger when model predicts <0.3 relevance score

### Key Insights

The framework demonstrates that explicit context management outperforms simple attention masking. By giving models agency over what to focus on, Sculptor exploits the model's internal understanding of task relevance. The proactive interference mitigation is particularly powerful: models actively suppress rather than merely ignore distracting content.

## Reference

**Sculptor: Empowering LLMs with Cognitive Agency via Active Context** (arXiv:2508.04664)

Introduces Active Context Management tools for fragmentation, summarization, hiding, and semantic search. Demonstrates improved reasoning at extended context lengths through explicit control of working memory and attention.
