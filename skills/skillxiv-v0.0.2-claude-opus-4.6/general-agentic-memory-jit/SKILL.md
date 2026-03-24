---
name: general-agentic-memory-jit
title: "General Agentic Memory Via Deep Research"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.18423"
keywords: [Agentic Memory, Just-in-Time Compilation, Deep Research, Agent Systems, Information Retrieval]
description: "Build persistent, lossless agent memory using just-in-time compilation: store complete history in a universal page-store while performing dynamic deep research at query time, enabling test-time scalability through iterative information synthesis and reflection."
---

# General Agentic Memory Via Deep Research

Traditional agent memory systems compress information ahead-of-time, causing unavoidable information loss when summarizing historical interactions. This skill shows how to implement a JIT-compiled memory architecture that preserves complete historical information while constructing optimized context dynamically at runtime—enabling agents to scale their reasoning effort through iterative research loops.

The key insight is reversing the memory compilation approach: instead of pre-compressing history (ahead-of-time compilation), maintain complete uncompressed records and construct task-relevant context on-demand through agentic reasoning. This enables performance optimization through reinforcement learning while avoiding context loss that plagues static, pre-summarized memories.

## Core Concept

General Agentic Memory (GAM) implements a two-component architecture inspired by just-in-time compiler principles:

1. **Memorizer (Offline Stage)**: Incrementally compresses crucial information from streaming interactions into a universal page-store, maintaining complete historical records without discarding original data.

2. **Researcher (Online Stage)**: Performs iterative deep research to retrieve and integrate task-relevant information from the page-store. Rather than relying on pre-constructed memory, this component dynamically searches, synthesizes, and reflects on gathered information until the client's needs are fully satisfied.

The system leverages modern LLMs' agentic capabilities and test-time scalability, enabling performance optimization through reinforcement learning while avoiding information loss limitations.

## Architecture Overview

- **Universal Page-Store**: Maintains complete, uncompressed historical sessions without information loss—the foundation for lossless memory realization
- **Page-Level Compression**: Incremental summarization of individual sessions with optional metadata and topic annotations
- **Search & Retrieval**: Agentic search through the page-store using semantic understanding and relevance ranking
- **Synthesis Loop**: Iterative refinement where the researcher synthesizes retrieved information, generates questions, and searches for clarifications
- **Reflection Component**: Critical evaluation of gathered information against the original query to determine if additional research is needed

## Implementation Steps

The memory system operates through the following stages during online inference:

**1. Initialize Search Context**

Create an initial formulation of the client's information need and establish search constraints (time window, relevance threshold).

```python
def initialize_research_context(client_query, page_store, max_iterations=5):
    """
    Set up the research context for deep memory search.
    Initializes search parameters and prepares the page-store for queries.
    """
    context = {
        'query': client_query,
        'retrieved_pages': [],
        'synthesized_info': '',
        'iteration': 0,
        'max_iterations': max_iterations
    }
    return context
```

**2. Agentic Search and Retrieval**

The researcher performs semantic search across the page-store, retrieving the most relevant pages according to the current information need.

```python
def agentic_search(context, page_store, embedder, top_k=5):
    """
    Perform semantic search for relevant pages.
    Encodes the current query need and retrieves pages by similarity.
    """
    query_embedding = embedder.encode(context['query'])
    candidates = page_store.retrieve_by_similarity(query_embedding, top_k)

    # Score candidates for relevance
    scored = [(page, score) for page, score in candidates]
    context['retrieved_pages'].extend(scored)
    return context
```

**3. Synthesis and Reflection**

Extract and synthesize information from retrieved pages, then assess whether the gathered information satisfies the original query.

```python
def synthesize_and_reflect(context, llm_agent):
    """
    Synthesize retrieved information and evaluate sufficiency.
    Uses the LLM to integrate multiple sources and identify gaps.
    """
    if not context['retrieved_pages']:
        return context

    # Combine retrieved page contents
    combined_content = '\n\n'.join(
        [page.content for page, _ in context['retrieved_pages']]
    )

    # Use LLM to synthesize and generate refined query if needed
    synthesis_prompt = f"""
    Based on the following retrieved information, synthesize an answer to: {context['query']}

    Retrieved information:
    {combined_content}

    If important information is missing, list follow-up questions to search for.
    """

    result = llm_agent.generate(synthesis_prompt)
    context['synthesized_info'] = result['synthesis']
    context['follow_up_questions'] = result.get('follow_up_questions', [])

    return context
```

**4. Iterative Refinement**

If reflection indicates gaps, refine the search query and retrieve additional pages.

```python
def deep_research_loop(client_query, page_store, llm_agent, embedder):
    """
    Execute the full deep research loop with iterative refinement.
    Continues searching and synthesizing until information need is satisfied.
    """
    context = initialize_research_context(client_query, page_store)

    while context['iteration'] < context['max_iterations']:
        # Search for relevant information
        context = agentic_search(context, page_store, embedder)

        # Synthesize and check if done
        context = synthesize_and_reflect(context, llm_agent)

        # If no follow-up questions, research is complete
        if not context.get('follow_up_questions'):
            break

        # Refine query for next iteration
        context['query'] = context['follow_up_questions'][0]
        context['iteration'] += 1

    return context['synthesized_info']
```

## Practical Guidance

**When to Use Deep Research Memory:**
- Agents need to reference historical interactions across long timescales
- Information loss from pre-compression would harm decision quality
- Reasoning tasks benefit from exhaustive context review
- Test-time computational budget is available for research loops

**When NOT to Use:**
- Real-time response constraints prohibit iterative search (>500ms latency requirements)
- Memory is small enough to fit entirely in context window
- Historical information is not needed (single-turn interactions)

**Key Hyperparameters:**
- `max_iterations`: Controls research depth; typically 3-5 iterations sufficient before diminishing returns
- `top_k`: Number of pages retrieved per search iteration; 5-10 balances coverage and redundancy
- `similarity_threshold`: Filter pages below relevance threshold to reduce noise
- `page_size`: Granularity of page-store segments; smaller pages enable fine-grained retrieval

**Pitfalls to Avoid:**
- Over-relying on pre-constructed summaries; ensure page-store maintains complete original content
- Infinite loops where researcher fails to synthesize sufficient information; enforce iteration limits
- Ignoring recency bias; older pages may be more valuable for historical context
- Combining too many pages in synthesis; limit to top-5 to maintain coherence

**Integration with Agents:**
This memory architecture integrates naturally with reinforcement learning during training—the researcher's synthesis process can be optimized to improve downstream task performance. The agentic search loop is compatible with tool-calling APIs; search can be implemented as an external function call returning page contents.

## Reference

Research paper: https://arxiv.org/abs/2511.18423
