---
name: pearl-personalized-streaming-video
title: "PEARL: Personalized Streaming Video Understanding — Problem Definition and Framework"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.20422"
keywords: [Personalized Video Understanding, Streaming Video, Real-Time Interaction, Video-Level Concepts, Frame-Level Concepts]
description: "Defines Personalized Streaming Video Understanding (PSVU) as a new task bridging static image personalization and video intelligence. PEARL-Bench (132 videos, 2173 annotations) establishes founding experiments. PEARL framework provides training-free plug-and-play strategy using dual-grained memory (concept + streaming) and concept-aware retrieval. Trigger: When building AI assistants that interact with streaming video and personalize on user-defined concepts, apply this problem definition and framework to support real-time multi-turn reasoning."
category: "Field Foundation"
---

## The Problem Statement

**What is this problem?**

Current AI personalization operates on static images or offline video, disconnecting it from real-world scenarios requiring continuous visual input and instant feedback. Users watch livestreams, security cameras, or personal recordings and expect AI assistants to understand:
- Concepts specific to their use case (custom objects, people, actions)
- Real-time events unfolding in the stream
- Historical context from earlier stream segments

No prior problem definition or benchmark existed for this scenario.

**Why is this important?**

Streaming video personalization is ubiquitous in practice: live event interpretation, security monitoring, personal video archives, accessibility assistance. Yet research focused on static image personalization, missing the unique challenges of temporal continuity and real-time interaction.

**What existing approaches are inadequate?**

- **Static image personalization**: Works on single frames, can't reason about temporal dynamics or streaming context
- **Offline video understanding**: Analyzes complete, pre-recorded video; doesn't handle streaming constraints or personalization
- **Generic video captioning**: No mechanism for user-defined concepts or real-time interaction

These approaches lack the infrastructure for continuous concept registration, temporal reasoning, and interactive multi-turn dialogue.

---

## The New Paradigm: Personalized Streaming Video Understanding (PSVU)

**What new problem class does PEARL introduce?**

PSVU is a distinct research area at the intersection of personalization, streaming inference, and video understanding. It defines three core task components:

1. **Concept-Definition QA**: Register new user-defined concepts (people, objects, actions) from video frames
2. **Real-Time QA**: Answer queries about immediate moments in the stream
3. **Past-Time QA**: Retrieve and reason about historical stream segments

**Key terminology:**

- **Frame-level concepts**: Static entities registered from single frames (specific persons, objects visible in a moment)
- **Video-level concepts**: Dynamic actions unfolding across time (a person walking, object being manipulated)
- **Streaming memory**: Running buffer of observed frames and their embeddings
- **Concept memory**: Repository of user-defined concepts and their representations
- **Dual-grained architecture**: System that maintains both frame-level and video-level concept understanding

**How does this reframe personalization?**

Before PSVU:
- Personalization was query-time only (user uploads image of person, find similar)
- Video processing was offline and batch-based
- No explicit support for concept registration during streaming

After PSVU:
- Concepts are registered interactively during streaming
- System maintains dual granularity (frame-level static, video-level dynamic)
- Real-time and historical queries operate on same concept vocabulary

---

## PEARL-Bench: Founding Benchmark

**Dataset composition:**
- **132 videos** capturing diverse streaming scenarios
- **2,173 fine-grained timestamp-annotated examples**
- Support for three query types: concept-definition, real-time, past-time
- Multi-turn interaction sequences (users register concepts, then query)

**Key benchmark characteristics:**
- Long-form streaming input (full video sequences, not clips)
- Frame-level annotations (specific moments)
- Video-level annotations (temporal spans)
- Concept registration examples (how users define new concepts)

**Why PEARL-Bench matters:**
Establishes canonical evaluation methodology for PSVU. Enables measurement of whether systems can simultaneously support personalization, streaming inference, and temporal reasoning.

---

## PEARL Framework: Architecture and Strategy

**Core Innovation**: Training-free, plug-and-play strategy

The PEARL framework achieves personalized streaming understanding without task-specific training via:

### 1. Dual-Grained Memory System

Decouples two types of information:

**Concept Memory**: Stores user-defined concepts and their semantic representations
- Registered during concept-definition QA
- Contains appearance, action, or attribute patterns
- Enables fast concept matching across frames

**Streaming Memory**: Maintains running observations from the video stream
- Frame embeddings and timestamps
- Short-term retention (recent frames prioritized)
- Provides temporal context for historical queries

Decoupling enables efficient retrieval without conflating dynamic observations with static concept definitions.

### 2. Concept-Aware Retrieval Algorithm

Query processing pipeline:

```python
# Retrieval-augmented approach for streaming personalization
# 1. Rewrite query using registered concepts
# 2. Retrieve relevant concept exemplars
# 3. Retrieve relevant stream clips
# 4. Answer query using retrieved context

def concept_aware_retrieval(user_query, concept_memory, streaming_memory):
    # Rewrite query using concept embeddings
    concept_rewritten_query = rewrite_query(user_query, concept_memory)

    # Retrieve relevant concept exemplars (who is this person/object?)
    relevant_concepts = retrieve_concepts(concept_rewritten_query, concept_memory)

    # Retrieve relevant clips from stream
    relevant_clips = retrieve_clips(
        query=concept_rewritten_query,
        concepts=relevant_concepts,
        stream=streaming_memory
    )

    # Generate answer
    answer = generate_answer(user_query, relevant_concepts, relevant_clips)
    return answer
```

### 3. Founding Experiments

**Test across three architectures:**
- GPT-4V (vision language model)
- Qwen-VL (alternative VLM)
- LLaVA (open-source variant)

**Results:**
- Frame-level QA: +13.79% average improvement over baseline
- Video-level QA: +12.80% average improvement over baseline

**Key finding**: Dual-grained memory + concept-aware retrieval provides consistent gains across architectures, validating the framework is architecture-agnostic.

---

## Opened Research Directions

1. **Real-time constraint handling**: How to maintain concept and streaming memory within latency budgets?
2. **Continual concept learning**: Can concepts evolve as the user sees new examples during streaming?
3. **Multi-user personalization**: How to handle shared streams with different personalization contexts per user?
4. **Temporal coherence**: How to ensure concept understanding remains consistent across long streams?
5. **Cross-stream generalization**: Do concepts learned on one stream transfer to similar content elsewhere?
6. **Streaming-specific architectures**: What model designs are optimized for incremental, real-time personalization?
7. **User study methodology**: How to evaluate perceived utility (not just accuracy) for real-time personalization?

---

## Vocabulary & Foundational Concepts

| Concept | Definition | Why It Matters |
|---------|-----------|----------------|
| **Streaming Memory** | Temporal buffer of observed frames and embeddings during online video processing | Enables real-time queries about recent observations without reprocessing |
| **Concept Memory** | Repository of user-defined concepts (people, objects, actions) registered during interaction | Enables personalization without requiring new model training |
| **Dual-Grained Understanding** | System that handles both frame-level (static) and video-level (dynamic) concepts | Real-world streaming contains both static entities and temporal actions |
| **Concept-Definition QA** | Query type where user registers a new concept from video frames | Unlocks personalization through interaction, not static configuration |
| **Real-Time QA** | Queries about immediate moments in the stream | Essential for live event interpretation and accessibility |
| **Past-Time QA** | Queries requiring historical reasoning across stream segments | Enables context-aware understanding beyond immediate moment |
| **Concept-Aware Retrieval** | Query rewriting using concept embeddings to improve relevance | Bridges user vocabulary (what they care about) and stream content |

---

## Pre vs. Post PEARL

**Before PEARL:**
- Personalization: static image-based, offline processing
- Video understanding: generic (no personalization), batch evaluation
- No established problem definition for streaming personalization
- No benchmark for evaluating this scenario

**After PEARL:**
- Personalization: interactive, real-time, concept-based
- Video understanding: supports both stream-level and concept-level reasoning
- PSVU is a defined research direction with clear problem scope
- PEARL-Bench enables systematic evaluation and comparison

---

## When to Use This Paradigm

**PSVU is appropriate when:**
- Your application requires real-time video understanding
- Users need to register custom concepts (not generic classes)
- Multi-turn interaction is expected (user asks follow-up questions)
- Temporal context matters (historical queries about the stream)
- Personalization is essential (generic models insufficient)

**Examples:**
- Live event interpretation (sports commentary, news monitoring)
- Security stream analysis (identify specific persons of interest)
- Personal video archives (find moments involving specific people/objects)
- Accessibility assistance (real-time scene description with personalization)

**PSVU is NOT appropriate when:**
- Your video is fully processed offline before querying
- Users want generic understanding (no personalization needed)
- Real-time latency isn't critical
- Single-query interaction (no multi-turn dialogue)

---

## Limitations & Future Work

**Current limitations:**
- PEARL-Bench covers 132 videos; scaling to diverse streaming domains needed
- Framework is training-free but relies on strong base VLMs
- Concept memory doesn't update across streams (new concepts reset)

**Open questions:**
- How to handle concept drift (person ages, object changes appearance)?
- Can retrieval be optimized for edge deployment?
- How does performance scale with stream length and concept memory size?

---

## Related Subfields This Opens

- **Streaming personalization**: Continuous concept registration and preference learning
- **Incremental video understanding**: Processing unbounded video streams efficiently
- **Interactive machine learning**: Systems that learn from user interaction during operation
- **Temporal reasoning for personalization**: Multi-turn dialogue with temporal context
- **Few-shot concept learning**: Registering concepts from minimal examples during streaming

## Reference

Paper: https://arxiv.org/abs/2603.20422
Benchmark: PEARL-Bench (132 videos, 2173 annotations)
Related field: Video understanding, personalized AI assistants
