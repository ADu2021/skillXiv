---
name: lightweight-memory-augmented
title: "LightMem: Lightweight and Efficient Memory-Augmented Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.18866"
keywords: [memory augmentation, efficient generation, token reduction, context compression, RAG]
description: "Reduce inference tokens by up to 38× and API calls by 30× through three-stage memory system: sensory compression, short-term consolidation, and offline long-term storage."
---

# Technique: Three-Stage Memory System — Efficient Context Management

Multi-turn conversations and long-horizon tasks require remembering context, but naive approaches accumulate tokens exponentially—each turn adds conversation history, doubling context length. LightMem addresses this through a **three-stage memory hierarchy** inspired by human memory: rapid filtering (sensory), structured consolidation (short-term), and offline processing (long-term).

The system filters irrelevant information early, organizes remaining context by topic, and separates online inference from offline consolidation. This achieves 38× token reduction and 30× fewer API calls while improving QA accuracy by up to 7.7%.

## Core Concept

LightMem operates on three principles:
- **Sensory Memory**: Lightweight compression that rapidly filters irrelevant information and groups by topic
- **Short-term Memory**: Organized consolidation of topic groups into structured summaries
- **Long-term Memory**: Offline procedures that separate consolidation from real-time inference
- **Lazy Retrieval**: Only fetch long-term memories when needed, don't load everything

The three-stage approach prevents context explosion while maintaining access to historical information when necessary.

## Architecture Overview

- **Input Processor**: Compress incoming text to key tokens/entities
- **Topic Detector**: Classify information into semantic topics
- **Sensory Buffer**: Lightweight sliding window that groups information
- **Short-term Consolidator**: Summarize topic groups on-demand
- **Long-term Storage**: Offline database of compressed memories
- **Retriever**: Fetch relevant long-term memories based on current query
- **Context Manager**: Assemble final context from sensory + retrieved long-term

## Implementation Steps

The key is implementing efficient filtering at each stage. This example shows the three-stage pipeline.

```python
import torch
from collections import defaultdict
from typing import List, Dict

class SensoryMemory:
    """
    Stage 1: Lightweight rapid filtering and topic grouping.
    """

    def __init__(self, max_tokens: int = 500, max_topics: int = 10):
        self.max_tokens = max_tokens
        self.max_topics = max_topics
        self.buffer = []
        self.topic_groups = defaultdict(list)

    def compress_and_group(self, text: str, topic_classifier) -> None:
        """
        Rapidly filter and group incoming information by topic.
        """
        # Extract key entities/tokens (fast, heuristic-based)
        key_tokens = extract_key_tokens(text, max_tokens=50)

        # Classify into topic
        topic = topic_classifier.predict(text)

        # Store in sensory buffer
        self.buffer.append({
            "text": text,
            "tokens": key_tokens,
            "topic": topic,
            "timestamp": len(self.buffer)
        })

        # Group by topic
        self.topic_groups[topic].append(len(self.buffer) - 1)

        # Prune oldest items if buffer exceeds max_tokens
        total_tokens = sum(len(item["tokens"]) for item in self.buffer)
        while total_tokens > self.max_tokens and self.buffer:
            old_item = self.buffer.pop(0)
            total_tokens -= len(old_item["tokens"])


class ShortTermMemory:
    """
    Stage 2: Consolidate topic groups into structured summaries.
    """

    def __init__(self, summarizer_model):
        self.summarizer = summarizer_model
        self.topic_summaries = {}

    def consolidate_topic_group(
        self,
        topic: str,
        topic_texts: List[str]
    ) -> str:
        """
        Summarize a group of texts from the same topic.
        """
        if not topic_texts:
            return ""

        # Combine topic texts
        combined = " ".join(topic_texts)

        # Summarize (offline, or cached if frequently used)
        summary_prompt = f"""
Summarize the following information in 2-3 sentences, keeping key facts:

{combined[:500]}  # Truncate if too long

Summary:
"""
        summary = self.summarizer.generate(summary_prompt, max_tokens=50)
        self.topic_summaries[topic] = summary
        return summary

    def get_short_term_context(self, active_topics: List[str]) -> str:
        """
        Retrieve consolidated summaries for active topics.
        """
        context_parts = []
        for topic in active_topics:
            if topic in self.topic_summaries:
                context_parts.append(f"[{topic}] {self.topic_summaries[topic]}")
        return "\n".join(context_parts)


class LongTermMemory:
    """
    Stage 3: Offline consolidation and efficient retrieval.
    """

    def __init__(self, embedding_model, db_path: str = "./ltm_db"):
        self.embedder = embedding_model
        self.db_path = db_path
        self.memory_vectors = []
        self.memory_texts = []

    def add_to_long_term(self, consolidated_text: str) -> None:
        """
        Offline: add consolidated information to long-term storage.
        (This runs asynchronously, not during inference)
        """
        # Embed for retrieval
        embedding = self.embedder.embed(consolidated_text)

        # Store
        self.memory_vectors.append(embedding)
        self.memory_texts.append(consolidated_text)

    def retrieve_relevant_memories(
        self,
        query: str,
        top_k: int = 3
    ) -> List[str]:
        """
        During inference: retrieve relevant long-term memories.
        """
        # Embed query
        query_embedding = self.embedder.embed(query)

        # Compute similarities
        similarities = torch.nn.functional.cosine_similarity(
            torch.tensor(query_embedding).unsqueeze(0),
            torch.tensor(self.memory_vectors)
        )

        # Retrieve top-k
        top_indices = torch.topk(similarities, k=min(top_k, len(self.memory_texts)))[1]
        relevant = [self.memory_texts[i] for i in top_indices.tolist()]

        return relevant


class LightMemoryManager:
    """
    Unified three-stage memory system.
    """

    def __init__(self, summarizer, embedder, topic_classifier):
        self.sensory = SensoryMemory()
        self.short_term = ShortTermMemory(summarizer)
        self.long_term = LongTermMemory(embedder)
        self.topic_classifier = topic_classifier

    def process_new_information(self, text: str) -> None:
        """
        Process incoming information through all stages.
        """
        # Stage 1: Sensory
        self.sensory.compress_and_group(text, self.topic_classifier)

        # Check if consolidation needed
        if len(self.sensory.buffer) > 20:
            # Stage 2: Short-term consolidation
            for topic, indices in self.sensory.topic_groups.items():
                topic_texts = [self.sensory.buffer[i]["text"] for i in indices]
                self.short_term.consolidate_topic_group(topic, topic_texts)

            # Periodically move old summaries to long-term (offline)
            for topic, summary in self.short_term.topic_summaries.items():
                if len(summary) > 50:  # Only store meaningful summaries
                    self.long_term.add_to_long_term(summary)

    def generate_context_for_query(self, query: str) -> str:
        """
        Assemble context for generation: sensory + short-term + relevant long-term.
        """
        # Sensory: current buffer (lightweight)
        sensory_context = " ".join(
            item["text"][:100] for item in self.sensory.buffer[-5:]  # Last 5 items
        )

        # Short-term: active topic summaries
        active_topics = list(self.sensory.topic_groups.keys())[:5]
        short_term_context = self.short_term.get_short_term_context(active_topics)

        # Long-term: retrieve relevant
        long_term_context = " ".join(
            self.long_term.retrieve_relevant_memories(query, top_k=2)
        )

        # Combine
        full_context = f"""
Sensory (Recent): {sensory_context}

Short-term (Organized): {short_term_context}

Long-term (Retrieved): {long_term_context}

Query: {query}
"""
        return full_context


def use_lightmem_in_conversation(
    user_inputs: List[str],
    model,
    memory_manager: LightMemoryManager
):
    """
    Multi-turn conversation with LightMem.
    """
    for user_input in user_inputs:
        # Process incoming information
        memory_manager.process_new_information(user_input)

        # Generate context
        context = memory_manager.generate_context_for_query(user_input)

        # Generate response
        response = model.generate(context, max_tokens=100)
        print(f"Response: {response}")
```

The three-stage design is crucial: Stage 1 filters fast (heuristics), Stage 2 organizes on-demand (summaries only when needed), Stage 3 stores and retrieves offline (no latency hit during inference).

## Practical Guidance

| Scenario | Token Reduction | API Call Reduction | Accuracy Change |
|----------|-----------------|-------------------|-----------------|
| Multi-turn QA | 20-30× | 20-30× | +5-8% |
| Document RAG | 10-15× | 10-15× | +3-5% |
| Online learning | 15-25× | 15-25× | +2-4% |

**When to Use:**
- Long-running conversations or sessions
- Token costs matter (pay-per-token APIs)
- Need to maintain historical context without explosion
- Multi-document or multi-topic interactions

**When NOT to Use:**
- Single-turn or short conversations
- Real-time applications where consolidation latency is problematic
- Tasks where full context is always needed
- Budget not constrained (simpler full-context approaches work)

**Common Pitfalls:**
- Sensory buffer too aggressive in filtering → loses important details
- Short-term summaries too compressed → lose nuance needed for reasoning
- Long-term retrieval misses relevant context → validate retriever quality
- Not separating online/offline → adds latency to inference
- Topic classifier too granular → fragments related information

## Reference

[LightMem: Lightweight and Efficient Memory-Augmented Generation](https://arxiv.org/abs/2510.18866)
