---
name: hypergraph-memory-rag
title: "Improving Multi-step RAG with Hypergraph-based Memory for Long-Context Complex Relational Modeling"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2512.23959"
keywords: [RAG, retrieval-augmented-generation, multi-step reasoning, hypergraph memory, knowledge representation, LLM]
description: "Build hypergraph-structured memory systems for multi-step RAG that capture high-order relationships between facts, enabling stronger reasoning across long contexts. Use when combining multiple retrieved documents in complex reasoning chains that require understanding connections between pieces of information."
---

## When to Use This Skill

- Multi-step QA systems requiring reasoning across multiple documents
- Long-context reasoning tasks that need to maintain relationships between facts
- Complex relational modeling where document connections matter
- Workflows combining retrieval with iterative refinement

## When NOT to Use This Skill

- Single-step information retrieval tasks
- Simple keyword-based lookup without reasoning requirements
- Tasks where retrieved chunks are independent
- Real-time systems with strict latency constraints (hypergraph operations add overhead)

## Core Concepts

Traditional RAG systems store retrieved information as isolated facts. HGMem instead represents this memory as a hypergraph where:
- **Nodes** represent facts, thoughts, or retrieved passages
- **Hyperedges** create higher-order interactions linking 3+ concepts together
- **Graph structure** evolves as new information is retrieved and integrated

This enables the system to form "stronger propositions for deeper reasoning" by understanding how multiple facts relate to each other.

## Implementation Pattern

The hypergraph memory approach proceeds through three phases:

**1. Fact Insertion**
Each retrieved document or generated thought is inserted as a node with semantic embedding and metadata about its provenance.

**2. Relationship Formation**
As new information arrives, the system identifies which existing nodes should be connected via hyperedges. This captures semantic or logical relationships (e.g., "Fact A explains Fact B", "Entity X appears in both C and D").

**3. Reasoning Over Hypergraph**
When generating the next reasoning step, traverse the hypergraph to gather contextually relevant clusters of connected facts rather than individual isolated pieces.

## Python Pseudocode Structure

```python
# Core hypergraph memory operations for RAG
class HypergraphMemory:
    def __init__(self, embedding_model):
        self.nodes = {}  # id -> {embedding, text, metadata}
        self.hyperedges = []  # list of node sets
        self.embedding_model = embedding_model

    def add_fact(self, text, source_id, metadata):
        """Insert a retrieved fact as a node"""
        embedding = self.embedding_model.encode(text)
        node_id = len(self.nodes)
        self.nodes[node_id] = {
            'text': text,
            'embedding': embedding,
            'source': source_id,
            'metadata': metadata
        }
        return node_id

    def form_hyperedge(self, node_ids, relationship_type):
        """Create a higher-order interaction between 3+ nodes"""
        if len(node_ids) < 3:
            return  # Require at least 3 nodes for hyperedge
        hyperedge = {
            'nodes': node_ids,
            'type': relationship_type,
            'timestamp': current_step
        }
        self.hyperedges.append(hyperedge)

    def retrieve_context_cluster(self, query_embedding, k_hyperedges=3):
        """Retrieve connected fact clusters relevant to query"""
        relevant_edges = self._find_relevant_hyperedges(query_embedding)
        context = []
        for edge in relevant_edges[:k_hyperedges]:
            cluster = [self.nodes[nid]['text'] for nid in edge['nodes']]
            context.extend(cluster)
        return context
```

## Integration with RAG Pipeline

1. **Retrieval Phase**: Standard dense retrieval (BM25/embedding) returns documents
2. **Memory Integration**: Insert retrieved chunks into hypergraph as nodes
3. **Relationship Detection**: Identify cross-document entities/concepts to form hyperedges
4. **Generation**: Access memory via hypergraph traversal instead of flat chunk list
5. **Iteration**: New retrieved documents in next step integrate into existing graph structure

## Key Benefits

- **Reasoning Depth**: Multi-hop relationships are explicit in the structure
- **Scalability**: Hyperedges scale to arbitrary relationships (not limited to pairwise)
- **Interpretability**: Graph structure reveals how the model connected pieces of information
- **Iterative Improvement**: Each reasoning step refines and extends the memory graph

## Trade-offs

| Aspect | Trade-off |
|--------|-----------|
| Speed | Hypergraph construction adds ~10-15% overhead vs. flat context |
| Memory | Storing relationships increases space proportional to connection density |
| Benefit | Better reasoning on complex multi-hop questions compensates |

## References

- Original paper: https://arxiv.org/abs/2512.23959
- Related work: Dense Passage Retrieval (DPR), RETRO, Self-Ask
- Implementation consideration: Efficient hypergraph libraries (NetworkX, DGL)
