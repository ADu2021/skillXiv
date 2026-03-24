---
name: arag-hierarchical-retrieval
title: "A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.03442"
keywords: [Retrieval-Augmented Generation, Agent Autonomy, Hierarchical Interfaces, Information Seeking, Tool Calling]
description: "Enable LLM agents to autonomously retrieve information across multiple granularities using keyword search, semantic search, and chunk read tools. Simple ReAct-based loop with hierarchical interfaces outperforms dense retrieval by allowing adaptive information seeking without complex graph construction."
---

# A-RAG: Agent-Driven Hierarchical Retrieval

Standard RAG systems retrieve all relevant passages upfront, but this assumes the retriever knows exactly what to find. A-RAG gives LLM agents autonomy to iteratively explore documents using three complementary tools: keyword search for entity lookups, semantic search for concept matching, and chunk read for full document access. This hierarchical approach mirrors human information-seeking behavior and enables agents to refine queries based on intermediate findings.

The key insight is that agents learn effective search strategies better than fixed retrievers. By exposing multiple retrieval granularities, the system allows adaptive information gathering without requiring complex graph construction or learning separate retrieval policies.

## Core Concept

A-RAG operates on three retrieval tools that agents can call autonomously:

1. **Keyword Search**: Exact text matching for specific entities and terms
2. **Semantic Search**: Embedding-based similarity to find conceptually related passages
3. **Chunk Read**: Access full document content after identifying relevant sections

Rather than ranking documents globally, agents iteratively refine their information gathering strategy, mimicking how humans explore documents interactively.

## Architecture Overview

- **Document Index**: Standard chunking (~1000 tokens) with keyword and embedding indices
- **Retrieval Tool Set**: Three callable tools with distinct semantics
- **Agent Loop**: ReAct-style reasoning with tool calls and observations
- **Tracking Mechanism**: Remembers previously-read chunks to avoid redundant queries
- **Integration Layer**: Works with existing LLM APIs that support tool calling

## Implementation

### Step 1: Create Hierarchical Document Index

Build indices supporting all three retrieval modalities.

```python
# Hierarchical document indexing
from sentence_transformers import SentenceTransformer

class HierarchicalDocumentIndex:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Index documents for hierarchical retrieval.

        Args:
            model_name: Sentence embedding model
        """
        self.chunks = {}  # chunk_id -> text
        self.chunk_embeddings = {}  # chunk_id -> embedding
        self.keyword_index = {}  # keyword -> [chunk_ids]
        self.embedding_model = SentenceTransformer(model_name)

    def index_document(self, doc_id: str, text: str,
                      chunk_size: int = 1000):
        """Break document into chunks and build indices."""
        # Split into chunks
        words = text.split()
        chunks_text = []

        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i+chunk_size])
            chunk_id = f"{doc_id}_{i//chunk_size}"

            self.chunks[chunk_id] = chunk_text
            chunks_text.append(chunk_text)

            # Index keywords (simple: split by spaces)
            for keyword in chunk_text.lower().split():
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(chunk_id)

        # Compute embeddings
        embeddings = self.embedding_model.encode(chunks_text)
        for chunk_id, embedding in zip(
            [f"{doc_id}_{i//chunk_size}" for i in range(0, len(words), chunk_size)],
            embeddings
        ):
            self.chunk_embeddings[chunk_id] = embedding

    def keyword_search(self, query: str, top_k: int = 5) -> List[str]:
        """Search by exact keyword matching."""
        query_terms = query.lower().split()
        chunk_scores = {}

        for term in query_terms:
            if term in self.keyword_index:
                for chunk_id in self.keyword_index[term]:
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + 1

        # Sort by frequency
        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, _ in ranked[:top_k]]

    def semantic_search(self, query: str, top_k: int = 5) -> List[str]:
        """Search by semantic similarity."""
        query_embedding = self.embedding_model.encode(query)

        # Compute similarity to all chunks
        similarities = {}
        for chunk_id, embedding in self.chunk_embeddings.items():
            sim = np.dot(query_embedding, embedding)
            similarities[chunk_id] = sim

        ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, _ in ranked[:top_k]]

    def read_chunk(self, chunk_id: str) -> str:
        """Retrieve full chunk content."""
        return self.chunks.get(chunk_id, "Chunk not found")
```

### Step 2: Implement Retrieval Tool Wrapper

Create tool interface that agents can call.

```python
# Tool wrapper for agent
class RetrievalToolset:
    def __init__(self, index: HierarchicalDocumentIndex):
        """Wrap index as callable tools."""
        self.index = index
        self.accessed_chunks = set()  # Track accessed chunks

    def keyword_search_tool(self, query: str, top_k: int = 5) -> str:
        """Tool: search documents by keyword."""
        chunk_ids = self.index.keyword_search(query, top_k=top_k)

        result = f"Found {len(chunk_ids)} results for '{query}':\n"
        for i, chunk_id in enumerate(chunk_ids, 1):
            preview = self.index.chunks[chunk_id][:100]
            result += f"{i}. {chunk_id}: {preview}...\n"

        return result

    def semantic_search_tool(self, query: str, top_k: int = 5) -> str:
        """Tool: search documents by semantic similarity."""
        chunk_ids = self.index.semantic_search(query, top_k=top_k)

        result = f"Found {len(chunk_ids)} semantically similar results:\n"
        for i, chunk_id in enumerate(chunk_ids, 1):
            preview = self.index.chunks[chunk_id][:100]
            result += f"{i}. {chunk_id}: {preview}...\n"

        return result

    def read_chunk_tool(self, chunk_id: str) -> str:
        """Tool: read full chunk content."""
        if chunk_id in self.accessed_chunks:
            return "Already read this chunk. Use a different query to find new information."

        content = self.index.read_chunk(chunk_id)
        self.accessed_chunks.add(chunk_id)

        return f"Content of {chunk_id}:\n{content}"

    def get_tools_schema(self) -> List[dict]:
        """Return OpenAI-style tool schema for agent."""
        return [
            {
                "name": "keyword_search",
                "description": "Search documents by exact keyword matching",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "description": "Number of results"}
                    }
                }
            },
            {
                "name": "semantic_search",
                "description": "Search documents by semantic similarity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "top_k": {"type": "integer", "description": "Number of results"}
                    }
                }
            },
            {
                "name": "read_chunk",
                "description": "Read full content of a document chunk",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "Chunk identifier"}
                    }
                }
            }
        ]
```

### Step 3: Implement ReAct-Based Agent Loop

Create agent that reasons over tool calls.

```python
# ReAct agent for information seeking
class RAGAgent:
    def __init__(self, model: str, toolset: RetrievalToolset,
                 max_steps: int = 10):
        """
        Agent for hierarchical retrieval reasoning.

        Args:
            model: LLM API (e.g., "gpt-4")
            toolset: Retrieval tools
            max_steps: Maximum reasoning steps
        """
        self.model = model
        self.toolset = toolset
        self.max_steps = max_steps
        self.history = []

    def run(self, query: str) -> str:
        """Execute agent loop to answer query."""
        messages = []
        step = 0

        # Initial prompt
        system_prompt = """You are an information-seeking agent. Answer questions by
        using the available retrieval tools to find relevant information.
        Use keyword_search for specific terms, semantic_search for concepts,
        and read_chunk to examine full documents.
        Only call read_chunk if you found a promising chunk from search results."""

        messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        while step < self.max_steps:
            # Get agent's response
            response = self._call_model(messages)

            if response.get("stop_reason") == "end_turn":
                # Agent finished reasoning
                return response.get("content", "")

            # Parse tool calls
            if "tool_calls" not in response:
                return response.get("content", "")

            # Execute tools
            tool_results = []
            for tool_call in response["tool_calls"]:
                result = self._execute_tool(tool_call)
                tool_results.append({
                    "tool_use_id": tool_call["id"],
                    "content": result
                })

            # Add to conversation
            messages.append({"role": "assistant", "content": response["content"]})
            messages.append({"role": "user", "content": tool_results})

            step += 1

        return "Max steps exceeded"

    def _call_model(self, messages: List[dict]) -> dict:
        """Call LLM with tools."""
        # Pseudo-code: actual implementation uses OpenAI API
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            tools=self.toolset.get_tools_schema(),
            tool_choice="auto",
            temperature=0.7
        )
        return response

    def _execute_tool(self, tool_call: dict) -> str:
        """Execute a tool call."""
        tool_name = tool_call["name"]
        params = tool_call["parameters"]

        if tool_name == "keyword_search":
            return self.toolset.keyword_search_tool(
                params["query"],
                params.get("top_k", 5)
            )
        elif tool_name == "semantic_search":
            return self.toolset.semantic_search_tool(
                params["query"],
                params.get("top_k", 5)
            )
        elif tool_name == "read_chunk":
            return self.toolset.read_chunk_tool(params["chunk_id"])

        return "Tool not found"
```

### Step 4: Full Pipeline Integration

Combine indexing, tools, and agent.

```python
# Full A-RAG pipeline
def build_arag_system(documents: List[str],
                     model: str = "gpt-4") -> RAGAgent:
    """
    Build complete A-RAG system from documents.

    Args:
        documents: List of document texts
        model: LLM to use

    Returns:
        Initialized RAG agent
    """
    # Build index
    index = HierarchicalDocumentIndex()
    for i, doc in enumerate(documents):
        index.index_document(f"doc_{i}", doc)

    # Create toolset
    toolset = RetrievalToolset(index)

    # Create agent
    agent = RAGAgent(model, toolset)

    return agent

def query_arag(agent: RAGAgent, question: str) -> str:
    """Query the A-RAG system."""
    return agent.run(question)
```

## Practical Guidance

**When to use A-RAG:**
- Large document collections where agents benefit from exploration
- Questions requiring cross-document synthesis
- Scenarios where query refinement helps (agent learns to search better)
- Open-ended information seeking where initial query may be imprecise

**When not to use:**
- Simple fact lookup where keyword search suffices
- Real-time systems where agent reasoning latency is prohibitive
- Scenarios needing guaranteed retrieval (agents may get stuck)
- Highly structured data better served by semantic search alone

**Common Pitfalls:**
- Agent getting stuck in loops: Set reasonable max_steps (5-10 typical)
- Too many tool calls: Penalize redundant searches; track accessed chunks
- Poor search results: Ensure index quality and semantic model are appropriate
- Incomplete answers: Agents may stop early; prompt for synthesis step at end

**Hyperparameter Guidelines:**

| Parameter | Range | Tuning |
|-----------|-------|--------|
| max_steps | 5-15 | Higher = more exploration; balance with latency |
| chunk_size | 500-1500 | Larger = faster semantic search; smaller = more precision |
| top_k (search) | 3-10 | Balance result diversity with noise |
| embedding model | small models for speed | Use small (MiniLM) for latency; large (e5) for quality |

## Reference

See the full paper at: https://arxiv.org/abs/2602.03442

Key results: Outperforms dense retrieval by enabling adaptive information seeking. Simplest agent loop backbone for reproducibility. Code and evaluation suite released on GitHub. Works with modern LLM APIs supporting tool calling.
