---
name: memory-operating-system
title: "MemOS: A Memory OS for AI System"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.03724"
keywords: [Memory Management, Knowledge Storage, LLM Optimization, Parameter Efficiency, Hybrid Memory]
description: "Treat memory as a manageable system resource for LLMs through unified management of plaintext, activation, and parameter-level memories with dynamic scheduling and lifecycle governance."
---

# MemOS: Operating System for AI Memory Management

Large language models incur high computational costs partly because knowledge exists across three forms: plaintext memories (documents, retrieval databases), activation-level memories (KV-caches, hidden states), and parameter-level memories (weights). Current systems manage these independently, leading to redundant storage, inefficient retrieval, and wasted computation. MemOS proposes a unified operating system architecture that treats memory as a managed resource, enabling cost-efficient storage, dynamic scheduling, and cross-type transformations. This approach reduces inference costs while maintaining or improving model performance.

The core innovation is recognizing that memory forms exist on a spectrum and can be dynamically transformed. Important context can move from plaintext (cheap storage, expensive retrieval) to activation memory (expensive storage, fast access) to parameters (static embedding). A proper OS layer optimizes these transitions based on frequency of use, storage constraints, and performance requirements.

## Core Concept

MemOS introduces the MemCube abstraction: each piece of knowledge is encapsulated with metadata indicating its type, governance attributes, access patterns, and current state. Rather than manually managing where knowledge lives, the system can automatically evolve memories across types. Frequently accessed plaintext documents can be compressed into parameters. Activation memories can be intelligently evicted or restructured. The scheduler makes these decisions based on cost models and performance metrics.

Three design principles guide the architecture:

- **Controllability**: Full lifecycle governance of memory from creation through archival
- **Plasticity**: Ability to restructure and migrate memories between types
- **Evolvability**: Dynamic transitions optimizing for changing workloads

## Architecture Overview

The system follows three layers:

- **Interface Layer**: Memory API for declarative operations, MemReader for natural language parsing of memory queries, and composable memory pipelines
- **Operation Layer**: MemOperator handles memory structuring and hybrid retrieval; MemScheduler dynamically selects memory types; MemLifecycle manages state transitions and governance
- **Infrastructure Layer**: MemGovernance for access control, MemVault for storage repositories, MemLoader/MemDumper for import/export, and MemStore for sharing mechanisms

## Implementation

Start with the MemCube abstraction as the fundamental unit:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Literal
from enum import Enum

class MemoryType(Enum):
    """Three orthogonal memory types in the system."""
    PLAINTEXT = "plaintext"      # External documents, retrievable
    ACTIVATION = "activation"     # Runtime states, KV-cache, hidden states
    PARAMETER = "parameter"       # Embedded in model weights

class MemCube:
    """
    Fundamental unit of memory with rich metadata.

    Encapsulates memory content with descriptive identifiers, governance
    attributes, and behavioral indicators enabling lifecycle control and
    cross-type transformations.
    """

    def __init__(self, content_id: str, content: str, memory_type: MemoryType):
        self.content_id = content_id
        self.content = content
        self.memory_type = memory_type

        # Metadata for lifecycle and optimization
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.access_frequency = 0.0  # Accesses per unit time
        self.storage_cost = self._estimate_storage_cost()
        self.retrieval_cost = self._estimate_retrieval_cost()
        self.governance_attrs = {}  # Access control, retention policies

    def _estimate_storage_cost(self) -> float:
        """Estimate cost of storing in current memory type."""
        if self.memory_type == MemoryType.PLAINTEXT:
            return len(self.content) * 0.01  # Cheap storage
        elif self.memory_type == MemoryType.ACTIVATION:
            return len(self.content) * 1.0   # Expensive in-GPU memory
        elif self.memory_type == MemoryType.PARAMETER:
            return len(self.content) * 0.5   # Moderate, persistent
        return 0.0

    def _estimate_retrieval_cost(self) -> float:
        """Estimate cost of accessing memory."""
        if self.memory_type == MemoryType.PLAINTEXT:
            return 0.5  # Slow retrieval (vector search, I/O)
        elif self.memory_type == MemoryType.ACTIVATION:
            return 0.01  # Fast (in-GPU access)
        elif self.memory_type == MemoryType.PARAMETER:
            return 0.1   # Fast forward pass through weights
        return 0.0

    def record_access(self):
        """Update access statistics for scheduling decisions."""
        self.last_accessed = datetime.now()
        self.access_count += 1
        time_delta = (datetime.now() - self.created_at).total_seconds()
        self.access_frequency = self.access_count / max(time_delta, 1)

    def get_cost_score(self) -> float:
        """Compute total cost: storage + retrieval weighted by usage."""
        return self.storage_cost + (self.retrieval_cost * self.access_frequency)
```

Implement the MemScheduler to make dynamic type decisions:

```python
from typing import Tuple

class MemScheduler:
    """
    Dynamically selects optimal memory type for each MemCube.

    Monitors access patterns, storage constraints, and performance metrics
    to decide whether knowledge should live in plaintext, activation, or
    parameter memory. Recommends transformations to minimize total cost.
    """

    def __init__(self, total_activation_budget: float = 1000,
                 total_parameter_budget: float = 50000):
        self.activation_budget = total_activation_budget
        self.activation_used = 0.0
        self.parameter_budget = total_parameter_budget
        self.parameter_used = 0.0

    def schedule(self, mem_cubes: List[MemCube]) -> List[Tuple[MemCube, MemoryType]]:
        """
        Assign optimal memory type to each cube given budget constraints.

        Uses access patterns and cost estimates to recommend memory type
        transitions that minimize total cost while respecting capacity limits.
        """
        # Sort by urgency (high-access, high-cost items first)
        sorted_cubes = sorted(
            mem_cubes,
            key=lambda m: m.get_cost_score() * m.access_frequency,
            reverse=True
        )

        assignments = []
        temp_activation_used = 0.0
        temp_parameter_used = 0.0

        for cube in sorted_cubes:
            # Try to allocate to optimal type
            if cube.access_frequency > 0.5:
                # Hot memory: prefer activation for speed
                if temp_activation_used + cube.storage_cost < self.activation_budget:
                    assignments.append((cube, MemoryType.ACTIVATION))
                    temp_activation_used += cube.storage_cost
                elif temp_parameter_used + cube.storage_cost < self.parameter_budget:
                    assignments.append((cube, MemoryType.PARAMETER))
                    temp_parameter_used += cube.storage_cost
                else:
                    assignments.append((cube, MemoryType.PLAINTEXT))
            else:
                # Cold memory: prefer plaintext for cost
                assignments.append((cube, MemoryType.PLAINTEXT))

        self.activation_used = temp_activation_used
        self.parameter_used = temp_parameter_used
        return assignments

    def get_memory_report(self) -> Dict[str, Any]:
        """Return current memory usage and utilization statistics."""
        return {
            'activation_used': self.activation_used,
            'activation_budget': self.activation_budget,
            'activation_utilization': self.activation_used / self.activation_budget,
            'parameter_used': self.parameter_used,
            'parameter_budget': self.parameter_budget,
            'parameter_utilization': self.parameter_used / self.parameter_budget
        }
```

Implement hybrid retrieval that respects memory types:

```python
import numpy as np
from typing import List, Tuple

class MemOperator:
    """
    Performs hybrid retrieval and structuring across memory types.

    Coordinates access to plaintext documents, activation caches, and
    parameter-embedded knowledge, optimizing for latency and accuracy.
    """

    def __init__(self, scheduler: MemScheduler):
        self.scheduler = scheduler
        self.plaintext_retriever = None  # Pluggable (vector DB, BM25)
        self.activation_cache = {}
        self.mem_cubes: List[MemCube] = []

    def add_memory(self, cube: MemCube):
        """Register a new memory unit in the system."""
        self.mem_cubes.append(cube)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve relevant memories across all types given a query.

        Searches plaintext documents and parameter memories according to
        scheduling decisions, returning ranked results by relevance.
        """
        results = []

        for cube in self.mem_cubes:
            cube.record_access()

            # Retrieve from plaintext using dense search
            if cube.memory_type == MemoryType.PLAINTEXT:
                # Placeholder: call vector DB retriever
                relevance = self.compute_relevance(query, cube.content)
                if relevance > 0.5:
                    results.append((cube.content, relevance))

            # Fast in-GPU activation retrieval
            elif cube.memory_type == MemoryType.ACTIVATION:
                relevance = self.compute_relevance(query, cube.content)
                results.append((cube.content, relevance))

            # Parameter-embedded knowledge (via forward pass)
            elif cube.memory_type == MemoryType.PARAMETER:
                relevance = self.compute_relevance(query, cube.content)
                if relevance > 0.3:  # Lower threshold for parameter memory
                    results.append((cube.content, relevance))

        # Sort by relevance and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def compute_relevance(self, query: str, text: str) -> float:
        """Compute semantic relevance between query and memory content."""
        # Placeholder: use embeddings or semantic similarity
        # In practice, use BERT embeddings, cosine similarity, etc.
        return np.random.random()  # Stub for demo

    def optimize_memory_types(self):
        """Reassign memory types based on current access patterns."""
        assignments = self.scheduler.schedule(self.mem_cubes)
        for cube, optimal_type in assignments:
            cube.memory_type = optimal_type
```

## Practical Guidance

**Hyperparameter Table:**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Activation memory budget | 1000 | 100-10000 | Total GPU memory for activation states |
| Parameter memory budget | 50000 | 1000-1M | Model parameter capacity for embedded knowledge |
| Access frequency threshold | 0.5 | 0.1-2.0 | Decides activation vs plaintext allocation |
| Plaintext retrieval cost | 0.5 | 0.1-1.0 | Relative cost for vector search |
| Activation cost multiplier | 1.0 | 0.5-5.0 | Multiplier on storage cost for GPU memory |

**When to Use:**
- You have heterogeneous memory requirements (retrieval DB, long-context, parameter knowledge)
- You want to optimize inference cost across multiple memory types
- You're managing large amounts of knowledge and need principled allocation
- You have budget constraints on GPU memory and storage
- You need to dynamically adapt memory as access patterns change

**When NOT to Use:**
- Your system has unlimited memory budgets
- Memory access patterns are stable and predictable (static assignment sufficient)
- You need sub-millisecond latency (scheduling overhead may hurt)
- Your knowledge is entirely in one type (homogeneous systems)
- You lack clear cost models for your memory types

**Common Pitfalls:**
- **Inaccurate cost estimates**: If storage/retrieval costs are wrong, scheduling makes poor decisions. Profile your actual system.
- **Insufficient budget**: If total budgets are too small, everything falls back to plaintext. Size budgets based on workload.
- **Access pattern volatility**: If patterns change rapidly, scheduling can thrash between types. Add hysteresis to transitions.
- **Memory type incompatibility**: Not all knowledge fits in parameter memory; some requires plaintext structure. Validate before assignment.
- **Governance conflicts**: Access control policies can conflict with optimization. Clarify retention and access requirements first.
- **Lack of monitoring**: Without visibility into actual costs and utilization, you can't tune the system. Implement metrics.

## Reference

Authors (2025). MemOS: A Memory OS for AI System. arXiv preprint arXiv:2507.03724. https://arxiv.org/abs/2507.03724
