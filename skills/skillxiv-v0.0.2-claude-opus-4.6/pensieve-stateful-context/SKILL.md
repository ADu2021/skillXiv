---
name: pensieve-stateful-context
title: "The Pensieve Paradigm: Stateful Language Models Mastering Context"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.12108"
keywords: [Stateful Language Models, Context Management, Memory Tools, Long-Horizon Tasks, Active State Engineering]
description: "Enable language models to actively manage their context using memory tools (read, index, note-take, delete). Models receive agency to self-engineer context rather than passively consume pre-staged information, maintaining compact high-quality reasoning states through strategic deletion of irrelevant information while preserving distilled notes."
---

# The Pensieve Paradigm: Stateful Language Models Mastering Context

## Problem Context

Current language models passively receive context that humans curate, creating a bottleneck: they cannot adapt information selection to specific reasoning needs. The Pensieve Paradigm shifts agency to the model itself: models receive memory tools and actively manage context throughout long-horizon tasks. The key insight is transforming from monotonic context accumulation (until context window exhaustion) to sawtooth profiles (strategically deleting irrelevant information while preserving high-value summaries).

## Core Concept

Models coordinate four memory operations: (1) **read**: retrieve chunks from document corpus, (2) **index**: catalog retrieved information, (3) **note**: take distilled notes on key facts, (4) **delete**: remove irrelevant context. This maintains a compact, high-quality reasoning state throughout extended tasks. The model gains the "wand" to use memory tools directly (unlike external retrieval systems that manipulate context without model input).

## Architecture Overview

- **Memory toolkit**: Read, index, note-take, delete operations
- **Tool orchestration**: Model decides which tool at each step
- **State management**: Track current context, identified facts, deleted items
- **Distillation**: Summarize information into notes before deletion
- **Long-horizon coordination**: Manage reasoning state across 100+ steps

## Implementation

### Step 1: Memory tool interfaces

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

@dataclass
class MemoryState:
    """Current state of model's working memory."""
    current_context: str       # Current active reasoning state
    indexed_facts: Dict[str, str]  # {fact_id: fact_text}
    notes: List[str]          # Distilled notes
    retrieved_docs: Dict[str, str]  # {doc_id: content}
    deleted_context: List[str]    # Log of deleted information

class MemoryToolkit:
    """Interfaces for memory management."""

    def __init__(self, document_corpus: Dict[str, str]):
        """
        Args:
            document_corpus: {doc_id: content} mapping
        """
        self.corpus = document_corpus
        self.state = MemoryState(
            current_context="",
            indexed_facts={},
            notes=[],
            retrieved_docs={},
            deleted_context=[]
        )

    def read(
        self,
        query: str,
        num_chunks: int = 3
    ) -> List[str]:
        """
        Retrieve relevant document chunks.

        Args:
            query: Natural language query
            num_chunks: Number of chunks to retrieve

        Returns:
            retrieved_chunks: List of document excerpts
        """
        # Simplified: keyword matching (would use semantic search)
        query_keywords = query.lower().split()
        scores = {}

        for doc_id, content in self.corpus.items():
            score = sum(
                content.lower().count(kw) for kw in query_keywords
            )
            if score > 0:
                scores[doc_id] = score

        # Get top-k documents
        top_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_chunks]
        retrieved = [self.corpus[doc_id] for doc_id, _ in top_docs]

        # Store in memory
        for doc_id, _ in top_docs:
            self.state.retrieved_docs[doc_id] = self.corpus[doc_id]

        return retrieved

    def index(
        self,
        facts: List[str],
        category: str = "general"
    ):
        """
        Index and catalog facts from current context.

        Args:
            facts: List of factual statements to index
            category: Topic category for organization
        """
        for fact_idx, fact in enumerate(facts):
            fact_id = f"{category}_{fact_idx}"
            self.state.indexed_facts[fact_id] = fact

    def note(
        self,
        summary: str,
        reference_ids: Optional[List[str]] = None
    ):
        """
        Take a distilled note on important information.

        Args:
            summary: Concise summary of key points
            reference_ids: Fact IDs this note summarizes
        """
        note_entry = {
            'summary': summary,
            'references': reference_ids or [],
            'timestamp': len(self.state.notes)
        }
        self.state.notes.append(json.dumps(note_entry))

    def delete(
        self,
        fact_ids: List[str]
    ) -> List[str]:
        """
        Delete irrelevant facts from indexed memory.

        Args:
            fact_ids: IDs of facts to delete

        Returns:
            deleted_facts: Deleted fact texts (for logging)
        """
        deleted = []
        for fact_id in fact_ids:
            if fact_id in self.state.indexed_facts:
                deleted.append(self.state.indexed_facts.pop(fact_id))
                self.state.deleted_context.append(fact_id)

        return deleted

    def get_current_state_summary(self) -> str:
        """Get compact representation of current memory state."""
        summary = f"""
## Current Reasoning State

**Notes**: {len(self.state.notes)} items
{chr(10).join(self.state.notes[-3:]) if self.state.notes else 'None'}

**Indexed Facts**: {len(self.state.indexed_facts)} items
{chr(10).join(list(self.state.indexed_facts.values())[-3:]) if self.state.indexed_facts else 'None'}

**Context Length**: {len(self.state.current_context)} characters
"""
        return summary
```

### Step 2: Model-directed memory operations

```python
class StateAwareLanguageModel:
    """Language model that directs memory tool usage."""

    def __init__(self, base_model, memory_toolkit: MemoryToolkit):
        self.model = base_model
        self.memory = memory_toolkit

    def forward_with_memory_planning(
        self,
        task: str,
        max_steps: int = 50,
        context_limit: int = 8000
    ) -> Dict:
        """
        Execute task with active memory management.

        Args:
            task: Reasoning task
            max_steps: Maximum reasoning steps
            context_limit: Token limit for context window

        Returns:
            result: {final_answer, memory_history, context_trajectory}
        """
        trajectory = []
        memory_history = []

        for step in range(max_steps):
            # Get current state summary
            state_summary = self.memory.get_current_state_summary()

            # Check if context is approaching limit
            context_length = len(state_summary) + len(self.memory.state.current_context)
            if context_length > context_limit:
                # Trigger memory cleanup
                self._cleanup_memory()

            # Generate next action
            prompt = f"""
Task: {task}

Current Memory State:
{state_summary}

What is your next action? Choose from:
1. read(query) - Retrieve documents
2. index(facts) - Catalog facts
3. note(summary) - Take summary note
4. delete(fact_ids) - Remove irrelevant facts
5. reason() - Continue reasoning
6. answer() - Provide final answer

Action:"""

            action_text = self.model.generate(prompt, max_tokens=100, temperature=0.7)
            trajectory.append({'step': step, 'action': action_text})

            # Execute tool call
            if 'read(' in action_text:
                query = self._extract_arg(action_text, 'read')
                chunks = self.memory.read(query)
                memory_history.append(f"Read: {query}")
            elif 'note(' in action_text:
                summary = self._extract_arg(action_text, 'note')
                self.memory.note(summary)
                memory_history.append(f"Note: {summary[:50]}...")
            elif 'delete(' in action_text:
                fact_ids = self._extract_arg(action_text, 'delete').split(',')
                self.memory.delete(fact_ids)
                memory_history.append(f"Delete: {len(fact_ids)} facts")
            elif 'answer(' in action_text:
                answer = self._extract_arg(action_text, 'answer')
                return {
                    'answer': answer,
                    'steps': len(trajectory),
                    'memory_ops': len(memory_history),
                    'final_state': self.memory.state
                }

        return {
            'answer': "Max steps reached",
            'steps': max_steps,
            'memory_ops': len(memory_history),
            'trajectory': trajectory
        }

    def _cleanup_memory(self):
        """Compact memory when context approaches limit."""
        # Identify less important facts
        important_ids = [
            fid for fid in list(self.memory.state.indexed_facts.keys())
            if self._score_fact_importance(fid) > 0.5
        ]

        # Delete low-importance facts, but note their key points first
        to_delete = [
            fid for fid in self.memory.state.indexed_facts.keys()
            if fid not in important_ids
        ]

        if to_delete:
            # Summarize before deletion
            summary = self._summarize_facts(to_delete)
            self.memory.note(summary, reference_ids=to_delete)
            self.memory.delete(to_delete)

    def _score_fact_importance(self, fact_id: str) -> float:
        """Score how important a fact is to current task."""
        # Simplified: could use semantic similarity
        return 0.5

    def _summarize_facts(self, fact_ids: List[str]) -> str:
        """Generate summary of facts being deleted."""
        facts = [self.memory.state.indexed_facts[fid] for fid in fact_ids if fid in self.memory.state.indexed_facts]
        summary_prompt = f"""Summarize these facts in one sentence:
{chr(10).join(facts)}

Summary:"""
        return self.model.generate(summary_prompt, max_tokens=30, temperature=0.3)

    def _extract_arg(self, action_text: str, tool_name: str) -> str:
        """Extract argument from tool call."""
        # Simple extraction: e.g., read("query") -> "query"
        import re
        match = re.search(f'{tool_name}\("([^"]+)"\)', action_text)
        return match.group(1) if match else ""
```

### Step 3: Context trajectory analysis

```python
class ContextTrajectoryAnalyzer:
    """Analyze how model uses context over time."""

    @staticmethod
    def plot_context_usage(trajectory: Dict) -> Dict:
        """
        Analyze context growth and deletion patterns.

        Returns metrics showing "sawtooth" pattern.
        """
        memory_ops = trajectory.get('memory_ops', 0)
        steps = trajectory.get('steps', 0)

        context_growth = []
        for step in trajectory.get('trajectory', []):
            if 'read' in step['action']:
                context_growth.append(1)  # Add
            elif 'delete' in step['action']:
                context_growth.append(-1)  # Remove
            else:
                context_growth.append(0)  # Neutral

        # Compute sawtooth pattern (ideal: frequent ups and downs)
        deletions = sum(1 for x in context_growth if x < 0)
        additions = sum(1 for x in context_growth if x > 0)

        return {
            'sawtooth_ratio': deletions / max(1, additions),  # Close to 1 is good
            'memory_operations': memory_ops,
            'steps_to_completion': steps,
            'avg_ops_per_step': memory_ops / max(1, steps)
        }

    @staticmethod
    def evaluate_context_efficiency(
        final_answer: str,
        context_ops: int,
        context_deletions: int
    ) -> float:
        """
        Score efficiency of context management.

        High score: achieved good answer with few operations and active deletion.
        """
        efficiency = (1.0 + context_deletions / max(1, context_ops)) / 2.0
        return min(1.0, efficiency)
```

### Step 4: Training with memory operations

```python
def train_stateful_lm(
    model,
    tasks: List[Dict],  # {task, gold_answer}
    memory_toolkit: MemoryToolkit,
    optimizer,
    num_epochs: int = 10,
    device: str = 'cuda'
):
    """
    Train model to use memory tools effectively.

    Reward for: correct answer, low context ops, high deletion ratio.
    """
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_reward = 0.0

        for task_dict in tasks:
            task = task_dict['task']
            gold_answer = task_dict['answer']

            # Run task with memory management
            state_aware = StateAwareLanguageModel(model, memory_toolkit)
            result = state_aware.forward_with_memory_planning(task)

            # Compute reward
            is_correct = result['answer'].lower() == gold_answer.lower()
            efficiency = ContextTrajectoryAnalyzer.evaluate_context_efficiency(
                result['answer'],
                result['memory_ops'],
                len(result['final_state'].deleted_context)
            )

            reward = float(is_correct) * 0.7 + efficiency * 0.3

            # Simple RL update (pseudo-code)
            # loss = -log_prob * reward
            total_reward += reward

        avg_reward = total_reward / len(tasks)
        print(f"Epoch {epoch + 1}: Avg Reward = {avg_reward:.4f}")

    return model
```

## Practical Guidance

**When to use**: Long-horizon reasoning (100+ steps); information-rich domains; tasks requiring selective fact retention

**Hyperparameters**:
- **context_limit**: 4K-32K tokens (window size)
- **cleanup_threshold**: 0.7-0.9 (when to trigger compression)
- **note_importance_weight**: 0.5-2.0 (value of notes vs. raw facts)
- **deletion_incentive**: 0.1-0.5 (bonus for removing facts)

**Key advantages**:
- Models adapt information selection to tasks
- Maintains compact reasoning state
- Handles 100+ step trajectories
- Transparent memory operations

**Common pitfalls**:
- Over-deletion → loses important context
- Under-deletion → context window exhaustion
- Notes too verbose → defeats compression
- Tool calls not grounded in actual retrieval

**Scaling**: Number of documents linearly affects retrieval cost; use semantic indexing for large corpora.

## Reference

Paper: https://arxiv.org/abs/2602.12108
Related work: Retrieval-augmented generation, active learning, state management
Benchmarks: Long-horizon question-answering, multi-document reasoning
Metaphor: Pensieve (Harry Potter) — memory tool controlled by the user
