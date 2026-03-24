---
name: agentfly-memory-augmented-agents
title: "AgentFly: Memory-Augmented Learning Without LLM Fine-tuning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.16153
keywords: [memory-augmented-learning, agent-adaptation, episodic-memory, reinforcement-learning, in-context-learning]
description: "Enable agent learning through episodic memory and neural case selection without fine-tuning the underlying LLM, achieving efficient continual adaptation via policy updates in memory space."
---

# AgentFly: Memory-Augmented Learning Without LLM Fine-tuning

## Core Concept

AgentFly (Memento) enables language model agents to learn and adapt without fine-tuning the base LLM. Instead, it uses memory-augmented online reinforcement learning with episodic memory storage and a learned case-selection policy. Past experiences are stored in differentiable or non-parametric memory, and a neural policy selects relevant experiences to guide decision-making. This approach achieves rapid adaptation on research-oriented tasks while maintaining the benefits of frozen, pre-trained models.

## Architecture Overview

- **Episodic Memory System**: Stores experience tuples (state, action, outcome)
- **Neural Case-Selection Policy**: Learns which memories are relevant for decisions
- **Memory-Augmented MDP (M-MDP)**: Markov decision process with memory state
- **Continual Adaptation**: Policy updates through memory rewriting
- **Frozen LLM Base**: No gradient updates to base model parameters

## Implementation Steps

### 1. Design Episodic Memory Structure

Create abstraction for experience storage:

```python
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import numpy as np
from collections import deque

@dataclass
class Experience:
    """Single experience stored in memory."""
    state: str  # Task context or observation
    action: str  # Agent action taken
    observation: str  # Result/feedback
    reward: float  # Task reward signal
    success: bool  # Was action successful?
    timestamp: float
    metadata: Dict[str, Any] = None  # Additional context

@dataclass
class MemoryTrace:
    """Linked traces of related experiences."""
    experiences: List[Experience]
    task_id: str
    success_rate: float
    last_accessed: float

class EpisodicMemory:
    """Storage for agent experiences."""

    def __init__(
        self,
        max_size: int = 10000,
        similarity_metric: str = "embedding"
    ):
        self.experiences: deque = deque(maxlen=max_size)
        self.max_size = max_size
        self.similarity_metric = similarity_metric
        self.task_groups: Dict[str, List[Experience]] = {}

    def store_experience(self, experience: Experience):
        """Add experience to memory."""
        self.experiences.append(experience)

        # Group by task for later retrieval
        task_id = experience.metadata.get("task_id", "default")
        if task_id not in self.task_groups:
            self.task_groups[task_id] = []
        self.task_groups[task_id].append(experience)

    def retrieve_similar(
        self,
        query_state: str,
        k: int = 5,
        filter_task: Optional[str] = None
    ) -> List[Tuple[Experience, float]]:
        """Retrieve k most similar experiences to query."""

        candidates = self.task_groups.get(filter_task) if filter_task else list(self.experiences)

        if not candidates:
            return []

        # Compute similarity scores
        similarities = []
        for exp in candidates:
            similarity = self._compute_similarity(query_state, exp.state)
            similarities.append((exp, similarity))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def _compute_similarity(self, state1: str, state2: str) -> float:
        """Compute similarity between two states."""

        if self.similarity_metric == "embedding":
            # Use embedding-based similarity
            emb1 = self._encode(state1)
            emb2 = self._encode(state2)
            return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)

        elif self.similarity_metric == "string":
            # Simple string overlap
            set1 = set(state1.split())
            set2 = set(state2.split())
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            return intersection / (union + 1e-8)

        return 0.0

    def _encode(self, text: str) -> np.ndarray:
        """Encode text to embedding."""
        # In practice: use sentence encoder
        pass

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Compute memory health metrics."""
        return {
            "total_experiences": len(self.experiences),
            "unique_tasks": len(self.task_groups),
            "avg_task_size": len(self.experiences) / max(len(self.task_groups), 1),
            "memory_utilization": len(self.experiences) / self.max_size
        }
```

### 2. Implement Neural Case-Selection Policy

Learn which memories to retrieve for decisions:

```python
import torch
import torch.nn as nn

class CaseSelectionPolicy(nn.Module):
    """Policy for selecting relevant cases from memory."""

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        max_cases: int = 5
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_cases = max_cases

        # Encoder for current state
        self.state_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # Relevance scorer for each candidate case
        self.relevance_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Selection gate for whether to use memory at all
        self.use_memory_gate = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        current_state: torch.Tensor,  # (embedding_dim,)
        candidate_cases: List[torch.Tensor],  # [(embedding_dim,), ...]
        return_scores: bool = False
    ) -> torch.Tensor:
        """
        Select top cases and return their indices/scores.
        """
        # Encode current state
        state_encoded = self.state_encoder(current_state)

        # Compute relevance for each candidate
        relevance_scores = []
        for case in candidate_cases:
            combined = torch.cat([state_encoded, case])
            score = self.relevance_scorer(combined)
            relevance_scores.append(score)

        relevance_scores = torch.stack(relevance_scores).squeeze(-1)

        # Decide whether to use memory
        memory_use_prob = self.use_memory_gate(state_encoded)

        # Select top-k
        topk_scores, topk_indices = torch.topk(
            relevance_scores,
            k=min(self.max_cases, len(candidate_cases))
        )

        if return_scores:
            return topk_indices, topk_scores

        return topk_indices

    def train_on_trajectory(
        self,
        states: List[torch.Tensor],
        selected_cases: List[int],
        outcomes: List[float],
        gamma: float = 0.99
    ):
        """Train policy on trajectory data."""

        # Compute returns
        returns = []
        G = 0
        for outcome in reversed(outcomes):
            G = outcome + gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss
        total_loss = 0.0
        for state, case_idx, ret in zip(states, selected_cases, returns):
            # Get probability of selected case
            with torch.no_grad():
                logits = self.relevance_scorer(state)

            # Compute policy gradient
            log_prob = torch.log_softmax(logits, dim=-1)[case_idx]
            loss = -log_prob * ret

            total_loss += loss

        return total_loss / len(states)
```

### 3. Implement Memory-Augmented MDP

Integrate memory into decision process:

```python
class MemoryAugmentedMDP:
    """Markov decision process with episodic memory."""

    def __init__(
        self,
        base_model: "LLM",
        memory: EpisodicMemory,
        selection_policy: CaseSelectionPolicy,
        max_memory_context_length: int = 2000
    ):
        self.base_model = base_model
        self.memory = memory
        self.selection_policy = selection_policy
        self.max_context_length = max_memory_context_length

    def get_action(
        self,
        task_description: str,
        current_observation: str,
        use_memory: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Select action considering memory context.
        """

        # Retrieve relevant experiences from memory
        relevant_experiences = []
        if use_memory and len(self.memory.experiences) > 0:
            relevant_experiences = self.memory.retrieve_similar(
                current_observation,
                k=5,
                filter_task=task_description
            )

        # Build context for LLM
        context = self._build_context(
            task_description,
            current_observation,
            relevant_experiences
        )

        # Get action from LLM
        prompt = f"{context}\n\nNext action:"
        action = self.base_model.generate(prompt, max_tokens=100)

        return action, {"context_length": len(context), "memories_used": len(relevant_experiences)}

    def _build_context(
        self,
        task: str,
        observation: str,
        experiences: List[Tuple[Experience, float]]
    ) -> str:
        """Build LLM context including relevant memories."""

        context = f"Task: {task}\n\nCurrent Observation: {observation}\n"

        if experiences:
            context += "\nRelevant Past Experiences:\n"
            for i, (exp, score) in enumerate(experiences[:3]):  # Top 3
                context += f"{i+1}. Previous: {exp.state}\n"
                context += f"   Action: {exp.action}\n"
                context += f"   Result: {exp.observation}\n"
                context += f"   Success: {exp.success}\n"

        return context

    def collect_rollout(
        self,
        task_description: str,
        max_steps: int = 20
    ) -> List[Experience]:
        """Collect trajectory using memory-augmented decisions."""

        trajectory = []
        observation = "Initial state"

        for step in range(max_steps):
            # Get action from MDP
            action, metadata = self.get_action(task_description, observation)

            # Execute action (simulated or real)
            new_observation, reward, done = self._execute_action(
                task_description,
                action
            )

            # Store experience
            experience = Experience(
                state=observation,
                action=action,
                observation=new_observation,
                reward=reward,
                success=(reward > 0.5),
                timestamp=0,
                metadata={"task_id": task_description, **metadata}
            )

            trajectory.append(experience)

            if done:
                break

            observation = new_observation

        return trajectory

    def _execute_action(
        self,
        task: str,
        action: str
    ) -> Tuple[str, float, bool]:
        """Execute action and get feedback."""
        # In practice: execute in real environment or simulator
        pass
```

### 4. Implement Continual Adaptation

Update memory and policy without LLM fine-tuning:

```python
class ContinualAdapter:
    """Handles continual learning without LLM updates."""

    def __init__(
        self,
        mdp: MemoryAugmentedMDP,
        selection_policy: CaseSelectionPolicy,
        learning_rate: float = 1e-4
    ):
        self.mdp = mdp
        self.selection_policy = selection_policy
        self.optimizer = torch.optim.Adam(
            selection_policy.parameters(),
            lr=learning_rate
        )

    def adapt_to_trajectory(self, trajectory: List[Experience]):
        """Update memory and policy based on new trajectory."""

        # Store all experiences in memory
        for exp in trajectory:
            self.mdp.memory.store_experience(exp)

        # Train selection policy on successful outcomes
        successful_trajectory = [e for e in trajectory if e.success]

        if successful_trajectory:
            self._train_selection_policy(successful_trajectory)

        # Clean up memory if needed
        if len(self.mdp.memory.experiences) > self.mdp.memory.max_size * 0.9:
            self._prune_memory()

    def _train_selection_policy(self, trajectory: List[Experience]):
        """Train policy on successful trajectory."""

        # Encode states
        states = [self._encode_state(e.state) for e in trajectory]
        outcomes = [e.reward for e in trajectory]

        # Get selected cases (which ones would the policy choose?)
        selected_cases = []
        for state in states:
            candidates = self._get_candidate_embeddings(state)
            indices = self.selection_policy(
                state, candidates, return_scores=False
            )
            selected_cases.append(indices[0].item())

        # Compute policy gradient loss
        loss = self.selection_policy.train_on_trajectory(
            states, selected_cases, outcomes
        )

        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.selection_policy.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def _prune_memory(self):
        """Remove least useful experiences."""
        # Keep only high-success experiences or recently accessed
        pass

    def _encode_state(self, state: str) -> torch.Tensor:
        """Encode state to embedding."""
        pass

    def _get_candidate_embeddings(self, state_embedding) -> List[torch.Tensor]:
        """Get embeddings of candidate memory cases."""
        pass
```

### 5. Evaluate Agent Learning

Measure adaptation efficiency without LLM fine-tuning:

```python
def evaluate_agentfly_learning(
    mdp: MemoryAugmentedMDP,
    test_tasks: List[Dict],
    num_rollouts_per_task: int = 3
) -> Dict[str, float]:
    """
    Evaluate memory-augmented agent performance.
    """

    task_successes = []
    memory_hit_rates = []
    trajectory_lengths = []

    for task in test_tasks:
        task_desc = task["description"]
        expected_solution = task["expected"]

        for _ in range(num_rollouts_per_task):
            # Collect trajectory
            trajectory = mdp.collect_rollout(task_desc, max_steps=50)

            # Evaluate success
            final_obs = trajectory[-1].observation if trajectory else ""
            success = evaluate_task_success(final_obs, expected_solution)
            task_successes.append(success)

            # Track memory efficiency
            memory_hits = sum(
                1 for exp in trajectory
                if exp.metadata.get("memories_used", 0) > 0
            )
            memory_hit_rates.append(memory_hits / len(trajectory))
            trajectory_lengths.append(len(trajectory))

            # Adapt to this trajectory
            mdp_adapter = ContinualAdapter(mdp, mdp.selection_policy)
            mdp_adapter.adapt_to_trajectory(trajectory)

    return {
        "avg_success_rate": sum(task_successes) / len(task_successes),
        "avg_memory_hit_rate": sum(memory_hit_rates) / len(memory_hit_rates),
        "avg_trajectory_length": sum(trajectory_lengths) / len(trajectory_lengths),
        "memory_size": len(mdp.memory.experiences)
    }
```

## Practical Guidance

### When to Use AgentFly

- Research tasks requiring rapid adaptation (GAIA, DeepResearcher)
- Scenarios where LLM fine-tuning is expensive/prohibited
- Agents needing to learn from few examples
- Continual learning systems with non-stationary tasks
- Open-source or proprietary model deployment

### When NOT to Use

- Tasks where base model knowledge is insufficient
- Extremely low-latency inference requirements
- Scenarios with no clear episodic structure
- Tasks requiring fundamental capability changes

### Key Hyperparameters

- **max_memory_size**: 5000-20000 experiences
- **memory_context_length**: 1000-3000 tokens
- **learning_rate**: 1e-4 to 1e-3
- **k (top cases retrieved)**: 3-5 usually sufficient
- **gamma (discount factor)**: 0.99

### Performance Expectations

- GAIA Benchmark: 87.88% Pass@3
- Adaptation Speed: 2-5 examples sufficient for new task class
- Memory Efficiency: Sub-linear growth with task diversity
- No LLM Fine-tuning: Entire approach requires 0 LLM updates

## Reference

Researchers. (2024). AgentFly: Fine-tuning LLM Agents without Fine-tuning LLMs. arXiv preprint arXiv:2508.16153.
