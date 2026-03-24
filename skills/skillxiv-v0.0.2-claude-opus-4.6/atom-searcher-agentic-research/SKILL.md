---
name: atom-searcher-agentic-research
title: "Atom-Searcher: Agentic Deep Research via Atomic Thought Reward"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.12800
keywords: [reinforcement-learning, agentic-reasoning, atomic-thoughts, reasoning-rewards, information-retrieval]
description: "Decompose agent reasoning into atomic thoughts guided by curriculum-based reasoning reward models, enabling multi-hop information retrieval and interpretable deep research."
---

# Atom-Searcher: Agentic Deep Research via Atomic Thought Reward

## Core Concept

Complex research requires multi-step reasoning: decomposing problems, retrieving information, refining hypotheses. Standard agent RL treats whole trajectories as units, missing opportunities to guide intermediate reasoning steps.

Atom-Searcher breaks reasoning into "Atomic Thoughts"—fine-grained functional units (search queries, hypothesis updates, result synthesis). Each receives guidance from a Reasoning Reward Model. A curriculum gradually shifts from process rewards (good intermediate steps) to outcome rewards (final answers), mirroring human learning.

## Architecture Overview

- **Atomic Thoughts**: Fine-grained reasoning units (search, compare, synthesize)
- **Reasoning Reward Models**: Evaluate quality of each atomic thought
- **Curriculum-Based Rewards**: Process rewards early, outcome rewards later
- **Multi-Hop Reasoning**: Track information dependencies across steps
- **Agent Planning**: Learn strategic information retrieval and hypothesis refinement
- **Interpretable Trajectories**: Each step's reasoning is visible and evaluable

## Implementation Steps

### 1. Define Atomic Thought Types

```python
from enum import Enum
from typing import Dict, List, Any

class AtomicThoughtType(Enum):
    SEARCH = "search"  # Search for information
    RETRIEVE = "retrieve"  # Fetch from database
    COMPARE = "compare"  # Compare evidence
    SYNTHESIZE = "synthesize"  # Combine findings
    HYPOTHESIZE = "hypothesize"  # Propose theory
    REFINE = "refine"  # Update hypothesis
    DECIDE = "decide"  # Make decision

class AtomicThought:
    """Represents one reasoning step"""
    def __init__(self, thought_type: AtomicThoughtType, content: str,
                context: Dict[str, Any] = None):
        self.type = thought_type
        self.content = content
        self.context = context or {}
        self.reasoning = ""  # Explanation of the thought
        self.reward = None

    def __repr__(self):
        return f"[{self.type.value.upper()}] {self.content}"

# Example trajectory
trajectory = [
    AtomicThought(AtomicThoughtType.SEARCH, "Find information about photosynthesis"),
    AtomicThought(AtomicThoughtType.RETRIEVE, "Retrieved 3 papers on photosynthesis"),
    AtomicThought(AtomicThoughtType.COMPARE, "Compare light and dark reactions"),
    AtomicThought(AtomicThoughtType.SYNTHESIZE, "Light reactions generate ATP/NADPH for dark reactions"),
    AtomicThought(AtomicThoughtType.DECIDE, "Answer: Light reactions energy, dark reactions synthesis")
]
```

### 2. Build Reasoning Reward Models

```python
import torch
import torch.nn as nn

class AtomicThoughtRewardModel(nn.Module):
    """Evaluate quality of individual atomic thoughts"""
    def __init__(self, hidden_size=768):
        super().__init__()

        # Encode thought content
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # Reward prediction
        self.reward_head = nn.Linear(256, 1)

        # Type-specific adjustments
        self.type_embeddings = nn.Embedding(
            num_embeddings=len(AtomicThoughtType),
            embedding_dim=64
        )

    def forward(self, thought_embedding: torch.Tensor,
               thought_type_idx: int, context_embedding: torch.Tensor = None):
        """
        Evaluate single atomic thought

        Args:
            thought_embedding: [hidden_size] embedding of thought content
            thought_type_idx: index of thought type
            context_embedding: [hidden_size] embedding of trajectory context
        """
        # Encode thought
        encoded = self.encoder(thought_embedding)

        # Add type information
        type_emb = self.type_embeddings(torch.tensor([thought_type_idx]))
        combined = torch.cat([encoded, type_emb.squeeze()], dim=-1)

        # Predict reward
        reward = torch.sigmoid(self.reward_head(combined))  # [0, 1]

        return reward

class ProcessVsOutcomeRewardScheduler:
    """Schedule which reward type to emphasize"""
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.process_ratio = 1.0  # Start with 100% process reward
        self.outcome_ratio = 0.0

    def get_weights(self, current_step: int) -> tuple:
        """Get process/outcome weights for current step"""
        # Linear curriculum: process -> outcome
        progress = current_step / self.total_steps
        self.process_ratio = max(0.0, 1.0 - progress)
        self.outcome_ratio = progress

        return self.process_ratio, self.outcome_ratio
```

### 3. Implement Atomic Thought Generation

```python
class AtomicThoughtGenerator:
    """Generate appropriate atomic thoughts"""
    def __init__(self, llm_model, tokenizer):
        self.llm = llm_model
        self.tokenizer = tokenizer

    def generate_thought(self, query: str, context: List[AtomicThought],
                        thought_type: AtomicThoughtType) -> AtomicThought:
        """Generate next atomic thought"""

        # Build context
        context_str = "Previous thoughts:\n"
        for prev_thought in context[-3:]:  # Last 3 thoughts
            context_str += f"- {prev_thought}\n"

        # Prompt for next thought
        prompt = f"""Given this research query and previous thoughts, generate the next {thought_type.value} thought.

Query: {query}

{context_str}

Generate a {thought_type.value} thought that advances the research:"""

        response = self.llm.generate(prompt, max_length=150)

        thought = AtomicThought(
            thought_type=thought_type,
            content=response,
            context={'query': query, 'previous': context}
        )

        return thought

    def execute_thought(self, thought: AtomicThought, tools_available: Dict):
        """Execute atomic thought (e.g., search, retrieve)"""
        if thought.type == AtomicThoughtType.SEARCH:
            # Execute search
            results = tools_available['search_engine'].search(thought.content)
            thought.reasoning = f"Retrieved {len(results)} results"
            return results

        elif thought.type == AtomicThoughtType.RETRIEVE:
            # Fetch from database
            data = tools_available['database'].query(thought.content)
            thought.reasoning = f"Retrieved {len(data)} items"
            return data

        elif thought.type == AtomicThoughtType.COMPARE:
            # Compare evidence
            thought.reasoning = "Compared using similarity metrics"
            return {'comparison': thought.content}

        elif thought.type == AtomicThoughtType.SYNTHESIZE:
            # Synthesize findings
            thought.reasoning = "Synthesized via semantic integration"
            return {'synthesis': thought.content}

        # ... other thought types
        return None
```

### 4. Train with Curriculum-Based Rewards

```python
def train_atom_searcher(agent_model, reward_model, scheduler,
                       research_tasks, num_epochs=10):
    """Train agent using curriculum of process then outcome rewards"""
    optimizer = torch.optim.AdamW(agent_model.parameters(), lr=1e-5)

    total_steps = num_epochs * len(research_tasks)
    steps = 0

    for epoch in range(num_epochs):
        for task in research_tasks:
            query = task['query']
            target_answer = task['target_answer']

            # Generate trajectory of atomic thoughts
            trajectory = []
            tools = {
                'search_engine': SearchEngine(),
                'database': Database()
            }

            context = []
            for step in range(10):  # Max 10 atomic thoughts
                # Decide next thought type
                thought_type = agent_model.decide_next_thought(query, context)

                # Generate thought
                generator = AtomicThoughtGenerator(agent_model.llm, agent_model.tokenizer)
                thought = generator.generate_thought(query, context, thought_type)

                # Execute thought
                results = generator.execute_thought(thought, tools)

                # Get embedding
                thought_embedding = agent_model.encode_thought(thought.content)

                # Evaluate thought with reward model
                thought_reward = reward_model(
                    thought_embedding,
                    thought_type.value,
                    context_embedding=agent_model.encode_context(context)
                )
                thought.reward = thought_reward.item()

                context.append(thought)
                trajectory.append(thought)

                if thought.type == AtomicThoughtType.DECIDE:
                    break

            # Compute trajectory reward
            final_answer = trajectory[-1].content if trajectory else ""
            outcome_correct = check_answer(final_answer, target_answer)

            # Get curriculum weights
            process_weight, outcome_weight = scheduler.get_weights(steps)

            # Loss: blend of process and outcome
            process_reward = sum(t.reward.item() for t in trajectory) / len(trajectory)
            outcome_reward = 1.0 if outcome_correct else 0.0

            combined_reward = (process_weight * process_reward +
                             outcome_weight * outcome_reward)

            # Policy gradient loss
            log_probs = agent_model.compute_trajectory_log_prob(trajectory)
            loss = -(log_probs * combined_reward)

            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            steps += 1

            if steps % 100 == 0:
                print(f"Steps {steps}: Process={process_weight:.2f}, "
                      f"Outcome={outcome_weight:.2f}, "
                      f"Loss={loss:.4f}")
```

### 5. Inference with Atom-Searcher

```python
def research_with_atom_searcher(agent_model, query: str,
                               max_thoughts: int = 10) -> Dict:
    """Perform deep research using atomic thoughts"""
    trajectory = []
    context = []

    tools = {
        'search_engine': SearchEngine(),
        'database': Database()
    }

    generator = AtomicThoughtGenerator(agent_model.llm, agent_model.tokenizer)

    for step in range(max_thoughts):
        # Decide next thought type
        thought_type = agent_model.decide_next_thought(query, context)

        print(f"Step {step + 1}: {thought_type.value.upper()}")

        # Generate and execute thought
        thought = generator.generate_thought(query, context, thought_type)
        results = generator.execute_thought(thought, tools)

        print(f"  {thought.content}")
        print(f"  -> {thought.reasoning}")

        context.append(thought)
        trajectory.append({
            'type': thought_type.value,
            'content': thought.content,
            'reasoning': thought.reasoning,
            'results': results
        })

        if thought_type == AtomicThoughtType.DECIDE:
            break

    # Extract final answer
    final_thought = trajectory[-1] if trajectory else None
    final_answer = final_thought['content'] if final_thought else "No answer"

    return {
        'query': query,
        'answer': final_answer,
        'trajectory': trajectory,
        'num_steps': len(trajectory)
    }
```

### 6. Evaluation on Deep Research

```python
def evaluate_atom_searcher(agent_model, benchmark_tasks):
    """Evaluate on multi-hop reasoning benchmarks"""
    correct = 0
    total_steps = 0

    for task in benchmark_tasks:
        result = research_with_atom_searcher(
            agent_model,
            task['query'],
            max_thoughts=10
        )

        # Check correctness
        is_correct = check_answer(result['answer'], task['target'])
        if is_correct:
            correct += 1

        total_steps += result['num_steps']

    accuracy = correct / len(benchmark_tasks) if benchmark_tasks else 0.0
    avg_steps = total_steps / len(benchmark_tasks) if benchmark_tasks else 0

    print(f"Accuracy: {accuracy * 100:.1f}%")
    print(f"Avg steps: {avg_steps:.1f}")

    return accuracy
```

## Practical Guidance

- **Curriculum Duration**: 50% process-focused, 50% outcome-focused training
- **Atomic Types**: 6-8 distinct thought types (search, compare, synthesize, etc.)
- **Context Window**: Keep last 3-5 thoughts as context
- **Max Thoughts**: 8-12 per query (limit reasoning depth)
- **Reward Model**: Train separately on process-quality labels

## Reference

Atom-Searcher (2508.12800): https://arxiv.org/abs/2508.12800

Guide agentic reasoning through fine-grained atomic thoughts with curriculum-based rewards, achieving interpretable multi-hop information retrieval and improved research task performance.
