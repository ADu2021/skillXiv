---
name: memory-indexed-experience-scaling
title: "Memex(RL): Scaling Long-Horizon LLM Agents via Indexed Experience Memory"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.04257"
keywords: [Long-Horizon Reasoning, Memory Management, Context Compression, Agent Scaling, Reinforcement Learning]
description: "Enable long-horizon agents to manage finite context by separating working memory from persistent storage. Use indexed summaries with pointers to archived evidence, treating memory operations as first-class agent actions learned via RL."
---

# Memex(RL): Indexed Experience Memory for Scaling Long-Horizon Agents

Long-horizon multi-step reasoning tasks overwhelm finite context windows: agents either repeat full conversation history (inefficient) or use lossy summaries (information loss). Memex introduces a memory architecture where agents explicitly manage both working context and long-term storage through learned operations. The system maintains compact in-context indexed summaries with pointers to externally archived artifacts, enabling full-fidelity evidence retrieval on demand.

The core innovation treats memory as a first-class agent capability: read operations dereference indices to retrieve exact past information; write operations compress state into indexed summaries. This mirrors how humans manage complex projects with external notes and references while keeping working memory focused.

## Core Concept

Memex decomposes memory into two complementary systems:

1. **Working Context**: Compact in-context summary (~1-2KB) containing indexed references to past interactions
2. **Experience Store**: External archive of full-fidelity artifacts (full trajectories, tool outputs, decision trees) under stable indices

Agents learn to:
- **Write**: Compress key information into indexed summary for future reference
- **Read**: Dereference indices to retrieve exact past information when needed
- **Manage**: Decide what to compress, when to retrieve, based on task demands

This separation enables agents to maintain reasoning coherence over thousands of tokens while staying within practical context limits.

## Architecture Overview

- **Input**: Long-horizon task with multi-step subtasks
- **Working Context Module**: Compressed summary with embedded indices
- **Experience Store**: Persistent key-value archive indexed by discrete IDs
- **Read/Write Operators**: Agent-learned memory operations
- **Action Space**: Extended with memory operations (read_idx, write_summary)
- **Output**: Decision with access to complete historical context

## Implementation Steps

**Step 1: Design experience store and indexing scheme**

Create a persistent storage system for archived experiences with stable access keys.

```python
class ExperienceStore:
    """External memory store for agent artifacts."""
    def __init__(self, max_size=10000):
        self.store = {}
        self.index_counter = 0
        self.max_size = max_size

    def write(self, artifact, artifact_type='trajectory'):
        """
        Archive an artifact and return stable index.
        artifact_type: 'trajectory', 'tool_output', 'decision_tree', etc.
        """
        if len(self.store) >= self.max_size:
            # Evict least-recently-used entry
            oldest_idx = min(self.store.keys(),
                            key=lambda k: self.store[k]['access_time'])
            del self.store[oldest_idx]

        idx = self.index_counter
        self.store[idx] = {
            'artifact': artifact,
            'type': artifact_type,
            'timestamp': time.time(),
            'access_time': time.time(),
            'access_count': 0
        }

        self.index_counter += 1
        return idx

    def read(self, idx):
        """Retrieve archived artifact by index."""
        if idx not in self.store:
            return None

        entry = self.store[idx]
        entry['access_time'] = time.time()
        entry['access_count'] += 1

        return entry['artifact']

    def list_indices(self):
        """Return available indices for agent querying."""
        return sorted(self.store.keys())
```

**Step 2: Create working context with indexed summaries**

Design a compact in-context representation that embeds indices without storing full artifacts.

```python
def create_indexed_summary(current_trajectory, experience_indices,
                          summary_max_tokens=200):
    """
    Create compact working context with embedded index references.

    Format:
    - Current step context (80-100 tokens)
    - Index references to archived experiences (20-40 tokens)
    - Metadata about available archived artifacts (remaining tokens)
    """
    # Part 1: Recent context (last 2-3 steps)
    recent_context = format_recent_trajectory(current_trajectory, depth=3)

    # Part 2: Archived experience index pointers
    index_pointers = []
    for idx in experience_indices[-5:]:  # Reference last 5 archived experiences
        summary = f"[exp_{idx}] "  # Compact reference
        index_pointers.append(summary)

    # Part 3: Metadata about available memories
    available_memories = f"Available: {len(experience_indices)} archived experiences"

    # Combine into compact summary
    working_context = f"""
Current Context:
{recent_context}

Available Memories:
{' '.join(index_pointers)}
{available_memories}

Agent Action Space: [act, read_exp_IDX, write_summary]
"""

    return working_context.strip()

def format_recent_trajectory(trajectory, depth=3):
    """Format last `depth` steps of trajectory for context."""
    recent = trajectory[-depth:]
    formatted = []
    for step in recent:
        formatted.append(f"Step: {step['action']}, Result: {step['result'][:50]}")
    return '\n'.join(formatted)
```

**Step 3: Implement read/write agent actions**

Extend agent action space to include memory operations as learned behaviors.

```python
class MemoryAugmentedAgent:
    """LLM agent with learned memory operations."""
    def __init__(self, llm_model, experience_store, working_context_size=200):
        self.llm = llm_model
        self.store = experience_store
        self.working_context_size = working_context_size

    def step(self, task_prompt, working_memory, trajectory):
        """
        Execute agent step with optional memory operations.
        Returns: (action, memory_operation)
        """
        # Construct full prompt with working memory
        indexed_summary = create_indexed_summary(trajectory, self.store.list_indices())

        full_prompt = f"""
Task: {task_prompt}

Working Memory:
{indexed_summary}

Your action (format: ACTION:value or READ:exp_IDX or WRITE:summary_text):
"""

        # Forward pass through LLM
        response = self.llm.generate(full_prompt, max_tokens=100)

        # Parse response for action type
        if response.startswith('READ:'):
            # Memory read operation
            exp_idx = int(response.split(':')[1])
            artifact = self.store.read(exp_idx)
            return ('read', artifact, exp_idx)

        elif response.startswith('WRITE:'):
            # Memory write operation
            summary = response.split(':', 1)[1]
            idx = self.store.write(summary, artifact_type='agent_summary')
            return ('write', idx)

        else:
            # Regular action
            return ('act', response)

    def compress_for_storage(self, trajectory_segment):
        """
        Compress trajectory segment into archived summary.
        Executed when agent issues WRITE action.
        """
        # Summarize key decisions and outcomes
        summary = f"""
Trajectory Summary:
- Steps: {len(trajectory_segment)}
- Key decisions: {[s['action'] for s in trajectory_segment[:3]]}
- Final outcome: {trajectory_segment[-1]['result']}
"""
        return summary
```

**Step 4: Learn memory operations via reinforcement learning**

Train agent to optimize when and what to read/write using RL.

```python
def compute_memory_cost(operation_type, access_count=0):
    """
    Compute cost of memory operation in tokens.
    - read: 1-2 tokens (index) + retrieved artifact tokens
    - write: summary length + index (2-3 tokens)
    """
    if operation_type == 'read':
        # Cost: index overhead + artifact retrieval
        return 2 + access_count  # Penalize repeated reads of same info

    elif operation_type == 'write':
        # Cost: summary compression + indexing overhead
        return 3 + 0  # Fixed overhead

    else:
        return 0  # No memory cost for regular actions

def compute_memory_reward(trajectory, task_success):
    """
    Reward successful task completion while penalizing memory operations.

    reward = task_success_bonus - memory_operation_cost
    """
    success_bonus = 10.0 if task_success else -1.0

    # Count memory operations in trajectory
    total_memory_cost = 0
    for step in trajectory:
        if step['action_type'] in ['read', 'write']:
            total_memory_cost += compute_memory_cost(step['action_type'])

    # Memory efficiency reward
    memory_penalty = 0.1 * total_memory_cost

    return success_bonus - memory_penalty

def memory_rl_update(agent, batch_trajectories, learning_rate=0.001):
    """
    RL update to optimize memory operations.
    Uses policy gradient with memory cost penalties.
    """
    total_pg_loss = 0.0

    for trajectory in batch_trajectories:
        task_success = trajectory[-1]['task_complete']
        reward = compute_memory_reward(trajectory, task_success)

        # Policy gradient for each step
        for t, step in enumerate(trajectory):
            # Log probability of chosen action
            logprob = agent.llm.compute_logprob(
                step['prompt'],
                step['action']
            )

            # PG loss: maximize log-prob of low-cost, high-reward actions
            step_memory_cost = compute_memory_cost(step['action_type'])
            adjusted_reward = reward - step_memory_cost

            pg_loss = -logprob * adjusted_reward
            total_pg_loss += pg_loss

    # Optimize
    agent.llm.backward(total_pg_loss / len(batch_trajectories))
    agent.llm.optimizer.step(learning_rate)

    return (total_pg_loss / len(batch_trajectories)).item()
```

**Step 5: Scaling evaluation on long-horizon tasks**

Benchmark memory-augmented agents on tasks requiring hundreds of steps.

```python
def evaluate_scaling(agent, task_sequence, max_steps=500):
    """
    Evaluate agent on long-horizon task.
    Compare: working context size, success rate, memory operations.
    """
    trajectory = []
    working_memory = []
    context_size = 0
    success = False

    for step_idx in range(max_steps):
        # Agent step
        action, memory_op = agent.step(
            task_sequence[step_idx],
            working_memory,
            trajectory
        )

        trajectory.append({
            'step': step_idx,
            'action': action,
            'memory_op': memory_op,
            'context_size': context_size
        })

        # Update working memory
        if memory_op[0] == 'write':
            working_memory.append(memory_op[1])
            context_size += len(str(memory_op[1]))

        # Check task completion
        if agent.check_success(action):
            success = True
            break

    return {
        'success': success,
        'steps_taken': len(trajectory),
        'peak_context_size': max(s['context_size'] for s in trajectory),
        'memory_ops': len([t for t in trajectory if t['memory_op'] is not None]),
        'trajectory': trajectory
    }
```

## Practical Guidance

**Hyperparameter Selection:**
- **Working context size**: 200-500 tokens. Balance information density vs. token overhead.
- **Experience store max size**: 1,000-10,000. Larger = more history available; higher eviction cost.
- **Memory operation cost weight**: 0.1-0.5. Higher = penalizes reads/writes more, encouraging efficient memory usage.
- **Access count penalty**: 0.1-0.5 per read. Discourages repeated reads of same information; encourages compression.

**When to Use:**
- Long-horizon multi-step reasoning (100+ steps)
- Tasks requiring history reference (e.g., negotiation, project planning)
- Scenarios where lossy compression is insufficient
- Agent systems with token budget constraints

**When NOT to Use:**
- Short tasks (<20 steps) where context window is not a bottleneck
- Real-time systems requiring minimal latency (memory operations add overhead)
- Tasks where all information is state-encoded (no explicit history dependency)

**Common Pitfalls:**
- **Memory leakage**: Archived summaries can drift from true history. Validate by spot-checking retrieved artifacts.
- **Access patterns**: Skewed access patterns can cause eviction of useful memories. Use LRU or frequency-based retention.
- **Compression degradation**: Over-compressed summaries lose critical details. Monitor task performance; increase context size if needed.
- **Index fragmentation**: After many reads/writes, indices become sparse. Periodic compaction of store helps.

## Reference

arXiv: https://arxiv.org/abs/2603.04257
