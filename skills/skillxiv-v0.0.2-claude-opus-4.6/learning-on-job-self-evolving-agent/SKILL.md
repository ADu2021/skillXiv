---
name: learning-on-job-self-evolving-agent
title: "Learning on the Job: An Experience-Driven Self-Evolving Agent for Long-Horizon Tasks"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.08002"
keywords: [Self-Evolution, Agent Learning, Memory Systems, Experience Reuse, Long-Horizon Tasks]
description: "Build autonomous agents that accumulate structured knowledge from task execution into hierarchical memory (strategic, procedural, tool) without human annotation, enabling knowledge transfer to unseen tasks."
---

# Technique: Hierarchical Memory for Autonomous Agent Self-Evolution

Task-specific fine-tuning of agents is expensive and doesn't generalize. Autonomous agents need to accumulate knowledge from their own experiences and apply it to new problems. Learning on the Job enables this through a hierarchical memory architecture that captures execution traces at multiple levels of abstraction.

The core insight is treating memory as natural language rather than parameters. After executing a task, agents extract high-level strategies, procedural steps, and tool patterns into structured memory. This memory becomes accessible for future tasks without retraining, enabling knowledge transfer across diverse problem domains.

## Core Concept

The system implements a "Plan-Execute-Reflect-Memorize" loop with three memory types:

1. **Strategic Memory**: High-level problem-solution mappings guiding overall approach
2. **Procedural Memory**: Step-by-step SOPs indexed by application domain
3. **Tool Memory**: Individual tool usage patterns and instructions

After each subtask, the Reflect Agent distills the trajectory into new memory entries, enabling seamless transfer across different LLMs without fine-tuning.

## Architecture Overview

- **Execution Phase**: Agent generates and executes actions within task environment
- **Reflection Phase**: Analyze trajectory to extract generalizable knowledge
- **Memory Update**: Store strategic insights, procedural steps, tool patterns
- **Retrieval Phase**: For new tasks, fetch relevant memory entries as context
- **Generalization**: Apply learned patterns to previously unseen challenges

## Implementation Steps

Define the hierarchical memory structure.

```python
class MemoryEntry:
    def __init__(self, entry_type, content, domain, success_rate=None):
        self.type = entry_type  # 'strategy', 'procedure', 'tool'
        self.content = content
        self.domain = domain
        self.success_rate = success_rate or 0.0

class HierarchicalMemory:
    def __init__(self):
        self.strategic = []  # High-level problem-solution pairs
        self.procedural = []  # Step-by-step procedures indexed by domain
        self.tools = []      # Tool usage patterns

    def add_strategic_memory(self, problem_pattern, solution_approach, domain):
        """Store high-level strategy."""
        entry = MemoryEntry('strategy', solution_approach, domain)
        self.strategic.append(entry)

    def add_procedural_memory(self, steps, application_domain, success_rate):
        """Store step-by-step procedure."""
        entry = MemoryEntry('procedure', steps, application_domain, success_rate)
        self.procedural.append(entry)

    def add_tool_memory(self, tool_name, usage_pattern, domain):
        """Store tool usage pattern."""
        entry = MemoryEntry('tool', {'name': tool_name, 'usage': usage_pattern},
                           domain)
        self.tools.append(entry)

    def retrieve_relevant_memory(self, task_description, memory_type='all'):
        """Retrieve memory entries relevant to task."""
        # Semantic similarity matching
        relevant = []

        if memory_type in ['all', 'strategy']:
            relevant.extend(self._semantic_match(task_description, self.strategic))

        if memory_type in ['all', 'procedure']:
            relevant.extend(self._semantic_match(task_description, self.procedural))

        if memory_type in ['all', 'tool']:
            relevant.extend(self._semantic_match(task_description, self.tools))

        return relevant

    def _semantic_match(self, query, memory_list, top_k=3):
        """Retrieve top-k semantically similar memories."""
        # Use embedding-based similarity
        from sklearn.metrics.pairwise import cosine_similarity

        if not memory_list:
            return []

        query_embedding = self._embed(query)
        scores = []

        for entry in memory_list:
            entry_embedding = self._embed(entry.content)
            similarity = cosine_similarity([query_embedding], [entry_embedding])[0][0]
            scores.append((entry, similarity))

        # Sort by similarity and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scores[:top_k]]

    def _embed(self, text):
        """Embed text for similarity computation."""
        # Use sentence transformer or similar
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(text)
```

Implement the Reflection module that extracts knowledge from execution traces.

```python
def extract_memory_from_trajectory(trajectory, task_description, llm_model):
    """
    Use LLM to reflect on execution trajectory and extract generalizable knowledge.

    Args:
        trajectory: Dict with 'actions', 'observations', 'results'
        task_description: Original task
        llm_model: Language model for reflection

    Returns:
        memories: Dict with 'strategy', 'procedures', 'tools'
    """

    # Format trajectory for LLM
    trajectory_text = f"""Task: {task_description}

Actions taken:
{format_actions(trajectory['actions'])}

Results achieved:
{trajectory['results']}

Extract generalizable knowledge:
1. What high-level strategy was effective?
2. What step-by-step procedures can be reused?
3. How were tools used effectively?
"""

    # Get reflection from LLM
    reflection = llm_model.generate(trajectory_text)

    # Parse reflection into memory entries
    memories = parse_reflection_to_memories(reflection)

    return memories


def format_actions(actions):
    """Format action list for LLM review."""
    formatted = []
    for i, action in enumerate(actions, 1):
        formatted.append(f"{i}. {action.get('type')}: {action.get('content')}")
    return "\n".join(formatted)


def parse_reflection_to_memories(reflection_text):
    """Parse LLM reflection into typed memory entries."""
    memories = {
        'strategy': [],
        'procedures': [],
        'tools': []
    }

    # Use regex or LLM to structure the reflection
    # Example: extract lines starting with "Strategy:", "Procedure:", "Tool:"
    lines = reflection_text.split('\n')

    current_type = None
    for line in lines:
        if 'Strategy:' in line:
            current_type = 'strategy'
            memories['strategy'].append(line.replace('Strategy:', '').strip())
        elif 'Procedure:' in line or 'Step:' in line:
            current_type = 'procedures'
            memories['procedures'].append(line.replace('Procedure:', '').strip())
        elif 'Tool:' in line:
            current_type = 'tools'
            memories['tools'].append(line.replace('Tool:', '').strip())

    return memories
```

Implement the Plan-Execute-Reflect-Memorize loop.

```python
def agent_execution_loop(agent, task, memory, max_steps=20):
    """
    Execute task with memory-guided planning and reflection.

    Args:
        agent: Agent policy
        task: Task description
        memory: Hierarchical memory
        max_steps: Maximum steps before timeout

    Returns:
        result: Task result
        trajectory: Execution trace for reflection
    """

    trajectory = {'actions': [], 'observations': [], 'results': None}

    # PLAN: Retrieve relevant memory for task
    relevant_memory = memory.retrieve_relevant_memory(task)
    memory_context = format_memory_for_context(relevant_memory)

    state = {'task': task, 'memory_context': memory_context}

    # EXECUTE: Run agent steps
    for step in range(max_steps):
        # Generate action
        action = agent.generate_action(state, memory_context)
        trajectory['actions'].append(action)

        # Execute action in environment
        observation = execute_action(action)
        trajectory['observations'].append(observation)

        # Check if task completed
        if is_task_complete(observation):
            trajectory['results'] = observation
            break

        # Update state
        state['last_observation'] = observation

    # REFLECT: Extract knowledge from trajectory
    extracted_memories = extract_memory_from_trajectory(
        trajectory, task, agent.llm_model
    )

    # MEMORIZE: Store in memory hierarchy
    for strategy in extracted_memories.get('strategy', []):
        memory.add_strategic_memory(task, strategy, extract_domain(task))

    for procedure in extracted_memories.get('procedures', []):
        memory.add_procedural_memory(procedure, extract_domain(task),
                                    success_rate=1.0 if trajectory['results'] else 0.0)

    for tool_usage in extracted_memories.get('tools', []):
        memory.add_tool_memory(tool_usage, extract_domain(task),
                              extract_domain(task))

    return trajectory['results'], trajectory


def format_memory_for_context(memory_entries):
    """Format retrieved memory entries as LLM context."""
    if not memory_entries:
        return "No relevant prior experiences."

    formatted = []
    for entry in memory_entries:
        formatted.append(f"{entry.type.upper()}: {entry.content}")

    return "\n".join(formatted)
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|---------------|-------|
| Memory embedding model | all-MiniLM-L6-v2 or similar | Balance quality vs. inference speed |
| Reflection LLM | Same as agent or smaller | Trade-offs between extraction quality and cost |
| Memory retention | Keep all entries; use success rate to rank | More memory enables better transfer |
| Memory update frequency | After each completed subtask | More frequent updates capture finer details |
| When to use | Long-horizon task sequences with pattern reuse | Project planning, web navigation, research tasks |
| When NOT to use | One-off tasks or non-repeating problem types | Memory overhead not justified |
| Common pitfall | Memory content becomes too generic | Enforce specificity in reflection prompts |

### When to Use Learning on the Job

- Agent systems solving sequences of related tasks
- Domains where patterns repeat across problems
- Scenarios where maintaining task-specific fine-tuning is impractical

### When NOT to Use Learning on the Job

- Single-task agents where memory transfer is unnecessary
- Domains with high task diversity and low pattern reuse
- Real-time systems where reflection overhead is problematic

### Common Pitfalls

- **Generic memory**: Reflection produces overly-abstract knowledge; ask for concrete procedures
- **Memory staleness**: Periodically prune low-success-rate memories
- **Encoding drift**: Use consistent embedding model across all memory operations
- **Scalability**: Very large memory requires efficient retrieval; consider periodic consolidation

## Reference

Paper: https://arxiv.org/abs/2510.08002
