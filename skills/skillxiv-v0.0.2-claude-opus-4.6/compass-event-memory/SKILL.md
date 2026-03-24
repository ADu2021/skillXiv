---
name: compass-event-memory
title: "Memory Matters More: Event-Centric Memory as a Logic Map for Agent Searching and Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.04726"
keywords: [agent-memory, event-graph, structured-retrieval, reasoning-over-memory, long-horizon-planning]
description: "Organize agent memory as an event graph with explicit logical relationships rather than flat embeddings. Framework incrementally segments experiences into events and links them through causal, temporal, and logical relations. Enables agents to navigate memory as a logic map for goal-directed searching and structured reasoning, improving performance on multi-hop reasoning and long-horizon planning tasks."
---

## Problem

Current LLM agent memory systems suffer from three limitations:

1. **Shallow Retrieval**: Similarity-based lookup returns semantically related but logically disconnected memories
2. **Lost Context**: Flat memory stores lose the narrative flow and causal relationships between experiences
3. **Poor Long-Horizon Planning**: Agents struggle to navigate sequences of memory spanning multiple steps because relationships aren't captured
4. **Inefficient Search**: Finding the "right" memory for multi-step reasoning requires expensive retrieval iterations

Agents need memory that captures not just what happened, but how events connect logically.

## Solution

**CompassMem** (Memory Matters More) organizes memory as an **Event Graph**:

1. **Event Segmentation**: Incrementally break experience sequences into discrete, self-contained events
2. **Logical Relations**: Link events through explicit relationships:
   - **Causal**: Event A led to Event B
   - **Temporal**: Event A happened before Event B
   - **Conditional**: Event B only occurred because Event A established preconditions
   - **Goal-Oriented**: Events form chains progressing toward specific goals
3. **Graph Navigation**: Agents traverse this logic map to gather relevant experiences, moving from high-level goals to supporting details

## When to Use

- **Multi-hop Reasoning**: Tasks requiring agents to chain knowledge across multiple memory segments
- **Long-Horizon Planning**: Agents coordinating actions over extended sequences
- **Dialogue Agents**: Maintaining context through multi-turn conversations with rich interaction history
- **Collaborative Agents**: Multiple agents sharing structured memory about past interactions
- **Question Answering**: Retrieving not just facts but the causal chain leading to conclusions

## When NOT to Use

- For simple retrieval tasks (single-fact lookup is more efficient with embeddings)
- In real-time systems where event segmentation overhead is prohibitive
- When memory access patterns are random (graph structure provides no advantage)
- For agents with limited interaction history (insufficient events to build meaningful graphs)

## Core Concepts

The framework operates on the principle that **memory structure mirrors reasoning structure**:

1. **Events as Units**: Each experience becomes a discrete, bounded event with input, process, and outcome
2. **Relationships as Reasoning Paths**: Logical connections between events form the routes agents traverse
3. **Graph Navigation as Planning**: Finding relevant memories becomes searching for paths in the event graph

## Key Implementation Pattern

Building and traversing CompassMem:

```python
# Conceptual: event-centric memory construction
class EventMemory:
    def __init__(self):
        self.events = []      # List of (description, metadata)
        self.relations = []   # List of (event_i, relation_type, event_j)

    def add_event(self, observation, action, outcome):
        event = {
            'observation': observation,
            'action': action,
            'outcome': outcome,
            'timestamp': time.now()
        }
        self.events.append(event)

    def link_events(self, event_i, event_j, relation_type):
        # relation_type: 'causal', 'temporal', 'conditional', 'goal_progression'
        self.relations.append((event_i, relation_type, event_j))

    def navigate_for_goal(self, goal):
        # Find events related to goal via explicit relations
        relevant_path = self.traverse_to_goal(goal)
        return relevant_path
```

Key mechanisms:
- Event boundaries determined by semantic shifts (state changes, action completions)
- Relation types encode reasoning patterns (causality enables counterfactual thinking, temporal captures dependencies)
- Traversal algorithms use relation types to guide relevant memory retrieval

## Expected Outcomes

- **Improved Multi-hop Reasoning**: 20-30% better performance on tasks requiring logical chaining
- **Efficient Long-Context Planning**: Agents navigate memory maps faster than iterative retrieval
- **Generalization Across Models**: Event graph structure works with different LLM architectures
- **Interpretability**: Explicit relations make agent reasoning paths transparent

## Limitations and Considerations

- Requires computational overhead to identify event boundaries and establish relations
- Manual relation type definition may need domain customization
- Event segmentation quality depends on input structure (clean logs are easier than raw observations)
- Large event graphs may require hierarchical compression for efficiency

## Integration Pattern

For a multi-turn dialogue agent:

1. **Segment Conversation**: Group dialogue into events (questions, answers, corrections)
2. **Establish Relations**: Link events with causal (question→answer), temporal (turn 1→turn 2), and goal (Q→resolved)
3. **Navigate on New Query**: Traverse event graph to find relevant prior interactions
4. **Use Path for Context**: Ground new response in context from event chain

This ensures the agent maintains coherent context across extended interactions.

## Related Work Context

CompassMem extends beyond vector retrieval by recognizing that reasoning structure should mirror memory structure. Unlike purely semantic approaches, event graphs enable agents to perform logical inference rather than just pattern matching.
