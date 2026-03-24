---
name: latent-collaboration-multiagent
title: "Latent Collaboration in Multi-Agent Systems"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.20639"
keywords: [Multi-Agent Communication, Latent Space Reasoning, Efficient Inference, Working Memory, KV Cache Sharing]
description: "Enable LLM agents to collaborate entirely in latent space by sharing layer-wise KV cache representations instead of text, achieving 4× speedup and 71-84% fewer output tokens while maintaining 14.6% higher accuracy through lossless information preservation in continuous embeddings."
---

# Latent Collaboration in Multi-Agent Systems

Traditional multi-agent systems communicate through text tokens, requiring expensive encoding/decoding cycles and losing information when discrete tokens cannot capture nuanced reasoning. This skill demonstrates how to enable agents to collaborate directly in continuous latent space, sharing internal representations (KV caches) that preserve complete semantic information while achieving dramatic efficiency gains.

The core insight is that latent representations are inherently more expressive than discrete tokens and preserve more information when passed between agents—enabling agents to "think" with each other through embedding space rather than natural language.

## Core Concept

Latent Collaboration (LatentMAS) enables multi-agent reasoning through:

1. **Latent Thought Generation**: Agents perform auto-regressive reasoning by feeding last-layer hidden states directly back as inputs, generating latent thoughts without explicit token decoding

2. **Working Memory Transfer**: Agents share layer-wise KV caches containing both initial context and newly generated latent representations, enabling lossless information preservation

3. **Alignment Matrix**: A learned (or precomputed) alignment transformation enables seamless information transfer between agent vocabularies

The system is training-free; it computes the alignment matrix once per task and reuses it across all agent interactions.

## Architecture Overview

- **Latent Thought Module**: Generates continuous embeddings representing reasoning without tokenization
- **KV Cache Management**: Tracks and shares cached key-value pairs across agents
- **Alignment Transformation**: Maps representations between different agent models
- **Sequential/Hierarchical Topologies**: Supports various multi-agent configurations
- **Inference Engine**: Routes intermediate representations between agents efficiently

## Implementation Steps

The latent collaboration system operates by sharing representations instead of tokens.

**1. Extract Last-Layer Hidden States**

Capture the continuous representations before tokenization to enable latent thinking.

```python
def extract_hidden_states(model_output, layer_idx=-1):
    """
    Extract last-layer hidden states (before output projection).
    These continuous representations are richer than discrete tokens.
    Args:
        model_output: LLM forward pass output
        layer_idx: which layer's hidden states to extract (-1 for last)
    Returns:
        hidden_states: (batch, seq_len, hidden_dim) continuous representations
    """
    # Access hidden states directly from model forward pass
    hidden_states = model_output.hidden_states[layer_idx]
    return hidden_states
```

**2. Implement Latent Thought Generation**

Enable agents to generate reasoning by directly iterating on hidden states.

```python
def generate_latent_thoughts(model, initial_hidden_states, num_thoughts=5):
    """
    Generate multi-step reasoning in latent space.
    Agent produces continuous thought embeddings without token decoding.
    Args:
        model: LLM with access to hidden states
        initial_hidden_states: (1, seq_len, hidden_dim)
        num_thoughts: number of latent reasoning steps
    Returns:
        thought_sequence: List of hidden state tensors representing reasoning steps
    """
    thought_sequence = [initial_hidden_states]

    for step in range(num_thoughts):
        # Current thoughts become input to next step
        current_hidden = thought_sequence[-1]

        # Forward pass: model processes latent thoughts
        # Instead of tokenizing, feed hidden states directly
        next_output = model.forward_from_hidden(current_hidden)

        # Extract new hidden states (next thought)
        next_hidden = extract_hidden_states(next_output, layer_idx=-1)

        thought_sequence.append(next_hidden)

    return thought_sequence
```

**3. Build and Manage KV Cache for Each Agent**

Maintain cached key-value pairs enabling efficient agent-to-agent communication.

```python
class SharedKVCache:
    """
    Manages KV caches for multi-agent collaboration.
    Agents inherit predecessor's cache, enabling lossless context preservation.
    """
    def __init__(self, num_layers, hidden_dim, cache_size=512):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Initialize caches (one per transformer layer)
        self.kv_cache = [
            {'k': None, 'v': None}
            for _ in range(num_layers)
        ]

    def append_kv(self, layer_idx, new_k, new_v):
        """Add new key-value pairs from agent reasoning step."""
        if self.kv_cache[layer_idx]['k'] is None:
            self.kv_cache[layer_idx]['k'] = new_k
            self.kv_cache[layer_idx]['v'] = new_v
        else:
            # Concatenate with existing cache
            self.kv_cache[layer_idx]['k'] = torch.cat(
                [self.kv_cache[layer_idx]['k'], new_k], dim=-2
            )
            self.kv_cache[layer_idx]['v'] = torch.cat(
                [self.kv_cache[layer_idx]['v'], new_v], dim=-2
            )

    def transfer_to_next_agent(self):
        """
        Return complete cache for next agent in chain.
        Next agent inherits all context (initial + current agent's reasoning).
        """
        return copy.deepcopy(self.kv_cache)

    def get_kv_for_layer(self, layer_idx):
        """Retrieve KV pair for specific layer."""
        return self.kv_cache[layer_idx]['k'], self.kv_cache[layer_idx]['v']
```

**4. Compute Alignment Transformation Between Models**

Create transformation enabling knowledge transfer between different agent models.

```python
def compute_alignment_matrix(model_a, model_b, reference_corpus, hidden_dim=768):
    """
    Compute learnable alignment matrix for mapping representations.
    Enables latent information transfer between agents with different architectures.
    Args:
        model_a, model_b: Two LLM models to align
        reference_corpus: Shared text samples for alignment computation
        hidden_dim: dimension of hidden states
    Returns:
        alignment_matrix: (hidden_dim, hidden_dim) transformation matrix
    """
    # Forward both models on reference corpus
    with torch.no_grad():
        hidden_a_list = []
        hidden_b_list = []

        for sample in reference_corpus:
            # Get representations from both models
            output_a = model_a(sample, output_hidden_states=True)
            output_b = model_b(sample, output_hidden_states=True)

            hidden_a = extract_hidden_states(output_a)
            hidden_b = extract_hidden_states(output_b)

            hidden_a_list.append(hidden_a.mean(dim=1))  # Average over sequence
            hidden_b_list.append(hidden_b.mean(dim=1))

        hidden_a_all = torch.cat(hidden_a_list, dim=0)
        hidden_b_all = torch.cat(hidden_b_list, dim=0)

    # Compute alignment via least-squares (or learned with gradient descent)
    # alignment_matrix @ hidden_a.T ≈ hidden_b.T
    alignment_matrix = torch.linalg.lstsq(
        hidden_a_all.T,
        hidden_b_all.T
    ).solution.T

    return alignment_matrix
```

**5. Sequential Multi-Agent Pipeline**

Chain agents together, sharing KV caches and latent representations.

```python
def run_sequential_latent_collaboration(agents, initial_query, alignment_matrices):
    """
    Execute sequential multi-agent reasoning in latent space.
    Each agent inherits complete context from predecessors.
    Args:
        agents: List of LLM models (potentially different architectures)
        initial_query: Input text/query
        alignment_matrices: Dict mapping (agent_i, agent_j) to alignment matrices
    Returns:
        final_answer: Text output from last agent
        reasoning_trace: Hidden state representations at each step
    """
    # Initialize first agent with query
    shared_cache = SharedKVCache(
        num_layers=agents[0].config.num_hidden_layers,
        hidden_dim=agents[0].config.hidden_size
    )

    initial_output = agents[0](
        initial_query,
        output_hidden_states=True,
        use_cache=True
    )

    current_hidden = extract_hidden_states(initial_output)
    reasoning_trace = [current_hidden]

    # Sequential agent chain
    for agent_idx in range(1, len(agents)):
        current_agent = agents[agent_idx]
        previous_agent = agents[agent_idx - 1]

        # Transfer cache to current agent
        inherited_cache = shared_cache.transfer_to_next_agent()

        # Align representations if agents have different architectures
        if agent_idx > 0 and (agent_idx - 1, agent_idx) in alignment_matrices:
            alignment = alignment_matrices[(agent_idx - 1, agent_idx)]
            current_hidden = current_hidden @ alignment.T

        # Current agent processes with inherited cache
        output = current_agent(
            inputs_embeds=current_hidden,  # Feed hidden states directly
            past_key_values=inherited_cache,
            output_hidden_states=True,
            use_cache=True
        )

        current_hidden = extract_hidden_states(output)
        reasoning_trace.append(current_hidden)

        # Update shared cache
        for layer_idx, layer_kv in enumerate(output.past_key_values):
            shared_cache.append_kv(layer_idx, layer_kv[0], layer_kv[1])

    # Final decoding (only once, from last agent)
    final_output = current_agent.generate(
        inputs_embeds=current_hidden,
        max_length=128,
        past_key_values=inherited_cache
    )

    return final_output, reasoning_trace
```

**6. Hierarchical Multi-Agent Architecture**

Support tree-structured agent topologies for complex reasoning.

```python
class HierarchicalLatentCollaboration:
    """
    Enables tree-structured agent collaboration with latent coordination.
    Leaf agents reason in parallel, then aggregate to parent agents.
    """
    def __init__(self, agent_tree, alignment_matrices):
        self.agent_tree = agent_tree  # Tree structure: {node_id: (agent, children)}
        self.alignment_matrices = alignment_matrices
        self.cache_tree = {}  # Cache for each node

    def forward(self, query, node_id='root'):
        """
        Forward pass through hierarchical agent tree.
        Children's outputs aggregate to parent's input.
        """
        agent, children = self.agent_tree[node_id]

        if not children:
            # Leaf agent: process query
            output = agent(query, output_hidden_states=True)
            hidden = extract_hidden_states(output)

        else:
            # Internal node: aggregate children's outputs
            child_hiddens = []

            for child_id in children:
                child_hidden, _ = self.forward(query, child_id)
                child_hiddens.append(child_hidden)

            # Aggregate children's latent representations
            aggregated = torch.cat(child_hiddens, dim=-1)

            # Process aggregated representation
            output = agent(
                inputs_embeds=aggregated,
                output_hidden_states=True
            )
            hidden = extract_hidden_states(output)

        self.cache_tree[node_id] = hidden
        return hidden, output
```

## Practical Guidance

**When to Use Latent Collaboration:**
- Multi-step reasoning where agents build on each other's analysis
- Heterogeneous agent architectures (different model sizes/types)
- Scenarios where efficiency is critical (4× speedup valuable)
- Complex reasoning tasks benefiting from specialized agent perspectives

**When NOT to Use:**
- Single-agent tasks (simpler solutions available)
- Scenarios requiring interpretable text-based communication logs
- Tasks where model architectures must remain isolated

**Key Hyperparameters:**
- `alignment_computation_strategy`: Least-squares vs. learned (least-squares simpler, faster)
- `cache_sharing_mode`: Full vs. selective layer sharing (full sharing faster but uses more memory)
- `num_agents`: Longer chains → better reasoning but higher latency
- `hidden_dim`: Model-dependent; larger dims preserve more information

**Efficiency Metrics:**
- Expected speedup: 4-4.3× for sequential chains of 3-5 agents
- Token reduction: 71-84% fewer output tokens vs. text-based coordination
- Memory overhead: ~15-20% additional for shared caches

**Integration Pattern:**
Latent collaboration integrates into multi-agent RL systems where agents coordinate on shared tasks. Use as internal reasoning mechanism alongside external action policies.

## Reference

Research paper: https://arxiv.org/abs/2511.20639
