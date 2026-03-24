---
name: treecua-gui-automation
title: "TreeCUA: Efficiently Scaling GUI Automation with Tree-Structured Evolution"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.09662"
keywords: [GUI Automation, Data Scaling, Tree Structures, Multi-Agent Framework, Trajectory Evolution]
description: "Scale GUI automation by organizing trajectories into tree structures for reuse and branching exploration, reducing data cost while maximizing step-level diversity through adaptive topology."
---

# TreeCUA: Efficiently Scaling GUI Automation with Tree-Structured Evolution

## Problem Context

Scaling GUI automation creates two critical bottlenecks: step redundancy (different applications repeatedly explore the same early functional entry points, wasting compute) and trajectory bias (models exhibit inherent bias toward high-frequency behaviors, limiting exploration of long-tail professional functionalities).

## Core Concept

TreeCUA organizes GUI automation trajectories into tree structures rather than linear chains. Intermediate states are replayed deterministically without system snapshots, enabling branching exploration. Adaptive topology concentrates early diversity while refining later steps, and global memory tracks explored prefixes to enforce novelty across independent trees.

## Architecture Overview

- **Multi-Agent Framework**: Exploration, verification, summary, and evaluation agents orchestrate trajectory generation
- **Node Reuse**: Deterministic replay restores intermediate states enabling branching without expensive snapshots
- **Adaptive Topology**: Temporal width decay reduces branching factor at deeper levels
- **Global Memory**: Tracks previously explored prefixes across trees to enforce novelty
- **Training**: Two-stage SFT + TreeCUA-DPO using branching nodes as natural preference pairs

## Implementation

**Phase 1: Tree Structure and Node Reuse**

```python
class GUIActionTree:
    def __init__(self, app, documentation):
        self.app = app
        self.documentation = documentation
        self.root = TreeNode(state=initial_app_state())
        self.global_memory = GlobalPrefixMemory()

    def deterministic_replay(self, action_sequence):
        """Replay sequence without system snapshots"""
        state = initial_app_state()

        for action in action_sequence:
            # Execute action deterministically
            state = self.app.execute(action, state)

        return state

    def explore_node(self, parent_node, depth, width_at_depth):
        """Generate child nodes at given depth"""
        # Check global memory for already-explored paths
        prefix = parent_node.path_from_root()

        if self.global_memory.is_explored(prefix):
            return []  # Skip redundant exploration

        # Replay to intermediate state
        state = self.deterministic_replay(prefix)

        children = []

        for _ in range(width_at_depth):
            # Exploration agent generates candidate action
            candidate = self.exploration_agent.propose_action(
                state, self.documentation, self.global_memory
            )

            # Verification agent validates
            is_valid = self.verification_agent.validate(
                candidate, state, self.app
            )

            if is_valid:
                # Summary agent extracts semantic intent
                intent = self.summary_agent.abstract_intent(candidate)

                # Evaluation agent filters low-quality
                quality_score = self.evaluation_agent.assess_quality(
                    intent, self.global_memory
                )

                if quality_score > threshold:
                    # Create child node
                    child = TreeNode(
                        parent=parent_node,
                        action=candidate,
                        semantic_intent=intent,
                        quality=quality_score
                    )
                    children.append(child)

            # Update global memory
            new_path = parent_node.path_from_root() + [candidate]
            self.global_memory.record(new_path)

        return children
```

**Phase 2: Adaptive Topology with Temporal Width Decay**

```python
def adaptive_width_schedule(max_depth, initial_width=5):
    """Compute branching factor at each depth"""
    # Temporal width decay: reduce branching at deeper levels
    widths = []

    for depth in range(max_depth):
        # Decay function: e^(-decay_rate * depth)
        decay_factor = np.exp(-0.3 * depth)
        width = int(initial_width * decay_factor)
        width = max(width, 1)  # At least 1
        widths.append(width)

    return widths

def build_tree(app, documentation, max_depth=8):
    tree = GUIActionTree(app, documentation)
    width_schedule = adaptive_width_schedule(max_depth, initial_width=5)

    nodes_to_process = [(tree.root, 0)]  # (node, depth)

    while nodes_to_process:
        node, depth = nodes_to_process.pop(0)

        if depth >= max_depth:
            continue

        # Explore with adaptive width
        width = width_schedule[depth]
        children = tree.explore_node(node, depth, width)

        for child in children:
            nodes_to_process.append((child, depth + 1))

    return tree
```

**Phase 3: Training Pipeline**

```python
def extract_training_data(tree):
    """Convert tree structure to training examples"""
    data = []

    def traverse(node):
        if node.is_root():
            return

        # Extract trajectory from root to node
        trajectory = node.path_from_root()
        action_sequence = [n.action for n in trajectory]
        intent_sequence = [n.semantic_intent for n in trajectory]

        data.append({
            'trajectory': action_sequence,
            'intents': intent_sequence,
            'quality': node.quality
        })

        for child in node.children:
            traverse(child)

    traverse(tree.root)
    return data

# Stage 1: Supervised Fine-Tuning
def sft_training(model, tree_data):
    for epoch in range(num_epochs):
        for sample in tree_data:
            trajectory = sample['trajectory']
            intents = sample['intents']

            # Predict next action
            logits = model.forward(current_state, context=intents)

            # Loss: cross-entropy on ground-truth actions
            loss = cross_entropy_loss(logits, trajectory)
            loss.backward()
            optimizer.step()

# Stage 2: TreeCUA-DPO
def treecua_dpo_training(model, tree):
    """Use sibling nodes as natural preference pairs"""
    for node in tree.all_nodes():
        if len(node.children) >= 2:
            # Create preference pairs from siblings
            for i, child_a in enumerate(node.children):
                for child_b in node.children[i+1:]:
                    # Prefer higher-quality child
                    if child_a.quality > child_b.quality:
                        preferred = child_a.path_from_root()
                        dispreferred = child_b.path_from_root()

                        # DPO loss
                        log_ratio = (
                            log_likelihood(model, preferred) -
                            log_likelihood(model, dispreferred)
                        )
                        loss = -log_sigmoid(log_ratio)
                        loss.backward()
                        optimizer.step()
```

## Practical Guidance

**When to use**: Deploy for GUI automation tasks requiring exploration across diverse application interfaces (e.g., e-commerce sites, SaaS platforms, system administration tools).

**Initial width**: Start with 3–5 branches at depth 0; decay factor around 0.3 provides good balance between early diversity and refinement depth.

**Documentation structure**: Format official docs as hierarchical structures; include API reference alongside task descriptions for guidance.

**Node quality scoring**: Combine semantic novelty (no similar actions in history), action validity, and task progress. Balance all three factors equally initially; adjust based on performance.

**Global memory efficiency**: Use bloom filters or MinHash for large-scale prefix tracking across thousands of trees.

## Reference

TreeCUA generates 708k step-level samples and 101k high-quality sub-trajectories through tree-structured evolution, reducing data cost by organizing exploration into reusable intermediate states. The adaptive topology mechanism concentrates computational budget on early decision points while maintaining refinement capacity for later steps, substantially improving both diversity and quality in GUI automation training data.
