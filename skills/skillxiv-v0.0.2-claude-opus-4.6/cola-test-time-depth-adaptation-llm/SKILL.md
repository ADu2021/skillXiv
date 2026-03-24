---
name: cola-test-time-depth-adaptation-llm
title: "Skip a Layer or Loop it? Test-Time Depth Adaptation of Pretrained LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.07996"
keywords: [LLM Architecture, Depth Adaptation, Test-Time Optimization, Monte Carlo Tree Search, Layer Reordering]
description: "Dynamically adapt LLM depth per input at test time by skipping, repeating, or reordering layers using MCTS search, correcting 60% of initially wrong predictions and processing 75% of correct predictions with shorter architectures without retraining."
---

# CoLa: Chain-of-Layers Architecture Search for Test-Time Adaptation

Standard language models apply all layers sequentially. But not every input needs every layer. Difficult examples benefit from depth, easy examples waste computation. CoLa (Chain-of-Layers) reframes inference as an architecture search problem: for each test input, find the optimal layer sequence by skipping unnecessary layers, repeating useful ones, or reordering them. Monte Carlo Tree Search efficiently explores this space without training, finding custom architectures that improve accuracy or reduce latency.

The method reveals that over 75% of correctly predicted samples could be processed through shorter paths, and over 60% of initially incorrect predictions become correct through architectural reconfiguration. This enables substantial efficiency gains or accuracy improvements depending on your objectives.

## Core Concept

The key insight is that transformer layers are relatively independent modules. A sequence of layers is not sacred; the same model can process inputs via different layer orderings. CoLa searches for the optimal "chain of layers" for each input: which layers to include, which to skip, which to repeat. This search happens at test time via MCTS, a planning algorithm that balances exploration (trying new architectures) and exploitation (focusing on promising ones).

For easy inputs, this finds shorter paths. For hard inputs, it identifies beneficial layer reorderings or repetitions. The method imposes no training overhead and works with frozen pretrained models.

## Architecture Overview

- **Layer Manipulation Space**: Skip, repeat, or reorder individual layers or layer blocks
- **MCTS Search**: Exploration-exploitation via Upper Confidence Bound over architecture space
- **State Representation**: Current layer index, already-processed layers, input progress
- **Action Space**: Skip 1-4 layers, repeat 1-4 layers, continue with next layer
- **Reward Signal**: Model accuracy on input (did you predict correctly?)
- **Simulation Budget**: 200 MCTS simulations per input, ~5x inference time typical
- **Compatible Layers**: Works with LLaMA-3, OLMoE, and other transformer families

## Implementation

### Step 1: Define Layer Manipulation Operations

Create the action space: all valid ways to modify layer sequences:

```python
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

class LayerAction(Enum):
    """Possible actions during layer sequence generation."""
    CONTINUE = "continue"      # Move to next layer
    SKIP = "skip"              # Skip ahead by k layers
    REPEAT = "repeat"          # Repeat current layer k times
    REORDER = "reorder"        # Reorder next k layers

@dataclass
class LayerState:
    """State in the layer sequence search space."""
    current_layer_idx: int      # Which layer we're at
    used_layers: List[int]      # Layers we've applied
    hidden_state: torch.Tensor  # Model state after layers used so far
    num_steps: int             # Computational steps used

class LayerSequenceBuilder:
    def __init__(self, model: nn.Module, num_layers: int = 32, max_skip: int = 4):
        self.model = model
        self.num_layers = num_layers
        self.max_skip = max_skip
        self.max_repeat = max_skip

    def get_valid_actions(self, state: LayerState) -> List[Tuple[LayerAction, int]]:
        """Return valid actions for current state."""
        actions = []

        # CONTINUE: always valid unless at end
        if state.current_layer_idx < self.num_layers - 1:
            actions.append((LayerAction.CONTINUE, 1))

        # SKIP: skip ahead k layers
        for k in range(1, self.max_skip + 1):
            if state.current_layer_idx + k < self.num_layers:
                actions.append((LayerAction.SKIP, k))

        # REPEAT: repeat current layer k times
        if state.current_layer_idx < self.num_layers:
            for k in range(1, self.max_repeat + 1):
                actions.append((LayerAction.REPEAT, k))

        # REORDER: reverse order of next k layers (simple reordering)
        for k in range(2, self.max_skip + 1):
            if state.current_layer_idx + k < self.num_layers:
                actions.append((LayerAction.REORDER, k))

        return actions

    def apply_action(self, state: LayerState,
                    action: LayerAction, param: int) -> LayerState:
        """Apply action to state, returning new state."""
        new_used_layers = state.used_layers.copy()
        new_idx = state.current_layer_idx

        if action == LayerAction.CONTINUE:
            new_used_layers.append(new_idx)
            new_idx += 1

        elif action == LayerAction.SKIP:
            new_idx += param

        elif action == LayerAction.REPEAT:
            new_used_layers.extend([new_idx] * param)

        elif action == LayerAction.REORDER:
            # Reverse order of next k layers
            for i in range(param):
                new_used_layers.append(new_idx + param - 1 - i)
            new_idx += param

        # Apply layers to hidden state
        hidden = state.hidden_state
        for layer_idx in new_used_layers[len(state.used_layers):]:
            layer = self.model.model.layers[layer_idx]
            with torch.no_grad():
                hidden = layer(hidden)[0]

        return LayerState(
            current_layer_idx=new_idx,
            used_layers=new_used_layers,
            hidden_state=hidden,
            num_steps=state.num_steps + 1
        )
```

### Step 2: Implement MCTS for Architecture Search

Search the space of layer sequences using Monte Carlo Tree Search:

```python
import math
import random
from collections import defaultdict

class MCTSNode:
    """Node in MCTS tree for layer sequences."""
    def __init__(self, state: LayerState):
        self.state = state
        self.children = {}  # Map action -> child node
        self.visit_count = 0
        self.value_sum = 0.0  # Sum of rewards

    def ucb_value(self, c: float = 1.41) -> float:
        """Compute Upper Confidence Bound for this node."""
        if self.visit_count == 0:
            return float('inf')
        exploitation = self.value_sum / self.visit_count
        exploration = c * math.sqrt(math.log(self.parent_visits) / self.visit_count)
        return exploitation + exploration

class LayerSequenceMCTS:
    def __init__(self, model: nn.Module, num_layers: int = 32):
        self.model = model
        self.builder = LayerSequenceBuilder(model, num_layers)
        self.root_nodes = {}  # Cache for initial states

    def search(self, initial_hidden: torch.Tensor,
              target_output: int,
              num_simulations: int = 200,
              max_depth: int = 32) -> List[int]:
        """
        Run MCTS to find optimal layer sequence.
        Returns list of layer indices in optimal order.
        """
        initial_state = LayerState(
            current_layer_idx=0,
            used_layers=[],
            hidden_state=initial_hidden,
            num_steps=0
        )

        root = MCTSNode(initial_state)

        for sim in range(num_simulations):
            # Selection + Expansion
            node = root
            while node.state.current_layer_idx < len(self.model.model.layers) - 1:
                actions = self.builder.get_valid_actions(node.state)

                if not actions:
                    break

                action_tuple = random.choice(actions)

                if action_tuple not in node.children:
                    # Expansion: create new child
                    new_state = self.builder.apply_action(
                        node.state, action_tuple[0], action_tuple[1]
                    )
                    node.children[action_tuple] = MCTSNode(new_state)

                node = node.children[action_tuple]

            # Simulation: run model with current layer sequence
            final_hidden = node.state.hidden_state
            logits = self.model.lm_head(final_hidden)
            prediction = torch.argmax(logits, dim=-1)
            accuracy = 1.0 if prediction == target_output else 0.0

            # Backpropagation
            while node:
                node.visit_count += 1
                node.value_sum += accuracy
                node = node.parent if hasattr(node, 'parent') else None

        # Return best action sequence
        best_sequence = self._extract_best_sequence(root)
        return best_sequence

    def _extract_best_sequence(self, root: MCTSNode) -> List[int]:
        """Extract the best layer sequence from MCTS tree."""
        sequence = []
        node = root

        while node.state.current_layer_idx < len(self.model.model.layers):
            # Choose child with highest visit count (most explored)
            if not node.children:
                break

            best_action = max(
                node.children.keys(),
                key=lambda a: node.children[a].visit_count
            )

            node = node.children[best_action]
            sequence.extend(node.state.used_layers[len(sequence):])

        return sequence
```

### Step 3: Evaluate Architecture Search Results

Measure accuracy and efficiency improvements from the searched architectures:

```python
def evaluate_cola_architecture(model: nn.Module,
                              dataset,
                              num_test_samples: int = 500,
                              num_simulations: int = 200) -> Dict:
    """
    Evaluate CoLa: compare default vs searched architectures.
    """
    default_depth = len(model.model.layers)
    improvements = {
        "correct_samples_shorter": 0,
        "incorrect_samples_fixed": 0,
        "avg_depth_reduction": 0,
        "total_samples": 0
    }

    default_correct = 0
    cola_correct = 0

    for sample_idx, (input_ids, labels) in enumerate(dataset.take(num_test_samples)):
        # Get embeddings
        embeddings = model.model.embed_tokens(input_ids)

        # Default: use all layers
        hidden = embeddings
        for layer in model.model.layers:
            hidden = layer(hidden)[0]
        default_logits = model.lm_head(hidden)
        default_pred = torch.argmax(default_logits, dim=-1)
        default_correct += (default_pred == labels).float().mean().item()

        # CoLa: search for optimal architecture
        mcts = LayerSequenceMCTS(model)
        optimal_layers = mcts.search(
            embeddings, labels.item(),
            num_simulations=num_simulations
        )

        # Apply optimal architecture
        hidden = embeddings
        for layer_idx in optimal_layers:
            hidden = model.model.layers[layer_idx](hidden)[0]
        cola_logits = model.lm_head(hidden)
        cola_pred = torch.argmax(cola_logits, dim=-1)
        cola_correct += (cola_pred == labels).float().mean().item()

        # Track metrics
        if default_pred == labels:
            if len(optimal_layers) < default_depth:
                improvements["correct_samples_shorter"] += 1
        else:
            if cola_pred == labels:
                improvements["incorrect_samples_fixed"] += 1

        improvements["avg_depth_reduction"] += (default_depth - len(optimal_layers))
        improvements["total_samples"] += 1

    improvements["avg_depth_reduction"] /= improvements["total_samples"]
    improvements["default_accuracy"] = default_correct / num_test_samples
    improvements["cola_accuracy"] = cola_correct / num_test_samples
    improvements["accuracy_improvement"] = (
        improvements["cola_accuracy"] - improvements["default_accuracy"]
    )

    return improvements
```

### Step 4: Efficient Inference with CoLa

Use the searched architecture for efficient inference:

```python
def inference_with_cola(model: nn.Module,
                       input_text: str,
                       num_simulations: int = 200,
                       return_metrics: bool = False) -> str:
    """
    Inference with dynamically adapted layer depth.
    """
    # Tokenize input
    tokenizer = model.tokenizer
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # Get embeddings
    embeddings = model.model.embed_tokens(input_ids)

    # Search for optimal layer sequence
    mcts = LayerSequenceMCTS(model, num_layers=len(model.model.layers))

    # Initial forward to get hidden state for MCTS
    initial_hidden = embeddings
    optimal_layers = mcts.search(
        initial_hidden,
        target_output=None,  # We're generating, not classifying
        num_simulations=num_simulations
    )

    # Apply layers in optimal order
    hidden = embeddings
    for layer_idx in optimal_layers:
        layer = model.model.layers[layer_idx]
        hidden = layer(hidden)[0]

    # Apply remaining layers if needed
    num_default_layers = len(model.model.layers)
    for layer_idx in range(num_default_layers):
        if layer_idx not in optimal_layers:
            layer = model.model.layers[layer_idx]
            hidden = layer(hidden)[0]

    # Final layer norm and head
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)
    output_ids = torch.argmax(logits, dim=-1)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if return_metrics:
        return output_text, {
            "layers_used": len(optimal_layers),
            "depth_reduction": (num_default_layers - len(optimal_layers)) / num_default_layers
        }
    else:
        return output_text
```

## Practical Guidance

| Parameter | Recommended Value | Notes |
|---|---|---|
| MCTS Simulations | 200 | Balance between search quality and inference latency |
| Max Skip Length | 4 | Prevents skipping too many useful layers |
| Max Repeat Count | 4 | Limits layer repetition to avoid divergence |
| UCB Exploration Constant | 1.41 | Standard value (sqrt(2) ≈ 1.41) |
| Search Depth | 32 | Matches typical LLM layer count |
| Inference Latency Multiplier | ~5× | 200 simulations ≈ 5× slower than single forward |
| Model Families | LLaMA-3, OLMoE, Mixtral | Tested on various architectures |

**When to use CoLa:**
- Inference scenarios where latency is flexible (batch processing, offline analysis)
- Applications wanting accuracy improvements over standard inference
- Scenarios exploring efficiency vs accuracy tradeoffs
- Evaluating which layers are actually useful for specific inputs
- Research into transformer layer importance and redundancy

**When NOT to use CoLa:**
- Real-time inference (200 MCTS simulations add 5× latency overhead)
- Latency-sensitive applications (streaming, interactive systems)
- Already-optimized models (pruning, distillation often better)
- Memory-constrained deployment (MCTS requires storing multiple states)
- Tasks where layer order matters (some specialized architectures)

**Common pitfalls:**
- MCTS simulations too low (< 100), missing good architectures
- UCB exploration constant too high, overly exploring suboptimal paths
- Not caching initial embeddings, recomputing for each simulation
- Assuming layer sequence independence (some correlations exist)
- Forgetting to apply remaining default layers after search
- Using identical random seeds across MCTS runs, missing diversity
- Not normalizing hidden states between layers, causing divergence

## Reference

Zhou, Y., Sun, S., Huang, Z., & Yang, S. (2025). Skip a Layer or Loop it? Test-Time Depth Adaptation of Pretrained LLMs. arXiv:2507.07996. https://arxiv.org/abs/2507.07996
