---
name: refusal-falls-off-cliff-safety-reasoning
title: "Refusal Falls off a Cliff: How Safety Alignment Fails in Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.06036"
keywords: [safety alignment, reasoning models, mechanistic interpretability, refusal cliff, attention analysis]
description: "Identify and patch critical safety vulnerabilities in large reasoning models. Via linear probing and causal intervention, locate specific attention heads responsible for alignment degradation at final tokens. Recover safety via 'Cliff-as-a-Judge' data curation targeting examples exhibiting largest refusal decline, achieving comparable improvements using only 1.7% of vanilla safety training data."
---

# Refusal Falls off a Cliff: Safety Alignment Failures in Reasoning Models

## Core Concept

Large reasoning models maintain refusal intentions during internal reasoning but experience sharp alignment degradation at generation's final tokens, allowing jailbreaks to succeed. This "refusal cliff" is not uniform failure but concentrated in specific attention heads. Mechanistic analysis identifies problematic heads; targeted data curation fixes them efficiently.

## Architecture Overview

- **Linear Probing**: Trace refusal intentions across token positions to locate cliff
- **Causal Intervention**: Identify specific attention heads causing degradation
- **Head Ablation**: Minimal (3%) head removal reduces attack success below 10%
- **Cliff-as-a-Judge Curation**: Automatically select training examples exhibiting largest refusal drop
- **Data Efficiency**: 1.7% of vanilla safety training data achieves comparable safety

## Implementation Steps

### 1. Linear Probing for Refusal Tracking

Train linear probes to extract refusal intention at each token position.

```python
import torch
import torch.nn as nn

class RefusalProbe:
    def __init__(self, hidden_dim=4096, num_layers=48):
        """
        Linear probe: map hidden states → refusal probability
        """
        self.probes = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def extract_refusal_scores(self, model, prompt, target_answer):
        """
        Extract refusal intention at each token position.
        """

        # Forward pass with hook to capture hidden states
        hidden_states_by_layer = {}

        def capture_hook(module, input, output):
            layer_idx = len(hidden_states_by_layer)
            hidden_states_by_layer[layer_idx] = output[0]  # [seq_len, batch, hidden]

        # Register hooks on transformer layers
        hooks = []
        for layer_idx, layer in enumerate(model.transformer.h):
            h = layer.register_forward_hook(capture_hook)
            hooks.append(h)

        # Generate tokens and capture states
        with torch.no_grad():
            tokens = model.tokenize(prompt)
            model(tokens)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Apply probes to extract refusal at each position
        refusal_trajectories = {}  # layer_idx → [seq_len] refusal scores

        for layer_idx in range(self.num_layers):
            hidden = hidden_states_by_layer.get(layer_idx)
            if hidden is None:
                continue

            # Apply linear probe
            refusal_logits = self.probes[layer_idx](hidden)
            refusal_probs = torch.sigmoid(refusal_logits)

            refusal_trajectories[layer_idx] = refusal_probs.squeeze().detach().cpu().numpy()

        return refusal_trajectories

    def detect_refusal_cliff(self, refusal_trajectories):
        """
        Identify where refusal intention drops sharply.
        """

        cliff_locations = {}

        for layer_idx, trajectory in refusal_trajectories.items():
            # Compute first derivative (change in refusal score)
            diffs = np.diff(trajectory)

            # Find largest negative jump (cliff)
            cliff_idx = np.argmin(diffs)  # Most negative
            cliff_magnitude = diffs[cliff_idx]

            if cliff_magnitude < -0.2:  # Significant drop
                cliff_locations[layer_idx] = {
                    'position': cliff_idx,
                    'magnitude': cliff_magnitude,
                    'pre_cliff_score': trajectory[cliff_idx],
                    'post_cliff_score': trajectory[cliff_idx + 1]
                }

        return cliff_locations
```

### 2. Causal Intervention Analysis

Identify which attention heads cause refusal degradation via ablation.

```python
class AttentionHeadAnalysis:
    def __init__(self, model):
        self.model = model
        self.num_heads = model.config.num_attention_heads
        self.num_layers = model.config.num_hidden_layers

    def ablate_attention_head(self, layer_idx, head_idx):
        """
        Ablate specific attention head by zeroing its outputs.
        """

        def ablation_hook(module, input, output):
            # output = (attn_output, attn_weights)
            attn_output, attn_weights = output

            # Zero out specific head
            head_dim = attn_output.shape[-1] // self.num_heads
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim

            attn_output[:, :, start:end] = 0

            return (attn_output, attn_weights)

        # Register ablation hook
        layer = self.model.transformer.h[layer_idx].self_attn
        hook = layer.register_forward_hook(ablation_hook)

        return hook

    def evaluate_head_importance(self, jailbreak_prompt, safety_loss_fn):
        """
        Measure each head's contribution to safety by ablating and measuring loss.
        """

        critical_heads = []

        for layer_idx in range(self.num_layers):
            for head_idx in range(self.num_heads):
                # Baseline safety performance
                with torch.no_grad():
                    baseline_output = self.model(jailbreak_prompt)
                    baseline_loss = safety_loss_fn(baseline_output)

                # With head ablated
                hook = self.ablate_attention_head(layer_idx, head_idx)

                with torch.no_grad():
                    ablated_output = self.model(jailbreak_prompt)
                    ablated_loss = safety_loss_fn(ablated_output)

                hook.remove()

                # Importance: how much does ablation hurt safety?
                importance = baseline_loss - ablated_loss  # Positive = important for safety

                if importance > 0.1:  # Threshold for critical heads
                    critical_heads.append({
                        'layer': layer_idx,
                        'head': head_idx,
                        'importance': importance
                    })

        # Sort by importance
        critical_heads.sort(key=lambda x: x['importance'], reverse=True)

        return critical_heads

    def batch_ablate_critical_heads(self, critical_heads, ablation_ratio=0.03):
        """
        Ablate top critical heads (e.g., 3% of total).
        """

        num_total_heads = self.num_layers * self.num_heads
        num_to_ablate = max(1, int(num_total_heads * ablation_ratio))

        heads_to_ablate = critical_heads[:num_to_ablate]

        # Zero out these heads permanently
        for head_info in heads_to_ablate:
            layer_idx = head_info['layer']
            head_idx = head_info['head']

            layer = self.model.transformer.h[layer_idx].self_attn
            head_dim = layer.hidden_size // self.num_heads
            start = head_idx * head_dim
            end = (head_idx + 1) * head_dim

            # Permanently zero weight
            with torch.no_grad():
                layer.dense.weight[:, start:end] = 0
                if layer.dense.bias is not None:
                    layer.dense.bias[start:end] = 0

        print(f"Ablated {len(heads_to_ablate)} critical heads")
        return self.model
```

### 3. Cliff-as-a-Judge Data Curation

Automatically select training examples exhibiting largest refusal degradation.

```python
def cliff_as_a_judge_curation(model, safety_training_pool, num_examples=None):
    """
    Data curation: select examples exhibiting largest refusal cliff.
    These examples are most important for safety training.
    """

    probe = RefusalProbe()
    curated_examples = []

    for example in safety_training_pool:
        prompt = example['harmful_prompt']
        target = example['safe_refusal']

        # Extract refusal trajectory
        refusal_scores = probe.extract_refusal_scores(model, prompt, target)

        # Find cliff magnitude
        cliff_magnitude = 0
        for layer_idx, trajectory in refusal_scores.items():
            diffs = np.diff(trajectory)
            worst_diff = np.min(diffs)
            cliff_magnitude = min(cliff_magnitude, worst_diff)

        # Score: larger cliff = more important
        cliff_score = abs(cliff_magnitude)  # 0-1 range

        curated_examples.append({
            'example': example,
            'cliff_score': cliff_score
        })

    # Sort by cliff score (largest cliffs first)
    curated_examples.sort(key=lambda x: x['cliff_score'], reverse=True)

    # Select top examples
    if num_examples is None:
        num_examples = int(0.017 * len(safety_training_pool))  # 1.7% of pool

    selected = curated_examples[:num_examples]

    print(f"Selected {len(selected)} examples (1.7% of pool) with largest refusal cliffs")
    return [ex['example'] for ex in selected]
```

### 4. Safety Training with Curated Data

Retrain on curated examples for efficient safety recovery.

```python
def train_safety_with_curated_data(model, curated_examples, num_epochs=3):
    """
    Train on cliff-detected examples for efficient safety improvement.
    """

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    safety_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0

        for example in curated_examples:
            prompt = example['harmful_prompt']
            safe_response = example['safe_refusal']

            # Forward pass
            logits = model(prompt)

            # Loss: predict safe response
            loss = safety_loss_fn(logits, safe_response)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Safety loss={total_loss/len(curated_examples):.4f}")

    return model

# Experimental results
results = {
    'head_ablation': {
        'heads_ablated': '3%',
        'attack_success_rate_before': '80%',
        'attack_success_rate_after': '<10%',
    },
    'data_curation': {
        'vanilla_safety_training': {
            'data_size': '100%',
            'safety_improvement': 'Baseline'
        },
        'cliff_as_a_judge': {
            'data_size': '1.7%',
            'safety_improvement': 'Comparable to vanilla',
            'token_cost': '~60x reduction'
        }
    }
}
```

## Practical Guidance

**Probe Training**: Train refusal probes on clean refusal examples (high safety score) vs jailbreak attempts (low score). Use held-out validation for probe quality.

**Head Identification**: Ablate iteratively; stop when safety improves sufficiently. 3% ablation (3-5 heads on 48-layer models) is typical sweet spot.

**Data Curation**: Cliff score correlates with retraining importance. Top 1-2% of examples by cliff score provide 80% of safety benefit.

**Training Efficiency**: Use smaller learning rate (1e-5 vs 1e-4) to avoid destabilizing base model while focusing on safety pathways.

## When to Use / When NOT to Use

**Use When**:
- Deploying reasoning models with safety requirements
- Attack vectors exploit final-token refusal degradation
- Data efficiency is critical (limited retraining budget)
- You need interpretable safety improvements (ablate specific heads)

**NOT For**:
- Non-reasoning models without clear refusal cliff
- Scenarios where broad retraining is feasible
- Domains requiring complete alignment review

## Reference

This skill synthesizes findings from "Refusal Falls off a Cliff: How Safety Alignment Fails in Reasoning" (arXiv:2510.06036). Mechanistic analysis reveals concentrated safety vulnerabilities fixable via targeted intervention.
