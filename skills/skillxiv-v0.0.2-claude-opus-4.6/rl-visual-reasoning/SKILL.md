---
name: rl-visual-reasoning
title: "What does RL improve for Visual Reasoning? A Frankenstein-Style Analysis"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.12395"
keywords: [Multimodal Learning, Reinforcement Learning, Visual Reasoning, Model Analysis, Layer Freezing]
description: "RL in vision-language models improves inference-time alignment between vision and reasoning in mid-to-late layers, not vision ability or reasoning separately. Strategic layer freezing enables diagnosis of which components contribute to RL gains."
---

# What RL Improves for Visual Reasoning: Frankenstein-Style Analysis

## Problem Context

Benchmark scores alone mask what RL actually improves in multimodal models. Standard supervised fine-tuning and RL finetuning both show improvements, but it's unclear whether RL enhances vision understanding, reasoning capability, or something else. Fine-grained evaluation reveals vision and reasoning abilities don't improve monotonically through the Base→SFT→RL pipeline. Understanding RL's true contribution requires beyond-benchmark analysis.

## Core Concept

Through systematic "Frankenstein-style" analysis (mixing layers from different model variants), the paper discovers that **RL doesn't uniformly enhance vision or reasoning, but rather improves alignment between them**. Specifically:

1. RL induces consistent inference-time shift in mid-to-late transformer layers
2. This shift increases attention from reasoning tokens to visual tokens
3. The shift is concentrated, affecting fewer parameter directions than SFT
4. The shift is necessary and sufficient for RL gains (shown via layer freezing)

This insight reveals RL's true function: optimizing coordination between vision and reasoning subsystems rather than improving individual capabilities.

## Architecture Overview

- **Layer-Wise Analysis**: Decompose model into blocks, probe each layer's contribution
- **Frankenstein Merging**: Mix layers from Base, SFT, and RL models systematically
- **Attention Pattern Tracking**: Monitor vision-to-reasoning token attention
- **Parameter Direction Analysis**: Identify dominant optimization directions
- **Layer Freezing Experiments**: Freeze specific layers to measure contribution
- **Fine-Grained Benchmarking**: Evaluate components beyond aggregate scores
- **Transfer Learning Validation**: Test if layer changes transfer between model families

## Implementation

Layer-wise decomposition and Frankenstein analysis:

```python
class FrankensteinAnalysis:
    """
    Mix layers from different model variants to isolate RL contributions.
    Enables fine-grained understanding beyond aggregate metrics.
    """

    def __init__(self, base_model, sft_model, rl_model):
        self.base = base_model
        self.sft = sft_model
        self.rl = rl_model
        self.num_layers = len(base_model.transformer.layers)

    def create_frankenstein_model(self, vision_rl_cutoff, reasoning_rl_cutoff):
        """
        Create hybrid model: use RL weights up to cutoff, base below.
        Tests which layers contribute to RL gains.
        """
        frankenstein = copy.deepcopy(self.base)

        # Replace vision-side layers up to cutoff with RL version
        for i in range(vision_rl_cutoff):
            frankenstein.vision_encoder.layers[i] = \
                self.rl.vision_encoder.layers[i]

        # Replace reasoning-side layers up to cutoff with RL version
        for i in range(reasoning_rl_cutoff):
            frankenstein.language_model.layers[i] = \
                self.rl.language_model.layers[i]

        return frankenstein

    def sweep_layer_cutoffs(self):
        """
        Systematically create Frankenstein models with varying cutoffs.
        Identify which layers are critical for RL improvement.
        """
        results = {}

        for v_cutoff in range(self.num_layers):
            for r_cutoff in range(self.num_layers):
                frankenstein = self.create_frankenstein_model(
                    v_cutoff, r_cutoff)

                # Evaluate on benchmark
                performance = evaluate_benchmark(frankenstein)

                results[(v_cutoff, r_cutoff)] = {
                    'performance': performance,
                    'vision_improved': performance > base_performance,
                    'reasoning_improved': performance > base_performance
                }

        return results

    def analyze_layer_contributions(self):
        """
        Determine which individual layers are necessary for RL gains.
        Identify bottleneck layers.
        """
        contributions = {}

        for layer_idx in range(self.num_layers):
            # Freeze layer (use base weights instead of RL)
            model_frozen = copy.deepcopy(self.rl)
            model_frozen.transformer.layers[layer_idx] = \
                self.base.transformer.layers[layer_idx]

            # Evaluate
            perf_frozen = evaluate_benchmark(model_frozen)

            # Contribution = performance drop when layer is frozen
            contribution = self.rl_performance - perf_frozen
            contributions[layer_idx] = contribution

        return contributions
```

Attention pattern analysis:

```python
class AttentionAnalysis:
    """
    Track how RL changes token attention patterns, especially
    vision-to-reasoning and reasoning-to-vision attention.
    """

    def __init__(self, base_model, sft_model, rl_model):
        self.base = base_model
        self.sft = sft_model
        self.rl = rl_model

    def extract_attention_patterns(self, model, input_ids, pixel_values):
        """
        Get attention weights for all layers and heads.
        """
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                output_attentions=True
            )
        return outputs.attentions  # List of [B, H, N, N] tensors

    def compute_vision_to_reasoning_attention(self, attention_weights,
                                               vision_length, text_length):
        """
        Extract portion of attention matrix from vision tokens to reasoning tokens.
        """
        # Attention matrix layout: [vision_tokens | reasoning_tokens]
        vision_end = vision_length
        reasoning_start = vision_length
        reasoning_end = vision_length + text_length

        v2r_attention = attention_weights[
            :, :, reasoning_start:reasoning_end, :vision_end]

        return v2r_attention

    def compare_attention_shifts(self, batch):
        """
        Compare how attention patterns shift: Base → SFT → RL
        Focus on vision-to-reasoning alignment.
        """
        input_ids = batch['input_ids']
        pixel_values = batch['pixel_values']

        # Extract attentions from all three models
        base_attn = self.extract_attention_patterns(
            self.base, input_ids, pixel_values)
        sft_attn = self.extract_attention_patterns(
            self.sft, input_ids, pixel_values)
        rl_attn = self.extract_attention_patterns(
            self.rl, input_ids, pixel_values)

        vision_length = pixel_values.shape[1]
        text_length = input_ids.shape[1]

        results = {
            'layer_wise_shifts': [],
            'mid_late_concentration': None
        }

        for layer_idx, (base_attn_layer, sft_attn_layer,
                        rl_attn_layer) in enumerate(
            zip(base_attn, sft_attn, rl_attn)):

            # Compute vision-to-reasoning attention
            base_v2r = self.compute_vision_to_reasoning_attention(
                base_attn_layer, vision_length, text_length)
            sft_v2r = self.compute_vision_to_reasoning_attention(
                sft_attn_layer, vision_length, text_length)
            rl_v2r = self.compute_vision_to_reasoning_attention(
                rl_attn_layer, vision_length, text_length)

            # Measure shift: how much more attention RL gives to vision
            shift_sft = (sft_v2r - base_v2r).mean()
            shift_rl = (rl_v2r - base_v2r).mean()

            results['layer_wise_shifts'].append({
                'layer': layer_idx,
                'sft_shift': shift_sft.item(),
                'rl_shift': shift_rl.item(),
                'is_mid_late': layer_idx >= len(base_attn) // 2
            })

        return results
```

Parameter direction analysis:

```python
class ParameterDirectionAnalysis:
    """
    Analyze how RL updates differ from SFT in terms of parameter space.
    RL uses fewer, more focused directions.
    """

    def __init__(self, base_model, sft_model, rl_model):
        self.base = base_model
        self.sft = sft_model
        self.rl = rl_model

    def compute_parameter_directions(self, model1, model2):
        """
        Compute direction in parameter space between two models.
        """
        directions = {}
        for name, param1 in model1.named_parameters():
            param2 = dict(model2.named_parameters())[name]
            # Direction: difference vector
            direction = param2 - param1
            directions[name] = direction
        return directions

    def analyze_direction_dominance(self):
        """
        Measure how concentrated RL updates are vs SFT.
        RL uses fewer dominant directions (more focused optimization).
        """
        sft_dirs = self.compute_parameter_directions(
            self.base, self.sft)
        rl_dirs = self.compute_parameter_directions(
            self.base, self.rl)

        # Compute PCA to find dominant directions
        sft_singular_values = compute_pca_spectrum(sft_dirs)
        rl_singular_values = compute_pca_spectrum(rl_dirs)

        # Measure concentration: how quickly singular values decay
        sft_concentration = sft_singular_values[0] / sft_singular_values.sum()
        rl_concentration = rl_singular_values[0] / rl_singular_values.sum()

        return {
            'sft_concentration': sft_concentration,
            'rl_concentration': rl_concentration,
            'rl_more_focused': rl_concentration > sft_concentration,
            'sft_singular_values': sft_singular_values,
            'rl_singular_values': rl_singular_values
        }
```

Layer freezing experiments:

```python
class LayerFreezeExperiments:
    """
    Freeze specific layers to determine necessity for RL gains.
    """

    def run_freeze_experiments(self, rl_model, base_model, benchmark):
        """
        For each layer, freeze it and measure performance drop.
        Identifies essential layers for RL gains.
        """
        baseline_rl_perf = evaluate(rl_model, benchmark)

        freeze_results = []

        for layer_idx in range(len(rl_model.transformer.layers)):
            # Create model with layer frozen
            model_frozen = copy.deepcopy(rl_model)

            # Replace RL layer weights with base weights
            model_frozen.transformer.layers[layer_idx] = \
                base_model.transformer.layers[layer_idx]

            # Evaluate frozen model
            perf_frozen = evaluate(model_frozen, benchmark)

            # Performance drop when layer frozen
            perf_drop = baseline_rl_perf - perf_frozen

            freeze_results.append({
                'layer': layer_idx,
                'perf_with_rl': baseline_rl_perf,
                'perf_frozen': perf_frozen,
                'perf_drop': perf_drop,
                'is_essential': perf_drop > threshold
            })

        return freeze_results
```

## Practical Guidance

**When to analyze**:
- Applying RL to multimodal models
- Need to understand what's actually improving
- Want to optimize selective component training
- Debugging unexpected RL behavior

**Key findings summary**:

1. **Vision ability doesn't improve monotonically**: Base→SFT→RL pipeline doesn't guarantee vision improvement
2. **Reasoning ability doesn't improve monotonically**: Same for reasoning
3. **Alignment improves consistently**: Vision-to-reasoning coordination improves across pipeline
4. **Mid-to-late concentration**: RL changes concentrated in middle and late layers, early layers unchanged
5. **Less diverse refinements**: RL uses fewer parameter directions than SFT

**Diagnostic workflow**:

1. Run Frankenstein analysis to identify critical layers
2. Analyze attention patterns in mid-to-late layers
3. Perform parameter direction analysis to measure optimization focus
4. Run layer freezing to validate findings
5. Apply insights to selective training (freeze early layers, fine-tune mid-late)

**Expected insights**:
- 10-30% of layers account for most RL gains
- Vision-to-reasoning attention increases 2-5× in critical layers
- RL parameter updates use 30-50% fewer dominant directions than SFT
- Layer freezing reveals 3-5 critical layers per model family

**Transfer properties**:
- Layer changes often transfer between model families (e.g., ViT→ConvNeXt)
- Attention pattern changes generalize well
- Parameter direction insights less transferable

## Reference

Frankenstein-style analysis reveals that RL improves multimodal models primarily through optimizing coordination between vision and reasoning components, particularly in mid-to-late layers. This insight enables more targeted training and better understanding of what RL actually contributes to model performance.
