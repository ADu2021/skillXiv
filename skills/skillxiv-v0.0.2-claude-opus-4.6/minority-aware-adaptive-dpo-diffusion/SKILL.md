---
name: minority-aware-adaptive-dpo-diffusion
title: "When Preferences Diverge: Aligning Diffusion Models with Minority-Aware Adaptive DPO"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2503.16921"
keywords: [Diffusion Models, Direct Preference Optimization, Minority-Aware Learning, Image Generation, Preference Alignment]
description: "Improve diffusion model alignment with human preferences by handling subjective and conflicting annotations. Adaptive-DPO incorporates minority-instance metrics (intra-annotator confidence and inter-annotator stability) to distinguish majority from minority samples, enhancing performance on both synthetic and real preference data."
---

## Core Concept

Adaptive-DPO addresses the challenge that human preferences for image generation are inherently subjective—different annotators may prefer different styles, compositions, or qualities. Standard Direct Preference Optimization (DPO) treats all preference pairs equally, but minority samples (where annotators disagree) can degrade model performance. Adaptive-DPO introduces annotation confidence and stability metrics to down-weight conflicting preferences while amplifying agreement, resulting in more aligned and robust models.

## Architecture Overview

The approach builds on Diffusion-DPO with additional components:

- **Intra-Annotator Confidence**: Measures consistency of individual annotators across multiple evaluations of same image pairs
- **Inter-Annotator Stability**: Quantifies agreement between different annotators on preference direction
- **Minority-Instance-Aware Metric**: Combines confidence and stability to identify unreliable preference labels
- **Adaptive DPO Loss**: Modified objective that reweights pairs based on annotation reliability
- **Dual Handling Strategy**: Enhances learning on majority (high confidence) samples while mitigating harm from minority samples

## Implementation Steps

### 1. Compute Intra-Annotator Confidence Scores

Measure individual annotator consistency by testing on repeated evaluations:

```python
# Compute intra-annotator confidence
import numpy as np
from scipy import stats

def compute_intra_annotator_confidence(preference_data, num_repeats=3):
    """
    Measure confidence of each annotator based on consistency.

    Args:
        preference_data: list of dicts with keys:
            - annotator_id: who made the judgment
            - image_pair: (image_a, image_b) tuple
            - preference: 0 or 1 (which image preferred)
            - timestamp: when judgment was made

        num_repeats: how many times each pair was re-judged

    Returns:
        confidence_scores: dict mapping annotator_id -> confidence [0, 1]
    """
    annotator_consistency = {}

    for annotator_id in set(a['annotator_id'] for a in preference_data):
        annotator_judgments = [p for p in preference_data
                              if p['annotator_id'] == annotator_id]

        # Group judgments by image pair
        pair_to_judgments = {}
        for judgment in annotator_judgments:
            pair_key = tuple(sorted([id(judgment['image_pair'][0]),
                                    id(judgment['image_pair'][1])]))
            if pair_key not in pair_to_judgments:
                pair_to_judgments[pair_key] = []
            pair_to_judgments[pair_key].append(judgment['preference'])

        # Compute consistency: how many pairs has agreement?
        consistent_pairs = 0
        total_repeated_pairs = 0

        for pair_key, preferences in pair_to_judgments.items():
            if len(preferences) >= 2:  # Repeated evaluation
                total_repeated_pairs += 1
                # Check if majority judgment is consistent
                most_common = stats.mode(preferences)[0]
                agreement_ratio = preferences.count(most_common) / len(preferences)
                if agreement_ratio >= 0.7:  # 70% agreement threshold
                    consistent_pairs += 1

        # Confidence is fraction of repeated pairs with consistency
        if total_repeated_pairs > 0:
            confidence = consistent_pairs / total_repeated_pairs
        else:
            confidence = 0.5  # Neutral if no repeats

        annotator_consistency[annotator_id] = confidence

    return annotator_consistency
```

### 2. Compute Inter-Annotator Stability

Measure agreement between different annotators on the same preference pairs:

```python
def compute_inter_annotator_stability(preference_data):
    """
    Measure annotation agreement across different annotators.

    Args:
        preference_data: list of preference judgments with annotator_id

    Returns:
        stability_scores: dict mapping pair_id -> stability [0, 1]
        pair_agreement: dict with voting results for each pair
    """
    # Group judgments by image pair (across all annotators)
    pair_to_judgments = {}

    for judgment in preference_data:
        pair_key = tuple(sorted([id(judgment['image_pair'][0]),
                                id(judgment['image_pair'][1])]))
        if pair_key not in pair_to_judgments:
            pair_to_judgments[pair_key] = []
        pair_to_judgments[pair_key].append(judgment['preference'])

    stability_scores = {}
    pair_agreement = {}

    for pair_key, judgments in pair_to_judgments.items():
        if len(judgments) < 2:  # Need multiple annotators
            stability_scores[pair_key] = 0.5
            pair_agreement[pair_key] = {
                'vote_0': 0,
                'vote_1': 0,
                'total': 0,
                'consensus': None,
            }
            continue

        # Count votes for each preference
        vote_counts = {0: judgments.count(0),
                      1: judgments.count(1)}

        # Stability = how strongly is consensus?
        total_votes = len(judgments)
        max_votes = max(vote_counts.values())
        stability = max_votes / total_votes

        stability_scores[pair_key] = stability

        # Determine consensus preference
        consensus_pref = 0 if vote_counts[0] > vote_counts[1] else 1

        pair_agreement[pair_key] = {
            'vote_0': vote_counts[0],
            'vote_1': vote_counts[1],
            'total': total_votes,
            'consensus': consensus_pref,
            'agreement_ratio': max_votes / total_votes,
        }

    return stability_scores, pair_agreement
```

### 3. Construct Minority-Instance-Aware Metric

Combine confidence and stability to identify unreliable annotations:

```python
def construct_minority_instance_metric(annotator_confidence, stability_scores,
                                      confidence_weight=0.5,
                                      stability_weight=0.5):
    """
    Create instance-level reliability metric for each preference pair.

    Args:
        annotator_confidence: dict of annotator_id -> confidence
        stability_scores: dict of pair_id -> inter-annotator stability
        confidence_weight: weight for annotator confidence in metric
        stability_weight: weight for inter-annotator stability

    Returns:
        instance_reliability: dict mapping pair_id -> reliability score [0, 1]
    """
    instance_reliability = {}

    for pair_id, stability in stability_scores.items():
        # Get average confidence of annotators for this pair
        # (simplified: use global mean, could be pair-specific)
        mean_confidence = np.mean(list(annotator_confidence.values()))

        # Weighted combination of confidence and stability
        reliability = (confidence_weight * mean_confidence +
                      stability_weight * stability)

        instance_reliability[pair_id] = reliability

    # Normalize to [0, 1]
    max_reliability = max(instance_reliability.values()) + 1e-8
    instance_reliability = {k: v / max_reliability
                           for k, v in instance_reliability.items()}

    return instance_reliability
```

### 4. Adaptive DPO Loss with Minority-Aware Weighting

Modify the DPO loss function to reweight samples based on annotation reliability:

```python
def adaptive_dpo_loss(model_logps, reference_logps, is_chosen,
                     instance_weights, beta=0.5, threshold=0.5):
    """
    Compute Adaptive-DPO loss with minority-instance weighting.

    Args:
        model_logps: log probabilities from current model [batch]
        reference_logps: log probabilities from reference model [batch]
        is_chosen: binary labels indicating preferred image [batch]
        instance_weights: reliability scores per sample [batch]
        beta: temperature parameter for preference optimization
        threshold: confidence threshold to distinguish majority/minority

    Returns:
        loss: scalar loss value
    """
    # Separate majority and minority samples
    is_majority = instance_weights >= threshold
    is_minority = instance_weights < threshold

    # Log probability ratios
    log_ratio = model_logps - reference_logps

    # Standard DPO objective
    # For chosen (preferred): maximize log_ratio
    # For not chosen: minimize log_ratio
    preferred_log_ratio = log_ratio * is_chosen + \
                         (log_ratio - 1.0) * (1 - is_chosen)

    # DPO loss: negative log of sigmoid of log probability difference
    dpo_loss = -torch.log(torch.sigmoid(beta * preferred_log_ratio) + 1e-8)

    # Adaptive weighting
    # Majority samples: use standard weight
    majority_weight = 1.0

    # Minority samples: reduce weight to mitigate negative impact
    minority_weight = instance_weights[is_minority]

    # Apply weights
    weighted_loss = dpo_loss.clone()
    weighted_loss[is_majority] *= majority_weight
    weighted_loss[is_minority] *= minority_weight

    # Compute final loss
    loss = weighted_loss.mean()

    # Optional: additional auxiliary term to enhance majority learning
    if is_majority.sum() > 0:
        majority_dpo_loss = dpo_loss[is_majority]
        # Higher weight on majority to strengthen agreement
        loss = 0.7 * loss + 0.3 * (2.0 * majority_dpo_loss.mean())

    return loss
```

### 5. Training Loop with Adaptive-DPO

Integrate the adaptive weighting into diffusion model fine-tuning:

```python
class AdaptiveDPOTrainer:
    """Train diffusion model with Adaptive-DPO for preference alignment."""

    def __init__(self, diffusion_model, reference_model, device='cuda'):
        self.model = diffusion_model.to(device)
        self.reference_model = reference_model.to(device)
        self.device = device

    def train_with_adaptive_dpo(self, preference_dataset, num_epochs=10,
                               batch_size=32, learning_rate=1e-4):
        """
        Train diffusion model with minority-aware preference optimization.

        Args:
            preference_dataset: PairwisePreferenceDataset with reliability scores
            num_epochs: training epochs
            batch_size: batch size
            learning_rate: optimizer learning rate
        """
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                     lr=learning_rate)

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(preference_dataset.batch_loader(
                                             batch_size=batch_size)):
                # Get batch components
                image_a = batch['image_a'].to(self.device)
                image_b = batch['image_b'].to(self.device)
                preferred = batch['preferred'].to(self.device)  # 0 or 1
                instance_reliability = batch['reliability'].to(self.device)

                # Compute model log probs
                # (simplified: log prob of denoising is inverse of loss)
                model_loss_a = self.model(image_a).loss
                model_loss_b = self.model(image_b).loss
                model_logps = -model_loss_a * (1 - preferred) + \
                             -model_loss_b * preferred

                # Compute reference log probs
                with torch.no_grad():
                    ref_loss_a = self.reference_model(image_a).loss
                    ref_loss_b = self.reference_model(image_b).loss
                    reference_logps = -ref_loss_a * (1 - preferred) + \
                                    -ref_loss_b * preferred

                # Compute is_chosen (which image was preferred)
                is_chosen = preferred.float()

                # Compute adaptive DPO loss
                loss = adaptive_dpo_loss(
                    model_logps, reference_logps, is_chosen,
                    instance_reliability,
                    beta=0.5,
                    threshold=0.5
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / (batch_idx + 1)
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}")
```

## Practical Guidance

### When to Use Adaptive-DPO

- Fine-tuning diffusion models with human preference data
- Preference datasets with known disagreement or subjectivity
- Want robust model alignment despite conflicting annotations
- Have multiple annotators rating same image pairs
- Need to balance quality improvement with safety/preference alignment

### When NOT to Use

- Preference data is nearly unanimous (little minority disagreement)
- Working with fully synthetic preference labels (no annotator variation)
- Insufficient compute for multiple reference models
- Need maximum speed in training (Adaptive-DPO adds overhead)

### Hyperparameters & Configuration

- **Beta (temperature)**: 0.5 typical; higher (0.7-1.0) for stronger alignment
- **Minority threshold**: 0.5 (50% of max reliability is minority cutoff)
- **Confidence weight**: 0.5 (equal to stability weight)
- **Stability weight**: 0.5 (equal to confidence weight)
- **Majority/minority loss ratio**: 0.7/0.3 (favor majority learning)
- **Minority weight scale**: 0.5-0.8 (down-weight uncertain samples)
- **Learning rate**: 1e-4 to 5e-5 (fine-tuning from pre-trained)
- **Batch size**: 16-32 (balance memory and gradient estimation)

### Common Pitfalls

- **Underestimating annotation disagreement**: Run inter-annotator agreement analysis first; if agreement > 95%, standard DPO may suffice
- **Threshold selection**: 0.5 reliability threshold is heuristic; visualize distribution and adjust
- **Reference model staleness**: Update reference model periodically (every N steps); don't freeze completely
- **Minority over-weighting**: Minority samples still matter; completely ignoring them (weight=0) hurts robustness
- **Insufficient annotators**: Inter-annotator stability requires 3+ annotators per pair; with fewer, confidence dominates
- **Mode collapse to majority**: Even with reweighting, model can overfit to majority preferences; use entropy regularization

## Reference

- Rafailov et al. 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model.
- Wallace et al. 2019. Analyzing Disagreement in Crowdsourced Annotations.
- Dhariwal & Nichol. 2021. Diffusion Models Beat GANs on Image Synthesis.
- Nichol et al. 2022. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models.
