---
name: config-knowledge-distillation
title: "ConfiG: Confidence-Guided Data Augmentation for Knowledge Distillation Under Covariate Shift"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.02294"
keywords: [knowledge-distillation, data-augmentation, covariate-shift, robustness]
description: "Improve student model robustness under covariate shift by using diffusion-based augmentation that targets spurious features via teacher-student disagreement."
---

# ConfiG: Confidence-Guided Data Augmentation for Knowledge Distillation

## Core Concept

When training data contains spurious features absent at test time, knowledge distillation can preserve these biases in student models. ConfiG addresses this by generating augmented images that maximize disagreement between teacher and student, targeting exactly the spurious correlations the student learned. This diffusion-based approach enables students to overcome dataset biases while maintaining knowledge transfer.

## Architecture Overview

- **Problem**: Knowledge distillation transfers spurious correlations from biased datasets, degrading generalization to unseen groups
- **Solution**: Confidence-guided diffusion generates adversarial examples leveraging teacher-student disagreement
- **Mechanism**: Optimize latent variables to maximize teacher confidence while minimizing student confidence, targeting spurious features
- **Theoretical Grounding**: Proposition 1 proves confidence-guided augmentation reduces distributional generalization gap
- **Synergy**: Works with model-centric bias mitigation (TAB, etc.), showing data and model approaches are complementary

## Implementation

### Step 1: Understand Covariate Shift and Generalization Decomposition

```python
import torch
import numpy as np
from typing import Tuple, List

class CovariateShiftAnalyzer:
    """Analyze how spurious features affect generalization"""

    def decompose_error(self, teacher_model, student_model,
                       train_data, test_data,
                       group_labels_test) -> Dict:
        """
        Decompose generalization error into two components:
        1. Teacher quality: how well teacher generalizes
        2. Distributional gap: how much student differs from teacher distribution

        Key insight: ConfiG targets reducing the gap, not improving teacher quality.
        """

        # Evaluate on training distribution (in-distribution)
        train_acc = self.evaluate_accuracy(student_model, train_data)

        # Evaluate on test distribution
        test_acc = self.evaluate_accuracy(student_model, test_data)

        # Evaluate per test group
        group_accs = {}
        for group_id in np.unique(group_labels_test):
            group_mask = group_labels_test == group_id
            group_test = test_data[group_mask]
            group_accs[group_id] = self.evaluate_accuracy(student_model, group_test)

        # Decompose error
        overall_gap = train_acc - test_acc
        group_gaps = {g: train_acc - acc for g, acc in group_accs.items()}

        print("=== Generalization Analysis ===")
        print(f"Overall train accuracy: {train_acc:.1%}")
        print(f"Overall test accuracy: {test_acc:.1%}")
        print(f"Overall gap: {overall_gap:.1%}")
        print("\nPer-group performance:")

        for group_id, acc in group_accs.items():
            print(f"  Group {group_id}: {acc:.1%} (gap: {group_gaps[group_id]:.1%})")

        return {
            'overall_gap': overall_gap,
            'group_gaps': group_gaps,
            'group_accs': group_accs,
        }

    def identify_spurious_correlations(self, train_data,
                                       train_labels,
                                       group_labels) -> List[Tuple]:
        """
        Identify features that are predictive in training but spurious.
        Example: Blond hair → Female in CelebA, but this breaks in real data.
        """

        spurious_correlations = []

        # Analyze correlations between features and groups
        unique_groups = np.unique(group_labels)

        for group_id in unique_groups:
            group_mask = group_labels == group_id

            # Extract feature statistics for this group
            group_data = train_data[group_mask]
            other_data = train_data[~group_mask]

            # Compute feature divergence between groups
            # Features with high divergence are likely spurious
            feature_divergence = self._compute_feature_divergence(
                group_data, other_data
            )

            # Identify top divergent features
            top_divergent = sorted(
                enumerate(feature_divergence),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            for feature_idx, divergence in top_divergent:
                spurious_correlations.append({
                    'group': group_id,
                    'feature': feature_idx,
                    'divergence': divergence,
                })

        return spurious_correlations

    def _compute_feature_divergence(self, group_data, other_data) -> np.ndarray:
        """Measure KL divergence of feature distributions"""

        # Simplified: compute histogram divergence per feature
        divergences = []

        for feature_idx in range(group_data.shape[1]):
            # Histogram of feature in group vs out
            hist_group = np.histogram(group_data[:, feature_idx], bins=20)[0]
            hist_other = np.histogram(other_data[:, feature_idx], bins=20)[0]

            # Normalize
            hist_group = hist_group / (np.sum(hist_group) + 1e-8)
            hist_other = hist_other / (np.sum(hist_other) + 1e-8)

            # KL divergence
            kl = np.sum(hist_group * np.log((hist_group + 1e-8) / (hist_other + 1e-8)))
            divergences.append(kl)

        return np.array(divergences)
```

### Step 2: Implement Confidence-Guided Augmentation

```python
import torch.nn.functional as F

class ConfidenceGuidedDiffusion:
    """Generate augmented images targeting spurious features"""

    def __init__(self, diffusion_model, teacher_model, student_model):
        self.diffusion = diffusion_model  # Pretrained diffusion (e.g., Stable Diffusion)
        self.teacher = teacher_model
        self.student = student_model

    def generate_augmented_sample(self, image: torch.Tensor,
                                 label: int,
                                 gamma: float = 2.0,
                                 num_iterations: int = 100) -> torch.Tensor:
        """
        Generate augmented image by optimizing latent vector z.

        Objective: maximize loss(z) = t(z)^γ + (1-f(z))^γ
        where:
          t(z) = teacher confidence on augmented sample
          f(z) = student confidence on augmented sample
          γ = 2.0 (empirically optimal)

        This targets spurious features the student learned.
        """

        # Initialize random latent in diffusion space
        z = torch.randn(1, 4, image.shape[1]//8, image.shape[2]//8)
        z.requires_grad = True

        # Optimizer for latent variables
        optimizer = torch.optim.Adam([z], lr=0.01)

        for iteration in range(num_iterations):
            # Decode latent to image
            augmented_image = self.diffusion.decode(z)

            # Get predictions
            with torch.no_grad():
                teacher_logits = self.teacher(augmented_image)
                student_logits = self.student(augmented_image)

            teacher_probs = F.softmax(teacher_logits, dim=-1)
            student_probs = F.softmax(student_logits, dim=-1)

            # Extract confidence for true label
            teacher_conf = teacher_probs[0, label]  # High = good, preserve label
            student_conf = student_probs[0, label]  # Low = good, challenge student

            # Confidence-guided loss
            loss = (teacher_conf ** gamma) + ((1.0 - student_conf) ** gamma)

            # Backward step
            optimizer.zero_grad()
            (-loss).backward()  # Maximize by minimizing negative
            optimizer.step()

            if (iteration + 1) % 20 == 0:
                print(f"Iter {iteration + 1}: "
                      f"teacher_conf={teacher_conf:.3f}, "
                      f"student_conf={student_conf:.3f}, "
                      f"loss={loss:.3f}")

        # Decode final latent
        with torch.no_grad():
            augmented = self.diffusion.decode(z)

        return augmented

    def generate_augmented_dataset(self, train_images: torch.Tensor,
                                  train_labels: torch.Tensor,
                                  augmentation_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate augmented samples for a subset of training data.
        Focus on misclassified or uncertain samples.
        """

        # Identify samples where student is uncertain
        with torch.no_grad():
            student_logits = self.student(train_images)
            student_probs = F.softmax(student_logits, dim=-1)
            student_confidence = torch.max(student_probs, dim=-1)[0]

        # Select samples with low confidence (more need augmentation)
        num_to_augment = int(len(train_images) * augmentation_ratio)
        uncertain_indices = torch.argsort(student_confidence)[:num_to_augment]

        # Generate augmentations
        augmented_images = []
        augmented_labels = []

        for idx in uncertain_indices:
            image = train_images[idx].unsqueeze(0)
            label = train_labels[idx].item()

            print(f"Generating augmentation {len(augmented_images) + 1}/{num_to_augment}")

            augmented = self.generate_augmented_sample(image, label)

            augmented_images.append(augmented)
            augmented_labels.append(label)

        # Concatenate original and augmented
        all_images = torch.cat([train_images] + augmented_images, dim=0)
        all_labels = torch.cat([
            train_labels,
            torch.tensor(augmented_labels, device=train_labels.device)
        ], dim=0)

        return all_images, all_labels
```

### Step 3: Implement Knowledge Distillation with ConfiG

```python
class KDWithConfiG:
    """Knowledge distillation enhanced with confidence-guided augmentation"""

    def __init__(self, teacher_model, student_model,
                 diffusion_model, temperature: float = 4.0):
        self.teacher = teacher_model
        self.student = student_model
        self.diffusion = diffusion_model
        self.temperature = temperature

        self.confidence_aug = ConfidenceGuidedDiffusion(
            diffusion_model, teacher_model, student_model
        )

        self.optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

    def distillation_loss(self, student_logits: torch.Tensor,
                         teacher_logits: torch.Tensor) -> torch.Tensor:
        """KL divergence between student and teacher distributions"""

        student_probs = F.softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        kl_loss = F.kl_div(
            torch.log(student_probs + 1e-8),
            teacher_probs.detach(),
            reduction='batchmean'
        )

        return kl_loss * (self.temperature ** 2)

    def training_step(self, batch_images: torch.Tensor,
                     batch_labels: torch.Tensor) -> float:
        """Single training step on original + augmented data"""

        # Forward pass
        student_logits = self.student(batch_images)

        with torch.no_grad():
            teacher_logits = self.teacher(batch_images)

        # Compute distillation loss
        loss = self.distillation_loss(student_logits, teacher_logits)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_with_augmentation(self, train_images: torch.Tensor,
                               train_labels: torch.Tensor,
                               num_epochs: int = 10) -> Dict:
        """Train student with periodic augmentation"""

        history = {'loss': []}

        for epoch in range(num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

            # Original training step
            loss = self.training_step(train_images, train_labels)
            history['loss'].append(loss)

            print(f"Original data loss: {loss:.4f}")

            # Every N epochs: generate augmentations and finetune
            if (epoch + 1) % 3 == 0:
                print("Generating confidence-guided augmentations...")

                aug_images, aug_labels = self.confidence_aug.generate_augmented_dataset(
                    train_images, train_labels, augmentation_ratio=0.3
                )

                print(f"Augmented dataset size: {len(aug_images)}")

                # Fine-tune on augmented data
                for aug_epoch in range(2):
                    aug_loss = self.training_step(aug_images, aug_labels)
                    print(f"  Augmented epoch {aug_epoch + 1}: loss={aug_loss:.4f}")

        return history
```

### Step 4: Integration with Model-Centric Bias Mitigation

```python
class HybridBiasMitigation:
    """Combine data-centric (ConfiG) and model-centric (TAB) approaches"""

    def __init__(self, teacher_model, student_model, diffusion_model):
        self.kd_config = KDWithConfiG(
            teacher_model, student_model, diffusion_model
        )

        # Model-centric: Train-aware batch normalization (TAB)
        self.model_centric = TrainAwareBatchNorm(student_model)

    def train_hybrid(self, train_images: torch.Tensor,
                    train_labels: torch.Tensor,
                    group_labels: torch.Tensor) -> Dict:
        """
        Data-centric: ConfiG augmentation targeting spurious features
        Model-centric: TAB encouraging invariant features
        """

        print("Starting hybrid bias mitigation training...")
        print("Data-centric: Confidence-guided augmentation")
        print("Model-centric: Train-aware batch normalization")

        # Phase 1: Data-centric augmentation
        kd_history = self.kd_config.train_with_augmentation(
            train_images, train_labels, num_epochs=10
        )

        # Phase 2: Model-centric refinement with TAB
        tab_history = self.model_centric.train_with_tab(
            train_images, train_labels, group_labels, num_epochs=5
        )

        return {
            'kd_history': kd_history,
            'tab_history': tab_history,
        }
```

## Practical Guidance

1. **Identify Spurious Correlations**: Start by analyzing your training data for features that correlate with labels but are likely absent in test data (e.g., background, lighting, specific objects).

2. **Teacher Quality Matters**: ConfiG assumes teacher model is robust. If teacher is biased, augmentation won't help. Use a well-trained teacher or synthetic data.

3. **Gamma Parameter**: gamma=2.0 is empirically optimal. Higher γ concentrates learning on high-disagreement samples; lower γ spreads it more broadly.

4. **Augmentation Ratio**: Augment 30-50% of training data, focusing on uncertain samples. Over-augmentation can degrade in-distribution performance.

5. **Hybrid Approach Works Best**: Combine data-centric (ConfiG) augmentation with model-centric approaches (TAB, group normalization). They're complementary and achieve better results together.

6. **Computational Cost**: Diffusion-based augmentation is expensive (100+ iterations per image). Pre-generate augmented dataset in batch, don't do online.

## Reference

- Paper: ConfiG (2506.02294)
- Key Innovation: Confidence-guided diffusion targeting spurious features
- Architecture: Teacher-student disagreement → augmentation objective
- Datasets: CelebA, SpuCo Birds, Spurious ImageNet
- Result: Superior performance under covariate shift compared to prior augmentation methods
