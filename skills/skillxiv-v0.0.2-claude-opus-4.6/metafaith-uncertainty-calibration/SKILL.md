---
name: metafaith-uncertainty-calibration
title: "MetaFaith: Faithful Natural Language Uncertainty Expression in LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2505.24858"
keywords: [Uncertainty Calibration, Confidence Expression, LLM Trustworthiness, Natural Language]
description: "Train LLMs to faithfully express uncertainty through natural language that accurately reflects their actual confidence, improving trustworthiness and reducing overconfidence."
---

# Train Models to Faithfully Express Uncertainty Through Language

A critical failure mode of language models is assertive language masking uncertainty. When an LLM says "The capital of France is Rome" with absolute confidence, users believe the wrong answer. MetaFaith addresses this through faithful confidence calibration: training models to use linguistic uncertainty expressions ("maybe," "probably," "I'm not sure") that genuinely reflect their actual confidence, not just hedge for safety.

The key insight is that this isn't about safety-washing—it's about honest epistemic communication. A well-calibrated model using uncertain language when actually uncertain builds appropriate user trust, while an overconfident model erodes it.

## Core Concept

Faithful uncertainty expression requires three components:

- **Intrinsic uncertainty**: Model's actual confidence (via likelihood, logits, or ensemble variance)
- **Linguistic expression**: Natural language markers of uncertainty used in response
- **Calibration**: Mapping between intrinsic uncertainty and language choice
- **Feedback training**: Learn to use uncertainty language proportional to actual uncertainty
- **User trust**: Appropriate reliance based on genuine confidence levels

The challenge is that standard training incentivizes confident language regardless of actual uncertainty. MetaFaith explicitly trains models to align linguistic expressions with true uncertainty.

## Architecture Overview

- **Uncertainty estimation module**: Compute intrinsic confidence (logits, ensemble methods, dropout-MC)
- **Linguistic calibration dataset**: Examples pairing questions, answers, uncertainty levels, and language
- **Expression classifier**: Identify uncertainty markers in generated text
- **Calibration loss**: Penalize misalignment between uncertainty and linguistic expression
- **Confidence scoring**: Validate using human judgments or downstream task performance
- **Evaluation suite**: Test alignment across diverse topics and difficulty levels

## Implementation

Build a framework for training faithful uncertainty expression:

```python
# MetaFaith: Faithful Natural Language Uncertainty Calibration
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict
import numpy as np

class UncertaintyCalibrator:
    """
    Train LLMs to express uncertainty faithfully in natural language.
    """
    def __init__(self, model_name="gpt2-large"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Uncertainty language markers (varying confidence levels)
        self.uncertainty_markers = {
            'certain': ['certainly', 'definitely', 'clearly', 'obviously'],
            'likely': ['likely', 'probably', 'presumably', 'it seems'],
            'uncertain': ['maybe', 'might', 'could be', 'not sure', 'unclear'],
            'very_uncertain': ['I don\'t know', 'I\'m confused', 'no idea', 'hard to say']
        }

    def estimate_intrinsic_uncertainty(self, prompt: str, candidate_answers: List[str]) -> dict:
        """
        Measure model's actual uncertainty for a question.
        Multiple estimation methods for robustness.
        """
        with torch.no_grad():
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

            # Method 1: Logit-based uncertainty (entropy)
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]  # Last token logits
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

            # Method 2: Top-k probability concentration
            top_k_probs, _ = torch.topk(probs, k=5, dim=-1)
            concentration = top_k_probs.sum(dim=-1)  # Higher = more certain

            # Method 3: Agreement across answer candidates
            agreement_scores = []
            for answer in candidate_answers:
                answer_ids = self.tokenizer.encode(answer)
                answer_ids = torch.tensor(answer_ids).to(input_ids.device)

                # Compute probability assigned to this answer
                answer_logits = logits[0, answer_ids].mean()
                agreement_scores.append(answer_logits.item())

            agreement = np.std(agreement_scores)  # High variance = disagreement

        # Aggregate uncertainty estimates
        # Normalize to 0-1 range
        normalized_entropy = entropy.item() / np.log(self.tokenizer.vocab_size)
        normalized_concentration = concentration.item()
        normalized_agreement = min(agreement / 5.0, 1.0)

        # Combine: higher = more uncertain
        combined_uncertainty = (normalized_entropy + (1 - normalized_concentration) +
                               normalized_agreement) / 3.0

        return {
            'entropy': normalized_entropy,
            'concentration': normalized_concentration,
            'agreement_variance': normalized_agreement,
            'combined': combined_uncertainty
        }

    def select_uncertainty_expression(self, uncertainty: float) -> str:
        """
        Choose natural language uncertainty marker based on intrinsic uncertainty.
        uncertainty: 0.0 (certain) to 1.0 (very uncertain)
        """
        if uncertainty < 0.2:
            markers = self.uncertainty_markers['certain']
        elif uncertainty < 0.4:
            markers = self.uncertainty_markers['likely']
        elif uncertainty < 0.7:
            markers = self.uncertainty_markers['uncertain']
        else:
            markers = self.uncertainty_markers['very_uncertain']

        # Randomly select from appropriate tier
        return np.random.choice(markers)

    def generate_with_uncertainty(self, prompt: str, max_length: int = 100) -> dict:
        """
        Generate response with faithful uncertainty expression.
        """
        # Estimate uncertainty for this query
        uncertainty = self.estimate_intrinsic_uncertainty(prompt, [])['combined']

        # Select appropriate uncertainty expression
        expression = self.select_uncertainty_expression(uncertainty)

        # Generate answer (simplified)
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output_ids = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9
        )

        response = self.tokenizer.decode(output_ids[0])

        # Inject uncertainty expression if not already present
        if not any(marker in response.lower() for markers in self.uncertainty_markers.values()
                  for marker in markers):
            # Prepend uncertainty expression
            response = f"{expression}, {response.lower()}"

        return {
            'response': response,
            'uncertainty': uncertainty,
            'expression': expression
        }
```

Implement a training procedure that rewards aligned uncertainty expression:

```python
def train_faithful_uncertainty_expression(model, dataset: List[Dict], num_epochs=10):
    """
    Train model to express uncertainty that matches its actual confidence.

    Dataset format:
    [
        {
            'prompt': "What is the capital of France?",
            'answer': "Paris",
            'uncertainty_label': 0.1,  # Human-judged uncertainty
            'correct_uncertainty_language': ['certainly', 'definitely']
        },
        ...
    ]
    """
    calibrator = UncertaintyCalibrator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)

    for epoch in range(num_epochs):
        total_loss = 0

        for example in dataset:
            prompt = example['prompt']
            answer = example['answer']
            true_uncertainty = example['uncertainty_label']
            correct_expressions = example['correct_uncertainty_language']

            # Step 1: Estimate intrinsic uncertainty
            estimated_uncertainty = calibrator.estimate_intrinsic_uncertainty(
                prompt, [answer]
            )['combined']

            # Step 2: Generate response with uncertainty
            input_ids = calibrator.tokenizer.encode(prompt, return_tensors='pt')
            outputs = model(input_ids, output_hidden_states=True)
            logits = outputs.logits

            # Step 3: Compute calibration loss
            # Loss component 1: Uncertainty estimation accuracy
            # Does estimated uncertainty match human judgment?
            uncertainty_mse_loss = (estimated_uncertainty - true_uncertainty) ** 2

            # Step 4: Language expression alignment loss
            # Does the model use appropriate uncertainty language?
            # Extract uncertainty markers from generated text
            response_ids = model.generate(input_ids, max_length=50)
            response_text = calibrator.tokenizer.decode(response_ids[0])

            # Check which uncertainty expressions appear
            expression_scores = {}
            for category, markers in calibrator.uncertainty_markers.items():
                for marker in markers:
                    if marker in response_text.lower():
                        expression_scores[category] = 1.0

            # Loss: wrong category gets penalized
            appropriate_category = map_uncertainty_to_category(true_uncertainty)
            expression_loss = 0.0
            for category, score in expression_scores.items():
                if category == appropriate_category:
                    # Reward using correct expression
                    expression_loss -= score * 0.1
                else:
                    # Penalize using wrong expression
                    expression_loss += score * 0.1

            # Combined loss
            total_loss_item = uncertainty_mse_loss + expression_loss * 0.5

            optimizer.zero_grad()
            total_loss_item.backward()
            optimizer.step()

            total_loss += total_loss_item.item()

        print(f"Epoch {epoch}: Loss = {total_loss / len(dataset):.4f}")

    return calibrator

def map_uncertainty_to_category(uncertainty: float) -> str:
    """Map continuous uncertainty to category"""
    if uncertainty < 0.2:
        return 'certain'
    elif uncertainty < 0.4:
        return 'likely'
    elif uncertainty < 0.7:
        return 'uncertain'
    else:
        return 'very_uncertain'
```

Implement evaluation of calibration quality:

```python
def evaluate_calibration(model, test_set: List[Dict]) -> Dict[str, float]:
    """
    Measure how well model's uncertainty expression matches actual confidence.
    """
    calibrator = UncertaintyCalibrator()

    predicted_uncertainties = []
    actual_uncertainties = []
    language_alignment_scores = []

    for example in test_set:
        # Get model's uncertainty estimate
        estimated_unc = calibrator.estimate_intrinsic_uncertainty(
            example['prompt'], [example['answer']]
        )['combined']

        # Get ground truth uncertainty (human judgment)
        true_unc = example['uncertainty_label']

        # Check if generated language matches uncertainty level
        response = calibrator.generate_with_uncertainty(example['prompt'])
        expression = response['expression']

        # Score: does expression align with actual uncertainty?
        appropriate_expression = check_expression_appropriateness(
            expression, true_unc, calibrator.uncertainty_markers
        )
        language_alignment_scores.append(appropriate_expression)

        predicted_uncertainties.append(estimated_unc)
        actual_uncertainties.append(true_unc)

    # Metrics
    predicted_uncertainties = np.array(predicted_uncertainties)
    actual_uncertainties = np.array(actual_uncertainties)
    language_alignment_scores = np.array(language_alignment_scores)

    # ECE: Expected Calibration Error
    # Bin predictions and check accuracy of confidence
    ece = compute_expected_calibration_error(predicted_uncertainties,
                                            actual_uncertainties)

    # MCE: Maximum Calibration Error (worst bin)
    mce = compute_max_calibration_error(predicted_uncertainties,
                                       actual_uncertainties)

    # Language alignment: what fraction expressed uncertainty appropriately?
    language_accuracy = language_alignment_scores.mean()

    return {
        'ece': ece,
        'mce': mce,
        'language_alignment': language_accuracy,
        'spearman_correlation': compute_spearman(predicted_uncertainties, actual_uncertainties)
    }

def check_expression_appropriateness(expression: str, uncertainty: float,
                                    markers_dict: Dict) -> float:
    """Score how appropriate an expression is for given uncertainty level"""
    category = map_uncertainty_to_category(uncertainty)
    correct_markers = markers_dict[category]

    if expression in correct_markers:
        return 1.0
    # Partial credit for adjacent categories
    elif uncertainty < 0.4 and expression in markers_dict['likely']:
        return 0.5
    elif uncertainty > 0.5 and expression in markers_dict['uncertain']:
        return 0.5
    else:
        return 0.0

def compute_expected_calibration_error(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Compute ECE: average calibration error across confidence bins"""
    num_bins = 10
    bin_edges = np.linspace(0, 1, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        mask = (predicted >= bin_edges[i]) & (predicted < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_confidence = predicted[mask].mean()
            bin_accuracy = (predicted[mask] == actual[mask]).mean()
            ece += np.abs(bin_confidence - bin_accuracy) * mask.sum() / len(predicted)

    return ece
```

## Practical Guidance

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Uncertainty weight | 0.3 - 0.7 | How much to penalize miscalibration |
| Expression loss weight | 0.1 - 0.5 | Language alignment vs. performance |
| Confidence bins | 5 - 10 | For evaluation and analysis |
| Entropy threshold | 0.3 - 0.7 | When to switch uncertainty tiers |
| Human annotation budget | 1000 - 5000 examples | Need calibrated ground truth labels |

**When to use MetaFaith:**
- Deploying LLMs where user trust matters
- Need reliable uncertainty signals for downstream systems
- Reducing overconfidence (model saying wrong things with certainty)
- Building human-AI collaborative systems
- Safety-critical applications needing honest confidence

**When NOT to use:**
- Use-case doesn't require confidence calibration
- Budget for creating uncertainty-labeled datasets unavailable
- Model performance (accuracy) is only metric that matters
- Users never see model uncertainty (just final answers)
- Uncertainty already well-calibrated in base model

**Common pitfalls:**
- Not separating intrinsic uncertainty from linguistic expression
- Training on weak human uncertainty judgments (need high-quality labels)
- Expression list too limited (needs diverse, natural language)
- Punishing calibration too heavily (hurts overall accuracy)
- Not evaluating on held-out test set (can overfit to training calibration)
- Assuming human judgment of uncertainty is ground truth (it's subjective)
- Not measuring language alignment separately from numerical calibration

## Reference

**MetaFaith: Faithful Natural Language Uncertainty Expression in LLMs**
https://arxiv.org/abs/2505.24858
