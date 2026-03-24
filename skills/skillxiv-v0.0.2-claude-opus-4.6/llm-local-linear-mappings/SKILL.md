---
name: llm-local-linear-mappings
title: "Large Language Models are Locally Linear Mappings"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2505.24293"
keywords: [Interpretability, LLM Mechanisms, Linear Systems, Transformer Analysis]
description: "Interpret LLM behavior as locally linear mappings between hidden representations, enabling mechanistic understanding of computation without examining individual weights or attention patterns."
---

# Interpret LLMs as Locally Linear Computational Systems

Despite advances in transformer interpretability, understanding how LLMs actually compute remains elusive. This work shows that LLM inference for a given input sequence can be accurately reconstructed as a linear system mapping from input embeddings to output logits. This linearity holds locally (for similar inputs) and provides an interpretable, mechanistic view of LLM computation.

The key insight is that the complex, nonlinear transformer can be well-approximated as a linear map in the region around a specific input. By studying these linear approximations, you can understand what information flows through the model and how it's combined to produce outputs—without analyzing millions of weight matrices or attention patterns individually.

## Core Concept

The locally linear model represents LLM computation as:

- **Input embedding space**: Tokenized text encoded as high-dimensional vectors
- **Linear transformation matrix**: Maps input embeddings to output logits
- **Local linearity**: Approximation holds for inputs near the original
- **Interpretable weights**: Weight values directly show information flow
- **Reconstruction error < 10%**: Linear model very accurately reproduces predictions
- **Mechanistic insight**: Understanding this linear map reveals how model processes information

This is profoundly different from weight analysis: instead of examining billions of parameters, you study a single linear map that captures the computation for your input.

## Architecture Overview

- **Input tokenization and embedding**: Standard LLM input processing
- **Sequence representation**: Tracking how information evolves through layers
- **Linear reconstruction module**: Fitting linear model to map embeddings → logits
- **Error analysis**: Measuring where linearity breaks down
- **Contribution analysis**: Which input elements matter most
- **Generalization bounds**: Understanding where linear approximation fails
- **Visualization and probing**: Making the linear structure interpretable

## Implementation

Build a framework for analyzing LLMs as locally linear systems:

```python
# Locally Linear LLM Analysis
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LinearRegression
import numpy as np

class LocallyLinearLLMAnalyzer:
    """
    Analyze LLM computation as locally linear mappings.
    """
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()

    def extract_hidden_states(self, prompt: str):
        """
        Extract hidden states at each layer for analysis.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_hidden_states=True,
                return_dict=True
            )

        hidden_states = outputs.hidden_states  # Tuple of (num_layers + 1, batch, seq, hidden_dim)
        logits = outputs.logits

        return hidden_states, logits, input_ids

    def fit_linear_model(self, prompt: str, candidate_outputs: torch.Tensor = None):
        """
        Fit a linear model from input embeddings to output logits.
        """
        hidden_states, logits, input_ids = self.extract_hidden_states(prompt)

        # Use input embeddings as features
        input_embeddings = hidden_states[0]  # (batch, seq, hidden_dim)
        batch_size, seq_len, hidden_dim = input_embeddings.shape

        # Flatten embeddings
        X = input_embeddings.reshape(batch_size * seq_len, hidden_dim).numpy()

        # Flatten logits for regression
        # Use last token's logits as target
        y = logits[:, -1, :].numpy()  # (batch, vocab_size)

        # Fit linear regression
        # X: (batch*seq, hidden_dim) -> y: (batch, vocab_size)
        linear_model = LinearRegression()

        # Regress each output dimension separately
        linear_models = []
        predictions = []

        for vocab_idx in range(y.shape[-1]):
            target = y[:, vocab_idx]

            # For single batch, we need another approach
            # Use dropout-based stochastic forward pass
            if batch_size == 1:
                # Generate multiple samples by using different random seeds
                X_samples = []
                y_samples = []

                for _ in range(10):  # 10 samples per token
                    self.model.train()  # Enable dropout
                    with torch.no_grad():
                        sample_hidden = self.extract_hidden_states(prompt)[0]
                    sample_embeddings = sample_hidden[0, -1, :].numpy()  # Last token
                    X_samples.append(sample_embeddings)

                    sample_logits = self.extract_hidden_states(prompt)[1][0, -1, vocab_idx].numpy()
                    y_samples.append(sample_logits)

                X_regression = np.array(X_samples)
                y_regression = np.array(y_samples)
            else:
                X_regression = X
                y_regression = target

            # Fit linear model for this output dimension
            model = LinearRegression()
            model.fit(X_regression, y_regression)
            linear_models.append(model)

            # Predict
            pred = model.predict(X_regression)
            predictions.append(pred)

        predictions = np.array(predictions).T

        return linear_models, X, y

    def measure_linearity_error(self, prompt: str):
        """
        Measure how well linear model approximates actual LLM outputs.
        """
        hidden_states, logits, input_ids = self.extract_hidden_states(prompt)

        # Get actual outputs
        actual_logits = logits[:, -1, :].detach().cpu().numpy()

        # Fit linear model
        linear_models, X, _ = self.fit_linear_model(prompt)

        # Get predictions from linear model
        input_embeddings = hidden_states[0][:, -1, :].numpy()
        predicted_logits = np.array([
            model.predict(input_embeddings.reshape(1, -1))[0]
            for model in linear_models
        ])

        # Compute reconstruction error
        mse = np.mean((actual_logits - predicted_logits) ** 2)
        relative_error = mse / (np.mean(actual_logits ** 2) + 1e-8)

        return {
            'mse': mse,
            'relative_error': relative_error,
            'predicted_logits': predicted_logits,
            'actual_logits': actual_logits
        }

    def analyze_information_flow(self, prompt: str):
        """
        Analyze how information flows from input to output via linear model.
        """
        linear_models, X, _ = self.fit_linear_model(prompt)

        # Extract weight matrix from linear model
        # Shows which input dimensions matter for each output
        weights = np.array([model.coef_ for model in linear_models])  # (vocab_size, hidden_dim)

        # Identify important input dimensions
        importance = np.abs(weights).max(axis=0)  # (hidden_dim,)
        important_indices = np.argsort(importance)[-10:]  # Top 10 dimensions

        return {
            'weights': weights,
            'input_importance': importance,
            'important_dimensions': important_indices,
            'top_importance': importance[important_indices]
        }

    def compare_linear_approximation(self, prompts: list):
        """
        Compare how well linear model generalizes across related prompts.
        """
        results = []

        # Fit on first prompt
        linear_models, X_train, y_train = self.fit_linear_model(prompts[0])

        # Test on other prompts
        for prompt in prompts[1:]:
            hidden_states, logits, _ = self.extract_hidden_states(prompt)
            input_embeddings = hidden_states[0][:, -1, :].numpy()
            actual_logits = logits[:, -1, :].detach().cpu().numpy()

            # Predict using models trained on first prompt
            predictions = np.array([
                model.predict(input_embeddings.reshape(1, -1))[0]
                for model in linear_models
            ])

            error = np.mean((actual_logits - predictions) ** 2)
            results.append({'prompt': prompt, 'generalization_error': error})

        return results
```

Implement analysis visualization and interpretation tools:

```python
def interpret_linear_computation(analyzer: LocallyLinearLLMAnalyzer, prompt: str):
    """
    Interpret what the linear model reveals about LLM computation.
    """
    # Measure linearity
    linearity = analyzer.measure_linearity_error(prompt)
    print(f"Linearity Error (relative): {linearity['relative_error']:.2%}")
    print(f"Model reconstructs output with {100 * (1 - linearity['relative_error']):.1f}% accuracy")

    # Analyze information flow
    flow = analyzer.analyze_information_flow(prompt)
    print(f"\nTop 5 important input dimensions: {flow['important_dimensions'][:5]}")

    # Generalization analysis
    related_prompts = generate_related_prompts(prompt, num_variations=5)
    generalization = analyzer.compare_linear_approximation([prompt] + related_prompts)

    avg_gen_error = np.mean([r['generalization_error'] for r in generalization[1:]])
    print(f"\nGeneralization error on similar prompts: {avg_gen_error:.4f}")

    return {
        'linearity': linearity,
        'information_flow': flow,
        'generalization': generalization
    }

def generate_related_prompts(prompt: str, num_variations: int) -> list:
    """
    Generate similar prompts to test generalization of linear model.
    """
    # Simple variations: paraphrase, context changes, etc.
    variations = [
        prompt,
        prompt.replace("?", "."),  # Remove question mark
        "The following question: " + prompt,
        prompt + " Can you answer this?",
    ]
    return variations[:num_variations]

def measure_linearity_bounds(analyzer, test_prompts: list) -> dict:
    """
    Determine bounds on where linearity assumptions hold.
    """
    errors = []
    generalization_distances = []

    for prompt in test_prompts:
        error = analyzer.measure_linearity_error(prompt)
        errors.append(error['relative_error'])

    # Analyze: how far can we go from training point before linearity fails?
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    print(f"Linearity holds with:")
    print(f"  Mean error: {mean_error:.2%}")
    print(f"  Std dev: {std_error:.2%}")
    print(f"  95% of errors below: {np.percentile(errors, 95):.2%}")

    return {
        'mean_error': mean_error,
        'std_error': std_error,
        'p95_error': np.percentile(errors, 95),
        'distribution': errors
    }

def explain_prediction_via_linearity(analyzer, prompt, target_token):
    """
    Explain a specific prediction using the linear model.
    """
    hidden_states, logits, input_ids = analyzer.extract_hidden_states(prompt)
    linear_models, X, _ = analyzer.fit_linear_model(prompt)

    # Get which tokens contributed most to target_token prediction
    weights = linear_models[target_token].coef_  # (hidden_dim,)

    # Show which input dimensions mattered
    top_indices = np.argsort(np.abs(weights))[-5:]

    explanation = f"""
Prediction for token '{target_token}' is explained by:
- Input dimensions: {top_indices}
- Contribution magnitudes: {weights[top_indices]}

This linear model suggests:
- These specific input embeddings matter most
- The computation is primarily a weighted sum of these elements
- Non-linearities are minimal for this region
"""
    return explanation
```

## Practical Guidance

| Aspect | Recommendation | Notes |
|--------|-----------------|-------|
| Number of samples for fitting | 10 - 100 | More samples = better fit, but more compute |
| Dimensionality of hidden states | Full | Don't reduce; linearity relies on full expressivity |
| Distance for local validity | Small ε | Linearity breaks down as you move far from base point |
| Relative error threshold | <10% | Linearity approximation valid if error is small |
| Generalization distance | Same model family | Linear fit may not transfer to different models |

**When to use locally linear analysis:**
- Need mechanistic understanding of LLM computation
- Want interpretable explanation of specific predictions
- Studying how information flows through models
- Analyzing failure modes or adversarial examples
- Building approximations or distillations

**When NOT to use:**
- Only care about final accuracy (interpretability not needed)
- Models exhibit strong non-linear behavior (check linearity error first)
- Computational budget for fitting linear models is tight
- Need global understanding (linearity is local, not global)
- Studying learned representations independent of computation

**Common pitfalls:**
- Assuming linearity holds globally (only true locally)
- Fitting on too few samples (unstable linear models)
- Not measuring reconstruction error (can't validate linearity)
- Using raw logits instead of normalizing (scale sensitivity)
- Generalizing insights beyond neighborhood of fitting point
- Confusing linear approximation with actual mechanism (still approximate)
- Not comparing to baselines (what's the accuracy gain from interpretability?)

## Reference

**Large Language Models are Locally Linear Mappings**
https://arxiv.org/abs/2505.24293
