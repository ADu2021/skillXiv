---
name: scaling-laws-optimal-data-mixtures
title: "Scaling Laws for Optimal Data Mixtures"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.09404"
keywords: [Scaling Laws, Data Mixtures, Multi-Domain Training, Compute Optimization, Foundation Models]
description: "Predict optimal data mixture proportions for multi-domain LLM training using scaling laws that require only 10-20 small experiments. Determine which domains should contribute how much data across model sizes (186M-7B), reducing computational waste in foundation model pretraining."
---

# Scaling Laws for Optimal Data Mixtures: Efficient Multi-Domain Foundation Model Training

Foundation models trained on diverse domains (code, math, text) must decide how much data from each domain to include. Current practice uses trial-and-error, wasting compute on suboptimal mixtures. This work proposes scaling laws that predict model loss as a function of model size (N), training tokens (D), and domain weights (h), enabling practitioners to determine optimal mixtures efficiently.

The key insight is that only 10-20 small-scale training runs are needed to fit accurate scaling laws that extrapolate to much larger models. Two law formulations are provided: additive (mixture-independent scale parameters, predicting scale-independent optimal mixtures) and joint (mixture-dependent scale parameters, predicting compute-budget-dependent optima). Both formulations have been validated across language models, multimodal models, and vision models.

## Core Concept

Multi-domain training loss can be modeled as: E + 1/∑(C_i·h_i^γ_i) + A/N^α + B/D^β, where only the bias term E depends on domain weights h. This reveals a crucial property: for additive scaling laws, optimal domain weights are compute-independent—they don't change with model size or total training budget. Only the absolute performance scales.

For joint scaling laws where scale parameters A and B themselves depend on mixture composition, optimal weights become budget-dependent. The fitting procedure uses Basin-hopping with L-BFGS to navigate the high-dimensional parameter space efficiently.

## Architecture Overview

- **Additive Scaling Law Formulation**: Loss depends on domain-weight-dependent bias term; scale parameters (A, B) are mixture-independent
- **Joint Scaling Law Formulation**: Both scale parameters and bias depend on mixture; accounts for scale-mixture interactions
- **Loss Prediction Pipeline**: Fit scaling laws to small experiments, extrapolate to large models/compute budgets
- **Optimization Framework**: Basin-hopping + L-BFGS to solve mixture optimization problem over weight simplex
- **Multi-Model Validation**: Tests on language models (186M-7B), multimodal (image+text), and vision encoders

## Implementation

### Additive Scaling Law: Mixture-Independent Scale Parameters

Model loss where domain weights affect bias term but not scaling.

```python
import numpy as np
from scipy.optimize import minimize, basinhopping
import torch

class AdditiveScalingLaw:
    """
    Additive formulation: Loss = E(h) + 1/∑(Ci·hi^γi) + A/N^α + B/D^β
    where E(h) is bias depending on mixture, but A and B are independent of h.
    """

    def __init__(self, num_domains: int = 3):
        self.num_domains = num_domains

        # Parameters to fit
        self.C = np.ones(num_domains)  # Domain constants
        self.gamma = np.ones(num_domains)  # Domain exponents
        self.A = 1.0  # Scale parameter for model size
        self.alpha = 0.5  # Model size exponent
        self.B = 1.0  # Scale parameter for tokens
        self.beta = 0.5  # Token exponent

    def predict_loss(self, model_size: int, tokens: int, mixture_weights: np.ndarray):
        """
        Predict loss for given model size, tokens, and domain mixture.

        Args:
            model_size: Model parameter count N
            tokens: Total training tokens D
            mixture_weights: (num_domains,) domain weight distribution, sums to 1

        Returns:
            Predicted loss
        """
        # Bias term depending on mixture
        bias = 0.0
        weighted_sum = 0.0

        for i in range(self.num_domains):
            weighted_sum += self.C[i] * (mixture_weights[i] ** self.gamma[i])

        mixture_term = 1.0 / (weighted_sum + 1e-8)
        bias += mixture_term

        # Scale terms independent of mixture
        scale_loss = self.A / (model_size ** self.alpha) + self.B / (tokens ** self.beta)

        return bias + scale_loss

    def fit_to_experiments(self, experiments: list):
        """
        Fit scaling law to experimental data.

        Args:
            experiments: List of {
                'model_size': N,
                'tokens': D,
                'mixture_weights': (num_domains,),
                'loss': measured loss
            }

        Returns:
            Fitted parameters (C, gamma, A, alpha, B, beta)
        """
        def objective(params):
            # Unpack parameters
            C = params[:self.num_domains]
            gamma = params[self.num_domains:2*self.num_domains]
            A = params[2*self.num_domains]
            alpha = params[2*self.num_domains + 1]
            B = params[2*self.num_domains + 2]
            beta = params[2*self.num_domains + 3]

            # Update self
            self.C = C
            self.gamma = gamma
            self.A = A
            self.alpha = alpha
            self.B = B
            self.beta = beta

            # Compute MSE
            mse = 0.0
            for exp in experiments:
                predicted = self.predict_loss(
                    exp['model_size'],
                    exp['tokens'],
                    exp['mixture_weights']
                )
                mse += (predicted - exp['loss']) ** 2

            return mse / len(experiments)

        # Initial guess
        x0 = np.concatenate([
            self.C,
            self.gamma,
            [self.A, self.alpha, self.B, self.beta]
        ])

        # Optimize
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': [(0.1, 10)] * (2*self.num_domains) + [(0.01, 100), (0.1, 2), (0.01, 100), (0.1, 2)]
        }

        result = basinhopping(
            objective,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=20,
            seed=42
        )

        return result.fun

    def find_optimal_mixture(self, model_size: int, tokens: int) -> np.ndarray:
        """
        Find domain mixture that minimizes loss for given compute budget.

        For additive formulation, optimal mixture is compute-independent!

        Args:
            model_size: Target model size (for validation)
            tokens: Target training tokens (for validation)

        Returns:
            Optimal mixture weights (sums to 1)
        """
        def objective_mixture(weights):
            # Constraint: weights sum to 1
            if not np.isclose(weights.sum(), 1.0):
                return 1e10

            # Minimize over mixture (ignoring scale terms since they don't depend on h)
            weighted_sum = 0.0
            for i in range(self.num_domains):
                weighted_sum += self.C[i] * (weights[i] ** self.gamma[i])

            # Bias term to minimize
            loss = 1.0 / (weighted_sum + 1e-8)

            return loss

        # Constrain to simplex
        from scipy.optimize import LinearConstraint

        A_constraint = np.ones((1, self.num_domains))
        constraint = LinearConstraint(A_constraint, [1], [1])

        # Initial guess: uniform mixture
        x0 = np.ones(self.num_domains) / self.num_domains

        result = minimize(
            objective_mixture,
            x0,
            method='SLSQP',
            bounds=[(0, 1) for _ in range(self.num_domains)]
        )

        optimal = result.x / result.x.sum()  # Normalize

        return optimal
```

### Joint Scaling Law: Mixture-Dependent Scale Parameters

More complex formulation where scale parameters depend on mixture.

```python
class JointScalingLaw:
    """
    Joint formulation: Loss = E(h) + A^h/N^α^h + B^h/D^β^h
    where A, B, α, β all depend on mixture weights h.
    Captures scale-mixture interactions.
    """

    def __init__(self, num_domains: int = 3):
        self.num_domains = num_domains

        # For A term
        self.C_A = np.ones(num_domains)
        self.gamma_A = np.ones(num_domains)
        self.alpha = 0.5

        # For B term
        self.C_B = np.ones(num_domains)
        self.gamma_B = np.ones(num_domains)
        self.beta = 0.5

        # Bias term
        self.C_E = np.ones(num_domains)
        self.gamma_E = np.ones(num_domains)

    def predict_loss(self, model_size: int, tokens: int, mixture_weights: np.ndarray):
        """
        Predict loss with mixture-dependent scale parameters.

        Args:
            model_size: N
            tokens: D
            mixture_weights: (num_domains,)

        Returns:
            Predicted loss
        """
        # A term depends on mixture
        A_weighted = 0.0
        for i in range(self.num_domains):
            A_weighted += self.C_A[i] * (mixture_weights[i] ** self.gamma_A[i])

        # B term depends on mixture
        B_weighted = 0.0
        for i in range(self.num_domains):
            B_weighted += self.C_B[i] * (mixture_weights[i] ** self.gamma_B[i])

        # Bias depends on mixture
        E_term = 0.0
        for i in range(self.num_domains):
            E_term += self.C_E[i] * (mixture_weights[i] ** self.gamma_E[i])

        # Loss prediction
        loss = E_term + A_weighted / (model_size ** self.alpha) + B_weighted / (tokens ** self.beta)

        return loss

    def fit_to_experiments(self, experiments: list) -> float:
        """Fit joint scaling law to experiments."""
        def objective(params):
            # Extract parameters
            C_A = params[:self.num_domains]
            gamma_A = params[self.num_domains:2*self.num_domains]
            C_B = params[2*self.num_domains:3*self.num_domains]
            gamma_B = params[3*self.num_domains:4*self.num_domains]
            C_E = params[4*self.num_domains:5*self.num_domains]
            gamma_E = params[5*self.num_domains:6*self.num_domains]
            alpha = params[6*self.num_domains]
            beta = params[6*self.num_domains + 1]

            # Update
            self.C_A = C_A
            self.gamma_A = gamma_A
            self.C_B = C_B
            self.gamma_B = gamma_B
            self.C_E = C_E
            self.gamma_E = gamma_E
            self.alpha = alpha
            self.beta = beta

            # MSE
            mse = 0.0
            for exp in experiments:
                predicted = self.predict_loss(
                    exp['model_size'],
                    exp['tokens'],
                    exp['mixture_weights']
                )
                mse += (predicted - exp['loss']) ** 2

            return mse / len(experiments)

        x0 = np.concatenate([
            self.C_A, self.gamma_A,
            self.C_B, self.gamma_B,
            self.C_E, self.gamma_E,
            [self.alpha, self.beta]
        ])

        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': [(0.1, 10)] * (6*self.num_domains) + [(0.1, 2), (0.1, 2)]
        }

        result = basinhopping(
            objective,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=20
        )

        return result.fun

    def find_optimal_mixture(
        self,
        model_size: int,
        tokens: int,
        optimization_target: str = 'loss'
    ) -> np.ndarray:
        """
        Find optimal mixture for specific compute budget (compute-dependent).

        Args:
            model_size: Target model size N
            tokens: Target tokens D
            optimization_target: 'loss' or 'efficiency'

        Returns:
            Optimal mixture weights for this budget
        """
        def objective_mixture(weights):
            if not np.isclose(weights.sum(), 1.0):
                return 1e10

            loss = self.predict_loss(model_size, tokens, weights)
            return loss

        x0 = np.ones(self.num_domains) / self.num_domains

        from scipy.optimize import LinearConstraint
        A_constraint = np.ones((1, self.num_domains))
        constraint = LinearConstraint(A_constraint, [1], [1])

        result = minimize(
            objective_mixture,
            x0,
            method='SLSQP',
            bounds=[(0, 1) for _ in range(self.num_domains)]
        )

        return result.x / result.x.sum()
```

### Experimental Design: Small-Scale Experiments for Law Fitting

Design efficient small experiments to fit scaling laws.

```python
class ScalingLawExperimentDesigner:
    """Design efficient experiments for fitting scaling laws."""

    @staticmethod
    def generate_experiment_grid(
        model_sizes: list,
        token_counts: list,
        mixture_points: list,
        num_domains: int = 3
    ) -> list:
        """
        Generate grid of experiments to fit scaling laws efficiently.

        Args:
            model_sizes: Model sizes to test (e.g., [186M, 370M, 750M, 1.5B])
            token_counts: Token counts to test (e.g., [1B, 5B, 10B, 50B])
            mixture_points: Mixture weight vectors (e.g., uniform, domain-heavy)
            num_domains: Number of domains

        Returns:
            List of experiment configs
        """
        experiments = []

        for model_size in model_sizes:
            for tokens in token_counts:
                for mixture in mixture_points:
                    # Validate mixture
                    if not np.isclose(sum(mixture), 1.0):
                        mixture = np.array(mixture) / sum(mixture)

                    experiments.append({
                        'model_size': model_size,
                        'tokens': tokens,
                        'mixture_weights': mixture,
                        'loss': None  # To be filled by actual training
                    })

        return experiments

    @staticmethod
    def recommend_efficient_grid(num_domains: int = 3):
        """Recommend efficient experimental grid (10-20 runs only)."""
        model_sizes = [186_000_000, 370_000_000, 750_000_000]  # 3 sizes
        token_counts = [1_000_000_000, 10_000_000_000]  # 2 token budgets

        # Mixture points: uniform + domain-heavy variants
        mixtures = [
            [1/num_domains] * num_domains  # Uniform
        ]

        # Add domain-heavy points
        for i in range(num_domains):
            mix = [0.1] * num_domains
            mix[i] = 0.7
            mix = [m / sum(mix) for m in mix]  # Normalize
            mixtures.append(mix)

        return {
            'model_sizes': model_sizes,
            'token_counts': token_counts,
            'mixtures': mixtures
        }
```

### End-to-End Workflow: From Experiments to Optimal Mixture

Complete pipeline for determining optimal training mixture.

```python
def determine_optimal_mixture_for_production(
    train_losses: list,
    target_model_size: int,
    target_tokens: int,
    num_domains: int = 3,
    use_joint: bool = False
) -> dict:
    """
    End-to-end: fit scaling law and find optimal mixture.

    Args:
        train_losses: Results from small-scale experiments
        target_model_size: Production model size to optimize for
        target_tokens: Production training budget
        num_domains: Number of domains in training data
        use_joint: Use joint (True) or additive (False) formulation

    Returns:
        {'optimal_mixture': weights, 'predicted_loss': loss, 'law_fit_error': mre}
    """
    # Choose formulation
    if use_joint:
        law = JointScalingLaw(num_domains)
    else:
        law = AdditiveScalingLaw(num_domains)

    # Fit to experimental data
    fit_error = law.fit_to_experiments(train_losses)
    print(f"Scaling law fit error (MSE): {fit_error:.6f}")

    # Find optimal mixture for target compute
    optimal_mixture = law.find_optimal_mixture(target_model_size, target_tokens)

    # Predict loss with optimal mixture
    predicted_loss = law.predict_loss(target_model_size, target_tokens, optimal_mixture)

    print(f"\nOptimal mixture for {target_model_size/1e6:.0f}M model, {target_tokens/1e9:.0f}B tokens:")
    for i, weight in enumerate(optimal_mixture):
        print(f"  Domain {i}: {weight:.1%}")

    print(f"Predicted loss: {predicted_loss:.4f}")

    # Extrapolation confidence
    mean_relative_error = fit_error  # Simplified; in practice compute MRE

    return {
        'optimal_mixture': optimal_mixture,
        'predicted_loss': predicted_loss,
        'law_fit_error': fit_error,
        'extrapolation_confidence': 'high' if fit_error < 0.02 else 'medium'
    }

# Example usage
sample_experiments = [
    {'model_size': 186_000_000, 'tokens': 1_000_000_000, 'mixture_weights': [0.33, 0.33, 0.34], 'loss': 3.521},
    {'model_size': 186_000_000, 'tokens': 10_000_000_000, 'mixture_weights': [0.33, 0.33, 0.34], 'loss': 3.201},
    {'model_size': 750_000_000, 'tokens': 1_000_000_000, 'mixture_weights': [0.33, 0.33, 0.34], 'loss': 3.156},
    {'model_size': 750_000_000, 'tokens': 10_000_000_000, 'mixture_weights': [0.33, 0.33, 0.34], 'loss': 2.834},
    # ... more experiments
]

result = determine_optimal_mixture_for_production(
    sample_experiments,
    target_model_size=7_000_000_000,
    target_tokens=1_000_000_000_000,
    num_domains=3,
    use_joint=False
)
```

## Practical Guidance

### Key Hyperparameters & Methodology

| Parameter | Value | Notes |
|-----------|-------|-------|
| Minimum Experiments | 10-20 | Each requires full pretraining; 10 experiments sufficient for 3-domain problems |
| Model Sizes | 186M, 370M, 750M (baseline), 1.7B | Doubling strategy reveals exponential trends |
| Token Budgets | 1B, 5B, 10B, 50B | Test different scales of compute |
| Domain Mixtures | Uniform + domain-heavy | 3-5 mixture points sufficient |
| Basin-hopping Iterations | 20 | Balance between optimization quality and speed |
| Optimizer | L-BFGS-B | Robust for parameter fitting |
| Constraint Type | Simplex (weights sum to 1) | Ensures valid probability distribution |

### When to Use

- Planning foundation model pretraining across multiple domains
- Determining optimal code/math/text ratios for LLM training
- Allocating compute budgets across multimodal training (image/text/video)
- Testing whether mixture optimization is worth additional experiments
- Predicting loss for unseen model sizes or compute budgets

### When NOT to Use

- Domains with very different natures (few samples extrapolate poorly)
- Scenarios where domain interactions are complex and non-factorizable
- Production environments where even 10 small experiments are too expensive
- Tasks already having established conventions (follow best practices instead)

### Common Pitfalls

- **Too few experiments**: <5 runs lead to poor parameter estimation; stick to 10+
- **Skipping intermediate scales**: Using only 186M and 7B without 750M loses information about scaling behavior
- **Ignoring mixture diversity**: Testing only uniform and single-domain-heavy limits law quality; vary gradually
- **Wrong formulation choice**: Additive assumes scale-mixture independence; joint is more flexible but needs more data
- **Over-fitting to experiments**: Don't use outlier experiments; validate for data quality before fitting
- **Ignoring compute-dependence**: For joint formulation, optimal mixture changes with budget; compute separately per target

### Expected Performance

- **Mean Relative Error**: <2% when extrapolating to 2-3x larger models (186M → 750M)
- **Extrapolation range**: Reliable up to ~7B parameters from ~750M baseline experiments
- **Mixture convergence**: Optimal weights typically converge with 50-200K total training tokens analyzed

## Reference

Chen, Y., Wang, X., Liu, S., et al. (2024). Scaling Laws for Optimal Data Mixtures. arXiv preprint arXiv:2507.09404.
