---
name: covrl-variational-reasoning
title: "Coupled Variational Reinforcement Learning for Language Model Reasoning Without External Verifiers"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.12576
keywords: [reinforcement-learning, language-models, reasoning, verifier-free, variational-inference]
description: "Enhance language model reasoning through coupled sampling from prior (question-only) and posterior (answer-conditioned) distributions. Construct composite distribution mixing both at token level using hybrid sampling. Combine reconstruction term, selective NLL loss, and KL regularization. Achieve 12.4% improvement over base model and 2.3% over comparable baselines."
---

## Skill Summary

Coupled Variational Reinforcement Learning (CoVRL) addresses limitations of previous verifier-free reasoning approaches by establishing coupling between two complementary sampling distributions. The prior distribution (question-only) reflects real inference conditions but offers limited guidance. The posterior distribution (answer-conditioned) provides better exploration but creates training-inference mismatch. CoVRL constructs a composite distribution mixing both distributions at the token level, employing a hybrid sampling strategy where each training example randomly samples from either prior or posterior. The optimization combines reconstruction term, selective negative log-likelihood loss on high-quality traces, and KL regularization ensuring learned patterns transfer to inference, achieving 12.4% improvement over base model.

## When To Use

- Training language models for reasoning without external verifiers
- Scenarios where answer conditioning during training aids exploration
- Projects seeking better training-inference alignment
- Research on verifier-free reasoning approaches

## When NOT To Use

- Applications where external verifiers are available (simpler approaches work)
- Domains not benefiting from answer-conditioned learning
- Scenarios where training-inference mismatch handling adds overhead
- Models where simple supervised finetuning suffices

## Core Technique

Three key components enable effective verifier-free reasoning training:

**1. Prior Distribution (Question-Only)**
Reflects real inference conditions but offers limited guidance. This distribution alone suffers from sparse reward signals and difficulty maintaining learning dynamics without answer context.

**2. Posterior Distribution (Answer-Conditioned)**
Provides richer guidance by conditioning on target answers. This enables better exploration and stronger learning signals but creates training-inference mismatch: during inference, no answer is available to condition on.

**3. Composite Distribution with Hybrid Sampling**
Establish coupling between prior and posterior by constructing composite distribution mixing both at the token level. Employ hybrid sampling strategy where each training example randomly samples from either:
- Prior: maintains inference-time realism
- Posterior: leverages answer guidance during training

This balances exploration during training with transferability to inference.

**4. Optimization Approach**
Combine three complementary terms:
- Reconstruction term: encourage correct answers
- Selective NLL loss: high-quality traces get standard supervision
- KL regularization: ensure learned patterns transfer to prior distribution

## Results

- 12.4% improvement over base model
- 2.3% improvement over comparable baselines
- Better transfer from training to inference compared to naive answer conditioning

## Implementation Notes

Design prior and posterior sampling distributions. Implement hybrid sampling strategy for training. Construct composite distribution from both. Combine reconstruction, selective NLL, and KL regularization objectives. Monitor transfer from training to inference-time performance.

## References

- Original paper: Coupled Variational RL (Dec 2025)
- Variational inference for language models
- Verifier-free reinforcement learning
