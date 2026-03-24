---
name: janus-moe-disaggregation
title: "Janus: Disaggregated Attention and Expert Layers for Scalable MoE Inference"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.13525
keywords: [mixture-of-experts, MoE-inference, distributed-inference, disaggregation, throughput-optimization]
description: "Enable scalable MoE inference by disaggregating attention and expert layers onto independent GPU sub-clusters. Use adaptive two-phase communication, activation load-balanced scheduling, and activation-aware expert management. Achieve 3.9× higher per-GPU throughput than state-of-the-art systems."
---

## Skill Summary

Janus addresses MoE inference scaling through disaggregated architecture separating attention and MoE layers onto independent GPU sub-clusters. The system combines adaptive two-phase communication minimizing cross-cluster transfers, activation load-balanced scheduling distributing requests intelligently, and activation-aware expert management dynamically adjusting replication. Results show 3.9× higher per-GPU throughput while maintaining latency SLOs.

## When To Use

- Deploying large sparse mixture-of-experts models with independent attention/expert scaling
- Scenarios where attention and MoE layers have different performance characteristics
- Projects with sufficient GPU clusters to justify disaggregation infrastructure
- Research on efficient MoE inference systems

## When NOT To Use

- Small models where disaggregation overhead exceeds benefits
- Single-GPU or tightly coupled systems where disaggregation creates bottlenecks
- Latency-critical applications where communication overhead is problematic
- Scenarios with fixed hardware constraints preventing disaggregation

## Core Technique

Four key components enable disaggregated MoE inference:

**1. Disaggregated Architecture**
Rather than deploying entire MoE model as monolithic unit, manage attention and MoE layers separately. This enables fine-grained, module-specific resource scaling based on distinct performance characteristics.

**2. Adaptive Two-Phase Communication**
Minimize overhead from frequent data transfers between attention and MoE instances:
- Intra-node aggregation via NVLink consolidates activations
- Bulk inter-node transfers reduce number of small cross-cluster messages
Adaptive routing switches between phases based on activation characteristics.

**3. Activation Load-Balanced Scheduling**
Lightweight GPU kernel scheduler distributes expert activation requests across MoE instances to minimize concurrently active experts per GPU. This reduces per-instance load and latency with negligible overhead.

**4. Activation-Aware Expert Management**
Dynamically adjust expert replication counts and placement based on activation patterns. Spread frequently co-activated experts across GPUs to reduce per-instance load, improving throughput.

## Implementation Notes

Design system separating attention and MoE layer clusters. Implement adaptive two-phase communication: intra-node via NVLink, inter-node in bulk transfers. Build activation load-balanced scheduler distributing requests intelligently. Track expert co-activation patterns and dynamically adjust replication. Monitor and optimize for your specific workload characteristics.

## References

- Original paper: Janus (Dec 2025)
- Mixture-of-experts systems and inference
- Distributed GPU systems and communication optimization
