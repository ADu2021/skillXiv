# SkillXiv v0.0.2 - Converted ArXiv Papers

Complete index of 11 converted arXiv papers into structured agent skills.

## Skills by Category

### LLM Training & Reproducibility
- **fp32-reproducible-llm-inference** - Numerical precision for deterministic LLM inference (2506.09501)
- **poet-orthogonal-llm-training** - Orthogonal reparameterization for stable training (2506.08001)
- **noloco-low-communication-training** - Distributed training without all-reduce (2506.10911)

### Reasoning & Extended Thinking
- **magistral-reasoning-rl** - Pure RL for reasoning models with curriculum learning (2506.10910)
- **resa-transparent-reasoning** - SAE-based reasoning transfer at low cost (2506.09967)
- **cort-code-reasoning** - Code-integrated reasoning with verification (2506.09820)
- **feedback-friction-llm-response** - Understanding feedback integration limits (2506.11930)

### Automated ML & Agents
- **automind-adaptive-data-science** - Knowledge-grounded LLM agents for ML (2506.10974)
- **swe-factory-benchmark-generation** - Automated GitHub issue resolution datasets (2506.10954)

### Vision & Multimodal
- **text-aware-image-restoration** - Diffusion models preserving text (2506.09993)
- **latte-flow-unified-multimodal** - Efficient unified understanding/generation (2506.06952)

## Quick Reference

| Skill | Focus | Lines | Size |
|-------|-------|-------|------|
| fp32-reproducible-llm-inference | Numerical stability | 182 | 8K |
| poet-orthogonal-llm-training | Training dynamics | 238 | 12K |
| magistral-reasoning-rl | Reasoning via RL | 251 | 12K |
| swe-factory-benchmark-generation | Dataset creation | 363 | 16K |
| text-aware-image-restoration | Image + text | 396 | 16K |
| resa-transparent-reasoning | Reasoning transfer | 338 | 16K |
| automind-adaptive-data-science | ML agents | 532 | 20K |
| latte-flow-unified-multimodal | Multimodal efficiency | 383 | 16K |
| noloco-low-communication-training | Distributed training | 391 | 16K |
| feedback-friction-llm-response | Feedback limitations | 520 | 20K |
| cort-code-reasoning | Code verification | 519 | 20K |

**Total: 4,313 lines, 168K**

## Usage

Each skill contains:
- **YAML frontmatter** with metadata (name, title, version, URL, keywords)
- **Core Concept** - High-level overview
- **Architecture Overview** - Key components and design decisions
- **Implementation** - Numbered steps with code examples (Python)
- **Practical Guidance** - When/how to apply the technique
- **Reference** - Technical background and citations

### Example: fp32-reproducible-llm-inference

```python
# Diagnose precision-related nondeterminism
precision_results = measure_reproducibility_drift(model, configs)

# Implement LayerCast: store weights in BF16, compute in FP32
optimizer = LayerCastOptimizer(model, compute_dtype=torch.float32)
optimizer.patch_matmul_layers()

# Configure deterministic inference
model = setup_deterministic_inference(model, use_fp32=True)
```

## Topics Covered

- Floating-point arithmetic and determinism
- Orthogonal matrix parameterization
- Reinforcement learning for reasoning
- Sparse autoencoders
- Diffusion models
- Distributed training optimization
- LLM agents and knowledge bases
- Code execution and verification
- Multimodal architectures
- Feedback integration limitations

## ArXiv Links

1. https://arxiv.org/abs/2506.09501 - Numerical precision
2. https://arxiv.org/abs/2506.09820 - Code-integrated reasoning
3. https://arxiv.org/abs/2506.08001 - Orthogonal training
4. https://arxiv.org/abs/2506.10910 - Magistral reasoning
5. https://arxiv.org/abs/2506.10954 - SWE-Factory datasets
6. https://arxiv.org/abs/2506.09993 - Text-aware restoration
7. https://arxiv.org/abs/2506.09967 - Resa reasoning
8. https://arxiv.org/abs/2506.10974 - AutoMind agents
9. https://arxiv.org/abs/2506.06952 - LaTtE-Flow multimodal
10. https://arxiv.org/abs/2506.10911 - NoLoCo training
11. https://arxiv.org/abs/2506.11930 - Feedback friction

---

**Generated:** 2026-03-23  
**Format Version:** 0.0.2  
**Engine:** skillxiv-v0.0.2-claude-opus-4.6  
**License:** MIT
