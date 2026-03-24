---
name: fusionroute-token-llm-collaboration
title: "Token-Level LLM Collaboration via FusionRoute"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.05106"
keywords: [Multi-Expert LLMs, Token-Level Routing, Model Collaboration, Inference Efficiency]
description: "Combine multiple specialized language models at token-level granularity without joint training or architectural compatibility. FusionRoute performs expert selection and generates complementary logits to overcome routing limitations, achieving superior cross-domain performance at inference time."
---

## When to Use This Skill
- Combining pre-trained specialized models (math experts, code specialists, reasoning models)
- Cross-domain tasks requiring diverse expertise (reasoning, coding, instruction following)
- Inference-time optimization where gradient access to experts is unavailable
- Scenarios where model merging or joint training is infeasible or expensive

## When NOT to Use This Skill
- Single-domain applications with single best-fit expert
- Token-level computation budget is critical (lightweight overhead still adds up at scale)
- Experts require synchronized updates or shared gradients
- Real-time latency constraints require minimal routing overhead

## Problem Summary
Organizations possess multiple specialized models but face a core dilemma: general-purpose models are expensive, while specialized models fail outside their training distributions. Prior token-level collaboration relies solely on expert selection between fixed outputs. However, a fundamental limitation exists: pure expert-only routing cannot attain optimal value functions unless strong assumptions hold. The routing-only approach treats expert outputs as immutable, limiting policy expressiveness.

## Solution: FusionRoute Dual-Function Framework

Perform simultaneous expert selection AND complementary logit generation at each token, expanding the achievable policy class beyond fixed expert outputs.

```python
class FusionRoute:
    def __init__(self, router, expert_models, num_experts):
        self.router = router  # Learned routing network
        self.experts = expert_models
        self.num_experts = num_experts

    def forward(self, token_embedding):
        """Select expert AND generate complementary logits"""
        # Step 1: Expert selection via routing weights
        routing_weights = self.router.select_expert(token_embedding)
        expert_idx = torch.argmax(routing_weights)

        # Step 2: Get selected expert's logits
        expert_output = self.experts[expert_idx](token_embedding)

        # Step 3: Generate complementary logits via router
        # These correct/refine the expert output via logit addition
        complement_logits = self.router.generate_complement(token_embedding, expert_idx)

        # Step 4: Combine via logit addition
        final_logits = expert_output + complement_logits
        return final_logits, routing_weights
```

**Theoretical Contribution:**
FusionRoute overcomes an identifiability problem: given value observations alone, identifying optimal expert sequences from routing weights is impossible. Expanding the policy class with trainable complementary logits resolves this fundamental limitation.

## Key Implementation Details

**Two-Phase Training Pipeline:**

**Phase 1: Supervised Fine-Tuning (SFT)**
- Router learns token-wise expert selection
- Training focuses on "informative tokens" where experts disagree
- Prevents gradient dominance from trivial agreements on easy tokens

```python
# Sample selection: informative tokens only
def select_informative_tokens(expert_outputs, entropy_threshold=0.5):
    """Select tokens where expert disagreement is high"""
    disagreement = compute_expert_disagreement(expert_outputs)
    entropy = -torch.sum(disagreement * torch.log(disagreement + 1e-8), dim=-1)
    return entropy > entropy_threshold
```

**Phase 2: Preference Optimization (CDPO)**
- Modified Direct Preference Optimization refines complementary logits
- Treats expert outputs as fixed to preserve routing quality
- Only base model parameters receive preference updates

**Mix Training Strategy:**
- Routing supervision updates all router parameters
- Preference optimization only updates base model parameters
- Prevents routing degradation from preference signal

## Practical Configuration

**Model Setup:**
- Tested with Llama-3 and Gemma-2 model families
- No gradient access required to expert models
- No architectural constraints (works with any pre-trained models)

**Training Data:**
- 200k SFT examples for router training
- 100k preference pairs for CDPO refinement
- Training time: Typical 24-48 GPU hours on 8×H100

**Benchmarks:**
- GSM8K (mathematical reasoning)
- MATH500 (advanced mathematics)
- MBPP (Python programming)
- HumanEval (code generation)
- IfEval (instruction following)

## Performance Results

**Cross-Domain Evaluation:**
- Outperforms token-level collaboration baselines
- Exceeds model merging approaches
- Competitive with jointly finetuned models
- Scaling benefits increase at larger model sizes

**Key Advantages:**
- No joint training overhead
- Works with fixed expert models
- Minimal inference latency (lightweight router)
- Seamlessly integrates existing off-the-shelf models

## Advantages Over Alternatives

- **vs. Expert Selection Only**: Expands policy expressiveness via complementary logits; pure routing cannot reach optimal value
- **vs. Model Merging**: No need to combine weights; preserves expert specialization
- **vs. Joint Training**: No gradient dependency on experts; works with closed-source models
- **vs. Sequence-Level Routing**: Token-level decisions enable fine-grained expert switches

## Deployment Strategy
1. Select pre-trained specialized models (no modification needed)
2. Train lightweight router with SFT + CDPO pipeline
3. Deploy with token-by-token routing and logit addition
4. Monitor expert invocation patterns to verify diversity
