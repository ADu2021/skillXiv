---
name: maps-multiagent-personality-reasoning
title: "MAPS: A Multi-Agent Framework Based on Big Seven Personality and Socratic Guidance for Multimodal Scientific Problem Solving"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2503.16905"
keywords: [Multi-Agent Systems, Personality Embeddings, Socratic Questioning, Multimodal Reasoning, Problem Solving]
description: "Coordinate specialized agents with distinct personality traits (Openness, Agreeableness, Conscientiousness, Extraversion) to solve complex scientific problems across text and vision, using a Critic agent to apply Socratic questioning for iterative refinement and error correction."
---

## Core Concept

MAPS decomposes multimodal scientific problem solving into specialized stages executed by agents with distinct personalities derived from Big Five personality theory. Each agent receives learned personality embeddings that shape its reasoning style. A Critic agent applies Socratic questioning to identify and correct flawed reasoning, enabling iterative refinement. The framework achieves state-of-the-art results on benchmarks like MathVista and OlympiadBench, surpassing human expert performance.

## Architecture Overview

The system orchestrates four core reasoning agents plus a Critic:

- **Interpreter Agent (Openness)**: Extracts visual semantics and structural information from diagrams and images
- **Aligner Agent (Agreeableness)**: Reconciles visual information with textual context and questions
- **Scholar Agent (Conscientiousness)**: Integrates domain-specific knowledge and ensures logical consistency
- **Solver Agent (Extraversion)**: Generates final answers through logical composition and synthesis
- **Critic Agent (Neuroticism)**: Evaluates confidence in each stage and triggers revision via Socratic questions

Each agent processes multimodal inputs (diagrams, text, questions) and produces intermediate reasoning outputs that feed into subsequent stages.

## Implementation

The personality embedding mechanism projects Big Five traits into the model's encoding space:

```python
import torch
import torch.nn as nn

class PersonalityEmbedding(nn.Module):
    """Projects personality traits into model embedding space."""
    def __init__(self, hidden_dim, num_traits=5):
        super().__init__()
        # Big Five: Openness, Conscientiousness, Extraversion,
        #          Agreeableness, Neuroticism
        self.trait_embeddings = nn.Parameter(torch.randn(num_traits, hidden_dim))
        self.projection = nn.Linear(num_traits, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, trait_vector):
        """
        Args:
            trait_vector: (B, 5) personality trait intensities [0, 1]
        Returns:
            embeddings: (B, D) projected personality embeddings
        """
        embedded = torch.matmul(trait_vector, self.trait_embeddings)
        projected = self.projection(trait_vector)
        combined = embedded + projected
        return self.norm(combined)
```

The four-stage reasoning pipeline is implemented as a sequential agent system:

```python
class ReasoningAgent(nn.Module):
    """Single reasoning stage with personality-conditioned processing."""
    def __init__(self, hidden_dim, personality_dim):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        self.personality_gate = nn.Linear(personality_dim, hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, visual_tokens, text_tokens, personality_embedding):
        """
        Args:
            visual_tokens: (B, N_v, D) image patch tokens
            text_tokens: (B, N_t, D) text embeddings
            personality_embedding: (B, D) personality trait projection
        Returns:
            output: (B, D) reasoning output
        """
        # Cross-modal fusion via attention
        query = visual_tokens.mean(dim=1, keepdim=True) + text_tokens.mean(dim=1, keepdim=True)
        attn_output, _ = self.cross_attention(query, visual_tokens, text_tokens)

        # Apply personality gate to condition reasoning
        personality_gate = torch.sigmoid(self.personality_gate(personality_embedding))
        gated = attn_output * personality_gate.unsqueeze(1)

        # Feed-forward with residual
        ffn_output = self.feed_forward(gated)
        output = self.norm2(ffn_output + attn_output)

        return output.squeeze(1)
```

The Critic agent implements Socratic questioning via confidence scoring and targeted revision:

```python
class CriticAgent(nn.Module):
    """Evaluates reasoning stages and triggers revision via Socratic questioning."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.confidence_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.question_generator = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, stage_outputs, threshold=0.7):
        """
        Args:
            stage_outputs: list of (B, D) outputs from each reasoning stage
            threshold: confidence threshold τ for triggering revision
        Returns:
            confidence_scores: (B, 4) confidence for each stage
            questions: (B, D) Socratic question embeddings
            needs_revision: (B, 4) boolean mask for stages needing revision
        """
        scores = []
        for output in stage_outputs:
            score = self.confidence_scorer(output)
            scores.append(score)

        confidence_scores = torch.cat(scores, dim=1)

        # Identify weakest stages
        needs_revision = confidence_scores < threshold

        # Generate Socratic questions for weak stages
        worst_idx = torch.argmin(confidence_scores, dim=1)
        questions = torch.stack([
            self.question_generator(stage_outputs[idx])
            for idx in worst_idx
        ])

        return confidence_scores, questions, needs_revision
```

The main reasoning loop orchestrates iterative refinement:

```python
def collaborative_reasoning_loop(
    visual_input, text_input, personality_embeddings,
    agents, critic, max_iterations=5, confidence_threshold=0.7
):
    """
    Args:
        visual_input: (B, N_v, D) image patches
        text_input: (B, N_t, D) text tokens
        personality_embeddings: dict of personality projections for each agent
        agents: dict of ReasoningAgent modules
        critic: CriticAgent module
        max_iterations: maximum refinement loops
        confidence_threshold: τ parameter
    Returns:
        final_answer: (B, D) solved output
        refinement_history: list of (confidence, questions) tuples
    """
    stage_outputs = {}
    refinement_history = []

    for iteration in range(max_iterations):
        # Execute four-stage pipeline
        interp_out = agents['interpreter'](
            visual_input, text_input, personality_embeddings['openness']
        )
        stage_outputs['interpreter'] = interp_out

        align_out = agents['aligner'](
            visual_input, text_input + interp_out.unsqueeze(1),
            personality_embeddings['agreeableness']
        )
        stage_outputs['aligner'] = align_out

        scholar_out = agents['scholar'](
            visual_input, text_input + align_out.unsqueeze(1),
            personality_embeddings['conscientiousness']
        )
        stage_outputs['scholar'] = scholar_out

        solver_out = agents['solver'](
            visual_input, text_input + scholar_out.unsqueeze(1),
            personality_embeddings['extraversion']
        )
        stage_outputs['solver'] = solver_out

        # Critic evaluation
        outputs_list = [stage_outputs[k] for k in ['interpreter', 'aligner', 'scholar', 'solver']]
        confidence, questions, needs_revision = critic(outputs_list, confidence_threshold)
        refinement_history.append((confidence, questions))

        # Check convergence
        if not needs_revision.any():
            break

        # Revise weakest stages with Socratic guidance
        weakest_stage = torch.argmax((~needs_revision).float(), dim=1)

    return solver_out, refinement_history
```

## Practical Guidance

**When to Use:**
- Solving complex scientific problems requiring multi-step reasoning (MathVista, OlympiadBench)
- Tasks combining diagrams, text, and domain knowledge
- Scenarios where iterative refinement improves solution quality
- When human-level or super-human performance is the goal

**When NOT to Use:**
- Simple classification or retrieval tasks that don't benefit from multi-stage reasoning
- Real-time applications where iterative refinement latency is unacceptable
- Tasks with no visual component (pure text reasoning may not need visual agent)
- Computationally constrained environments (4 agents + Critic = higher cost)

**Key Hyperparameters:**
- `confidence_threshold` (τ): 0.6-0.8 controls revision frequency; lower = more iterations
- `max_iterations`: 3-5 typically sufficient; 10+ rarely needed
- `hidden_dim`: 768-1024 for competitive performance
- `num_heads`: 8-16 for multi-head attention in agents

**Common Pitfalls:**
- Setting τ too low causes excessive revision iterations, increasing latency
- Using identical personality embeddings defeats the purpose of role specialization
- Not providing sufficient domain knowledge to the Scholar agent
- Critic threshold tuning requires task-specific validation

## Performance Notes

- Achieves 56.31% accuracy on benchmark suite (MathVista, OlympiadBench, EMMA)
- Surpasses human expert performance by 3.58%
- 15.84% improvement over previous SOTA
- Generalizes across GPT-4o, Gemini, Qwen base models
- Each additional iteration adds ~200ms latency but can improve accuracy 2-5%

## References

- Big Five personality theory (McCrae & Costa)
- Socratic questioning method for guided learning
- Multimodal transformers and cross-modal attention
- Chain-of-thought prompting and reasoning decomposition
- MathVista and OlympiadBench benchmarks
