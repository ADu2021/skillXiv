---
name: test-time-self-improving-video
title: "VISTA: A Test-Time Self-Improving Video Generation Agent"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.15831"
keywords: [video generation, test-time improvement, agentic feedback, iterative refinement, self-play]
description: "Generate high-quality videos through iterative test-time refinement: agents critique outputs on visual/audio/context fidelity and a reasoning agent synthesizes feedback to improve prompts across multiple generations."
---

# Technique: Test-Time Self-Improving Video Agent — Iterative Refinement Without Retraining

Video generation quality depends critically on precise prompt engineering, but most users struggle to write optimal prompts on the first try. VISTA addresses this by creating a multi-agent system that runs autonomously at test time, iteratively refining prompts based on synthetic critique until reaching high-quality outputs.

Rather than requiring user feedback or external training, VISTA uses specialized agent critique on three quality dimensions—visual fidelity, audio coherence, and contextual alignment—then synthesizes this feedback into improved prompts. This enables quality improvement without retraining the base video model.

## Core Concept

VISTA operates through a closed-loop refinement cycle:
- **Temporal Planning**: Convert user idea into structured temporal structure
- **Generation & Selection**: Generate candidate videos, identify best via pairwise tournament
- **Multi-Agent Critique**: Three agents (visual, audio, context) evaluate independently
- **Prompt Synthesis**: Reasoning agent integrates critiques and rewrites prompts
- **Iterative Cycles**: Repeat until convergence or quality threshold

The system achieves 60% win rate against baselines and 66.4% human preference rate by working entirely at test time without model fine-tuning.

## Architecture Overview

- **Prompt Encoder**: Convert user input and feedback into structured temporal plans
- **Video Generator**: Off-the-shelf text-to-video model (e.g., OpenAI Sora, Runway)
- **Selection Tournament**: Pairwise comparisons to rank generated videos
- **Visual Agent**: Evaluates scene composition, object clarity, visual coherence
- **Audio Agent**: Checks audio-visual synchronization, sound quality, speech clarity
- **Context Agent**: Verifies narrative consistency, temporal flow, semantic alignment
- **Reasoning Agent**: Aggregates all critiques and generates improved prompts
- **Iteration Controller**: Decides when to stop (quality reached or budget exhausted)

## Implementation Steps

The core loop generates videos, collects critique, and synthesizes improved prompts. This example shows how to implement the critique-synthesis pipeline.

```python
from typing import List, Dict, Tuple
import json

class VideoAgentCritique:
    """Multi-agent critique system for video generation."""

    def __init__(self, visual_agent, audio_agent, context_agent, reasoning_agent):
        self.visual = visual_agent      # LLM with vision understanding
        self.audio = audio_agent        # LLM with audio expertise
        self.context = context_agent    # LLM for narrative analysis
        self.reasoning = reasoning_agent # LLM for synthesis

    def critique_video(self, video_frames, audio, current_prompt):
        """
        Evaluate video on three dimensions independently.
        Returns structured critique for each agent.
        """
        # Visual critique: scene composition, clarity, consistency
        visual_critique = self.visual.evaluate(
            prompt=(
                "Evaluate this video's visual quality. Check: scene composition, "
                "object clarity, visual consistency across frames, color grading, "
                "lighting coherence. Rate each 1-5 and highlight issues."
            ),
            video_frames=video_frames,
            current_prompt=current_prompt
        )

        # Audio critique: sync, quality, speech
        audio_critique = self.audio.evaluate(
            prompt=(
                "Evaluate audio quality: Are sounds synchronized with visuals? "
                "Is speech clear? Is music/ambient sound appropriate? "
                "Rate audio-visual sync 1-5. List specific issues."
            ),
            audio=audio,
            video_frames=video_frames,
            current_prompt=current_prompt
        )

        # Context critique: narrative, temporal flow
        context_critique = self.context.evaluate(
            prompt=(
                "Analyze narrative coherence: Does the video follow the prompt? "
                "Is temporal flow logical? Are characters/objects consistent? "
                "Rate narrative alignment 1-5. List inconsistencies."
            ),
            video_frames=video_frames,
            current_prompt=current_prompt
        )

        return {
            "visual": visual_critique,
            "audio": audio_critique,
            "context": context_critique
        }

    def synthesize_feedback(
        self,
        critiques: Dict,
        current_prompt: str,
        iteration: int
    ) -> str:
        """
        Aggregate multi-agent critiques and generate improved prompt.
        """
        synthesis_prompt = f"""
You are a video prompt engineer synthesizing feedback from three specialists.

Current Prompt: {current_prompt}

Iteration: {iteration}

Visual Feedback:
{json.dumps(critiques['visual'], indent=2)}

Audio Feedback:
{json.dumps(critiques['audio'], indent=2)}

Context Feedback:
{json.dumps(critiques['context'], indent=2)}

Based on this feedback, generate an improved prompt that:
1. Preserves the core intent
2. Addresses the top visual issues
3. Improves audio coherence
4. Strengthens narrative alignment
5. Is specific and actionable

Return only the improved prompt, no explanation.
"""
        improved_prompt = self.reasoning.generate(synthesis_prompt)
        return improved_prompt


def self_improving_video_generation(
    user_prompt: str,
    video_generator,
    critique_system: VideoAgentCritique,
    max_iterations: int = 3,
    quality_threshold: float = 4.2  # Out of 5
):
    """
    Generate video and iteratively improve via multi-agent critique.
    """
    current_prompt = user_prompt
    best_video = None
    best_score = 0.0

    for iteration in range(max_iterations):
        # Generate video candidates
        videos = [
            video_generator.generate(current_prompt)
            for _ in range(2)  # Generate 2 candidates per iteration
        ]

        # Select best via pairwise tournament
        selected_video = select_best_video(videos)

        # Evaluate quality
        quality_score = evaluate_video_quality(selected_video)

        if quality_score > best_score:
            best_video = selected_video
            best_score = quality_score

        # Check convergence
        if quality_score >= quality_threshold:
            print(f"Quality threshold reached at iteration {iteration}")
            break

        # Critique and refine
        critiques = critique_system.critique_video(
            video_frames=selected_video['frames'],
            audio=selected_video['audio'],
            current_prompt=current_prompt
        )

        current_prompt = critique_system.synthesize_feedback(
            critiques,
            current_prompt,
            iteration=iteration + 1
        )

        print(f"Iteration {iteration + 1}: Score={quality_score:.2f}, "
              f"Refined prompt={current_prompt[:100]}...")

    return best_video, best_score, current_prompt
```

The key insight is that critique doesn't require perfect agents—even simple LLM-based evaluators can identify genuine improvements. Three independent critiques provide redundancy; if two agents agree on an issue, the synthesis agent prioritizes it.

## Practical Guidance

| Video Type | Iterations | Expected Improvement |
|-----------|-----------|--------------------:|
| Simple scene | 1-2 | +15-25% quality |
| Complex narrative | 2-3 | +25-40% quality |
| Multi-shot cinematic | 3-4 | +30-50% quality |

**When to Use:**
- Generating videos for which high quality matters (marketing, content)
- You can tolerate 2-3× generation cost for better results
- Base video model is already reasonably capable
- You have specialized agents (visual, audio, context) available

**When NOT to Use:**
- Real-time video generation (iterations take seconds/minutes)
- Base video model is poor (iteration can't fix fundamental limitations)
- Cost per generation is prohibitive (test-time refinement multiplies cost)
- Prompt is already well-specified (diminishing returns)

**Common Pitfalls:**
- Critique agents too harsh or lenient → biases refinement (calibrate on validation set)
- Synthesis agent stuck in local optima → add random perturbations to prompts occasionally
- Not tracking best video separately → always use best_score, not latest_score for quality assessment
- Over-iterating → set quality threshold and early stopping condition
- Specialized agents lack proper context → provide full video frames + previous critique history

## Reference

[VISTA: A Test-Time Self-Improving Video Generation Agent](https://arxiv.org/abs/2510.15831)
