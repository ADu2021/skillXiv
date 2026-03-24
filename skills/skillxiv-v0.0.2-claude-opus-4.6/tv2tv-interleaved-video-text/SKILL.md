---
name: tv2tv-interleaved-video-text
title: "TV2TV: Unified Framework for Interleaved Language and Video Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.05103
keywords: [video-generation, language-generation, multimodal, flow-matching, user-control]
description: "Mixture-of-Transformers jointly learning language modeling and video flow matching, enabling interleaved text-video generation where semantic decisions happen in language, pixel generation in video, and users can intervene textually at any step."
---

## Summary

TV2TV introduces a unified framework that decomposes video generation into interleaved text and video generation stages. The approach employs a Mixture-of-Transformers architecture jointly learning language modeling and video flow matching, enabling models to "think in words" about subsequent content before "acting in pixels" to produce frames. This factorization enables dynamic user control through textual interventions at any generation step.

## Core Technique

**Interleaved Generation Stages:**
1. **Language Stage:** Model reasons about what should happen next in compact text
2. **Video Stage:** Model generates pixels matching the language description
3. **User Intervention:** User can provide text feedback to steer generation

**Mixture-of-Transformers:** Separate but jointly-trained transformers:
- **Language Transformer:** Autoregressive language modeling
- **Video Transformer:** Flow matching for pixel generation
- **Routing:** Mixture gates determine which transformer to use

**Dynamic User Control:** At any point, user can provide text instructions modifying the generation path without restarting.

## Implementation

**Mixture-of-Transformers architecture:**
```python
class MixtureOfTransformers:
    def __init__(self):
        self.language_transformer = LanguageTransformer()
        self.video_transformer = VideoTransformer()
        self.router = Router()

    def forward(self, context, modality_hint='auto'):
        # Route to appropriate transformer
        gate = self.router(context)

        if gate > 0.5 or modality_hint == 'language':
            # Generate text description
            output = self.language_transformer(context)
        else:
            # Generate video frames
            output = self.video_transformer(context)

        return output, gate
```

**Interleaved generation loop:**
```python
def generate_interleaved(initial_prompt, max_steps=100):
    context = initial_prompt
    generated = []

    for step in range(max_steps):
        # Decide: language or video?
        output, gate = model(context)

        if gate > 0.5:  # Language stage
            # Generate text description of next scene
            next_description = output['text']
            generated.append(('text', next_description))
            context = concat(context, next_description)

        else:  # Video stage
            # Generate frames matching current description
            next_frames = output['video']
            generated.append(('video', next_frames))
            context = concat(context, next_frames)

        # Allow user intervention
        user_input = get_user_feedback()
        if user_input:
            context = concat(context, user_input)
            generated.append(('user', user_input))

    return generated
```

**Joint training objective:**
```python
def joint_loss(language_logits, video_flow, language_target, video_target, gate):
    # Language modeling loss
    language_loss = cross_entropy(language_logits, language_target)

    # Video flow matching loss
    video_loss = mse(video_flow, video_target)

    # Balance losses based on router gate
    combined_loss = gate * language_loss + (1 - gate) * video_loss

    return combined_loss
```

**User-driven steering:**
```python
def apply_user_direction(context, user_text):
    # Reweight mixture gates based on user preference
    # Language for semantic control, video for visual refinement

    if 'make' in user_text or 'change' in user_text:
        # Force language stage for semantic changes
        gate_override = 1.0
    elif 'adjust' in user_text:
        # Allow video stage for visual refinement
        gate_override = 0.3

    # Continue generation with modified routing
    return context, gate_override
```

## When to Use

- Video generation requiring fine-grained user control
- Scenarios where semantic planning precedes pixel generation
- Applications allowing interactive multi-turn generation
- Tasks benefiting from interpretable language descriptions

## When NOT to Use

- Fully autonomous video generation without user input
- Real-time generation where language-video interleaving adds latency
- Scenarios where single-stage generation is faster
- Applications where text descriptions interfere with visual quality

## Key References

- Mixture-of-Experts for multimodal routing
- Interleaved generation and autoregressive models
- Flow matching for video generation
- Interactive generation and user control
