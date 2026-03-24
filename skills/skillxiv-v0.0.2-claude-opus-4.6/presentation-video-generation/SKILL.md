---
name: presentation-video-generation
title: "PresentAgent: Multimodal Agent for Presentation Video Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.04036"
keywords: [Presentation Generation, Multimodal AI, Video Synthesis, Document Processing, Audio-Visual Alignment]
description: "Transform lengthy documents into fully narrated presentation videos with synchronized audio-visual delivery. Automatically segments content, generates visuals, synthesizes speech, and composes final video."
---

# PresentAgent: Automatic Presentation Video Generation from Documents

Creating presentation videos from documents requires coordinating multiple modalities: extracting key ideas, designing visuals, writing scripts, synthesizing speech, and synchronizing everything. Current approaches either generate static slides or disconnected video clips. PresentAgent bridges this gap by orchestrating the entire workflow—from document parsing to final video composition—maintaining semantic coherence and temporal alignment throughout.

The core insight is that presentation generation is a modular pipeline problem. Each stage (segmentation, visual design, narration, assembly) has distinct requirements and can be optimized independently, yet must coordinate carefully through shared semantic understanding.

## Core Concept

PresentAgent operates as a four-stage sequential pipeline where each stage builds on the previous one while maintaining focus on the original document's intent:

1. **Document segmentation** breaks lengthy inputs into coherent content blocks
2. **Slide generation** creates visually-aligned layouts for each block
3. **Narration synthesis** converts key messages into natural spoken audio
4. **Video assembly** synchronizes visuals and audio into a final presentation

This modular design allows each component to be replaced or upgraded independently while maintaining end-to-end coherence.

## Architecture Overview

- **Document parser**: Extracts semantic structure through outline planning
- **Slide renderer**: Generates layout-guided visual frames with text and imagery
- **Script generator**: LLM-based conversion of content to oral-style narration scripts
- **Text-to-speech engine**: Synthesizes high-quality audio from scripts
- **Video compositor**: Aligns slides temporally with audio duration
- **PresentEval framework**: Vision-language model evaluation of content fidelity, visual clarity, and comprehension

## Implementation

Start by parsing your document and extracting the outline structure:

```python
import json
from presentagent.parser import DocumentParser
from presentagent.planner import ContentPlanner

# Load and segment the document
parser = DocumentParser()
document_text = open("report.pdf").read()

# Extract semantic structure into content blocks
outline = parser.create_outline(document_text)
content_blocks = parser.segment_by_outline(document_text, outline)

# Each block becomes one or more slides
for i, block in enumerate(content_blocks):
    print(f"Block {i}: {block['title']}")
    print(f"Content: {block['text'][:100]}...")
```

Next, generate visual slides for each content block using layout guidance:

```python
from presentagent.renderer import SlideRenderer

renderer = SlideRenderer(theme="professional")

slides = []
for block in content_blocks:
    # Determine layout based on content type
    layout_type = determine_layout(block)  # "title", "content", "image", etc.

    slide = renderer.render(
        title=block["title"],
        content=block["text"],
        layout=layout_type,
        images=retrieve_relevant_images(block["title"])
    )
    slides.append(slide)
```

Convert slide content to narration scripts using an LLM, then synthesize audio:

```python
from presentagent.narration import NarrationGenerator
from presentagent.tts import TextToSpeech

narrator = NarrationGenerator(model="gpt-4")
tts = TextToSpeech(voice="professional")

audio_segments = []
for slide in slides:
    # Convert visual content to spoken narration
    script = narrator.generate_script(
        title=slide["title"],
        content=slide["content"]
    )

    # Synthesize audio with natural pacing
    audio = tts.synthesize(script, duration_adjustment=1.2)
    audio_segments.append(audio)
```

Finally, compose slides and audio into a synchronized video:

```python
from presentagent.compositor import VideoCompositor

compositor = VideoCompositor(resolution="1080p", fps=30)

# Build video with automatic timing based on audio duration
video = compositor.compose(
    slides=slides,
    audio_segments=audio_segments,
    transitions="fade",
    timing="audio-driven"  # Each slide duration matches its audio
)

video.export("presentation.mp4")
```

## Practical Guidance

### When to Use

Use PresentAgent for:
- Converting research papers or reports into presentation videos
- Automating briefing video generation for large teams
- Creating tutorial videos from technical documentation
- Generating training materials from course materials
- Producing narrated slide decks without manual effort

### When NOT to Use

Avoid this system for:
- Highly stylized presentations requiring custom branding
- Videos requiring speaker presence or facial expressions
- Content with complex animations or interactive elements
- Presentations where human narration tone is critical
- Documents with poor structure or ambiguous semantic flow

### Key Design Choices

| Component | Decision | Rationale |
|-----------|----------|-----------|
| Segmentation | Outline-based | Preserves document structure and semantic units |
| Narration | LLM-generated scripts | Converts visual content naturally to speech |
| Audio | TTS with pacing control | Maintains professional delivery speed |
| Timing | Audio-driven slide duration | Natural pacing without awkward static frames |
| Evaluation | Vision-language models | Assess final output quality without human raters |

### Common Pitfalls

1. **Breaking semantic units**: Forcing one slide per paragraph creates choppy presentations. Respect the document's natural content boundaries.
2. **Over-scripting**: Generated narration that reads exactly like slide text sounds robotic. Add connecting language and natural transitions.
3. **Ignoring audio duration**: Fixed slide durations cause jarring transitions. Always align visuals to audio length.
4. **Missing visual coherence**: Random images don't improve clarity. Select visuals that reinforce the specific message of each slide.
5. **Forgetting accessibility**: Always include captions synchronized with audio for accessibility.

### Evaluation Metrics

The paper introduces PresentEval, which assesses presentations on:
- **Content Fidelity**: Does the video accurately represent the original document?
- **Visual Clarity**: Are slides legible and visually well-organized?
- **Audience Comprehension**: Can viewers understand the key messages?

Evaluate your generated videos using these dimensions before deployment.

## Reference

"PresentAgent: Multimodal Agent for Presentation Video Generation" - [arXiv:2507.04036](https://arxiv.org/abs/2507.04036)
