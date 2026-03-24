---
name: duetsvg-multimodal-svg
title: "DuetSVG: Unified Multimodal SVG Generation with Internal Visual Guidance"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2512.10894
keywords: [SVG generation, multimodal models, visual guidance, vector graphics, test-time scaling]
description: "Generate SVGs through simultaneous image and SVG token generation with internal visual guidance. DuetSVG overcomes text-only limitations by leveraging visual predictions to enhance SVG coherence—ideal when visual quality and geometric correctness matter."
---

## Overview

DuetSVG addresses limitations in vision-language-only SVG generation by simultaneously generating both image tokens and SVG tokens in an integrated process. Test-time scaling uses the model's own visual predictions as guidance to enhance generation quality.

## When to Use

- Scalable vector graphics generation from text/images
- Need for visually coherent SVGs
- Scenarios requiring geometric correctness
- Applications with diverse SVG generation tasks
- Need for semantic coherence and visual appeal

## When NOT to Use

- Raster image generation (use diffusion models)
- Simple geometric shapes (simpler approaches sufficient)
- Real-time SVG generation with latency constraints

## Core Technique

Unified multimodal generation with test-time scaling:

```python
# DuetSVG: Unified multimodal SVG generation
class DuetSVGGenerator:
    def __init__(self):
        self.image_encoder = ImageEncoder()
        self.svg_tokenizer = SVGTokenizer()
        self.multimodal_decoder = MultimodalDecoder()

    def generate_svg_with_visual_guidance(self, text_prompt):
        """
        Generate SVG jointly with image tokens.
        Use visual predictions to guide SVG generation.
        """
        # Initialize generation
        generated_image_tokens = []
        generated_svg_tokens = []

        for step in range(max_steps):
            # Joint decoding: image and SVG tokens together
            next_image_token = self.multimodal_decoder.predict_image_token(
                generated_image_tokens,
                generated_svg_tokens,
                text_prompt
            )

            next_svg_token = self.multimodal_decoder.predict_svg_token(
                generated_image_tokens,
                generated_svg_tokens,
                text_prompt
            )

            # Internal visual guidance: use predicted image to guide SVG
            if self.should_apply_visual_guidance(step):
                # Decode partial image
                partial_image = self.decode_image_tokens(
                    generated_image_tokens + [next_image_token]
                )

                # Analyze visual properties
                visual_features = self.extract_visual_features(
                    partial_image
                )

                # Re-score SVG token based on visual consistency
                next_svg_token = self.rescore_svg_token(
                    next_svg_token,
                    visual_features,
                    partial_image
                )

            generated_image_tokens.append(next_image_token)
            generated_svg_tokens.append(next_svg_token)

            # Check for completion
            if self.is_complete(generated_svg_tokens):
                break

        # Decode final SVG
        final_svg = self.svg_tokenizer.decode(generated_svg_tokens)
        final_image = self.decode_image_tokens(generated_image_tokens)

        return final_svg, final_image

    def extract_visual_features(self, image):
        """Extract relevant visual properties from partial rendering."""
        features = {
            'color_palette': self.extract_colors(image),
            'spatial_layout': self.extract_layout(image),
            'object_positions': self.detect_objects(image),
            'visual_style': self.analyze_style(image)
        }
        return features

    def rescore_svg_token(self, token, visual_features, partial_image):
        """Re-evaluate SVG token based on visual consistency."""
        # Generate alternative SVG tokens
        alternatives = self.generate_alternatives(token)

        scores = []
        for alt in alternatives:
            # Simulate adding this SVG element
            test_svg = self.simulate_svg_addition(alt, partial_image)

            # Score consistency with visual features
            consistency = self.compute_visual_consistency(
                test_svg,
                visual_features,
                partial_image
            )

            scores.append(consistency)

        # Select highest-scoring alternative
        best_token = alternatives[torch.argmax(torch.tensor(scores))]
        return best_token

    def test_time_scaling(self, text_prompt, budget=10):
        """
        Test-time scaling: use remaining budget to refine SVG.
        Can sample multiple SVGs and select best.
        """
        candidates = []

        for sample_idx in range(budget):
            # Generate SVG (stochastic sampling)
            svg, image = self.generate_svg_with_visual_guidance(
                text_prompt
            )

            candidates.append((svg, image))

        # Score candidates
        best_svg = self.select_best_candidate(
            candidates,
            text_prompt
        )

        return best_svg

    def select_best_candidate(self, candidates, text_prompt):
        """Score and select best generated SVG."""
        scores = []

        for svg, image in candidates:
            # Score: visual quality, semantic correctness, coherence
            visual_quality = self.score_visual_quality(image)
            semantic_match = self.score_semantic_match(svg, text_prompt)
            internal_consistency = self.score_consistency(svg, image)

            total_score = (
                0.4 * visual_quality +
                0.4 * semantic_match +
                0.2 * internal_consistency
            )

            scores.append(total_score)

        best_idx = torch.argmax(torch.tensor(scores))
        return candidates[best_idx][0]
```

## Key Results

- Visually coherent and geometrically correct SVGs
- Semantic coherence through visual guidance
- Diverse SVG generation capabilities
- Test-time scaling improves quality

## References

- Original paper: https://arxiv.org/abs/2512.10894
- Focus: Vector graphics generation
- Domain: Generative models, SVG synthesis
