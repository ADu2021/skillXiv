---
name: samtok-mask-tokenization
title: "SAMTok: Representing Any Mask with Two Words"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.16093"
keywords: [mask-tokenization, multimodal-models, segmentation, discrete-representation, pixel-tasks]
description: "Tokenize any region mask into two special tokens, enabling pixel-wise tasks like segmentation and region understanding in multimodal language models without architectural changes. Use when adapting vision-language models to perform pixel-level understanding and manipulation."
---

# SAMTok: Efficient Mask Tokenization

This skill demonstrates how to represent arbitrary region masks with just two special tokens, enabling vision-language models to perform pixel-wise tasks like segmentation and spatial understanding through standard language prediction.

## When to Use
- Adding segmentation capability to existing vision-language models
- Building models for region-level visual understanding
- Interactive image editing and manipulation with language models
- Pixel-level tasks without modifying model architecture
- Systems needing both global image and local region understanding

## When NOT to Use
- Simple image classification (doesn't need masks)
- Extremely high-resolution pixel-perfect tasks (two tokens may be limited)
- Tasks requiring dense pixel-level predictions (continuous masks better)
- Systems where architectural modifications aren't a constraint

## Key Concept
SAMTok enables language models to reason about image regions by converting arbitrary masks into just two special tokens. Instead of representing masks as:
- Dense pixel-level features (expensive)
- Continuous coordinate regression (hard for LMs)
- Complex polygon representations (non-standard)

SAMTok produces: `<REGION_START> ... <REGION_END>` where the content between encodes the mask in a format LMs understand.

This integrates seamlessly with existing language model prediction—segmentation becomes token prediction.

## Implementation Pattern

Implement mask tokenization for language models:

```python
# Pseudocode for SAMTok mask tokenization
class SAMTok:
    def __init__(self, vocab_size, num_mask_tokens=2):
        self.vocab_size = vocab_size
        self.region_start_token = vocab_size + 1
        self.region_end_token = vocab_size + 2

    def mask_to_tokens(self, mask):
        # Convert binary mask to compact token representation
        # mask: (H, W) binary or probability map
        # returns: list of tokens representing the mask

        # 1. Compress mask using lossless encoding
        compressed = self.compress_mask(mask)

        # 2. Convert to token sequence
        tokens = []
        tokens.append(self.region_start_token)

        # Encode compressed mask as token sequence
        for chunk in compressed:
            token = self.encode_chunk(chunk)
            tokens.append(token)

        tokens.append(self.region_end_token)

        return tokens

    def compress_mask(self, mask):
        # Lossless compression: identify mask boundary points
        # Store only contours/key coordinates, not full pixel map

        boundary_points = self.extract_contour(mask)

        # Quantize to lower resolution for efficiency
        quantized = self.quantize_points(boundary_points, grid_size=32)

        return quantized

    def encode_chunk(self, chunk):
        # Convert chunk (quantized coordinates) to token ID
        # Use hash or direct encoding
        return hash(tuple(chunk)) % (self.vocab_size - 2)

    def tokens_to_mask(self, tokens, target_shape):
        # Reverse: reconstruct mask from tokens
        # Assume tokens between REGION_START and REGION_END
        # represent the mask

        mask_tokens = tokens[tokens.index(self.region_start_token) + 1 :
                            tokens.index(self.region_end_token)]

        # Decode token sequence back to boundary points
        boundary_points = []
        for token in mask_tokens:
            chunk = self.decode_token(token)
            boundary_points.extend(chunk)

        # Reconstruct full mask from boundary points
        mask = self.reconstruct_from_boundary(boundary_points, target_shape)

        return mask

    def integrate_with_lm(self, image_tokens, region_mask):
        # In vision-language model forward pass:
        # 1. Get image embedding
        # 2. Append mask tokens
        # 3. Process with language model

        mask_tokens = self.mask_to_tokens(region_mask)

        # Combine in sequence
        full_sequence = image_tokens + mask_tokens + [output_token]

        return full_sequence
```

The beauty: masks integrate as just two special tokens plus efficient encodings.

## Key Results
- Pixel-wise tasks work in standard language models
- No architectural modifications needed
- Efficient: masks represented with minimal tokens
- Natural integration with language-based reasoning about regions

## Research Context
SAMTok shows that pixel-level understanding doesn't require dense representations. By using compact token-based encoding for masks, language models can naturally reason about regions, enabling segmentation and spatial understanding through language prediction rather than specialized pixel-level decoders.
