---
name: sttm-spatio-temporal-token-merging-video
title: "Multi-Granular Spatio-Temporal Token Merging for Training-Free Acceleration of Video LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.07990"
keywords: [Video LLM, Token Reduction, Spatio-Temporal Merging, Training-Free Acceleration, KV Cache Efficiency]
description: "Accelerate video LLMs by 2× with minimal accuracy loss using training-free spatio-temporal token merging that exploits video redundancy through quadtree-based spatial compression and temporal frame similarity, enabling efficient multi-turn reasoning with KV cache reuse."
---

# STTM: Training-Free Acceleration of Video LLMs via Token Merging

Video LLMs process thousands of tokens per frame (video + text), causing quadratic KV cache memory growth and slow inference. STTM addresses this through principled token reduction that exploits video structure: spatial redundancy within frames and temporal redundancy across frames. Unlike query-aware methods that cannot reuse cached computations across questions, STTM is query-agnostic, enabling efficient KV cache reuse in multi-turn conversations.

The method uses quadtree-based hierarchical search for spatial merging (preserving fine details where needed) and union-find for efficient temporal merging. At 50% token reduction, accuracy drops only 0.5% while achieving 2× speedup.

## Core Concept

Video is inherently redundant: regions of uniform color or texture don't need individual tokens, and consecutive frames have high similarity. STTM exploits both by (1) merging spatially uniform tokens within each frame via quadtree decomposition, and (2) merging temporally similar tokens across frames. Critically, this is done without query information, so the same compressed representation works for any downstream question.

The method operates early in the LLM pipeline (typically layer 3 for 7B models), replacing expensive token representations with merged tokens before costly attention operations.

## Architecture Overview

- **Quadtree Spatial Merging**: Hierarchical frame decomposition with adaptive granularity
- **Temporal Merging**: Identifies similar tokens across consecutive frames, chains them in graphs
- **Token Reordering**: Z-shaped spatial ordering with temporal precedence for coherent linearization
- **Positional Embedding Adjustment**: Handles merged tokens in the positional encoding space
- **Insertion Point**: Early LLM layers (typically layer 3) before expensive multi-head attention
- **KV Cache Reuse**: Same merged tokens work across multiple questions without regeneration

## Implementation

### Step 1: Build Quadtree for Spatial Token Merging

Construct a quadtree for each frame where nodes represent token groups at different spatial granularities:

```python
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple

class QuadTreeNode:
    def __init__(self, x_min: int, y_min: int, x_max: int, y_max: int,
                 tokens: torch.Tensor, depth: int = 0):
        self.x_min, self.y_min = x_min, y_min
        self.x_max, self.y_max = x_max, y_max
        self.tokens = tokens  # Tokens in this region
        self.depth = depth
        self.children = []

    def get_merged_token(self) -> torch.Tensor:
        """Get representative token for this region (mean pooling)."""
        return self.tokens.mean(dim=0, keepdim=True)

def build_spatial_quadtree(frame_tokens: torch.Tensor,
                          similarity_threshold: float = 0.9) -> QuadTreeNode:
    """
    Build quadtree for frame tokens with adaptive granularity.
    Merges tokens in regions of high similarity; splits diverse regions.

    frame_tokens: [H, W, D] tensor of token embeddings
    """
    h, w, d = frame_tokens.shape

    def recursive_build(x_min, y_min, x_max, y_max, depth=0):
        """Recursively build tree, splitting diverse regions."""
        region = frame_tokens[y_min:y_max, x_min:x_max]
        region_flat = region.reshape(-1, d)

        # Check if region is uniform (similar tokens)
        if region_flat.shape[0] <= 1:  # Single token, leaf node
            return QuadTreeNode(x_min, y_min, x_max, y_max,
                               region_flat.unsqueeze(0), depth)

        # Compute pairwise similarity
        sim = F.cosine_similarity(
            region_flat.unsqueeze(1),  # [N, 1, D]
            region_flat.unsqueeze(0),  # [1, N, D]
            dim=-1
        )  # [N, N]

        # If all tokens in region are similar, merge
        mean_sim = sim.mean().item()
        if mean_sim >= similarity_threshold or depth >= 3:  # Max depth
            merged = region_flat
            return QuadTreeNode(x_min, y_min, x_max, y_max, merged, depth)

        # Otherwise, split into 4 quadrants and recurse
        mid_x = (x_min + x_max) // 2
        mid_y = (y_min + y_max) // 2

        node = QuadTreeNode(x_min, y_min, x_max, y_max, region_flat, depth)

        # Recursively build children
        node.children.append(recursive_build(x_min, y_min, mid_x, mid_y, depth + 1))
        node.children.append(recursive_build(mid_x, y_min, x_max, mid_y, depth + 1))
        node.children.append(recursive_build(x_min, mid_y, mid_x, y_max, depth + 1))
        node.children.append(recursive_build(mid_x, mid_y, x_max, y_max, depth + 1))

        return node

    root = recursive_build(0, 0, w, h)
    return root

def extract_quadtree_tokens(root: QuadTreeNode) -> Tuple[torch.Tensor, List[Tuple]]:
    """Extract merged tokens and their spatial information from tree."""
    tokens = []
    spatial_info = []  # (x, y, size) for each token

    def traverse(node):
        if not node.children:  # Leaf node
            merged = node.get_merged_token()
            tokens.append(merged)
            cx = (node.x_min + node.x_max) / 2
            cy = (node.y_min + node.y_max) / 2
            size = (node.x_max - node.x_min + node.y_max - node.y_min) / 2
            spatial_info.append((cx, cy, size))
        else:
            for child in node.children:
                traverse(child)

    traverse(root)
    return torch.cat(tokens, dim=0), spatial_info
```

### Step 2: Temporal Token Merging Across Frames

Identify similar tokens across consecutive frames and merge them, chaining tokens in graphs:

```python
class UnionFind:
    """Union-Find data structure for efficient temporal merging."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

def merge_tokens_temporal(frame_tokens_list: List[torch.Tensor],
                         temporal_threshold: float = 0.85) -> Dict:
    """
    Merge tokens across consecutive frames using union-find.
    Returns mapping of frame-local tokens to merged representatives.
    """
    num_frames = len(frame_tokens_list)

    # Build spatial trees for each frame
    trees = [build_spatial_quadtree(ft) for ft in frame_tokens_list]
    frames_tokens = []
    for tree in trees:
        tokens, _ = extract_quadtree_tokens(tree)
        frames_tokens.append(tokens)

    # Union-Find for token grouping
    total_tokens = sum(t.shape[0] for t in frames_tokens)
    uf = UnionFind(total_tokens)

    token_idx = 0
    frame_token_map = {}  # Map frame_id -> local_token_id -> global_id

    for frame_id, tokens in enumerate(frames_tokens):
        frame_token_map[frame_id] = list(range(token_idx, token_idx + tokens.shape[0]))
        token_idx += tokens.shape[0]

    # Temporal merging: compare adjacent frames
    for frame_id in range(num_frames - 1):
        tokens_curr = frames_tokens[frame_id]
        tokens_next = frames_tokens[frame_id + 1]

        # Compute similarity between all pairs
        sim = F.cosine_similarity(
            tokens_curr.unsqueeze(1),  # [N_curr, 1, D]
            tokens_next.unsqueeze(0),  # [1, N_next, D]
            dim=-1
        )  # [N_curr, N_next]

        # Greedy matching: each token in frame_curr matches at most one in frame_next
        for i in range(tokens_curr.shape[0]):
            j = torch.argmax(sim[i]).item()
            if sim[i, j] >= temporal_threshold:
                global_i = frame_token_map[frame_id][i]
                global_j = frame_token_map[frame_id + 1][j]
                uf.union(global_i, global_j)

    # Build merged token groups
    merged_groups = {}
    for i in range(total_tokens):
        root = uf.find(i)
        if root not in merged_groups:
            merged_groups[root] = []
        merged_groups[root].append(i)

    return merged_groups, frame_token_map

def get_merged_tokens(frames_tokens: List[torch.Tensor],
                     merged_groups: Dict) -> torch.Tensor:
    """Compute final merged token representations."""
    all_tokens = torch.cat(frames_tokens, dim=0)

    merged_tokens = []
    for root_idx in sorted(merged_groups.keys()):
        indices = merged_groups[root_idx]
        merged = all_tokens[indices].mean(dim=0)
        merged_tokens.append(merged)

    return torch.stack(merged_tokens) if merged_tokens else torch.tensor([])
```

### Step 3: Token Reordering with Positional Embeddings

Reorder merged tokens in a coherent pattern (Z-curve spatial + temporal precedence) and adjust positional embeddings:

```python
def reorder_tokens_zorder(spatial_info: List[Tuple],
                         temporal_order: List[int]) -> List[int]:
    """
    Reorder tokens in Z-order (Morton order) within temporal precedence.
    Z-order preserves spatial locality for attention efficiency.
    """
    def xy_to_z(x, y):
        """Convert 2D coordinates to Z-order curve index."""
        z = 0
        for i in range(16):  # 16-bit precision
            z |= ((int(x) >> i) & 1) << (2 * i)
            z |= ((int(y) >> i) & 1) << (2 * i + 1)
        return z

    # Sort tokens by (temporal_frame, z_order)
    sorted_indices = sorted(
        range(len(spatial_info)),
        key=lambda i: (temporal_order[i], xy_to_z(spatial_info[i][0],
                                                   spatial_info[i][1]))
    )

    return sorted_indices

def adjust_positional_embeddings(merged_token_count: int,
                                original_seq_length: int) -> torch.Tensor:
    """
    Adjust positional embeddings for merged tokens.
    Interpolate from original position space to compressed space.
    """
    pos_orig = torch.arange(original_seq_length).float()
    pos_merged = torch.linspace(0, original_seq_length - 1,
                                merged_token_count)

    # Interpolate embeddings (example using linear interpolation)
    pos_embeddings_merged = torch.nn.functional.interpolate(
        pos_orig.unsqueeze(0).unsqueeze(0),
        size=merged_token_count,
        mode='linear',
        align_corners=False
    ).squeeze()

    return pos_embeddings_merged
```

### Step 4: Insert into LLM Early Layers

Apply token merging early in the model pipeline (typically layer 3-4 for 7B models) to maximize speedup:

```python
import torch.nn as nn

def apply_sttm_to_video_llm(model: nn.Module,
                           video_frames: torch.Tensor,
                           insertion_layer: int = 3) -> torch.Tensor:
    """
    Apply STTM: merge video tokens before processing through LLM layers.
    video_frames: [num_frames, H, W, D] tensor
    """
    num_frames = video_frames.shape[0]

    # Step 1: Spatial merging (quadtree per frame)
    frame_tokens_list = []
    spatial_info_list = []

    for frame_id in range(num_frames):
        frame = video_frames[frame_id]  # [H, W, D]
        tree = build_spatial_quadtree(frame)
        tokens, spatial_info = extract_quadtree_tokens(tree)
        frame_tokens_list.append(tokens)
        spatial_info_list.append(spatial_info)

    # Step 2: Temporal merging
    merged_groups, frame_token_map = merge_tokens_temporal(frame_tokens_list)
    merged_tokens = get_merged_tokens(frame_tokens_list, merged_groups)

    # Step 3: Reorder tokens
    temporal_order = []
    for frame_id, local_tokens in enumerate(frame_token_map.values()):
        temporal_order.extend([frame_id] * len(local_tokens))

    reordered_indices = reorder_tokens_zorder(spatial_info_list[0],
                                              temporal_order)

    # Apply reordering
    merged_tokens_reordered = merged_tokens[reordered_indices]

    # Step 4: Adjust positional embeddings
    original_length = sum(t.shape[0] for t in frame_tokens_list)
    pos_embeddings = adjust_positional_embeddings(
        merged_tokens_reordered.shape[0],
        original_length
    )

    # Add positional embeddings
    merged_tokens_with_pos = merged_tokens_reordered + pos_embeddings.unsqueeze(-1)

    return merged_tokens_with_pos

# Example integration with LLM
def forward_with_sttm(model: nn.Module,
                     video_input: torch.Tensor,
                     text_input: torch.Tensor,
                     compression_ratio: float = 0.5,
                     insertion_layer: int = 3):
    """Forward pass with STTM token merging applied at insertion_layer."""

    # Tokenize video
    video_tokens = model.encode_video(video_input)

    # Apply STTM
    video_tokens_merged = apply_sttm_to_video_llm(
        model, video_tokens, insertion_layer
    )

    # Tokenize text
    text_tokens = model.encode_text(text_input)

    # Concatenate
    combined_tokens = torch.cat([video_tokens_merged, text_tokens], dim=0)

    # Pass through LLM layers (with insertion_layer applying merging)
    output = model.lm_head(combined_tokens)

    return output
```

## Practical Guidance

| Parameter | Recommended Value | Notes |
|---|---|---|
| Spatial Similarity Threshold | 0.9 | Higher = more aggressive merging |
| Temporal Similarity Threshold | 0.85 | Controls frame-to-frame merging |
| Quadtree Max Depth | 3 | Limits minimum merge region size |
| Token Reduction Ratio | 50% | 0.5× = 2× speedup with <0.5% accuracy drop |
| Insertion Layer | 3-4 (7B models) | Optimal balance of speedup vs quality |
| KV Cache Saving | Up to 2× | Enabled by query-agnostic design |
| Z-Order Bit Precision | 16 bits | Sufficient for spatial coherence |
| Positional Embedding | Linear interpolation | Smooth adaptation to merged length |

**When to use STTM:**
- Video QA requiring real-time or near-real-time inference
- Multi-turn video conversations where KV cache reuse matters
- Scenarios with limited GPU memory (KV cache is bottleneck)
- Applications processing many short video clips in batches
- When maintaining high accuracy is critical (0.5% loss at 50% reduction)

**When NOT to use STTM:**
- Extremely fine-grained tasks needing every pixel (medical imaging)
- Very short videos (< 5 frames) where merging overhead dominates
- Scenarios requiring exact token-level interpretability
- Tasks where temporal alignment is critical (action localization)

**Common pitfalls:**
- Similarity thresholds too high (0.95+), preventing merging
- Similarity thresholds too low (0.7), over-aggressive merging
- Quadtree max depth too shallow, losing spatial details
- Inserting merging too late (layer 10+), missing speedup benefits
- Not handling edge cases (single-token regions, empty frames)
- Forgetting to synchronize positional embeddings with merged tokens
- KV cache not actually reused across questions (implementation issue)

## Reference

Li, C., Zhang, X., Wu, Y., & Tan, M. (2025). Multi-Granular Spatio-Temporal Token Merging for Training-Free Acceleration of Video LLMs. arXiv:2507.07990. https://arxiv.org/abs/2507.07990
