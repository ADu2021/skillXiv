---
name: agent-ocr-history-compression
title: "AgentOCR: Reimagining Agent History via Optical Self-Compression"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.04786"
keywords: [agent-efficiency, history-compression, token-optimization, visual-encoding, long-context]
description: "Compress agent interaction history by converting observation-action sequences into compact visual representations (images), leveraging visual tokens' superior information density. Implements segment optical caching with 20x rendering speedup and enables dynamic compression rates. Preserves over 95% of agent performance while reducing token consumption by 50%+, enabling agents to maintain longer interaction histories within fixed budgets."
---

## Problem

Agent interaction histories grow rapidly, creating bottlenecks:

1. **Token Explosion**: Long sequences of observations and actions consume enormous token counts
2. **Memory Pressure**: Multi-turn agent executions accumulate context that exceeds token limits
3. **Computational Cost**: Each token in history requires reprocessing in forward passes
4. **Limited Horizon**: Token constraints force agents to forget recent history or truncate interactions
5. **Inefficient Encoding**: Text-based history is redundant (JSON records, timestamped logs)

Agents need ways to compress histories without losing critical information for decision-making.

## Solution

**AgentOCR** introduces **Optical Self-Compression** for agent histories:

1. **Visual Representation**: Convert observation-action sequences into rendered images that capture state visually
   - Renders agent state, actions taken, and outcomes as structured visual layouts
   - Leverages fact that vision models process visual information with higher information density than text
2. **Segment Optical Caching**: Decomposes history into hashable segments with visual cache
   - Segments: [state → action → outcome] → render to image
   - Cache: map segment hash → cached image rendering
   - 20x speedup through cache hits and vectorized rendering
3. **Agentic Self-Compression**: Agent learns to dynamically emit compression rates
   - Trade-off: compress aggressively to preserve computation budget, or maintain detail for critical decisions
   - Agent optimizes compression rate given current task demands

## When to Use

- **Long-Horizon Agents**: Tasks spanning 100+ interaction steps
- **Token-Constrained Deployment**: Systems with fixed token budgets (mobile, inference servers)
- **Multi-Turn Applications**: Dialogue agents, interactive tools with extended conversations
- **Expensive Computation**: GPU-limited systems where per-token cost matters
- **History-Heavy Reasoning**: Agents whose future decisions depend on cumulative history

## When NOT to Use

- For short-horizon tasks (compression overhead exceeds savings)
- When exact textual history is required for auditing
- In systems with unlimited token budgets
- For tasks where visual compression might lose critical information

## Core Concepts

The framework operates on the principle that **visual encoding is information-dense**:

1. **Visual > Text**: Images compress agent states more efficiently than textual logs
2. **Segment Caching**: Repeated state patterns can be cached and reused
3. **Dynamic Compression**: Agents learn when to compress aggressively vs. preserve detail
4. **Performance Preservation**: Aggressive compression doesn't significantly harm agent performance

## Key Implementation Pattern

Implementing agent history compression:

```python
# Conceptual: optical self-compression for agent histories
class CompressedAgentMemory:
    def __init__(self):
        self.segments = []        # [state, action, outcome] tuples
        self.visual_cache = {}    # hash -> rendered image

    def record_step(self, state, action, outcome):
        segment = (state, action, outcome)

        # Compute segment hash for caching
        segment_hash = hash(segment)

        # Render or retrieve from cache
        if segment_hash in self.visual_cache:
            visual = self.visual_cache[segment_hash]
        else:
            visual = self.render_segment(segment)
            self.visual_cache[segment_hash] = visual

        self.segments.append({
            'segment': segment,
            'visual': visual,
            'hash': segment_hash
        })

    def compress_history(self, compression_rate):
        """
        compression_rate: 0.0 (no compression) to 1.0 (maximum)
        """
        if compression_rate == 0.0:
            return self.segments  # Full history as text

        # Sample segments based on importance
        num_keep = int(len(self.segments) * (1 - compression_rate))
        important_segments = self.select_important(num_keep)

        # Convert kept segments to visuals
        compressed = [seg['visual'] for seg in important_segments]

        return compressed
```

Key mechanisms:
- Segment rendering: state → structured visual layout
- Hash-based caching: avoid re-rendering duplicate states
- Importance sampling: prioritize critical decision points
- Dynamic compression: agent chooses compression rate

## Expected Outcomes

- **50%+ Token Reduction**: Compressed visual histories use half the tokens
- **95%+ Performance Retention**: Agent task performance barely degrades despite compression
- **20x Rendering Speedup**: Caching eliminates redundant visual rendering
- **Longer Horizons**: Fixed token budget enables more interaction steps
- **Flexible Trade-Offs**: Agents adapt compression dynamically

## Limitations and Considerations

- Visual rendering requires computational overhead (though caching mitigates)
- Some information loss inevitable; compression rate must be tuned per task
- Visual encoding assumes vision-capable agents (text-only models don't benefit)
- Cached segments may become stale if environment state representation changes

## Integration Pattern

For a long-horizon web search agent:

1. **Record Steps**: Each search iteration records [query, results, decision]
2. **Visual Encoding**: Render query-results-decision as structured image
3. **Cache Hit**: Repeated search patterns hit cache, skip rendering
4. **Compress When Needed**: As history grows, compress older segments visually
5. **Use Compressed History**: Future reasoning uses compressed visuals + recent text

This maintains decision-making quality while managing token budget.

## Compression Rate Tuning

Start with 0.3-0.5 compression rate for most tasks:
- 0.0-0.2: Minimal compression, preserve detail (critical decisions)
- 0.3-0.5: Moderate compression, good balance
- 0.6-0.8: Aggressive compression, tight token budgets
- 0.9+: Extreme compression, only for tasks tolerant of history loss

## Related Work Context

AgentOCR advances agent efficiency by recognizing that history storage and processing is a core bottleneck. By shifting from text-based logs to visual representations, it enables longer-horizon agents within fixed computational budgets. This infrastructure improvement indirectly supports more complex agent behavior.
