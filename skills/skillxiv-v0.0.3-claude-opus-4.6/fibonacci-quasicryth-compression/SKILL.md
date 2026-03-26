---
name: fibonacci-quasicryth-compression
title: "Aperiodic Structures Never Collapse: Fibonacci Hierarchies in Lossless Compression"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.14999"
keywords: [Lossless Compression, Quasicrystals, Hierarchical Coding, Mathematical Foundations, Algebraic Number Theory]
description: "Introduce Quasicryth, a text compressor using Fibonacci quasicrystal tilings for phrase-level compression. Prove that aperiodic structures never structurally collapse at depth, enabling compression at arbitrary hierarchy levels. Bridge quasicrystal mathematics with practical compression, achieving 22.59% ratio on enwik9 with unbounded scaling advantages."
---

# Aperiodic Structures in Lossless Compression

## Problem Formulation: Hierarchy Collapse

Traditional periodic tilings used in hierarchical compression have a fundamental limitation: they collapse structurally within O(log p) levels, where p is the period. Beyond that depth, one tile type vanishes, making further compression impossible.

**Example:** Binary hierarchies collapse at depth log₂(W) (word vocabulary size). The codebook loses structural variety, forcing fallback to escapes.

## New Paradigm: Aperiodic Fibonacci Tilings

Fibonacci quasicrystals offer a mathematically-proven alternative: structures that never collapse at any depth, enabling compression at arbitrarily deep hierarchy levels.

## Founding Mathematical Results

**Main Theorem (Aperiodic Hierarchy Advantage):**
The Fibonacci tiling uniquely satisfies five properties simultaneously:

1. **Non-Collapse at All Depths:** Both tile types persist indefinitely
   - Unlike periodic structures that collapse within O(log p) levels
   - Guaranteed by golden-ratio eigenvalue φ (Pisot-Vijayaraghavan number)

2. **Scale-Invariant Coverage:** Potential word coverage remains constant across hierarchy levels
   - ~0.724W words available at each level regardless of depth
   - Proven via Golden Compensation Theorem

3. **Maximum Codebook Efficiency:** Exactly Fm+1 distinct patterns at level m
   - Minimal for aperiodic sequences
   - Maximizes phrase reuse probability

4. **Bounded Overhead:** Flag entropy capped at 1/φ ≈ 0.618 bits/word
   - Constant regardless of hierarchy depth
   - Periodic structures have unbounded overhead growth

5. **Strict Entropy Advantage:** Lower per-word coding entropy than periodic alternatives for long-range sources
   - Exploits statistical correlations across deeper hierarchy levels

**Mathematical Framework:**

```
Substitution Matrix Analysis:
  σ: L → LS, S → L (Fibonacci substitution)
  Eigenvalue = φ (golden ratio)

Sturmian Sequences:
  Minimal factor complexity p(n) = n+1
  Maximum phrase reuse potential

Weyl Equidistribution:
  Irrational tiling slopes guarantee uniform density
  at all scales
```

## Practical Implementation: Quasicryth v5.6

**Hierarchy Configuration:**
- 10-level hierarchy with phrase lengths: {2, 3, 5, 8, 13, 21, 34, 55, 89, 144} words
- 36 distinct tilings:
  - 12 golden-ratio optimized
  - 6 non-golden variants
  - 18 optimized via greedy search

**Encoding Strategy:**
- Multi-level adaptive arithmetic coding with order-2 context models
- Separate LZMA-compressed escape stream for out-of-vocabulary words
- Lazy evaluation: activate deep levels only when shallow codebooks saturate

```python
# Quasicryth compression pipeline:
# Level 1 (2-word phrases): Most frequent, low cost
# Level 2 (3-word phrases): Medium frequency phrases
# ...
# Level 10 (144-word phrases): Rare deep patterns
#
# Each level maintains Fibonacci-structured tiling
# ensuring that phrase vocabularies never collapse
#
# Arithmetic coding casts: each level encoded
# with context-aware probability models trained
# on compressed training data
```

## Performance Results

**Compression on enwik9 (1 GB English Wikipedia):**
- **Achieved Ratio:** 22.59%
- **Deep Level Activation:** Levels 9-10 (89-gram, 144-gram phrases) uniquely activate here
  - 89-gram phrases: 5,369 hits
  - 144-gram phrases: 2,026 hits
- **Advantage Over Periodic Baseline:** 11+ million bytes saved vs Period-5 periodic tiling

**Scaling Property:**
As dataset size increases:
- Periodic tilings: compression ratio plateaus (hierarchy collapses)
- Fibonacci tilings: continuous improvement (new patterns activate at deeper levels)

## Why This Matters

**Bridge Between Mathematics and Practice:**

1. **Theoretical Grounding:** Proves that structural properties from quasicrystal geometry (physics/materials science) solve fundamental CS problems

2. **Unbounded Improvement:** Unlike fixed-depth hierarchies, Fibonacci structures keep improving with larger datasets—architectural advantage grows

3. **Generalization:** Same principle applies to:
   - Hierarchical tokenization in LLMs (variable-length codes)
   - Semantic hierarchies in knowledge graphs
   - Phylogenetic tree compression in biology

## Opened Research Directions

1. **Adaptation to Other Domains:** Can Fibonacci tilings improve hierarchical clustering, multi-level hashing, or tree-based indexing?

2. **Hybrid Approaches:** Combine Fibonacci hierarchies with modern neural compression or learned arithmetic coding

3. **Scaling Experiments:** Test on larger corpora (10+ GB) to validate unbounded scaling advantage

4. **Physical Quasicrystals Inspiration:** Other aperiodic structures (Penrose tilings, Thue-Morse sequences) may have complementary properties

## Vocabulary & Founding Concepts

- **Quasicrystal:** Aperiodic ordered structure without translational symmetry (discovered in materials; novel application here)
- **Pisot-Vijayaraghavan Number:** Algebraic integer with all conjugates <1; ensures stability (φ is PV number)
- **Sturmian Sequence:** Infinite binary sequence with minimal complexity; encodes Fibonacci tiling structure
- **Golden Compensation Theorem (Novel):** Exponential growth in phrase length (Fm~φ^m) exactly cancels exponential decrease in position counts (φ^-(m-1)), yielding level-independent reuse capacity
