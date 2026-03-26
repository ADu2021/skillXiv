---
name: bubblerag-evidence-driven-graphs
title: "BubbleRAG: Evidence-Driven Retrieval-Augmented Generation for Black-Box Knowledge Graphs"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2603.20309
keywords: [RAG, Knowledge Graphs, Evidence Retrieval, Black-Box Graphs, Hallucination Reduction]
description: "Address hallucinations in LLM QA over black-box knowledge graphs using evidence-driven retrieval. Formalize Optimal Informative Subgraph Retrieval and employ bubble expansion to discover candidate evidence graphs, achieving state-of-the-art multi-hop QA performance."
---

## Component ID
Bubble Expansion Evidence Retrieval for Black-Box Knowledge Graph Subgraph Selection

## Motivation
Retrieving relevant subgraphs from knowledge graphs with unknown schema and structure introduces three sources of uncertainty: semantic instantiation uncertainty (mapping entities to graph nodes), structural path uncertainty (finding multi-hop connection patterns), and evidential comparison uncertainty (ranking candidate subgraphs). Existing retrieval approaches fail when graph structure is opaque and require significant prior knowledge.

## The Modification
BubbleRAG formalizes the challenge as **Optimal Informative Subgraph Retrieval** and proposes a training-free pipeline combining:

1. **Heuristic Bubble Expansion** - Discover candidate evidence graphs (CEGs) through iterative neighborhood expansion around seed entities. Bubbles grow by following graph edges without knowledge of full schema, enabling exploration even with structural uncertainty.

2. **Semantic Anchor Grouping** - Ground semantic entities in graph nodes despite instantiation uncertainty, clustering related entities and enriching them with cross-references.

3. **Composite Ranking** - Rank candidate subgraphs using task-aware signals (relevance to question, coverage of reasoning paths) balancing recall and precision.

4. **Reasoning-Aware Expansion** - Prioritize expansion directions that align with the reasoning structure required for the question (multi-hop reasoning patterns).

## Ablation Results
The paper demonstrates:
- State-of-the-art performance on multi-hop question answering tasks
- Effective hallucination reduction through grounded evidence retrieval
- Performance maintained across diverse black-box knowledge graph structures
- Training-free pipeline achieves competitive or superior results to fine-tuned baselines
- Bubble expansion efficiently explores graph neighborhoods without schema knowledge

## Conditions
- Applicable to knowledge graphs with unknown or partially-known schema
- Works best when questions require multi-hop reasoning (2+ steps)
- Assumes entity mentions in questions are matchable to some graph node representation
- Training-free design minimizes data requirements but assumes graph is queryable at inference time

## Drop-In Checklist
- [ ] Implement semantic anchor detection and entity-to-node grounding
- [ ] Add bubble expansion algorithm for neighborhood exploration
- [ ] Create composite ranking function (relevance + coverage + reasoning alignment)
- [ ] Implement reasoning-aware expansion (prioritize task-relevant directions)
- [ ] Integrate evidence graph selection with downstream LLM QA
- [ ] Test on multi-hop QA benchmarks with black-box graphs
- [ ] Validate hallucination reduction compared to baseline RAG
