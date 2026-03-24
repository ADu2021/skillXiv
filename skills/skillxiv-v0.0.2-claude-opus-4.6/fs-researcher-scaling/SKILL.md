---
name: fs-researcher-scaling
title: "FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.01566"
keywords: [Test-Time Scaling, External Memory, Multi-Agent, Long-Horizon Reasoning, Knowledge Accumulation]
description: "Scale research agent capability using persistent filesystem as external memory. Dual-agent architecture with context builder accumulating knowledge and report writer composing outputs enables computation scaling beyond context windows."
---

# FS-Researcher: Test-Time Scaling via External Memory

## Problem
Research tasks require accumulating knowledge far exceeding model context windows. Deep research needs iterative refinement, synthesis across sources, and integration with external findings.

Single-pass generation cannot match quality of human iterative research. Standard scaling approaches hit context limits.

## Core Concept
FS-Researcher uses the filesystem as durable external memory and coordination medium. A Context Builder agent accumulates knowledge by web browsing and extracting information; a Report Writer composes final output using the persistent knowledge base.

Allocating more computation rounds to the Context Builder directly improves downstream report quality, demonstrating effective test-time scaling.

## Architecture Overview

- **Context Builder Agent**: Web browsing, information extraction, hierarchical knowledge base construction
- **Report Writer Agent**: Section-by-section composition using persistent knowledge base
- **Filesystem Coordination**: Shared external memory enabling cross-session persistence
- **Hierarchical Storage**: Organize findings into nested structures
- **Iterative Refinement**: Multiple rounds of context building improve knowledge base quality
- **Section-by-Section Writing**: Incremental report composition with knowledge base lookups

## Implementation

### Step 1: Build Knowledge Base Storage
Create hierarchical filesystem structure for knowledge accumulation.

```python
import os
import json
from pathlib import Path

class KnowledgeBaseStorage:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def add_finding(self, topic, finding_text, source_url, metadata=None):
        """Store a research finding."""
        topic_path = self.base_path / topic.replace(' ', '_')
        topic_path.mkdir(parents=True, exist_ok=True)

        # Create structured finding record
        finding = {
            'text': finding_text,
            'source': source_url,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        # Append to topic findings
        findings_file = topic_path / 'findings.jsonl'
        with open(findings_file, 'a') as f:
            f.write(json.dumps(finding) + '\n')

    def search_findings(self, topic, query):
        """Search findings within a topic."""
        topic_path = self.base_path / topic.replace(' ', '_')
        findings_file = topic_path / 'findings.jsonl'

        if not findings_file.exists():
            return []

        results = []
        with open(findings_file, 'r') as f:
            for line in f:
                finding = json.loads(line)
                if query.lower() in finding['text'].lower():
                    results.append(finding)

        return results

    def get_all_findings_for_topic(self, topic):
        """Retrieve all findings for a topic."""
        topic_path = self.base_path / topic.replace(' ', '_')
        findings_file = topic_path / 'findings.jsonl'

        if not findings_file.exists():
            return []

        findings = []
        with open(findings_file, 'r') as f:
            for line in f:
                findings.append(json.loads(line))

        return findings
```

### Step 2: Context Builder Agent
Systematically accumulate knowledge through web search and extraction.

```python
def context_builder_agent(research_query, kb_storage, search_engine, num_rounds=3):
    """Iteratively expand knowledge base for research topic."""
    for round_num in range(num_rounds):
        # Generate search queries based on current knowledge
        current_findings = kb_storage.get_all_findings_for_topic(research_query)

        if round_num == 0:
            # Initial searches
            search_queries = generate_initial_queries(research_query)
        else:
            # Refinement searches based on gaps
            gaps = identify_knowledge_gaps(current_findings)
            search_queries = generate_refinement_queries(gaps)

        # Search and extract
        for query in search_queries:
            results = search_engine.search(query)

            for result in results[:5]:  # Top 5 results
                # Extract key information
                extracted_text = extract_key_info(result['content'])

                # Store finding
                kb_storage.add_finding(
                    topic=research_query,
                    finding_text=extracted_text,
                    source_url=result['url'],
                    metadata={'round': round_num, 'query': query}
                )

    return kb_storage
```

### Step 3: Report Writer Agent
Compose research report using knowledge base.

```python
def report_writer_agent(research_query, kb_storage, report_model, num_sections=5):
    """Write comprehensive research report section-by-section."""
    # Generate section outlines
    all_findings = kb_storage.get_all_findings_for_topic(research_query)
    section_topics = generate_section_outline(research_query, len(all_findings))

    report = ""

    for section_idx, section_topic in enumerate(section_topics):
        # Find relevant findings for this section
        relevant_findings = kb_storage.search_findings(research_query, section_topic)

        # Synthesize findings into section context
        context = synthesize_findings(relevant_findings)

        # Generate section
        section_prompt = f"""Write a well-organized research section on: {section_topic}

Supporting findings:
{context}

Write a comprehensive, well-sourced section:"""

        section_text = report_model.generate(section_prompt, max_tokens=800)

        # Append section with citations
        report += f"\n## {section_topic}\n{section_text}\n"

    return report
```

### Step 4: Test-Time Scaling Integration
Verify that additional context building rounds improve output quality.

```python
def evaluate_scaling_correlation(research_query, kb_storage, report_model, max_rounds=5):
    """Validate positive correlation between computation and quality."""
    quality_scores = []

    for num_rounds in range(1, max_rounds + 1):
        # Build with varying rounds
        kb_storage = context_builder_agent(research_query, kb_storage, num_rounds=num_rounds)

        # Generate report
        report = report_writer_agent(research_query, kb_storage, report_model)

        # Evaluate quality
        score = evaluate_report_quality(report, research_query)
        quality_scores.append({
            'rounds': num_rounds,
            'quality_score': score
        })

    # Verify positive trend
    correlation = compute_correlation([s['rounds'] for s in quality_scores],
                                      [s['quality_score'] for s in quality_scores])

    return quality_scores, correlation > 0
```

## Practical Guidance

### Hyperparameter Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Context builder rounds | 3-10 | More rounds improve knowledge base |
| Search results per query | 5-10 | Balance coverage and cost |
| Extraction chunk size | 200-500 tokens | Manageable summary units |
| Report sections | 4-8 | Typical research paper structure |
| Knowledge base retention | All findings | No pruning needed |

### When to Use

- Deep research tasks requiring comprehensive knowledge
- Multi-source synthesis (literature reviews, market research)
- Long-horizon fact-finding where quality improves iteratively
- Knowledge-intensive tasks exceeding context windows
- Systems with persistent storage available

### When Not to Use

- Single-source lookups (web search sufficient)
- Real-time tasks requiring immediate responses
- Highly dynamic domains where information becomes stale
- Environments without persistent storage
- Tasks with clear, bounded information needs

### Common Pitfalls

1. **Knowledge base bloat**: Unfiltered findings accumulate noise. Implement quality filters or deduplication.
2. **Search query repetition**: Multiple rounds may repeat searches. Track already-queried topics.
3. **Synthesis hallucination**: Report writer may invent citations. Require explicit source anchoring.
4. **Scaling plateau**: Improvement eventually plateaus. Validate diminishing returns and stop early.

## Reference
FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks
https://arxiv.org/abs/2602.01566
