---
name: docdancer-document-agent
title: "DocDancer: Towards Agentic Document-Grounded Information Seeking"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.05163"
keywords: [Document Question Answering, Agent Design, Information Seeking, Synthetic Data Generation]
description: "Build open-source agents for document question-answering by modeling DocQA as information-seeking with explicit tool utilization. DocDancer uses an exploration-then-synthesis pipeline to generate high-quality training data, addressing the scarcity that limits agent-based document understanding systems."
---

## When to Use This Skill
- Document question-answering systems with limited training data
- Applications requiring tool-driven exploration (highlighting, extracting, reasoning)
- Long-document understanding where sequential processing is necessary
- Scenarios where synthetic data generation can reduce annotation costs
- Building open-source DocQA agents without proprietary models

## When NOT to Use This Skill
- Single-turn simple fact lookup (standard retrieval sufficient)
- Applications with abundant labeled DocQA training data
- Systems with hard latency constraints (multi-step reasoning is slower)
- Short-document scenarios (tool exploration adds overhead)

## Problem Summary
Existing document question-answering (DocQA) agents suffer from two critical limitations: (1) they lack effective tool utilization, relying on implicit understanding instead of explicit document exploration, and (2) they depend heavily on closed-source models, limiting accessibility and adaptability. The fundamental barrier is scarcity of high-quality training data for DocQA agents—annotation is expensive and difficult at scale.

## Solution: Tool-Driven DocQA with Synthetic Data

Model DocQA as information-seeking with explicit tool integration, then generate synthetic training data through an exploration-then-synthesis pipeline.

```python
class DocDancerAgent:
    def __init__(self, base_llm, document):
        self.llm = base_llm
        self.document = document
        self.interaction_history = []

    def answer_document_question(self, question):
        """Tool-driven exploration followed by answer synthesis"""

        # Phase 1: Exploration
        exploration_steps = self.explore_document(question)
        # exploration_steps = [
        #     {"action": "highlight", "text": "...", "rationale": "..."},
        #     {"action": "extract", "content": "...", "rationale": "..."},
        #     {"action": "reason", "inference": "...", "rationale": "..."}
        # ]

        # Phase 2: Synthesis
        answer = self.synthesize_answer(question, exploration_steps)

        self.interaction_history.append({
            "question": question,
            "exploration": exploration_steps,
            "answer": answer
        })

        return answer

    def explore_document(self, question):
        """Sequential tool invocation for information gathering"""
        steps = []
        context = f"Question: {question}\nDocument: {self.document[:2000]}..."

        for exploration_turn in range(max_exploration_steps):
            # Decide which tool to use next
            tool_decision = self.llm.generate(f"""
            Current exploration state:
            {format_exploration_history(steps)}

            Question: {question}

            What's the next exploration action?
            Options:
            - highlight: Mark important text regions
            - extract: Pull out specific information
            - reason: Make inference from gathered info
            - stop: Sufficient information gathered
            """)

            action = parse_tool_action(tool_decision)

            if action == "stop":
                break

            # Execute chosen tool
            if action == "highlight":
                highlighted_text = self.identify_relevant_sections(question, context)
                steps.append({
                    "action": "highlight",
                    "text": highlighted_text,
                    "rationale": tool_decision
                })
            elif action == "extract":
                extracted_content = self.extract_key_information(question, context)
                steps.append({
                    "action": "extract",
                    "content": extracted_content,
                    "rationale": tool_decision
                })
            elif action == "reason":
                inference = self.llm.generate(f"""
                Based on gathered evidence:
                {format_exploration_steps(steps)}

                Make an inference relevant to: {question}
                """)
                steps.append({
                    "action": "reason",
                    "inference": inference,
                    "rationale": tool_decision
                })

        return steps

    def synthesize_answer(self, question, exploration_steps):
        """Combine exploration traces into final answer"""
        synthesis_prompt = f"""
        Question: {question}

        Exploration process:
        {format_exploration_steps(exploration_steps)}

        Based on this exploration, provide the final answer.
        """
        return self.llm.generate(synthesis_prompt)
```

## Key Implementation Details

**Tool-Driven Architecture:**
- Explicit tools: highlight, extract, reason, summarize
- Sequential tool invocation with history tracking
- Each tool provides interpretable signals for debugging

**Exploration-Then-Synthesis Pipeline:**
Generates high-quality synthetic training data:

```python
def generate_synthetic_training_data(document, gold_answer, num_samples=100):
    """Generate diverse question-exploration-answer triplets"""
    synthetic_data = []

    for sample_idx in range(num_samples):
        # Generate question variants that require document exploration
        question = generate_question_from_answer(gold_answer, document)

        # Simulate diverse exploration strategies
        exploration_trajectories = []
        for strategy in ["sequential", "selective", "comprehensive"]:
            trajectory = simulate_exploration(
                question, document, gold_answer, strategy
            )
            exploration_trajectories.append(trajectory)

        # Create training examples from best exploration
        best_trajectory = select_best_trajectory(
            exploration_trajectories, gold_answer
        )

        synthetic_data.append({
            "question": question,
            "document": document,
            "exploration": best_trajectory,
            "answer": gold_answer
        })

    return synthetic_data
```

## Training Data Methodology

**Data Scarcity Problem:**
- MMLongBench-Doc: Limited labeled examples
- Manual annotation is expensive
- Diversity of document types and questions is limited

**Solution: Synthetic Generation**
- Start with reference answers
- Generate plausible questions backward
- Simulate exploration trajectories
- Create diverse exploration styles
- Output: 100x more training examples than labeled data

## Performance Evaluation

**Benchmarks:**
- MMLongBench-Doc: Multimodal document QA
- DocBench: Document understanding across domains

**Comparison:**
- Outperforms closed-source baselines
- Open-source implementation enables reproducibility
- Tool-explicit approach provides interpretability

## Advantages Over Baselines

- **vs. LLM-Only**: Tool-driven exploration prevents hallucination
- **vs. Closed-Source**: Open-source enables customization
- **vs. Retrieval-Only**: Reasoning capability over multi-hop questions
- **vs. Unaided Agents**: Synthetic data provides training signal

## Deployment Strategy

1. **Document Processing**: Load and preprocess target documents
2. **Tool Integration**: Implement highlight, extract, reason tools
3. **Data Generation**: Create synthetic training examples
4. **Agent Training**: Fine-tune on exploration-answer pairs
5. **Evaluation**: Test on held-out document QA tasks
6. **Iteration**: Analyze errors to improve exploration strategies

## Open-Source Implementation
Full codebase released with:
- Synthetic data generation scripts
- Tool implementations
- Training utilities for benchmark datasets
- Evaluation metrics
