---
name: superwriter-longform
title: "SuperWriter: Reflection-Driven Long-Form Generation with LLMs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.04180"
keywords: [long-form-generation, reflection, tree-search, direct-preference-optimization]
description: "Generate coherent, consistent long-form text through structured planning, hierarchical reflection, and Monte Carlo tree search-guided optimization."
---

# SuperWriter: Reflection-Driven Long-Form Generation

## Core Concept

SuperWriter-Agent demonstrates that long-form text generation quality improves through explicit structured thinking stages that mimic professional writing workflows. By combining hierarchical direct preference optimization with Monte Carlo tree search (MCTS) for propagating quality signals, a 7B SuperWriter model outperforms larger baselines across diverse long-form writing tasks.

## Architecture Overview

- **Structured Thinking Stages**: Decompose generation into planning, drafting, and refinement phases
- **Hierarchical Direct Preference Optimization (DPO)**: Use MCTS to propagate quality judgments from document level down to paragraph and sentence levels
- **Reflection Mechanism**: Generate internal critiques and improvement suggestions to guide refinement
- **SuperWriter-LM**: Fine-tuned 7B model trained on structured reasoning dataset
- **Goal**: Maintain coherence and logical consistency across extended sequences

## Implementation

### Step 1: Prepare Structured Thinking Dataset

```python
from typing import List, Dict
import json

class LongFormDatasetBuilder:
    def __init__(self):
        self.structured_templates = {
            'article': self.article_structure(),
            'report': self.report_structure(),
            'narrative': self.narrative_structure(),
        }

    def article_structure(self):
        """Extract structure from well-written articles"""
        return {
            'outline': 'Hierarchical topic breakdown',
            'introduction': 'Hook + thesis statement',
            'body_paragraphs': 'Topic sentence + evidence + analysis',
            'conclusion': 'Summary + forward thinking',
        }

    def build_structured_examples(self, raw_documents: List[str]):
        """Decompose long documents into thinking stages"""
        structured_data = []

        for doc in raw_documents:
            # Stage 1: Extract or generate outline
            outline = self.extract_outline(doc)

            # Stage 2: Identify structural elements
            structure = self.identify_structure(doc)

            # Stage 3: Create supervision signal
            example = {
                'original_document': doc,
                'outline': outline,
                'structure_analysis': structure,
                'thinking_chain': self.generate_thinking_chain(doc),
                'coherence_score': self.assess_coherence(doc),
            }

            structured_data.append(example)

        return structured_data

    def generate_thinking_chain(self, document: str) -> Dict:
        """Create explicit reasoning about document construction"""
        return {
            'planning': {
                'topic': self.extract_main_topic(document),
                'key_points': self.extract_key_points(document),
                'target_audience': self.infer_audience(document),
                'scope': self.estimate_scope(document),
            },
            'drafting': {
                'main_claims': self.extract_claims(document),
                'supporting_evidence': self.extract_evidence(document),
                'flow_transitions': self.identify_transitions(document),
            },
            'refinement': {
                'clarity_improvements': self.suggest_clarity_fixes(document),
                'consistency_checks': self.find_inconsistencies(document),
                'polish': self.suggest_style_improvements(document),
            }
        }

# Build dataset with thinking annotations
builder = LongFormDatasetBuilder()
structured_dataset = builder.build_structured_examples(documents)
```

### Step 2: Implement Multi-Stage Generation

```python
class SuperWriterAgent:
    def __init__(self, base_model_name='Qwen-7B'):
        self.model = load_model(base_model_name)
        self.planner = self.PlanningModule()
        self.drafter = self.DraftingModule()
        self.reflector = self.ReflectionModule()

    class PlanningModule:
        def generate_outline(self, topic: str, target_length: int) -> Dict:
            """Stage 1: Generate structured outline"""
            prompt = f"""Given topic: {topic}
Target length: {target_length} words

Generate a detailed outline with:
1. Main thesis
2. Key sections (3-5)
3. Key points per section (2-3 each)
4. Logical flow between sections

Format as nested structure."""

            outline = generate_completion(prompt)
            return self.parse_outline(outline)

        def parse_outline(self, outline_text: str) -> Dict:
            """Structure outline into actionable sections"""
            sections = []
            current_section = None

            for line in outline_text.split('\n'):
                if line.startswith('1.'):
                    current_section = {'title': line[2:], 'points': []}
                elif line.startswith('  -'):
                    current_section['points'].append(line[3:])
                elif current_section:
                    sections.append(current_section)

            return {'sections': sections}

    class DraftingModule:
        def draft_from_outline(self, outline: Dict, topic: str) -> str:
            """Stage 2: Generate initial draft following outline"""
            draft_text = ""

            # Generate introduction
            intro_prompt = f"Write introduction for: {topic}\nThesis: {outline['thesis']}"
            introduction = generate_completion(intro_prompt)
            draft_text += introduction + "\n\n"

            # Generate body sections
            for section in outline['sections']:
                section_prompt = f"""Write section: {section['title']}
Key points to cover:
{chr(10).join(f"- {p}" for p in section['points'])}

Maintain coherence with previous content."""

                section_text = generate_completion(section_prompt)
                draft_text += section_text + "\n\n"

            # Generate conclusion
            conclusion_prompt = f"Write conclusion summarizing: {outline['thesis']}"
            conclusion = generate_completion(conclusion_prompt)
            draft_text += conclusion

            return draft_text

    class ReflectionModule:
        def generate_critique(self, draft_text: str) -> Dict:
            """Stage 3a: Introspect and identify weaknesses"""
            critique_prompt = f"""Review this draft:
{draft_text}

Identify:
1. Logical gaps or unclear transitions
2. Inconsistent arguments
3. Weak supporting evidence
4. Clarity issues
5. Structural problems"""

            critique = generate_completion(critique_prompt)

            return {
                'critique': critique,
                'severity_scores': self.score_issues(critique),
                'priority_fixes': self.prioritize_fixes(critique),
            }

        def score_issues(self, critique: str) -> Dict:
            """Rate severity of identified issues"""
            issues = {}

            for issue_type in ['logical_gaps', 'inconsistency', 'clarity']:
                score_prompt = f"Rate severity of {issue_type} issues: {critique}"
                score = extract_numeric_score(generate_completion(score_prompt))
                issues[issue_type] = score

            return issues

        def suggest_refinements(self, draft: str, critique: Dict) -> str:
            """Generate improved version based on critique"""
            refinement_prompt = f"""Original draft:
{draft}

Issues to address:
{critique['critique']}

Priority fixes:
{critique['priority_fixes']}

Generate improved version."""

            refined = generate_completion(refinement_prompt)
            return refined

    def generate_longform(self, topic: str, target_length: int = 2000) -> Dict:
        """Complete multi-stage generation pipeline"""

        # Stage 1: Planning
        outline = self.planner.generate_outline(topic, target_length)

        # Stage 2: Drafting
        draft = self.drafter.draft_from_outline(outline, topic)

        # Stage 3: Reflection and Refinement
        critique = self.reflector.generate_critique(draft)
        refined = self.reflector.suggest_refinements(draft, critique)

        return {
            'outline': outline,
            'draft': draft,
            'critique': critique,
            'refined': refined,
            'final': refined,  # Use refined version as final
        }
```

### Step 3: Implement Hierarchical DPO with MCTS

```python
import numpy as np
from collections import defaultdict

class HierarchicalDPOWithMCTS:
    def __init__(self, base_model, value_model):
        self.base_model = base_model
        self.value_model = value_model  # Evaluates quality at different levels

    def evaluate_document_quality(self, document: str) -> float:
        """Document-level quality assessment"""
        metrics = {
            'coherence': self.assess_coherence(document),
            'consistency': self.assess_consistency(document),
            'completeness': self.assess_completeness(document),
            'clarity': self.assess_clarity(document),
        }

        return np.mean(list(metrics.values()))

    def hierarchical_evaluation(self, document: str) -> Dict:
        """Multi-level quality scoring"""
        paragraphs = document.split('\n\n')
        sentences = [s for para in paragraphs for s in para.split('. ')]

        return {
            'document_score': self.evaluate_document_quality(document),
            'paragraph_scores': [self.assess_paragraph(p) for p in paragraphs],
            'sentence_scores': [self.assess_sentence(s) for s in sentences],
        }

    def mcts_preference_propagation(self, document_pair, max_iterations=100):
        """Use MCTS to propagate document-level preference down to paragraphs"""

        better_doc = self.select_better_document(document_pair)
        worse_doc = document_pair[1] if document_pair[0] == better_doc else document_pair[0]

        better_paras = better_doc.split('\n\n')
        worse_paras = worse_doc.split('\n\n')

        paragraph_preferences = []

        for b_para, w_para in zip(better_paras, worse_paras):
            # MCTS: simulate edits to understand which paragraph structure works better
            visit_count = defaultdict(int)
            value_sum = defaultdict(float)

            for iteration in range(max_iterations):
                # Simulate: progressively transform worse_para toward better_para structure
                simulated = self.simulate_paragraph_edit(w_para, b_para)

                # Evaluate quality
                quality = self.assess_paragraph(simulated)

                # Update statistics
                edit_signature = hash(simulated)
                visit_count[edit_signature] += 1
                value_sum[edit_signature] += quality

            # Select best structural transformation
            best_edit = max(visit_count.keys(),
                          key=lambda x: value_sum[x] / visit_count[x])

            paragraph_preferences.append({
                'better': b_para,
                'worse': w_para,
                'best_transformation': best_edit,
                'learned_preference': value_sum[best_edit] / visit_count[best_edit]
            })

        return paragraph_preferences

    def direct_preference_optimization_step(self, document_pair, optimizer):
        """DPO training step using hierarchical preferences"""

        # Get hierarchical preferences
        prefs = self.hierarchical_evaluation(document_pair[0])
        worse_prefs = self.hierarchical_evaluation(document_pair[1])

        # Propagate preferences via MCTS
        para_preferences = self.mcts_preference_propagation(document_pair)

        # Compute preference loss
        # Better doc should have higher model likelihood
        log_prob_better = self.base_model.log_probability(document_pair[0])
        log_prob_worse = self.base_model.log_probability(document_pair[1])

        # DPO loss: maximize relative preference
        dpo_loss = -np.log(
            torch.sigmoid(log_prob_better - log_prob_worse)
        )

        optimizer.zero_grad()
        dpo_loss.backward()
        optimizer.step()

        return dpo_loss.item()
```

## Practical Guidance

1. **Multi-Stage Decomposition**: Always split long-form generation into explicit stages: planning (outline), drafting (section by section), reflection (critique), refinement (editing based on critique).

2. **Enforce Outline Adherence**: Have the drafter explicitly reference outline sections to maintain logical flow. Check that each paragraph maps to outline points.

3. **Implement Reflection Loop**: Generate internal critiques before refinement. This "thinking aloud" about weaknesses significantly improves final output quality.

4. **MCTS for Quality Propagation**: Use Monte Carlo tree search to understand which paragraph-level structures align with document-level quality. This avoids needing paragraph-level human annotations.

5. **Hierarchical Optimization**: Train with preferences at multiple levels—document, paragraph, sentence—rather than just document-level scores. This helps internalize writing patterns at all scales.

6. **Target Model Size**: A 7B model with structured training outperforms much larger generalist models on long-form tasks. Quality of training process matters more than parameter count.

## Reference

- Paper: SuperWriter (2506.04180)
- Architecture: 7B SuperWriter-LM with planning, drafting, reflection modules
- Method: Hierarchical DPO with MCTS preference propagation
- Key Insight: Explicit structured thinking improves long-form generation coherence
