---
name: hop-skip-overthink-diagnosis
title: Hop Skip Overthink - Diagnosing Reasoning Models Multi-Hop
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.04699
keywords: [reasoning-diagnosis, error-analysis, hallucination, multi-hop-qa]
description: "Novel error categorization framework examining failures across hops (diversity), coverage, and overthinking. Combines human annotation with automated metrics to diagnose why reasoning models hallucinate on multi-step tasks."
---

# Hop Skip Overthink: Diagnosing Reasoning Models Multi-Hop

## Core Concept

Reasoning models hallucinate more on multi-hop questions than general models, yet root causes remain unclear. This work introduces a nuanced error categorization framework that goes beyond accuracy metrics to diagnose failure modes. By examining three dimensions—hop diversity/uniqueness, information coverage, and cognitive efficiency—the framework provides actionable guidance for improving reasoning model robustness.

## Architecture Overview

- **Hop Analysis**: Examines diversity and uniqueness of source documents used
- **Coverage Assessment**: Measures completeness in capturing relevant information
- **Overthinking Detection**: Identifies cognitive inefficiency in reasoning processes
- **Hybrid Evaluation**: Combines human annotation with automated metrics
- **Error Categorization**: Maps failures to specific improvement opportunities

## Implementation Steps

### Step 1: Build Error Categorization Framework

Create structured system for classifying reasoning failures.

```python
from enum import Enum
from typing import Dict, List, Tuple
from dataclasses import dataclass

class ErrorType(Enum):
    """Types of errors in reasoning models."""
    HOP_DIVERSITY = "hop_diversity"  # Using same sources repeatedly
    COVERAGE_GAP = "coverage_gap"     # Missing relevant information
    OVERTHINKING = "overthinking"     # Inefficient reasoning
    FACTUAL_ERROR = "factual_error"   # Wrong facts
    LOGICAL_ERROR = "logical_error"   # Wrong reasoning

@dataclass
class ReasoningError:
    """Characterized reasoning error."""
    error_type: ErrorType
    question: str
    model_answer: str
    gold_answer: str
    evidence: List[str]
    severity: float  # 0-1
    reasoning: str

class ErrorCategorizer:
    """
    Categorize reasoning model errors across multiple dimensions.
    """

    def __init__(self, model, evidence_retriever):
        self.model = model
        self.retriever = evidence_retriever

    def categorize_failure(
        self,
        question: str,
        model_output: str,
        gold_output: str,
        retrieved_evidence: List[str]
    ) -> List[ReasoningError]:
        """
        Analyze why model failed on this question.

        Args:
            question: Original question
            model_output: Model's response
            gold_output: Correct answer
            retrieved_evidence: Evidence documents available

        Returns:
            List of identified errors
        """
        errors = []

        # Error 1: Hop Diversity Issues
        hop_error = self._analyze_hop_diversity(
            question,
            model_output,
            retrieved_evidence
        )
        if hop_error:
            errors.append(hop_error)

        # Error 2: Coverage Gaps
        coverage_error = self._analyze_coverage(
            question,
            model_output,
            gold_output,
            retrieved_evidence
        )
        if coverage_error:
            errors.append(coverage_error)

        # Error 3: Overthinking
        overthink_error = self._analyze_overthinking(
            question,
            model_output,
            gold_output
        )
        if overthink_error:
            errors.append(overthink_error)

        return errors

    def _analyze_hop_diversity(
        self,
        question: str,
        model_output: str,
        retrieved_evidence: List[str]
    ) -> ReasoningError:
        """
        Check if model reuses same documents (low hop diversity).

        Args:
            question: Question asked
            model_output: Model's reasoning/answer
            retrieved_evidence: Available evidence

        Returns:
            Error if low diversity detected
        """
        # Extract which documents are referenced in model output
        referenced_docs = self._extract_referenced_documents(model_output, retrieved_evidence)

        # Measure diversity: are references from many different documents?
        num_unique_docs = len(set(referenced_docs))
        num_total_references = len(referenced_docs)

        if num_total_references == 0:
            return None  # No citations, can't evaluate

        diversity_ratio = num_unique_docs / num_total_references

        # Low diversity if using same document repeatedly
        if diversity_ratio < 0.6:  # Less than 60% unique
            return ReasoningError(
                error_type=ErrorType.HOP_DIVERSITY,
                question=question,
                model_answer=model_output,
                gold_answer="",
                evidence=retrieved_evidence,
                severity=1.0 - diversity_ratio,
                reasoning=f"Model reused {num_unique_docs} unique docs in {num_total_references} references"
            )

        return None

    def _analyze_coverage(
        self,
        question: str,
        model_output: str,
        gold_output: str,
        retrieved_evidence: List[str]
    ) -> ReasoningError:
        """
        Check if model missed relevant information (coverage gap).

        Args:
            question: Original question
            model_output: Model's response
            gold_output: Correct answer
            retrieved_evidence: Available evidence

        Returns:
            Error if coverage gap detected
        """
        # Extract key facts from gold answer
        gold_facts = self._extract_facts(gold_output)

        # Check which facts are mentioned in model output
        model_facts = self._extract_facts(model_output)

        covered_facts = len(set(gold_facts) & set(model_facts))
        total_facts = len(gold_facts)

        coverage_ratio = covered_facts / total_facts if total_facts > 0 else 0

        # Coverage gap if missing significant facts
        if coverage_ratio < 0.8:
            missing_facts = set(gold_facts) - set(model_facts)

            return ReasoningError(
                error_type=ErrorType.COVERAGE_GAP,
                question=question,
                model_answer=model_output,
                gold_answer=gold_output,
                evidence=retrieved_evidence,
                severity=1.0 - coverage_ratio,
                reasoning=f"Missing {len(missing_facts)} key facts: {missing_facts}"
            )

        return None

    def _analyze_overthinking(
        self,
        question: str,
        model_output: str,
        gold_output: str
    ) -> ReasoningError:
        """
        Detect cognitive inefficiency (overthinking).

        Args:
            question: Original question
            model_output: Model's reasoning trace
            gold_output: Correct answer

        Returns:
            Error if overthinking detected
        """
        # Measure reasoning efficiency
        reasoning_steps = model_output.count("Step") + model_output.count("Therefore")

        # Check for circular reasoning or repetition
        sentences = model_output.split(".")
        unique_sentences = len(set(sentences))
        repetition_ratio = 1.0 - (unique_sentences / len(sentences))

        # Overthinking if excessive steps or repetition
        if reasoning_steps > 10 or repetition_ratio > 0.3:
            return ReasoningError(
                error_type=ErrorType.OVERTHINKING,
                question=question,
                model_answer=model_output,
                gold_answer=gold_output,
                evidence=[],
                severity=min(repetition_ratio, reasoning_steps / 10.0),
                reasoning=f"Excessive steps ({reasoning_steps}) or repetition ({repetition_ratio:.1%})"
            )

        return None

    def _extract_referenced_documents(self, text: str, documents: List[str]) -> List[int]:
        """Extract which documents are referenced in text."""
        referenced = []

        for i, doc in enumerate(documents):
            if doc in text or any(
                key_phrase in text.lower()
                for key_phrase in self._extract_key_phrases(doc)[:3]
            ):
                referenced.append(i)

        return referenced

    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        import re
        # Simple: sentences with assertions
        sentences = re.split(r'[.!?]+', text)

        facts = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) > 3 and any(
                verb in sent.lower() for verb in ["is", "are", "was", "were"]
            ):
                facts.append(sent)

        return facts

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        # Simple: proper nouns and important words
        words = text.split()
        return [w for w in words if w[0].isupper() or len(w) > 6][:5]
```

### Step 2: Implement Hybrid Human-Automated Evaluation

Combine automated metrics with human judgment.

```python
class HybridErrorEvaluator:
    """
    Hybrid evaluation combining automated and human judgment.
    """

    def __init__(self, categorizer: ErrorCategorizer):
        self.categorizer = categorizer
        self.error_statistics = {}

    def evaluate_errors_hybrid(
        self,
        test_examples: List[Dict],
        human_judgments: List[Dict] = None
    ) -> Dict:
        """
        Evaluate errors using hybrid approach.

        Args:
            test_examples: Test cases with model outputs
            human_judgments: Optional human annotation of errors

        Returns:
            Comprehensive error analysis
        """
        automated_errors = []

        for example in test_examples:
            errors = self.categorizer.categorize_failure(
                example["question"],
                example["model_output"],
                example["gold_output"],
                example.get("evidence", [])
            )

            automated_errors.extend(errors)

        # Combine with human judgments if available
        if human_judgments:
            automated_errors = self._reconcile_with_human_judgment(
                automated_errors,
                human_judgments
            )

        # Aggregate statistics
        error_stats = self._aggregate_error_statistics(automated_errors)

        return error_stats

    def _reconcile_with_human_judgment(
        self,
        automated_errors: List[ReasoningError],
        human_judgments: List[Dict]
    ) -> List[ReasoningError]:
        """Reconcile automated and human error assessments."""
        for error in automated_errors:
            # Find corresponding human judgment
            matching_human = next(
                (h for h in human_judgments if h["question"] == error.question),
                None
            )

            if matching_human:
                # Adjust severity based on human agreement
                if matching_human.get("agrees_with_error"):
                    error.severity = min(1.0, error.severity * 1.2)
                else:
                    error.severity = max(0.0, error.severity * 0.7)

        return automated_errors

    def _aggregate_error_statistics(self, errors: List[ReasoningError]) -> Dict:
        """Aggregate error statistics."""
        stats = {
            "total_errors": len(errors),
            "errors_by_type": {},
            "avg_severity": 0.0,
            "recommendations": []
        }

        # Count by error type
        for error_type in ErrorType:
            count = sum(1 for e in errors if e.error_type == error_type)
            if count > 0:
                stats["errors_by_type"][error_type.value] = {
                    "count": count,
                    "percentage": count / len(errors) if errors else 0
                }

        # Average severity
        if errors:
            stats["avg_severity"] = sum(e.severity for e in errors) / len(errors)

        # Generate recommendations
        stats["recommendations"] = self._generate_recommendations(stats)

        return stats

    def _generate_recommendations(self, stats: Dict) -> List[str]:
        """Generate improvement recommendations based on error patterns."""
        recommendations = []

        for error_type, type_stats in stats["errors_by_type"].items():
            percentage = type_stats["percentage"]

            if error_type == "hop_diversity" and percentage > 0.3:
                recommendations.append(
                    "Improve multi-hop reasoning: model reuses same sources"
                )

            elif error_type == "coverage_gap" and percentage > 0.3:
                recommendations.append(
                    "Enhance information coverage: ensure all relevant facts extracted"
                )

            elif error_type == "overthinking" and percentage > 0.3:
                recommendations.append(
                    "Reduce reasoning overhead: eliminate circular/repetitive steps"
                )

        return recommendations
```

### Step 3: Diagnostic Report Generation

Create human-readable diagnostic reports.

```python
def generate_diagnostic_report(
    model,
    test_examples: List[Dict],
    evidence_retriever
) -> Dict:
    """
    Generate comprehensive diagnostic report.

    Args:
        model: Reasoning model to evaluate
        test_examples: Test cases
        evidence_retriever: Retrieval system

    Returns:
        Diagnostic report with analysis and recommendations
    """
    categorizer = ErrorCategorizer(model, evidence_retriever)
    evaluator = HybridErrorEvaluator(categorizer)

    # Evaluate all examples
    error_stats = evaluator.evaluate_errors_hybrid(test_examples)

    # Build report
    report = {
        "model_performance": {
            "total_examples": len(test_examples),
            "errors_detected": error_stats["total_errors"],
            "error_rate": error_stats["total_errors"] / len(test_examples) if test_examples else 0
        },
        "error_breakdown": error_stats["errors_by_type"],
        "severity": {
            "average": error_stats["avg_severity"],
            "critical": sum(1 for e in test_examples if e.get("severity", 0) > 0.8),
            "moderate": sum(1 for e in test_examples if 0.4 <= e.get("severity", 0) <= 0.8)
        },
        "recommendations": error_stats["recommendations"],
        "detailed_errors": test_examples[:10]  # Show first 10 failures
    }

    return report
```

## Practical Guidance

### When to Use Hop Skip Overthink Framework

- **Debugging reasoning models**: Understanding why models fail on multi-hop questions
- **Model comparison**: Diagnostic-level analysis beyond accuracy metrics
- **Improvement prioritization**: Identifying high-impact areas for enhancement
- **Error analysis studies**: Academic/research understanding of model behavior

### When NOT to Use Framework

- **Real-time inference**: Framework adds overhead unsuitable for latency-sensitive apps
- **Simple classification**: Overkill for single-hop or fully observable problems
- **Automated quality assurance**: Human annotation component limits scalability

### Hyperparameter Recommendations

- **Hop diversity threshold**: 0.6 (use 60%+ unique sources)
- **Coverage threshold**: 0.8 (capture 80%+ key facts)
- **Overthinking step count**: 10 steps (red flag for excessive reasoning)
- **Repetition threshold**: 0.3 (more than 30% repeated content)

### Key Insights

The key insight is that reasoning failures have multiple independent causes, not a single root. By separating hop diversity, coverage, and overthinking diagnostics, the framework enables targeted improvements. Traditional accuracy metrics mask these specific failure modes, making diagnosis difficult.

## Reference

**Hop Skip Overthink: Diagnosing Reasoning Models Multi-Hop** (arXiv:2508.04699)

Introduces nuanced error categorization framework examining failures across hops (diversity), coverage (completeness), and overthinking (efficiency). Combines human annotation with automated metrics to provide actionable diagnostics for improving reasoning model robustness.
