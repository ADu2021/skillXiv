---
name: autonomous-research-framework
title: "Idea2Story: An Automated Pipeline for Transforming Research Concepts into Complete Scientific Narratives"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.20833"
keywords: [autonomous-science, research-automation, knowledge-graph, research-patterns, scientific-discovery]
description: "Build autonomous research agents using pre-computed knowledge graphs instead of online reasoning. Extract methodological patterns from literature, organize them into structured knowledge, and enable agents to align user research intents with established paradigms for efficient, grounded research planning and execution."
---

## Problem

Autonomous research agents using online reasoning suffer from high computational costs, context window limitations, and brittle reasoning. They repeatedly read and summarize literature, generating intermediate reasoning that often leads to hallucinations and failed research plans. This runtime-centric approach doesn't scale.

## Solution

Implement Idea2Story: a pre-computation-driven framework that shifts the burden from runtime to offline knowledge construction:

1. **Literature Collection**: Systematically gather peer-reviewed papers with peer review feedback
2. **Methodological Extraction**: Extract core research methods and techniques from papers
3. **Pattern Composition**: Create reusable research patterns by composing methodological units
4. **Knowledge Graph Organization**: Build structured graph of research paradigms and relationships
5. **Runtime Grounding**: At execution time, align user research intents to established paradigms instead of generating from scratch

This shifts the computational model from expensive online reasoning to efficient offline indexing and retrieval.

## When to Use

- Building autonomous research systems for specific domains
- Scaling research automation across large literature bases
- Scientific discovery with reliable, grounded reasoning
- Situations where research methodology is well-established (mature fields)
- Systems where reproducibility and grounding in prior work is critical

## When NOT to Use

- Novel, emerging research areas (limited literature for patterns)
- Real-time current events research (requires recent data)
- Highly creative or exploratory research (pattern reuse may limit novelty)
- Scenarios requiring custom methodologies outside established paradigms

## Implementation

### Step 1: Build the Literature Collection Pipeline

Systematically gather and metadata extraction from papers.

```python
class LiteratureCollector:
    """Gather papers and extract review feedback"""

    def __init__(self, data_sources):
        self.papers = []
        self.review_feedback = {}
        self.data_sources = data_sources  # arXiv, conferences, etc.

    def collect_papers(self, query, num_papers=1000):
        """
        Retrieve papers matching research query
        """
        for source in self.data_sources:
            results = source.search(query, limit=num_papers // len(self.data_sources))

            for paper in results:
                paper_data = {
                    "id": paper["id"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                    "full_text": self.extract_text(paper),
                    "authors": paper["authors"],
                    "publication_date": paper["date"],
                    "citations": self.get_citations(paper["id"])
                }
                self.papers.append(paper_data)

        return len(self.papers)

    def collect_review_feedback(self, paper_id):
        """
        Extract peer review comments and decisions
        """
        feedback = {
            "decision": "",  # accept/reject/revise
            "comments": [],
            "strengths": [],
            "weaknesses": []
        }

        # Try to find reviews (from review repositories, OpenReview, etc.)
        reviews = self.search_reviews(paper_id)

        for review in reviews:
            feedback["decision"] = review.get("decision")
            feedback["comments"].extend(review.get("comments", []))
            feedback["strengths"].extend(review.get("strengths", []))
            feedback["weaknesses"].extend(review.get("weaknesses", []))

        self.review_feedback[paper_id] = feedback
        return feedback

    def extract_text(self, paper):
        """Extract full text from paper PDF/HTML"""
        # Use PDF extraction or arXiv HTML
        return paper.get("full_text", "")
```

### Step 2: Extract Core Methodological Units

Identify and isolate research methods from papers.

```python
class MethodologicalExtractor:
    """Extract research methods and techniques from papers"""

    def __init__(self):
        self.extracted_methods = []

    def extract_methods_from_paper(self, paper):
        """
        Identify methodological components in a paper
        Returns list of (method_type, description, implementation_details)
        """
        methods = []

        text = paper["full_text"]
        abstract = paper["abstract"]

        # Extract problem statement
        problem = self.extract_problem_statement(paper)

        # Extract methodology section
        methodology_text = self.extract_section(text, ["Methods", "Methodology", "Approach"])

        # Identify technique names
        techniques = self.identify_techniques(methodology_text)

        # Extract algorithm descriptions
        algorithms = self.extract_algorithms(methodology_text)

        # Extract evaluation methodology
        evaluation = self.extract_evaluation_approach(paper)

        for technique in techniques:
            method = {
                "type": "technique",
                "name": technique,
                "description": self.describe_technique(technique, methodology_text),
                "problem_addressed": problem,
                "evaluation_metrics": evaluation["metrics"],
                "source_paper": paper["id"]
            }
            methods.append(method)

        for algorithm in algorithms:
            method = {
                "type": "algorithm",
                "name": algorithm["name"],
                "pseudocode": algorithm["pseudocode"],
                "complexity": algorithm.get("complexity"),
                "problem_addressed": problem,
                "source_paper": paper["id"]
            }
            methods.append(method)

        return methods

    def identify_techniques(self, methodology_text):
        """Find technique names using NER and keywords"""
        # Look for patterns like "We propose X", "Using X method", "X approach"
        patterns = [
            r"propose[d]?\s+(?:a|an)\s+([^.]+)",
            r"introduce[d]?\s+(?:a|an)\s+([^.]+)",
            r"(?:using|employing)\s+(?:a|an)\s+([^.]+)",
        ]

        techniques = []
        for pattern in patterns:
            matches = re.findall(pattern, methodology_text, re.IGNORECASE)
            techniques.extend(matches)

        return list(set(techniques))

    def extract_algorithms(self, methodology_text):
        """Extract algorithm pseudocode and descriptions"""
        algorithms = []

        # Look for Algorithm: or pseudocode blocks
        algo_pattern = r"Algorithm[^\n]*:\s*([^.]+\.)"

        for match in re.finditer(algo_pattern, methodology_text):
            algo_desc = match.group(1)
            algorithm = {
                "name": self.extract_algo_name(algo_desc),
                "pseudocode": algo_desc,
                "complexity": self.estimate_complexity(algo_desc)
            }
            algorithms.append(algorithm)

        return algorithms

    def extract_evaluation_approach(self, paper):
        """Extract how the paper evaluated their method"""
        eval_section = self.extract_section(
            paper["full_text"],
            ["Evaluation", "Experiments", "Results"]
        )

        evaluation = {
            "metrics": self.extract_metrics(eval_section),
            "baselines": self.extract_baselines(eval_section),
            "datasets": self.extract_datasets(paper),
            "improvements": self.extract_improvements(eval_section)
        }

        return evaluation
```

### Step 3: Compose Research Patterns

Combine extracted methods into reusable patterns.

```python
class ResearchPatternComposer:
    """Create composite research patterns from methodological units"""

    def __init__(self):
        self.patterns = []

    def compose_patterns_from_methods(self, extracted_methods):
        """
        Group related methods into coherent research patterns
        A pattern is a reusable sequence: Problem -> Techniques -> Evaluation
        """
        # Group by problem type
        problem_groups = {}

        for method in extracted_methods:
            problem = method["problem_addressed"]
            if problem not in problem_groups:
                problem_groups[problem] = []
            problem_groups[problem].append(method)

        # Create patterns from each problem group
        patterns = []
        for problem, methods in problem_groups.items():
            # Extract techniques
            techniques = [m for m in methods if m["type"] == "technique"]
            algorithms = [m for m in methods if m["type"] == "algorithm"]

            # Extract evaluation
            evaluation_metrics = set()
            for method in methods:
                evaluation_metrics.update(method.get("evaluation_metrics", []))

            pattern = {
                "problem": problem,
                "techniques": techniques,
                "algorithms": algorithms,
                "evaluation_metrics": list(evaluation_metrics),
                "source_papers": list(set(m["source_paper"] for m in methods)),
                "pattern_id": f"pattern_{len(patterns)}"
            }

            patterns.append(pattern)

        self.patterns.extend(patterns)
        return patterns

    def rank_patterns_by_effectiveness(self):
        """
        Rank patterns based on citation count and review feedback
        """
        for pattern in self.patterns:
            effectiveness_score = 0

            # Score by citation count
            for paper_id in pattern["source_papers"]:
                effectiveness_score += self.get_citation_count(paper_id)

            # Score by review feedback (positive comments)
            for paper_id in pattern["source_papers"]:
                review_feedback = self.get_review_feedback(paper_id)
                positive_comments = len(review_feedback.get("strengths", []))
                effectiveness_score += positive_comments

            pattern["effectiveness_score"] = effectiveness_score

        # Sort descending by effectiveness
        self.patterns.sort(key=lambda p: p["effectiveness_score"], reverse=True)
```

### Step 4: Build Methodological Knowledge Graph

Organize patterns and their relationships into a queryable graph.

```python
class MethodologicalKnowledgeGraph:
    """Structured graph of research patterns and paradigms"""

    def __init__(self):
        self.nodes = {}  # pattern_id -> pattern_data
        self.edges = []  # relationships between patterns
        self.problem_index = {}  # problem_type -> [pattern_ids]
        self.technique_index = {}  # technique_name -> [pattern_ids]

    def add_pattern(self, pattern):
        """Add research pattern to graph"""
        pattern_id = pattern["pattern_id"]
        self.nodes[pattern_id] = pattern

        # Index by problem
        problem = pattern["problem"]
        if problem not in self.problem_index:
            self.problem_index[problem] = []
        self.problem_index[problem].append(pattern_id)

        # Index by techniques
        for technique in pattern["techniques"]:
            tech_name = technique["name"]
            if tech_name not in self.technique_index:
                self.technique_index[tech_name] = []
            self.technique_index[tech_name].append(pattern_id)

    def find_patterns_for_problem(self, problem):
        """Retrieve patterns for specific research problem"""
        pattern_ids = self.problem_index.get(problem, [])
        return [self.nodes[pid] for pid in pattern_ids]

    def find_patterns_by_technique(self, technique):
        """Find patterns using specific technique"""
        pattern_ids = self.technique_index.get(technique, [])
        return [self.nodes[pid] for pid in pattern_ids]

    def get_related_patterns(self, pattern_id, depth=1):
        """Find related patterns (same problem, overlapping techniques)"""
        pattern = self.nodes[pattern_id]
        problem = pattern["problem"]

        # Related = addressing same problem
        related_ids = self.problem_index.get(problem, [])
        return [self.nodes[pid] for pid in related_ids if pid != pattern_id]
```

### Step 5: Align Research Intent to Paradigms at Runtime

At execution time, ground agent planning in pre-built knowledge.

```python
class ResearchIntentAligner:
    """Align user research intent to established paradigms"""

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def align_intent_to_patterns(self, user_query):
        """
        Map user research intent to patterns in knowledge graph
        User: "I want to improve model efficiency on language tasks"
        -> Aligned to: efficiency patterns for NLP
        """
        # Parse user intent
        problem_inferred = self.infer_problem_from_query(user_query)
        constraints = self.extract_constraints(user_query)

        # Find matching patterns
        matching_patterns = self.kg.find_patterns_for_problem(problem_inferred)

        # Filter by constraints (if any)
        if constraints:
            matching_patterns = [
                p for p in matching_patterns
                if self.matches_constraints(p, constraints)
            ]

        # Rank by effectiveness and relevance
        matching_patterns.sort(
            key=lambda p: (p["effectiveness_score"], self.relevance_score(p, user_query)),
            reverse=True
        )

        return matching_patterns[:3]  # Top 3 most relevant

    def generate_research_plan(self, user_query, selected_pattern):
        """
        Generate concrete research plan based on aligned pattern
        Instead of LLM generating from scratch, use pattern as blueprint
        """
        problem = selected_pattern["problem"]
        techniques = selected_pattern["techniques"]
        evaluation = selected_pattern["evaluation_metrics"]

        plan = {
            "step_1_problem_definition": {
                "problem_from_pattern": problem,
                "customization": f"Adapt to: {user_query}"
            },
            "step_2_methodology": {
                "techniques": [t["name"] for t in techniques],
                "reference_implementations": [t["description"] for t in techniques]
            },
            "step_3_evaluation": {
                "metrics": evaluation,
                "datasets": selected_pattern.get("source_papers", [])
            }
        }

        return plan
```

## Key Advantages Over Online Reasoning

- **Computational Efficiency**: No online literature search/reasoning during execution
- **Context Window**: Pre-computed patterns fit in limited token budgets
- **Grounding**: All patterns verified against peer-reviewed literature
- **Reproducibility**: Plans reference established methodologies
- **Scalability**: Offline knowledge graph scales better than online reasoning

## Workflow Summary

```
Papers -> Extract Methods -> Compose Patterns -> Build Knowledge Graph
                                                        |
                                                        v
User Query -> Parse Intent -> Align to Patterns -> Generate Plan
```

## References

- arXiv:2601.20833: Idea2Story framework for automated research discovery
- Demonstrates pre-computation vs. online reasoning for autonomous science
- Empirically shows coherent, high-quality research pattern generation
