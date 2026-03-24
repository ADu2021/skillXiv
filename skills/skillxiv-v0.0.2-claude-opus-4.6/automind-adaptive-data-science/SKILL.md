---
name: automind-adaptive-data-science
title: "AutoMind: Adaptive Knowledgeable Agent for Automated Data Science"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.10974"
keywords: [automated machine learning, LLM agents, knowledge base, agentic search, code generation]
description: "Build LLM-driven data science agents grounded in empirical knowledge through expert knowledge base, tree search algorithms, and complexity-adaptive code generation, surpassing SOTA by 8% on MLE-Bench."
---

# AutoMind: Adaptive Knowledgeable Agent for Automated Data Science

## Core Concept

AutoMind addresses limitations in rigid LLM-driven data science workflows through three innovations: an expert knowledge base curated from Kaggle competitions and peer-reviewed papers, an agentic tree search algorithm exploring multiple solution paths, and adaptive code generation that scales complexity with task difficulty. The system achieves 41.2% human participant surpass rate on official leaderboards and outperforms prior state-of-the-art by 8% on MLE-Bench.

## Architecture Overview

- **Expert Knowledge Base**: 3,237 Kaggle posts + papers from KDD, ICLR, NeurIPS, ICML, EMNLP; hierarchically labeled for efficient retrieval
- **Agentic Knowledgeable Tree Search**: Explores solution tree with three action types—drafting (create branches), improving (refine), debugging (fix failures). Stochastic heuristic guides exploration
- **Self-Adaptive Code Generation**: Switches between one-pass (simple plans) and stepwise (complex plans) based on solution complexity scoring. AST checking validates syntax before execution
- **Dynamic Solution Space**: Manages plans, code, validation metrics; greedy selection optimizes top solutions while exploring alternatives
- **Scalable Model Support**: GPT-4o, o3-mini, DeepSeek-V3; reduces token cost by 9.6% vs baseline

## Implementation

### Step 1: Expert Knowledge Base Construction

```python
import json
from collections import defaultdict

class KnowledgeBase:
    """
    Curates and indexes expert knowledge from diverse sources.
    Supports hierarchical query and efficient retrieval.
    """

    def __init__(self):
        self.knowledge_index = defaultdict(list)
        self.category_hierarchy = {
            'feature_engineering': [
                'categorical_encoding', 'feature_scaling', 'polynomial_features',
                'interaction_features', 'domain_specific_features'
            ],
            'model_selection': [
                'tree_based', 'linear_models', 'neural_networks',
                'ensemble_methods', 'domain_adapted'
            ],
            'hyperparameter_tuning': [
                'grid_search', 'random_search', 'bayesian_optimization',
                'learning_rate', 'regularization'
            ],
            'data_preprocessing': [
                'missing_value_handling', 'outlier_detection', 'normalization',
                'dimensionality_reduction', 'sampling_strategies'
            ]
        }

    def ingest_kaggle_posts(self, post_list):
        """
        Index Kaggle competition posts and solutions.
        Post format: {'id', 'title', 'content', 'competition', 'votes'}
        """

        for post in post_list:
            # Extract entities and techniques from post
            entities = self._extract_entities(post['content'])

            for entity in entities:
                category = self._categorize_entity(entity)
                self.knowledge_index[category].append({
                    'source': 'kaggle',
                    'post_id': post['id'],
                    'competition': post['competition'],
                    'content': post['content'],
                    'votes': post['votes'],
                    'technique': entity
                })

    def ingest_academic_papers(self, papers):
        """
        Index papers from top-tier ML conferences.
        Paper format: {'title', 'abstract', 'pdf_url', 'venue', 'year'}
        """

        for paper in papers:
            # Extract methodologies and benchmarks
            methodologies = self._extract_methodologies(paper['abstract'])

            for method in methodologies:
                category = self._categorize_method(method)
                self.knowledge_index[category].append({
                    'source': 'academic',
                    'paper_title': paper['title'],
                    'venue': paper['venue'],
                    'year': paper['year'],
                    'abstract': paper['abstract'],
                    'methodology': method
                })

    def retrieve(self, query, category=None, top_k=5):
        """
        Retrieve relevant knowledge items.
        Supports hierarchical category filtering.
        """

        results = []

        # Determine relevant categories
        target_categories = [category] if category else list(self.knowledge_index.keys())

        for cat in target_categories:
            for item in self.knowledge_index[cat]:
                # Simple relevance scoring: TF-IDF-like
                relevance = self._score_relevance(query, item)
                results.append((relevance, item))

        # Sort by relevance and return top-k
        results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in results[:top_k]]

    def _extract_entities(self, text):
        """Extract technical entities from text."""
        # Placeholder: would use NER or keyword extraction
        return []

    def _categorize_entity(self, entity):
        """Determine category for entity."""
        if 'encode' in entity.lower():
            return 'feature_engineering'
        elif 'xgb' in entity.lower() or 'random' in entity.lower():
            return 'model_selection'
        return 'feature_engineering'

    def _extract_methodologies(self, abstract):
        """Extract methodologies from paper abstract."""
        return []

    def _categorize_method(self, method):
        """Determine category for method."""
        return 'feature_engineering'

    def _score_relevance(self, query, item):
        """Score relevance of item to query."""
        return 0.5  # Placeholder
```

### Step 2: Agentic Tree Search Algorithm

```python
import torch
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class SolutionNode:
    """Represents a solution in the search tree."""
    plan: str  # High-level approach
    code: str  # Implementation
    validation_metric: float  # Accuracy/F1/RMSE
    parent: Optional['SolutionNode'] = None
    children: List['SolutionNode'] = None

class AgenticTreeSearch:
    """
    Explores solution space using three action types:
    - Draft: Create new solution branches
    - Improve: Refine existing solutions
    - Debug: Fix execution failures
    """

    def __init__(self, knowledge_base, model_name='gpt-4o'):
        self.kb = knowledge_base
        self.model = model_name
        self.root = None
        self.best_solution = None
        self.best_metric = float('-inf')

    def search(self, task_description, max_iterations=10):
        """
        Main search loop: iteratively explore and refine solutions.
        """

        # Initialize root: generate initial solution
        initial_solution = self._draft_solution(task_description)
        self.root = SolutionNode(
            plan=initial_solution['plan'],
            code=initial_solution['code'],
            validation_metric=0.0
        )

        current_node = self.root

        for iteration in range(max_iterations):
            print(f"Iteration {iteration}: Best metric={self.best_metric:.4f}")

            # Choose action: draft, improve, or debug
            action = self._select_action(current_node)

            if action == 'draft':
                # Create multiple new solution branches
                for _ in range(2):
                    new_solution = self._draft_solution(
                        task_description,
                        parent_context=current_node
                    )
                    new_node = SolutionNode(
                        plan=new_solution['plan'],
                        code=new_solution['code'],
                        validation_metric=0.0,
                        parent=current_node
                    )
                    current_node.children.append(new_node)

            elif action == 'improve':
                # Refine current best solution
                improved = self._improve_solution(
                    current_node.plan,
                    current_node.code
                )
                current_node.plan = improved['plan']
                current_node.code = improved['code']

            elif action == 'debug':
                # Fix execution errors
                if current_node.validation_metric == float('-inf'):  # Error state
                    debugged = self._debug_solution(
                        current_node.code,
                        error_log=current_node.error_log
                    )
                    current_node.code = debugged['code']

            # Evaluate current node
            metric = self._execute_and_validate(current_node.code)
            current_node.validation_metric = metric

            # Update best solution
            if metric > self.best_metric:
                self.best_metric = metric
                self.best_solution = current_node
                print(f"New best: {metric:.4f}")

            # Move to next promising node (greedy selection)
            current_node = self._select_next_node()

        return self.best_solution

    def _draft_solution(self, task, parent_context=None):
        """Generate new solution draft using knowledge base."""

        # Retrieve relevant knowledge
        knowledge = self.kb.retrieve(task, top_k=3)

        draft_prompt = f"""
        Task: {task}

        Relevant knowledge:
        {self._format_knowledge(knowledge)}

        Generate a solution with:
        1. High-level plan (strategy, approach)
        2. Python code implementation

        Output JSON:
        {{"plan": "...", "code": "..."}}
        """

        response = self._llm_call(draft_prompt)
        return self._parse_solution(response)

    def _improve_solution(self, plan, code):
        """Refine solution iteratively."""

        improve_prompt = f"""
        Current plan: {plan}

        Current code:
        {code}

        Suggest improvements:
        1. Efficiency enhancements
        2. Code quality improvements
        3. Better ML practices

        Provide improved plan and code.
        """

        response = self._llm_call(improve_prompt)
        return self._parse_solution(response)

    def _debug_solution(self, code, error_log):
        """Fix execution errors."""

        debug_prompt = f"""
        Code:
        {code}

        Error:
        {error_log}

        Fix the error and provide corrected code.
        """

        response = self._llm_call(debug_prompt)
        return {'code': response}

    def _select_action(self, current_node):
        """Select next action based on stochastic heuristic policy."""

        # Simple policy: draft if few children, improve if no error, debug if error
        if len(current_node.children) < 2:
            return 'draft'
        elif current_node.validation_metric == float('-inf'):
            return 'debug'
        else:
            return 'improve'

    def _select_next_node(self):
        """Greedy selection: move to best unexplored node."""

        candidates = []
        for node in self._get_all_nodes():
            if node.validation_metric == float('-inf'):  # Unevaluated
                candidates.append(node)

        if candidates:
            return candidates[0]

        # If all evaluated, pick best
        return max(self._get_all_nodes(),
                  key=lambda n: n.validation_metric)

    def _execute_and_validate(self, code):
        """Execute code and return validation metric."""
        # Placeholder: would execute code and measure performance
        return 0.75

    def _llm_call(self, prompt):
        """Call LLM."""
        return "response"

    def _format_knowledge(self, knowledge):
        """Format knowledge for prompt."""
        return '\n'.join([f"- {k['technique']}" for k in knowledge])

    def _parse_solution(self, response):
        """Parse LLM response into solution dict."""
        return {'plan': 'plan', 'code': 'code'}

    def _get_all_nodes(self):
        """Collect all nodes in tree."""
        def dfs(node):
            if node is None:
                return []
            result = [node]
            for child in (node.children or []):
                result.extend(dfs(child))
            return result

        return dfs(self.root)
```

### Step 3: Self-Adaptive Code Generation

```python
import ast

class AdaptiveCodeGenerator:
    """
    Generates code that adapts complexity to task difficulty.
    Switches between one-pass (simple) and stepwise (complex).
    """

    def generate_code(self, plan, task_complexity_score):
        """
        Generate code with complexity matching task difficulty.
        Complexity score: 0-1 (0=simple, 1=very complex)
        """

        if task_complexity_score < 0.4:
            return self._generate_one_pass_code(plan)
        else:
            return self._generate_stepwise_code(plan)

    def _generate_one_pass_code(self, plan):
        """
        Simple implementation: single coherent script.
        For straightforward tasks (basic preprocessing, standard models).
        """

        code_template = f"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and preprocess
X = load_features()
y = load_labels()

X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {{accuracy:.4f}}")
"""

        return code_template

    def _generate_stepwise_code(self, plan):
        """
        Complex implementation: decomposed into substeps.
        For intricate tasks (feature engineering, ensemble, hyperparameter tuning).
        """

        code_template = f"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# Step 1: Advanced feature engineering
def engineer_features(X):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    return X_poly

# Step 2: Normalize
X = load_features()
y = load_labels()
X_eng = engineer_features(X)
X_scaled = StandardScaler().fit_transform(X_eng)

# Step 3: Ensemble training with cross-validation
models = [
    RandomForestClassifier(n_estimators=100),
    GradientBoostingClassifier(n_estimators=100)
]

scores = []
for model in models:
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    scores.append(cv_scores.mean())
    print(f"{{model.__class__.__name__}}: {{cv_scores.mean():.4f}}")

print(f"Best model score: {{max(scores):.4f}}")
"""

        return code_template

    def validate_and_refine(self, code):
        """
        Validate code syntax and refine if needed.
        Uses AST parsing to catch errors before execution.
        """

        try:
            ast.parse(code)
            return code, True
        except SyntaxError as e:
            print(f"Syntax error: {e}")

            # Attempt automated fix
            fixed_code = self._fix_syntax_error(code, e)

            # Recursive validation
            return self.validate_and_refine(fixed_code)

    def _fix_syntax_error(self, code, error):
        """Attempt to fix common syntax errors."""
        # Placeholder: would use heuristics to fix
        return code

def score_plan_complexity(plan):
    """
    Score plan complexity 0-1 using professional rubric.
    Considers: feature engineering scope, model selection, hyperparameter tuning.
    """

    complexity_factors = {
        'feature_engineering': 0.3,
        'model_selection': 0.3,
        'hyperparameter_tuning': 0.2,
        'ensemble_methods': 0.2
    }

    total_score = 0.0
    for factor, weight in complexity_factors.items():
        if factor in plan.lower():
            total_score += weight

    return min(total_score, 1.0)
```

## Practical Guidance

**Knowledge Base Curation**:
- Kaggle: Focus on recent competitions; weight by votes/popularity
- Papers: Filter for methodology papers (not just benchmarks); include implementation details
- Organization: Hierarchical categorization enables targeted retrieval
- Update frequency: Monthly updates with new competition solutions

**Search Hyperparameters**:
- Max iterations: 8-12 (balance quality vs cost)
- Branching factor: 2-3 new solutions per draft action
- Metric threshold: Accept solution if metric > 0.7 of domain SOTA

**Cost Optimization**:
- Knowledge retrieval: TF-IDF indexes reduce LLM query overhead
- Code validation: AST parsing catches errors before execution (saves rerun cost)
- Model selection: Use cheaper model (o3-mini) for most steps; GPT-4o for complex planning

**When to Use AutoMind**:
- Structured tabular data problems (heterogeneous features, missing values)
- Limited time budgets (20-hour leaderboard competitions)
- Ensemble/stacking opportunities (AutoMind explores combinations)
- Transfer learning scenarios (knowledge base accelerates solution discovery)

## Reference

- Agentic tree search: Combines exploration (drafting) with exploitation (improving best solutions)
- Stochastic heuristic policy: Non-deterministic action selection enables diverse solution exploration
- AST validation: Abstract Syntax Tree parsing catches errors without execution overhead
