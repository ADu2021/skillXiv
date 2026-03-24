---
name: swe-debate-multi-agent-issue-resolution
title: SWE-Debate Competitive Multi-Agent Debate for Issue Resolution
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2507.23348
keywords: [software-engineering, multi-agent, debate, consensus, code-analysis]
description: "Framework orchestrating competitive debate among specialized agents with different reasoning perspectives. Generates multiple fault propagation traces via code dependency graphs, then resolves to consolidated fixes through structured multi-round competition."
---

## SWE-Debate: Competitive Multi-Agent Debate for Software Issue Resolution

SWE-Debate represents a paradigm shift in automated software engineering by replacing independent agent exploration with structured competitive debate. Multiple specialized agents with different reasoning perspectives analyze code, propose hypotheses, and engage in structured rounds of argument and counter-argument to identify the best fixes.

### Core Concept

The fundamental insight is that software bug localization and fixing benefit from diverse analytical perspectives. Rather than a single agent's trial-and-error search, SWE-Debate:

- **Generates multiple fault hypotheses** by traversing code dependency graphs
- **Specializes agents** to adopt different reasoning perspectives (conservative, aggressive, pattern-based)
- **Orchestrates competitive debate** where agents challenge each other's hypotheses
- **Converges on consensus** through structured rounds of argumentation
- **Produces consolidated fix plans** that synthesize insights from all perspectives

### Architecture Overview

The framework consists of:

- **Code Dependency Graph Builder**: Constructs static call and data dependencies
- **Fault Propagation Tracer**: Generates multiple hypotheses by tracing backward/forward from observed failures
- **Specialized Agent Pool**: Creates agents with different reasoning styles (conservative/aggressive/pattern-matching)
- **Debate Orchestrator**: Manages multi-round structured argumentation
- **Consensus Module**: Extracts agreement on root causes and fixes
- **MCTS Code Modifier**: Executes final fixes through Monte Carlo tree search

### Implementation Steps

**Step 1: Build code dependency graph and extract propagation traces**

Create a structural representation of code dependencies:

```python
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
import ast
import re

@dataclass
class CodeLocation:
    """Represents a location in source code"""
    file: str
    function: str
    line: int

@dataclass
class DependencyEdge:
    """Dependency relationship between code locations"""
    source: CodeLocation
    target: CodeLocation
    edge_type: str  # 'call', 'data_flow', 'control_flow'
    strength: float  # 0-1, confidence in dependency

class DependencyGraphBuilder:
    """Builds static analysis graph of code dependencies"""

    def __init__(self):
        self.dependencies: Dict[CodeLocation, List[DependencyEdge]] = {}
        self.all_locations: Set[CodeLocation] = set()

    def analyze_repository(self, repo_path: str):
        """Scan repository and build dependency graph"""
        import os

        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):  # Example: Python files
                    filepath = os.path.join(root, file)
                    self._analyze_file(filepath)

    def _analyze_file(self, filepath: str):
        """Extract dependencies from single file"""
        try:
            with open(filepath, 'r') as f:
                content = f.read()

            tree = ast.parse(content)
            self._visit_ast(tree, filepath)

        except Exception:
            pass  # Skip files that don't parse

    def _visit_ast(self, node, filepath: str, parent_func: str = None):
        """Walk AST and extract call relationships"""
        if isinstance(node, ast.FunctionDef):
            parent_func = node.name

            # Analyze function calls within
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        callee = child.func.id
                        source = CodeLocation(filepath, parent_func, node.lineno)
                        target = CodeLocation(filepath, callee, 0)

                        edge = DependencyEdge(
                            source=source,
                            target=target,
                            edge_type='call',
                            strength=0.9
                        )

                        if source not in self.dependencies:
                            self.dependencies[source] = []

                        self.dependencies[source].append(edge)
                        self.all_locations.add(source)
                        self.all_locations.add(target)

        for child in ast.iter_child_nodes(node):
            self._visit_ast(child, filepath, parent_func)

    def get_dependents(self, location: CodeLocation) -> List[CodeLocation]:
        """Find all code locations that depend on given location"""
        dependents = []

        for source, edges in self.dependencies.items():
            for edge in edges:
                if edge.target == location:
                    dependents.append(source)

        return dependents

    def get_dependencies(self, location: CodeLocation) -> List[CodeLocation]:
        """Find all code locations that given location depends on"""
        edges = self.dependencies.get(location, [])
        return [edge.target for edge in edges]


class FaultPropagationTracer:
    """Traces fault propagation through code to generate hypotheses"""

    def __init__(self, graph_builder: DependencyGraphBuilder):
        self.graph = graph_builder

    def trace_backward(self, failure_location: CodeLocation,
                      max_depth: int = 5) -> List[List[CodeLocation]]:
        """
        Trace backward from failure point to find root causes.

        Returns multiple paths representing different hypotheses.
        """
        paths = []
        queue = [(failure_location, [failure_location], 0)]

        while queue:
            current, path, depth = queue.pop(0)

            if depth > max_depth:
                paths.append(path)
                continue

            dependencies = self.graph.get_dependencies(current)

            if not dependencies:
                # Reached a source location
                paths.append(path)
            else:
                for dep in dependencies[:3]:  # Limit branching
                    queue.append((dep, path + [dep], depth + 1))

        return paths

    def trace_forward(self, suspect_location: CodeLocation,
                     max_depth: int = 5) -> List[List[CodeLocation]]:
        """
        Trace forward to find all code affected by suspect location.

        Helps understand impact of potential bug.
        """
        paths = []
        queue = [(suspect_location, [suspect_location], 0)]

        while queue:
            current, path, depth = queue.pop(0)

            if depth > max_depth:
                paths.append(path)
                continue

            dependents = self.graph.get_dependents(current)

            if not dependents:
                paths.append(path)
            else:
                for dep in dependents[:3]:
                    queue.append((dep, path + [dep], depth + 1))

        return paths

    def generate_hypotheses(self, error_location: CodeLocation,
                           error_message: str) -> List[Dict]:
        """
        Generate root cause hypotheses by tracing propagation.

        Returns ranked list of hypotheses about bug location.
        """
        hypotheses = []

        # Get backward traces (where bug might originate)
        backward_paths = self.trace_backward(error_location)

        for path in backward_paths:
            hypothesis = {
                'root_cause': path[0] if path else error_location,
                'propagation_path': path,
                'evidence': f"Error at {error_location} traces back through {len(path)} locations",
                'confidence': max(0, 1.0 - len(path) * 0.1)  # Deeper = less confident
            }
            hypotheses.append(hypothesis)

        # Sort by confidence
        hypotheses.sort(key=lambda h: h['confidence'], reverse=True)

        return hypotheses[:5]  # Top 5 hypotheses
```

This creates structural understanding of code for better diagnosis.

**Step 2: Implement specialized debate agents**

Create agents with different analytical perspectives:

```python
class DebateAgent:
    """Base class for specialized debate agents"""

    def __init__(self, llm, agent_type: str = 'balanced'):
        self.llm = llm
        self.agent_type = agent_type  # conservative, aggressive, pattern-based
        self.reasoning_history: List[str] = []

    def analyze_issue(self, issue_description: str,
                     hypotheses: List[Dict],
                     code_context: str) -> Dict:
        """
        Analyze issue from agent's perspective.

        Returns reasoning and proposed fix.
        """
        prompt = self._build_analysis_prompt(issue_description, hypotheses,
                                            code_context)

        analysis = self.llm.generate(prompt, max_tokens=1500)

        return {
            'agent_type': self.agent_type,
            'analysis': analysis,
            'preferred_hypothesis': self._extract_preferred(analysis, hypotheses),
            'reasoning': analysis
        }

    def _build_analysis_prompt(self, issue: str, hypotheses: List[Dict],
                               code: str) -> str:
        """Build analysis prompt tailored to agent type"""

        if self.agent_type == 'conservative':
            style = "Focus on the most obviously broken code with clear evidence."
        elif self.agent_type == 'aggressive':
            style = "Consider unconventional fixes and distant root causes."
        elif self.agent_type == 'pattern-based':
            style = "Look for common error patterns and standard fixes."
        else:
            style = "Balanced analysis"

        prompt = f"""{style}

Issue: {issue}

Candidate hypotheses:
{chr(10).join(f"- {h['root_cause']}: {h['evidence']}" for h in hypotheses)}

Code context:
{code}

Provide your analysis:"""

        return prompt

    def _extract_preferred(self, analysis: str,
                          hypotheses: List[Dict]) -> Dict:
        """Extract which hypothesis agent prefers from analysis"""
        if hypotheses:
            return hypotheses[0]  # Simplified
        return {}

    def propose_fix(self, issue: str, hypothesis: Dict) -> str:
        """Generate fix proposal for accepted hypothesis"""

        prompt = f"""Based on root cause {hypothesis['root_cause']},
propose a specific code fix:

Issue: {issue}

Generate concrete code changes:"""

        fix = self.llm.generate(prompt, max_tokens=1000)

        return fix

class ConservativeAgent(DebateAgent):
    """Agent that prefers obvious, low-risk fixes"""
    def __init__(self, llm):
        super().__init__(llm, agent_type='conservative')

class AggressiveAgent(DebateAgent):
    """Agent willing to explore unconventional fixes"""
    def __init__(self, llm):
        super().__init__(llm, agent_type='aggressive')

class PatternAgent(DebateAgent):
    """Agent that applies known bug patterns and standard fixes"""
    def __init__(self, llm):
        super().__init__(llm, agent_type='pattern-based')
```

This creates agents with different analytical biases.

**Step 3: Orchestrate multi-round debate**

Manage structured argumentation between agents:

```python
class DebateOrchestrator:
    """Manages multi-round debate between specialized agents"""

    def __init__(self, agents: List[DebateAgent]):
        self.agents = agents
        self.debate_rounds: List[Dict] = []

    def run_debate(self, issue: Dict, hypotheses: List[Dict],
                  code_context: str, num_rounds: int = 3) -> Dict:
        """
        Run structured debate to reach consensus.

        Returns consolidated understanding and fix plan.
        """

        # Round 1: Initial analysis by all agents
        initial_analyses = []
        for agent in self.agents:
            analysis = agent.analyze_issue(
                issue['description'],
                hypotheses,
                code_context
            )
            initial_analyses.append(analysis)

        # Rounds 2+: Agents debate and refine positions
        for round_num in range(1, num_rounds):
            round_results = self._run_debate_round(
                initial_analyses,
                round_num,
                issue,
                hypotheses
            )
            self.debate_rounds.append(round_results)
            initial_analyses = round_results['refined_analyses']

        # Extract consensus
        consensus = self._extract_consensus(initial_analyses, hypotheses)

        return {
            'debate_rounds': self.debate_rounds,
            'consensus': consensus,
            'accepted_hypothesis': consensus['root_cause']
        }

    def _run_debate_round(self, current_analyses: List[Dict],
                         round_num: int,
                         issue: Dict,
                         hypotheses: List[Dict]) -> Dict:
        """
        Run one round of debate: agents challenge each other.
        """

        refined_analyses = []

        for i, agent in enumerate(self.agents):
            # Get arguments from other agents
            other_arguments = [
                a['analysis'] for j, a in enumerate(current_analyses)
                if j != i
            ]

            # Build challenge prompt
            prompt = f"""Round {round_num} of debate.

Your previous analysis:
{current_analyses[i]['analysis']}

Other agents' arguments:
{chr(10).join(f"Agent {j}: {arg[:200]}" for j, arg in enumerate(other_arguments))}

Address or refine your position:"""

            refined = agent.llm.generate(prompt, max_tokens=1000)

            refined_analyses.append({
                'agent_type': agent.agent_type,
                'analysis': refined,
                'round': round_num
            })

        return {
            'round': round_num,
            'refined_analyses': refined_analyses
        }

    def _extract_consensus(self, final_analyses: List[Dict],
                          hypotheses: List[Dict]) -> Dict:
        """
        Extract consensus from agent positions.

        Identify which hypothesis has strongest support.
        """

        hypothesis_votes = {}

        for analysis in final_analyses:
            # Count support for each hypothesis
            text = analysis['analysis']

            for hypothesis in hypotheses:
                if hypothesis['root_cause'].function in text:
                    key = hypothesis['root_cause']
                    hypothesis_votes[key] = hypothesis_votes.get(key, 0) + 1

        # Most supported hypothesis
        if hypothesis_votes:
            best = max(hypothesis_votes.items(), key=lambda x: x[1])
            consensus_hypothesis = next(
                h for h in hypotheses
                if h['root_cause'] == best[0]
            )
        else:
            consensus_hypothesis = hypotheses[0] if hypotheses else None

        return {
            'root_cause': consensus_hypothesis,
            'support_level': len(final_analyses),
            'agreement_ratio': hypothesis_votes.get(
                consensus_hypothesis['root_cause'], 0) / len(final_analyses) if final_analyses else 0
        }
```

This orchestrates competitive analysis and consensus-building.

**Step 4: Implement consensus-based fix generation**

Generate fixes that synthesize agent insights:

```python
class ConsensusFixGenerator:
    """Generates fixes based on debate consensus"""

    def __init__(self, llm):
        self.llm = llm

    def generate_consolidated_fix(self, consensus: Dict,
                                 issue: Dict,
                                 code_context: str) -> str:
        """
        Generate fix proposal incorporating consensus insights.
        """

        root_cause = consensus['root_cause']

        prompt = f"""Based on consensus analysis, fix this issue:

Root cause: {root_cause['root_cause']}
Evidence: {root_cause['evidence']}

Issue: {issue['description']}

Code context:
{code_context}

Generate a specific, minimal code fix:"""

        fix_proposal = self.llm.generate(prompt, max_tokens=1500)

        return fix_proposal

    def verify_fix(self, fix_proposal: str, tests: List[str]) -> Dict:
        """
        Verify proposed fix against test cases.

        Returns verification results.
        """
        results = {
            'fix': fix_proposal,
            'test_results': {},
            'viable': True
        }

        # Simplified: real version would actually run tests
        for test in tests:
            results['test_results'][test] = True

        return results
```

**Step 5: Integrate MCTS for code modification**

Apply Monte Carlo tree search to explore code modification space:

```python
class MCTSCodeModifier:
    """Explores code modifications using Monte Carlo tree search"""

    def __init__(self, llm):
        self.llm = llm

    def search_modifications(self, original_code: str,
                            fix_proposal: str,
                            tests: List[str],
                            max_iterations: int = 100) -> str:
        """
        Use MCTS to find optimal code modifications.

        Args:
            original_code: Starting code
            fix_proposal: Suggested fix direction
            tests: Test cases to evaluate
            max_iterations: MCTS iterations

        Returns:
            Best code modification found
        """

        current_state = {'code': original_code, 'test_pass_rate': 0}
        best_state = current_state

        for iteration in range(max_iterations):
            # Selection & Expansion
            candidate_modifications = self._generate_modifications(
                current_state['code'],
                fix_proposal
            )

            # Simulation
            best_candidate = None
            best_score = -float('inf')

            for mod in candidate_modifications[:3]:  # Limit branching
                score = self._simulate_modification(mod, tests)

                if score > best_score:
                    best_score = score
                    best_candidate = mod

            # Backpropagation
            if best_score > best_state['test_pass_rate']:
                best_state = {
                    'code': best_candidate,
                    'test_pass_rate': best_score
                }
                current_state = best_state

        return best_state['code']

    def _generate_modifications(self, code: str, fix_idea: str) -> List[str]:
        """Generate candidate code modifications"""
        prompt = f"""Generate 3 alternative ways to implement this fix:

Current code:
{code}

Fix idea:
{fix_idea}

Provide 3 different implementations:"""

        modifications_text = self.llm.generate(prompt, max_tokens=1500)

        # Parse out individual modifications (simplified)
        return modifications_text.split('---')

    def _simulate_modification(self, code: str, tests: List[str]) -> float:
        """Evaluate modification against test suite"""
        # Simplified: real version would run tests
        # Return pass rate 0-1
        return 0.7
```

This enables exploration of modification space to find best fixes.

### Practical Guidance

**When to use SWE-Debate:**
- Complex software issues with multiple possible root causes
- Teams wanting to avoid false certainty in fault localization
- Codebases with complex dependencies
- When consensus and justification matter
- Issues requiring discussion of trade-offs

**When NOT to use SWE-Debate:**
- Simple, obvious bugs with clear fixes
- Real-time repair systems needing fast decisions
- Single-agent approaches sufficient for problem class
- When debate overhead isn't justified

**Key hyperparameters:**

- `num_debate_rounds`: 2-4 typical; more for complex issues
- `num_agents`: 3-5 specialized agents effective
- `max_mcts_iterations`: 50-200 depending on complexity
- `hypothesis_depth`: 3-7 for dependency tracing

**Expected characteristics:**

- Debate overhead: ~2-3x single agent cost
- Consensus agreement: 60-80% typical on root cause
- Fix quality: 10-20% improvement vs single agent
- Computation: Multiple agents in parallel can reduce wall time

### Reference

SWE-Debate: Competitive Multi-Agent Debate for Software Issue Resolution. arXiv:2507.23348
