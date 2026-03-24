---
name: tool-integrated-rl-repo-search
title: Tool-Integrated RL for Repository Bug Localization
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.03012
keywords: [reinforcement-learning, code-search, bug-localization, tool-use]
description: "Two-stage post-training framework combining rejection-sampled SFT and RL for LLM-guided repository code search and issue localization."
---

## Tool-Integrated RL for Repo Deep Search

This framework teaches LLMs to locate buggy code in repositories through structured tool use and reinforcement learning. The two-stage approach combines supervised fine-tuning on high-quality trajectories with RL-based exploration, enabling models to learn sophisticated repository navigation strategies.

### Core Concept

Developers spend significant time locating bugs in code repositories—a task requiring systematic exploration of file structure, function definitions, and import relationships. Rather than hoping LLMs can solve this through in-context learning alone, this framework explicitly trains them to use navigation tools effectively. The key insight: start with supervised learning of basic tool use, then deepen understanding through RL rewards that measure accuracy and ranking quality.

### Architecture Overview

- **Lightweight RepoSearcher Tools**: Six minimalist tools (GetRepoStructure, GetImportOfFile, SearchClass, SearchFunction, SearchClassMethod, Exit)
- **Stage 1 (SFT)**: Rejection sampling: keep only trajectories that correctly identify buggy code for supervised fine-tuning
- **Stage 2 (RL)**: Reward-based policy optimization using nDCG@k metric to balance correctness and ranking quality
- **Ranking-Aware Rewards**: Unlike binary correctness, rewards account for ranking of candidate functions

### Implementation Steps

**Step 1: Define RepoSearcher Tool Interface**

Create a minimal but complete tool set for repository navigation:

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import os

class RepoSearcherTool(ABC):
    """Base class for repository search tools."""
    @abstractmethod
    def execute(self, repo_path: str, query: str) -> str:
        pass

class GetRepoStructure(RepoSearcherTool):
    """Retrieve directory structure and top-level files."""
    def execute(self, repo_path: str, query: str = None) -> str:
        """
        Returns: Formatted tree view of repo structure.
        """
        tree = []
        for root, dirs, files in os.walk(repo_path):
            # Limit depth to avoid overwhelming output
            depth = root.replace(repo_path, '').count(os.sep)
            if depth > 2:
                continue

            indent = '  ' * depth
            tree.append(f"{indent}{os.path.basename(root)}/")
            subindent = '  ' * (depth + 1)
            for file in files[:5]:  # Show first 5 files only
                tree.append(f"{subindent}{file}")

        return "\n".join(tree)

class GetImportOfFile(RepoSearcherTool):
    """Get import statements and dependencies from a file."""
    def execute(self, repo_path: str, file_path: str) -> str:
        """
        Args:
            file_path: Path to file relative to repo
        Returns: List of imports/dependencies
        """
        full_path = os.path.join(repo_path, file_path)
        imports = []

        try:
            with open(full_path, 'r') as f:
                for line in f:
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        imports.append(line.strip())
        except FileNotFoundError:
            return f"File not found: {file_path}"

        return "\n".join(imports[:20])  # Limit output

class SearchClass(RepoSearcherTool):
    """Search for class definitions matching a query."""
    def execute(self, repo_path: str, class_name: str) -> str:
        """
        Args:
            class_name: Name or substring of class to find
        Returns: List of matching class definitions with file locations
        """
        matches = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if not file.endswith('.py'):
                    continue
                filepath = os.path.join(root, file)

                try:
                    with open(filepath, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            if f'class {class_name}' in line:
                                rel_path = os.path.relpath(filepath, repo_path)
                                matches.append(f"{rel_path}:{line_num} - {line.strip()}")
                except:
                    pass

        return "\n".join(matches[:10]) if matches else "No matching classes found"

class SearchFunction(RepoSearcherTool):
    """Search for function definitions."""
    def execute(self, repo_path: str, func_name: str) -> str:
        """Search for functions matching query."""
        matches = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if not file.endswith('.py'):
                    continue
                filepath = os.path.join(root, file)

                try:
                    with open(filepath, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            if f'def {func_name}' in line:
                                rel_path = os.path.relpath(filepath, repo_path)
                                matches.append(f"{rel_path}:{line_num} - {line.strip()}")
                except:
                    pass

        return "\n".join(matches[:10]) if matches else "No matching functions found"

class SearchClassMethod(RepoSearcherTool):
    """Search for methods within a class."""
    def execute(self, repo_path: str, query: str) -> str:
        """
        Args:
            query: Format "ClassName.method_name"
        Returns: Method locations and signatures
        """
        class_name, method_name = query.split('.')
        # Implementation would parse AST and find methods
        # Simplified here
        return f"Searching for method {method_name} in class {class_name}"

class Exit(RepoSearcherTool):
    """Signal that agent has completed search and will provide answer."""
    def execute(self, repo_path: str, query: str = None) -> str:
        return "Search complete. Ready to provide answer."

class RepoSearcher:
    """Manager for repository search tools."""
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.tools = {
            'GetRepoStructure': GetRepoStructure(),
            'GetImportOfFile': GetImportOfFile(),
            'SearchClass': SearchClass(),
            'SearchFunction': SearchFunction(),
            'SearchClassMethod': SearchClassMethod(),
            'Exit': Exit(),
        }

    def execute_tool(self, tool_name: str, query: str) -> str:
        """Execute a tool and return result."""
        if tool_name not in self.tools:
            return f"Tool not found: {tool_name}"
        return self.tools[tool_name].execute(self.repo_path, query)
```

**Step 2: Implement Stage 1 - Rejection-Sampled SFT**

Generate and filter high-quality training trajectories:

```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Trajectory:
    """Represents a sequence of tool calls and responses."""
    steps: List[Tuple[str, str, str]]  # (tool_name, input, response)
    final_answer: str
    is_correct: bool
    ground_truth: str

class SFTDataCollector:
    """
    Generate trajectories using base LLM, keep only correct ones.
    Rejection sampling: sample trajectories and filter by correctness.
    """
    def __init__(self, model, repo_searcher, num_samples=10):
        self.model = model
        self.repo_searcher = repo_searcher
        self.num_samples = num_samples

    def generate_trajectories(self, issue_description: str, ground_truth_functions: List[str]) -> List[Trajectory]:
        """
        Sample multiple trajectories and filter for correctness.
        Only keep trajectories that identify ground truth as top prediction.
        """
        trajectories = []

        for _ in range(self.num_samples):
            # Initialize trajectory
            steps = []
            current_state = f"Issue: {issue_description}"

            # Run agent until Exit tool is called
            for step in range(10):  # Max 10 steps
                # LLM decides which tool to use
                prompt = f"""Repository: {self.repo_searcher.repo_path}
Current state: {current_state}
Available tools: GetRepoStructure, GetImportOfFile, SearchClass, SearchFunction, SearchClassMethod, Exit

Which tool would you use next? Respond with:
TOOL: <tool_name>
INPUT: <query>"""

                response = self.model.generate(prompt)
                tool_name, tool_input = self._parse_response(response)

                if tool_name == 'Exit':
                    break

                # Execute tool
                tool_result = self.repo_searcher.execute_tool(tool_name, tool_input)
                steps.append((tool_name, tool_input, tool_result))
                current_state += f"\n{tool_name}({tool_input}) -> {tool_result[:200]}"

            # Generate final answer
            final_prompt = f"""Based on your repository exploration: {current_state}

What are the functions containing the bug? List function names."""
            final_answer = self.model.generate(final_prompt)
            predicted_functions = self._extract_functions(final_answer)

            # Check correctness: is at least one ground truth in predictions?
            is_correct = any(gt in predicted_functions for gt in ground_truth_functions)

            trajectory = Trajectory(
                steps=steps,
                final_answer=final_answer,
                is_correct=is_correct,
                ground_truth=', '.join(ground_truth_functions)
            )
            trajectories.append(trajectory)

        # Rejection sampling: keep only correct trajectories
        correct_trajectories = [t for t in trajectories if t.is_correct]
        return correct_trajectories

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse tool selection from LLM response."""
        lines = response.split('\n')
        tool_name = ""
        tool_input = ""

        for line in lines:
            if line.startswith('TOOL:'):
                tool_name = line.replace('TOOL:', '').strip()
            elif line.startswith('INPUT:'):
                tool_input = line.replace('INPUT:', '').strip()

        return tool_name, tool_input

    def _extract_functions(self, text: str) -> List[str]:
        """Extract function names from model output."""
        # Simple heuristic: look for patterns like "function_name"
        import re
        functions = re.findall(r'`(\w+)\`', text)
        return functions or [text.split('\n')[0].strip()]

def create_sft_dataset(model, issues_and_functions, repo_searcher):
    """Create SFT dataset by collecting and filtering trajectories."""
    collector = SFTDataCollector(model, repo_searcher, num_samples=5)

    sft_data = []
    for issue, ground_truth_funcs in issues_and_functions:
        trajectories = collector.generate_trajectories(issue, ground_truth_funcs)
        for trajectory in trajectories:
            # Format as SFT example
            sft_data.append({
                'prompt': f"Locate buggy code for: {issue}",
                'trajectory': trajectory.steps,
                'answer': trajectory.final_answer
            })

    return sft_data
```

**Step 3: Implement Stage 2 - RL with nDCG Rewards**

Train using ranking-aware rewards:

```python
import numpy as np
from typing import List

def compute_ndcg_at_k(predictions: List[str], ground_truth: List[str], k: int = 5) -> float:
    """
    Compute nDCG@k metric: Normalized Discounted Cumulative Gain.
    Rewards both correctness and ranking quality.

    A correct function at rank 1 gets higher reward than at rank 5.
    """
    # Binary relevance: 1 if function is ground truth, 0 otherwise
    relevance = [1.0 if pred in ground_truth else 0.0 for pred in predictions[:k]]

    # Discounted cumulative gain
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))

    # Ideal DCG: perfect ranking would have all ground truth first
    ideal_relevance = [1.0] * min(k, len(ground_truth))
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))

    # Normalized
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return ndcg

class RLTrainer:
    """
    Train model using RL with nDCG rewards.
    Reinforcement learning explores beyond supervised trajectories.
    """
    def __init__(self, model, repo_searcher, learning_rate=1e-5):
        self.model = model
        self.repo_searcher = repo_searcher
        self.learning_rate = learning_rate

    def compute_reward(self, predictions: List[str], ground_truth: List[str],
                      trajectory_length: int) -> float:
        """
        Reward = nDCG@5 - length_penalty

        Encourages both accuracy and efficiency.
        """
        ndcg = compute_ndcg_at_k(predictions, ground_truth, k=5)
        length_penalty = 0.01 * trajectory_length  # Penalize long trajectories

        reward = ndcg - length_penalty
        return reward

    def run_episode(self, issue: str, ground_truth: List[str]) -> Tuple[float, List[Trajectory]]:
        """
        Run one RL episode: agent explores, collects reward.
        """
        trajectories = []

        for _ in range(3):  # Sample 3 trajectories per issue
            steps = []
            current_state = f"Issue: {issue}"

            # Agent acts (sample from policy)
            for step in range(10):
                prompt = f"Issue: {issue}\nCurrent state: {current_state}\nNext tool?"
                tool_name, tool_input = self._sample_action(prompt)

                if tool_name == 'Exit':
                    break

                tool_result = self.repo_searcher.execute_tool(tool_name, tool_input)
                steps.append((tool_name, tool_input, tool_result))
                current_state += f"\n{tool_name}({tool_input})"

            # Get final prediction
            final_answer = self.model.generate(f"{current_state}\nFunctions with bug?")
            predictions = self._extract_functions(final_answer)

            # Compute reward
            reward = self.compute_reward(predictions, ground_truth, len(steps))

            trajectory = Trajectory(
                steps=steps,
                final_answer=final_answer,
                is_correct=any(p in ground_truth for p in predictions),
                ground_truth=', '.join(ground_truth)
            )
            trajectory.reward = reward
            trajectories.append(trajectory)

        return sum(t.reward for t in trajectories) / len(trajectories), trajectories

    def _sample_action(self, prompt: str) -> Tuple[str, str]:
        """Sample tool selection from model."""
        # Implementation would use model.sample() instead of generate()
        # to enable exploration during RL
        return "SearchFunction", "query"

    def _extract_functions(self, text: str) -> List[str]:
        """Extract predicted function names."""
        pass

    def train(self, issues_and_ground_truth, num_epochs: int = 3):
        """
        RL training loop.
        """
        for epoch in range(num_epochs):
            total_reward = 0.0

            for issue, ground_truth in issues_and_ground_truth:
                episode_reward, trajectories = self.run_episode(issue, ground_truth)
                total_reward += episode_reward

                # Advantage estimation and policy update
                for trajectory in trajectories:
                    advantage = trajectory.reward - total_reward  # Baseline
                    # GRPO update (Group Relative Policy Optimization)
                    self._update_policy(trajectory, advantage)

            print(f"Epoch {epoch}: Avg reward {total_reward / len(issues_and_ground_truth):.3f}")

    def _update_policy(self, trajectory: Trajectory, advantage: float):
        """Update model weights based on trajectory and advantage."""
        # Would implement GRPO or similar algorithm
        pass
```

**Step 4: Evaluate on Benchmarks**

Measure performance improvement:

```python
def evaluate_model(model, repo_searcher, test_issues, test_ground_truth) -> Dict:
    """Evaluate model on test set, measuring localization accuracy."""
    predictions = []

    for issue in test_issues:
        # Run model once (no sampling)
        trajectory = []
        current_state = f"Issue: {issue}"

        for _ in range(10):
            tool_name, tool_input = model.act(current_state)
            if tool_name == 'Exit':
                break
            result = repo_searcher.execute_tool(tool_name, tool_input)
            trajectory.append((tool_name, tool_input, result))
            current_state += f"\n{tool_name}({tool_input})"

        final_answer = model.generate(f"{current_state}\nFunctions?")
        predictions.append(model.extract_functions(final_answer))

    # Compute metrics
    ndcg_scores = [
        compute_ndcg_at_k(pred, truth, k=5)
        for pred, truth in zip(predictions, test_ground_truth)
    ]

    return {
        'mean_ndcg@5': np.mean(ndcg_scores),
        'success_rate': sum(1 for pred, truth in zip(predictions, test_ground_truth)
                           if any(p in truth for p in pred)) / len(predictions),
    }
```

### Practical Guidance

**When to Use:**
- Bug localization in large codebases (>100K lines)
- Scenarios where ground truth function labels are available for training
- Applications emphasizing systematic code exploration over pattern matching
- Cases where ranking quality matters (top-k function suggestions)

**When NOT to Use:**
- Small codebases where simple string search suffices
- Scenarios with limited labeled training data (<50 examples)
- Real-time systems requiring <100ms latency (requires full trajectory generation)
- Domains without clear hierarchical structure (unstructured data)

**Hyperparameters:**

| Parameter | Default | Impact |
|-----------|---------|--------|
| `num_sft_samples` | 10 | Trajectories sampled per issue in Stage 1; higher = better SFT data quality |
| `sft_reject_threshold` | 1.0 | Keep only perfectly correct trajectories; lower = more training data |
| `rl_episodes_per_issue` | 3 | Trajectories sampled per issue in RL; balance exploration vs. computation |
| `ndcg_k` | 5 | Top-k functions evaluated in reward; match use case requirements |
| `length_penalty` | 0.01 | Penalize long trajectories; higher = encourage efficiency |

### Reference

**Paper**: Tool-integrated RL for Repo Deep Search (2508.03012)
- Two-stage training: SFT + RL with ranking-aware rewards
- nDCG@k metric balances correctness and ranking quality
- 32B ToolTrain model matches Claude-3.7-Sonnet on function-level localization
