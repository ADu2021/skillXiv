---
name: cyber-zero-runtime-free-agent-training
title: Cyber-Zero Training Cybersecurity Agents Without Runtime
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.00910
keywords: [cybersecurity, agent-training, synthetic-data, persona-simulation, runtime-free]
description: "Framework for training cybersecurity agents without access to live environments. Uses CTF writeups and persona-driven LLM simulation to synthesize training trajectories, achieving performance matching proprietary systems like Claude-3.5-Sonnet."
---

## Cyber-Zero: Training Cybersecurity Agents Without Runtime

Cyber-Zero addresses a critical challenge in cybersecurity agent development: the lack of accessible runtime environments for training. Rather than requiring live CTF platforms or complex setups, the system leverages public documentation and persona-driven simulation to create high-quality synthetic training data, enabling agents to match proprietary system performance.

### Core Concept

The fundamental insight is that cybersecurity knowledge is extensively documented in CTF (Capture-The-Flag) writeups and solutions. Cyber-Zero:

- **Extracts knowledge** from public CTF writeups and documentation
- **Simulates agent interactions** using persona-driven LLM simulation
- **Synthesizes trajectories** of realistic long-horizon problem-solving
- **Trains agents** on synthetic data without runtime access
- **Achieves parity** with systems that have access to live environments

### Architecture Overview

The framework consists of:

- **CTF Knowledge Extractor**: Parses writeups to identify problem types and solutions
- **Persona-Driven Simulator**: Uses LLMs with specific personas to simulate agent interactions
- **Trajectory Synthesizer**: Generates plausible multi-step trajectories
- **Data Quality Validator**: Ensures synthetic trajectories are realistic
- **Agent Training Pipeline**: Fine-tunes agents on synthesized data

### Implementation Steps

**Step 1: Extract knowledge from CTF writeups**

Parse CTF solutions to identify problem types and solution patterns:

```python
import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class CTFProblem:
    """Represents a CTF challenge and its solution"""
    name: str
    category: str  # 'crypto', 'web', 'reverse', 'pwn', 'forensics'
    description: str
    solution_steps: List[str]
    tools_used: List[str]
    vulnerability_type: str
    difficulty: str

class CTFKnowledgeExtractor:
    """Extracts knowledge from CTF writeups"""

    def __init__(self):
        self.problems: List[CTFProblem] = []
        self.vulnerability_index: Dict[str, List[CTFProblem]] = {}

    def parse_writeup(self, writeup_text: str) -> Optional[CTFProblem]:
        """
        Parse CTF writeup to extract problem and solution.

        Args:
            writeup_text: Complete writeup or solution document

        Returns:
            CTFProblem with extracted information
        """
        # Extract title
        title_match = re.search(r'^#+\s*(.+?)$', writeup_text, re.MULTILINE)
        title = title_match.group(1) if title_match else "Unknown"

        # Detect category by content analysis
        category = self._detect_category(writeup_text)

        # Extract problem description
        description = self._extract_description(writeup_text)

        # Extract solution steps
        solution_steps = self._extract_solution_steps(writeup_text)

        # Identify tools used
        tools = self._identify_tools(writeup_text, solution_steps)

        # Determine vulnerability type
        vulnerability = self._extract_vulnerability(writeup_text)

        # Assess difficulty
        difficulty = self._assess_difficulty(solution_steps)

        problem = CTFProblem(
            name=title,
            category=category,
            description=description,
            solution_steps=solution_steps,
            tools_used=tools,
            vulnerability_type=vulnerability,
            difficulty=difficulty
        )

        self.problems.append(problem)

        # Index by vulnerability
        if vulnerability not in self.vulnerability_index:
            self.vulnerability_index[vulnerability] = []

        self.vulnerability_index[vulnerability].append(problem)

        return problem

    def _detect_category(self, text: str) -> str:
        """Detect challenge category from text"""

        categories = {
            'crypto': ['encryption', 'rsa', 'aes', 'hash', 'cipher'],
            'web': ['http', 'sql', 'xss', 'csrf', 'web server'],
            'reverse': ['binary', 'assembly', 'decompile', 'ida'],
            'pwn': ['buffer', 'overflow', 'rop', 'shellcode', 'exploit'],
            'forensics': ['file', 'memory', 'log', 'metadata']
        }

        text_lower = text.lower()

        for category, keywords in categories.items():
            if any(kw in text_lower for kw in keywords):
                return category

        return 'misc'

    def _extract_description(self, text: str) -> str:
        """Extract problem description"""

        lines = text.split('\n')
        description_lines = []

        in_description = False

        for line in lines:
            if 'description' in line.lower() or 'challenge' in line.lower():
                in_description = True
            elif in_description and line.strip() and not line.startswith('#'):
                description_lines.append(line)
            elif in_description and line.strip().startswith('#'):
                break

        return '\n'.join(description_lines[:5])

    def _extract_solution_steps(self, text: str) -> List[str]:
        """Extract step-by-step solution"""

        steps = []

        # Look for numbered steps or code blocks
        step_matches = re.findall(
            r'(\d+)[.)]\s*(.+?)(?=\n\d+[.)]|\n\n|\Z)',
            text,
            re.DOTALL
        )

        for num, step_text in step_matches:
            # Clean step text
            step = step_text.strip()[:200]
            if step:
                steps.append(step)

        # If no numbered steps, extract code blocks as steps
        if not steps:
            code_blocks = re.findall(r'```.*?\n(.+?)\n```', text, re.DOTALL)
            steps = [b[:200] for b in code_blocks[:5]]

        return steps

    def _identify_tools(self, text: str, steps: List[str]) -> List[str]:
        """Identify tools used in solution"""

        tools = set()

        # Common CTF tools
        tool_patterns = {
            'python': r'python\s+\w+',
            'ida': r'IDA|ida pro',
            'ghidra': r'Ghidra',
            'gdb': r'gdb|GDB',
            'strings': r'strings',
            'hexdump': r'hexdump',
            'wireshark': r'Wireshark',
            'sqlmap': r'sqlmap',
            'burp': r'Burp Suite',
            'pwntools': r'pwntools|pwn'
        }

        full_text = text + ' ' + ' '.join(steps)

        for tool, pattern in tool_patterns.items():
            if re.search(pattern, full_text, re.IGNORECASE):
                tools.add(tool)

        return list(tools)

    def _extract_vulnerability(self, text: str) -> str:
        """Extract vulnerability type"""

        vulnerabilities = [
            'buffer_overflow', 'sql_injection', 'xss', 'rsa_weakness',
            'weak_crypto', 'file_upload', 'privilege_escalation'
        ]

        text_lower = text.lower()

        for vuln in vulnerabilities:
            if vuln.replace('_', ' ') in text_lower or vuln in text_lower:
                return vuln

        return 'unknown'

    def _assess_difficulty(self, steps: List[str]) -> str:
        """Assess difficulty from solution complexity"""

        if len(steps) <= 2:
            return 'easy'
        elif len(steps) <= 5:
            return 'medium'
        else:
            return 'hard'

    def get_problems_by_category(self, category: str) -> List[CTFProblem]:
        """Get all problems in category"""
        return [p for p in self.problems if p.category == category]

    def get_problems_by_vulnerability(self,
                                     vuln_type: str) -> List[CTFProblem]:
        """Get all problems with specific vulnerability"""
        return self.vulnerability_index.get(vuln_type, [])
```

**Step 2: Implement persona-driven LLM simulator**

Create LLM-based agent simulation with different personas:

```python
from enum import Enum

class AgentPersona(Enum):
    """Different agent personas for simulation"""
    CAUTIOUS = "thorough, methodical, checks assumptions"
    AGGRESSIVE = "tries shortcuts, tests hypotheses quickly"
    EXPERT = "applies known patterns, recognizes problem types"

class PersonaDrivenSimulator:
    """Simulates agent interaction using LLM with personas"""

    def __init__(self, llm):
        self.llm = llm

    def simulate_agent_step(self, problem: CTFProblem,
                          current_state: Dict,
                          persona: AgentPersona = AgentPersona.EXPERT) -> Tuple[str, Dict]:
        """
        Simulate single agent action step.

        Args:
            problem: CTF problem to solve
            current_state: Current progress/findings
            persona: Agent personality style

        Returns:
            (action, updated_state)
        """

        # Build context
        context = self._build_context(problem, current_state, persona)

        # Generate action
        prompt = f"""{context}

Based on the current situation, what is your next action? Be specific and actionable.

Action:"""

        action = self.llm.generate(prompt, max_tokens=200).strip()

        # Simulate effect of action
        new_state = self._simulate_action_effect(action, current_state, problem)

        return action, new_state

    def _build_context(self, problem: CTFProblem,
                      current_state: Dict,
                      persona: AgentPersona) -> str:
        """Build simulation context for LLM"""

        persona_desc = {
            AgentPersona.CAUTIOUS: "You are a cautious, methodical analyst",
            AgentPersona.AGGRESSIVE: "You are an aggressive problem solver",
            AgentPersona.EXPERT: "You are an expert in cybersecurity"
        }[persona]

        context = f"""{persona_desc}.

Challenge: {problem.name}
Category: {problem.category}
Description: {problem.description}

Vulnerability Type: {problem.vulnerability_type}
Tools Available: {', '.join(problem.tools_used)}

Current Progress:
{json.dumps(current_state, indent=2)}

Solution Steps (for guidance):
{chr(10).join(f'- {s}' for s in problem.solution_steps[:3])}"""

        return context

    def _simulate_action_effect(self, action: str,
                               current_state: Dict,
                               problem: CTFProblem) -> Dict:
        """Simulate the effect of an action"""

        new_state = current_state.copy()

        # Simple heuristics for effect simulation
        action_lower = action.lower()

        if 'scan' in action_lower or 'nmap' in action_lower:
            new_state['scanned'] = True
            new_state['ports_discovered'] = [22, 80, 443]
        elif 'connect' in action_lower or 'telnet' in action_lower:
            new_state['connected'] = True
        elif 'find' in action_lower or 'grep' in action_lower:
            new_state['findings'] = (new_state.get('findings', 0) + 1)
        elif 'exploit' in action_lower:
            new_state['exploit_attempted'] = True
            new_state['success'] = True  # Simplified

        return new_state
```

**Step 3: Generate synthetic trajectories**

Create realistic multi-step problem-solving sequences:

```python
@dataclass
class Trajectory:
    """Multi-step solution trajectory"""
    problem: CTFProblem
    steps: List[Tuple[str, Dict]]  # (action, resulting_state)
    success: bool
    num_steps: int

class TrajectoryGenerator:
    """Generates synthetic problem-solving trajectories"""

    def __init__(self, extractor: CTFKnowledgeExtractor,
                 simulator: PersonaDrivenSimulator):
        self.extractor = extractor
        self.simulator = simulator

    def generate_trajectory(self, problem: CTFProblem,
                          persona: AgentPersona = AgentPersona.EXPERT,
                          max_steps: int = 20) -> Trajectory:
        """
        Generate complete trajectory for solving a problem.

        Args:
            problem: CTF problem to solve
            persona: Agent persona to use
            max_steps: Maximum trajectory length

        Returns:
            Complete trajectory from start to goal
        """

        steps = []
        current_state = {
            'started': True,
            'found_flags': [],
            'tools_used': []
        }

        success = False

        for step_num in range(max_steps):
            # Simulate next step
            action, new_state = self.simulator.simulate_agent_step(
                problem,
                current_state,
                persona
            )

            steps.append((action, new_state))
            current_state = new_state

            # Check if goal reached (simplified)
            if 'flag' in action.lower() or new_state.get('success'):
                success = True
                break

            # Bail if stalled
            if step_num > 10 and not new_state.get('progress', False):
                break

        return Trajectory(
            problem=problem,
            steps=steps,
            success=success,
            num_steps=len(steps)
        )

    def generate_batch_trajectories(self, problems: List[CTFProblem],
                                    batch_size: int = 10) -> List[Trajectory]:
        """Generate trajectories for multiple problems"""

        trajectories = []

        for problem in problems[:batch_size]:
            # Generate with each persona
            for persona in AgentPersona:
                traj = self.generate_trajectory(problem, persona=persona)
                trajectories.append(traj)

        return trajectories
```

**Step 4: Validate trajectory quality**

Ensure synthetic data is realistic and useful:

```python
class TrajectoryValidator:
    """Validates quality of synthetic trajectories"""

    def __init__(self):
        self.realistic_patterns = self._load_patterns()

    def validate_trajectory(self, trajectory: Trajectory) -> Dict:
        """
        Validate trajectory for realism and utility.

        Returns validation score and issues.
        """

        issues = []
        scores = {
            'realism': 0.0,
            'diversity': 0.0,
            'utility': 0.0,
            'overall': 0.0
        }

        # Check realism
        realism_score = self._check_realism(trajectory)
        if realism_score < 0.5:
            issues.append("Low action realism")
        scores['realism'] = realism_score

        # Check diversity
        actions = [step[0] for step in trajectory.steps]
        diversity = len(set(actions)) / len(actions) if actions else 0
        if diversity < 0.3:
            issues.append("Low action diversity")
        scores['diversity'] = diversity

        # Check utility
        if trajectory.success:
            utility = 0.9
        else:
            utility = 0.3
        scores['utility'] = utility

        # Overall
        scores['overall'] = (
            0.4 * realism_score + 0.3 * diversity + 0.3 * utility
        )

        return {
            'valid': scores['overall'] > 0.5,
            'scores': scores,
            'issues': issues
        }

    def _check_realism(self, trajectory: Trajectory) -> float:
        """Check if actions follow realistic patterns"""

        realism_score = 0.0
        valid_actions = 0

        for action, _ in trajectory.steps:
            if self._is_realistic_action(action):
                valid_actions += 1

        realism_score = valid_actions / len(trajectory.steps) if trajectory.steps else 0

        return realism_score

    def _is_realistic_action(self, action: str) -> bool:
        """Check if action is realistically described"""

        realistic_patterns = [
            r'(scan|nmap)',
            r'(analyze|examine)',
            r'(connect|telnet)',
            r'(crack|bruteforce)',
            r'(exploit|execute)',
            r'(find|search)',
            r'(decrypt|decode)'
        ]

        action_lower = action.lower()

        for pattern in realistic_patterns:
            if re.search(pattern, action_lower):
                return True

        return False

    def _load_patterns(self) -> List[str]:
        """Load realistic action patterns"""
        return []

    def filter_valid_trajectories(self,
                                 trajectories: List[Trajectory],
                                 min_score: float = 0.5) -> List[Trajectory]:
        """Filter trajectories by quality"""

        valid = []

        for trajectory in trajectories:
            validation = self.validate_trajectory(trajectory)

            if validation['scores']['overall'] > min_score:
                valid.append(trajectory)

        return valid
```

**Step 5: Integrate agent training**

Train agents on synthetic trajectories:

```python
class CyberZeroTrainer:
    """Trains cybersecurity agents on synthetic trajectories"""

    def __init__(self, agent_model, trajectory_dataset: List[Trajectory]):
        self.agent = agent_model
        self.trajectories = trajectory_dataset

    def prepare_training_data(self) -> Tuple[List, List]:
        """
        Convert trajectories to training data.

        Returns (problem_contexts, action_sequences)
        """

        contexts = []
        sequences = []

        for trajectory in self.trajectories:
            problem = trajectory.problem
            actions = [step[0] for step in trajectory.steps]

            # Context: problem description and category
            context = f"{problem.category}: {problem.description}"
            contexts.append(context)

            # Action sequence
            sequences.append(actions)

        return contexts, sequences

    def train(self, learning_rate: float = 1e-5, num_epochs: int = 3):
        """Train agent on trajectories"""

        contexts, action_sequences = self.prepare_training_data()

        optimizer = torch.optim.AdamW(
            self.agent.parameters(),
            lr=learning_rate
        )

        for epoch in range(num_epochs):
            total_loss = 0.0

            for context, actions in zip(contexts, action_sequences):
                # Encode problem
                problem_encoding = self.agent.encode_problem(context)

                # Predict next action at each step
                for step_idx, target_action in enumerate(actions):
                    # Get prediction
                    pred_action = self.agent.predict_next_action(
                        problem_encoding,
                        actions[:step_idx]
                    )

                    # Compute loss
                    loss = self._compute_loss(pred_action, target_action)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            avg_loss = total_loss / len(contexts)
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

    def _compute_loss(self, prediction, target) -> torch.Tensor:
        """Compute training loss"""
        # Simplified: real version would use cross-entropy
        return torch.tensor(0.0)
```

### Practical Guidance

**When to use Cyber-Zero:**
- Training agents without access to live CTF platforms
- Building cybersecurity training systems
- Creating agents for security research
- Democratizing access to security agent capabilities
- Scenarios where environment setup is prohibitive

**When NOT to use Cyber-Zero:**
- Systems requiring real-world environment interaction
- Tasks needing actual vulnerability exploitation
- Real-time security operations requiring live data
- Scenarios where synthetic data accuracy is critical

**Key hyperparameters:**

- `max_trajectory_steps`: 15-25 typical for realistic solutions
- `num_personas`: 3 provides good diversity
- `validation_min_score`: 0.5-0.6 typical threshold
- `training_epochs`: 2-5 typical for convergence

**Expected performance:**

- Achieves parity with Claude-3.5-Sonnet on CTF tasks
- 13.1% absolute improvement on standard benchmarks
- Training data synthesis: 1000+ trajectories from writeups
- Agent training time: Hours on single GPU

**Dataset characteristics:**

- Typical: 100-500 CTF writeups per category
- Generates 3-5 trajectories per writeup per persona
- Quality filtering keeps 60-70% of generated trajectories
- Results in 10K+ training examples per domain

### Reference

Cyber-Zero: Training Cybersecurity Agents without Runtime. arXiv:2508.00910
