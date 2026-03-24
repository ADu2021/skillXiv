---
name: cove-tool-use-training
title: "CoVe: Training Interactive Tool-Use Agents via Constraint-Guided Verification"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.01940"
keywords: [Tool Use, Agent Training, Data Synthesis, Verification, Interactive Agents]
description: "CoVe synthesizes high-quality tool-use training data using explicit task constraints as both generation guidance and verification validators, enabling effective agent training without manual curation."
---

# Technique: Constraint-Guided Verification for Tool-Use Agent Training

Training agents to use tools (APIs, functions, domain-specific commands) is notoriously difficult. The challenge: creating diverse, realistic interaction trajectories where agents navigate complex, ambiguous user requests through deterministic actions is expensive and error-prone. Manual curation scales poorly, and unconstrained data synthesis produces trajectories that violate domain logic or are unrealistic.

CoVe solves this by embedding explicit task constraints (business rules, domain requirements) directly into the data synthesis process. Constraints serve dual purposes: (1) guiding generation of sophisticated, realistic trajectories, and (2) providing deterministic verification that outputs are correct. This eliminates manual annotation while ensuring data quality.

## Core Concept

The core insight: constraints are semantic specifications that can guide both generation and validation. Rather than generating trajectories unconstrained and hoping they're correct, define constraints that the agent must satisfy (e.g., "booking a flight must include selecting date, passengers, and payment"). Use these constraints to:

1. **Guide generation**: LLM generates trajectories that respect constraints
2. **Verify correctness**: Automatically check that trajectories satisfy all constraints
3. **Train agents**: High-quality data with verified correctness enables effective SFT and RL

This creates a virtuous cycle: better constraints yield better data, which trains better agents.

## Architecture Overview

- **Constraint Definition**: Specify task requirements (preconditions, actions, postconditions)
- **Trajectory Generation**: LLM generates interaction sequences respecting constraints
- **Constraint Verification**: Deterministic validator confirms trajectories satisfy all constraints
- **Training Data**: Filtered, verified trajectories for SFT and RL
- **Agent Training**: Standard supervised fine-tuning on verified trajectories

## Implementation Steps

CoVe involves defining constraints, generating trajectories, and training agents. Here's how to implement it:

Define explicit constraints that specify valid task trajectories. Constraints encode domain logic:

```python
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Constraint:
    """Base class for task constraints."""
    name: str
    description: str

    def verify(self, trajectory: List[Dict]) -> bool:
        """Check if trajectory satisfies this constraint."""
        raise NotImplementedError

class AirlineBookingConstraints:
    """Constraints for airline booking tasks."""

    class HasFlight(Constraint):
        def __init__(self):
            super().__init__(
                "has_flight",
                "Trajectory must include flight selection action"
            )

        def verify(self, trajectory: List[Dict]) -> bool:
            actions = [step.get('action') for step in trajectory]
            return 'select_flight' in actions or 'confirm_flight' in actions

    class HasPassengers(Constraint):
        def __init__(self):
            super().__init__(
                "has_passengers",
                "Must specify passenger details"
            )

        def verify(self, trajectory: List[Dict]) -> bool:
            for step in trajectory:
                if step.get('action') == 'enter_passenger_info':
                    return 'passengers' in step and len(step['passengers']) > 0
            return False

    class ValidPayment(Constraint):
        def __init__(self):
            super().__init__(
                "valid_payment",
                "Payment must be processed"
            )

        def verify(self, trajectory: List[Dict]) -> bool:
            for step in trajectory:
                if step.get('action') == 'process_payment':
                    payment = step.get('payment_method')
                    return payment in ['credit_card', 'debit_card', 'wallet']
            return False

    def __init__(self):
        self.constraints = [
            self.HasFlight(),
            self.HasPassengers(),
            self.ValidPayment(),
        ]

    def verify_trajectory(self, trajectory: List[Dict]) -> bool:
        """Check if trajectory satisfies all constraints."""
        return all(c.verify(trajectory) for c in self.constraints)

    def get_constraint_prompt(self) -> str:
        """Generate prompt guidance for trajectory generation."""
        prompt = "Generate a valid airline booking trajectory that:\n"
        for i, constraint in enumerate(self.constraints, 1):
            prompt += f"{i}. {constraint.description}\n"
        return prompt
```

Generate trajectories using an LLM, guided by constraints:

```python
def generate_tool_trajectories(
    model,
    constraints,
    num_trajectories=100,
    temperature=0.8,
):
    """
    Generate synthetic tool-use trajectories respecting constraints.
    """
    constraint_guidance = constraints.get_constraint_prompt()

    prompt_template = f"""
    {constraint_guidance}

    Generate a realistic user request and corresponding agent interaction trajectory.
    The trajectory should show the agent using tools to complete the task.

    Format:
    USER_REQUEST: [User's initial request]
    TRAJECTORY:
    [Step 1]: action=..., parameters={{...}}
    [Step 2]: action=..., parameters={{...}}
    ...
    CONFIRMATION: [Final booking/result]
    """

    trajectories = []
    verified_count = 0

    for _ in range(num_trajectories):
        # Generate trajectory
        generation = model.generate(
            prompt_template,
            max_length=500,
            temperature=temperature,
            num_return_sequences=1,
        )[0]

        # Parse trajectory
        parsed = parse_trajectory_output(generation)

        # Verify constraints
        if constraints.verify_trajectory(parsed['steps']):
            trajectories.append(parsed)
            verified_count += 1

    print(f"Generated {verified_count}/{num_trajectories} valid trajectories")
    return trajectories

def parse_trajectory_output(text: str) -> Dict:
    """Parse model output into structured trajectory."""
    lines = text.split('\n')
    trajectory = {
        'request': '',
        'steps': [],
        'confirmation': ''
    }

    current_section = None
    for line in lines:
        if 'USER_REQUEST:' in line:
            trajectory['request'] = line.split('USER_REQUEST:')[1].strip()
            current_section = 'request'
        elif 'TRAJECTORY:' in line:
            current_section = 'steps'
        elif 'CONFIRMATION:' in line:
            trajectory['confirmation'] = line.split('CONFIRMATION:')[1].strip()
            current_section = 'confirmation'
        elif current_section == 'steps' and line.strip():
            # Parse step: [Step N]: action=..., parameters={...}
            step_data = parse_step_line(line)
            trajectory['steps'].append(step_data)

    return trajectory

def parse_step_line(line: str) -> Dict:
    """Parse individual trajectory step."""
    import re
    # Example: [Step 1]: action=select_flight, parameters={'flight_id': 'AA123'}
    match = re.search(r'action=(\w+),\s*parameters=(\{.*\})', line)
    if match:
        return {
            'action': match.group(1),
            'parameters': eval(match.group(2))  # In practice, use safer parsing
        }
    return {}
```

Create training data from verified trajectories:

```python
def create_training_data(trajectories, constraints):
    """
    Convert verified trajectories into supervised fine-tuning examples.
    """
    training_examples = []

    for trajectory in trajectories:
        # Verify one more time before adding to training
        if not constraints.verify_trajectory(trajectory['steps']):
            continue

        # Create prompt-response pairs
        user_request = trajectory['request']

        # Multi-turn conversation: user request + agent actions
        conversation = [
            {'role': 'user', 'content': user_request}
        ]

        for step in trajectory['steps']:
            action_str = f"Action: {step['action']}\n"
            params_str = f"Parameters: {step['parameters']}"
            conversation.append({
                'role': 'assistant',
                'content': f"{action_str}{params_str}"
            })

        training_examples.append({
            'conversation': conversation,
            'trajectory': trajectory,
            'verified': True
        })

    return training_examples

# Usage
constraints = AirlineBookingConstraints()
trajectories = generate_tool_trajectories(
    model,
    constraints,
    num_trajectories=1000
)
training_data = create_training_data(trajectories, constraints)
```

Train agent on verified trajectories:

```python
def train_tool_use_agent(
    model,
    training_data,
    num_epochs=3,
    learning_rate=1e-5,
):
    """
    Fine-tune model on constraint-verified tool-use data.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for example in training_data:
            # Format conversation for training
            conversation_text = format_conversation(example['conversation'])

            # Standard language modeling loss
            outputs = model(conversation_text)
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    return model

def format_conversation(conversation):
    """Format multi-turn conversation for LLM fine-tuning."""
    formatted = ""
    for turn in conversation:
        role = turn['role'].upper()
        content = turn['content']
        formatted += f"{role}: {content}\n"
    return formatted
```

## Practical Guidance

**When to Use:**
- Tool-use agents (booking, shopping, information retrieval)
- Domain-specific tasks with clear business rules
- When manual trajectory curation is expensive
- For production agent deployment

**When NOT to Use:**
- Open-ended generation tasks without clear constraints
- When defining comprehensive constraints is infeasible
- Real-time systems needing immediate deployment (generation takes time)

**Constraint Design:**
- Start with 3–5 core constraints covering essential requirements
- Make constraints verifiable (deterministic checks, no subjective judgment)
- Include preconditions (what must be true before task) and postconditions (what must be true after)
- Test constraints on manually curated examples first

**Generation and Filtering:**
- Generate 5–10x more trajectories than you need to account for verification failures
- Monitor verification success rate; if <50%, constraints may be too strict
- Increase temperature slightly (0.7–0.9) to encourage diverse trajectories

**Training:**
- Use standard SFT first, then optional RL for refinement
- 4B parameter models can achieve competitive results with ~4000 verified examples
- Test on target tasks to ensure agent learns tool-use patterns

**Results:**
- Airline booking: 43% success (4B model vs. larger baselines)
- Retail tasks: 59.4% success
- Competitive with models 17x larger when using constraint-verified data

---

**Reference:** [CoVe: Training Interactive Tool-Use Agents via Constraint-Guided Verification](https://arxiv.org/abs/2603.01940)
