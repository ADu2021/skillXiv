---
name: open-cua-computer-agents
title: OpenCUA - Open Foundations for Computer-Use Agents
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.09123
keywords: [computer-use-agents, agent-scaling, chain-of-thought, dataset-annotation, reflection]
description: "Scales computer-use agent capabilities through reflective Chain-of-Thought reasoning in large-scale annotated datasets spanning multiple operating systems and 200+ applications."
---

## OpenCUA: Open Foundations for Computer-Use Agents

### Core Concept

OpenCUA provides comprehensive foundations for computer-use agents through three interconnected components: an annotation infrastructure that captures human-computer interactions, the AgentNet dataset spanning 3 operating systems and 200+ applications, and a transformation pipeline converting demonstrations into state-action pairs with reflective Chain-of-Thought reasoning. This approach enables robust agent scaling with improved reasoning patterns.

### Architecture Overview

- **Human Demonstration Capture**: Seamlessly record human interactions with computers
- **Large-Scale AgentNet Dataset**: 3 operating systems, 200+ applications/websites
- **Reflective CoT Transformation**: Convert raw demonstrations to reasoning-annotated pairs
- **State-Action Representation**: Structured data for agent training
- **Multi-OS Support**: Windows, macOS, Linux compatibility

### Implementation Steps

**Step 1: Implement Interaction Capture Infrastructure**

Record human demonstrations:

```python
# Pseudocode for demonstration capture
class InteractionCapture:
    def __init__(self):
        super().__init__()
        self.current_session = None
        self.interactions = []

    def capture_session_start(self, app_name, window_title):
        """
        Initialize capture for new application interaction.

        Args:
            app_name: Name of application
            window_title: Window title at start

        Returns:
            session_id: Unique session identifier
        """
        session_id = str(uuid.uuid4())

        self.current_session = {
            'session_id': session_id,
            'app_name': app_name,
            'window_title': window_title,
            'start_time': time.time(),
            'interactions': [],
            'screenshots': []
        }

        return session_id

    def capture_action(self, action_type, details):
        """
        Capture single user action.

        Args:
            action_type: Type of action (click, type, scroll, etc)
            details: Action-specific details

        Returns:
            action_record: Captured action with metadata
        """
        action_record = {
            'timestamp': time.time(),
            'type': action_type,
            'details': details,
            'screenshot_before': self._capture_screenshot(),
            'window_state': self._get_window_state()
        }

        self.current_session['interactions'].append(action_record)
        self.interactions.append(action_record)

        return action_record

    def capture_click(self, x, y, button='left'):
        """
        Capture mouse click action.
        """
        action = {
            'type': 'click',
            'x': x,
            'y': y,
            'button': button
        }

        return self.capture_action('click', action)

    def capture_type(self, text):
        """
        Capture text input action.
        """
        action = {
            'type': 'type',
            'text': text,
            'length': len(text)
        }

        return self.capture_action('type', action)

    def capture_scroll(self, direction, amount):
        """
        Capture scroll action.
        """
        action = {
            'type': 'scroll',
            'direction': direction,  # up, down, left, right
            'amount': amount
        }

        return self.capture_action('scroll', action)

    def capture_screenshot(self):
        """
        Capture current screen.
        """
        import pyautogui

        screenshot = pyautogui.screenshot()
        self.current_session['screenshots'].append({
            'timestamp': time.time(),
            'image': screenshot
        })

        return screenshot

    def _capture_screenshot(self):
        """
        Capture screenshot for interaction record.
        """
        import pyautogui
        return pyautogui.screenshot()

    def _get_window_state(self):
        """
        Get current window state.
        """
        import pygetwindow

        active_window = pygetwindow.getActiveWindow()
        return {
            'title': active_window.title if active_window else None,
            'geometry': (active_window.left, active_window.top,
                        active_window.width, active_window.height)
            if active_window else None
        }

    def end_session(self):
        """
        Finalize current session.

        Returns:
            session: Completed session with all interactions
        """
        if self.current_session:
            self.current_session['end_time'] = time.time()
            self.current_session['duration'] = (
                self.current_session['end_time'] - self.current_session['start_time']
            )

            completed = self.current_session.copy()
            self.current_session = None

            return completed

        return None
```

**Step 2: Create Reflective CoT Annotation Pipeline**

Convert demonstrations to reasoning-annotated training data:

```python
# Pseudocode for reflective CoT generation
class ReflectiveCoTAnnotator:
    def __init__(self, reasoning_model):
        super().__init__()
        self.reasoning_model = reasoning_model

    def generate_cot_for_interaction(self, interaction_sequence, goal):
        """
        Generate Chain-of-Thought explanation for action sequence.

        Args:
            interaction_sequence: List of captured actions
            goal: High-level goal being accomplished

        Returns:
            annotated_sequence: Actions with reasoning
        """
        annotated = []

        context = f"Goal: {goal}\n\nActions taken:"

        for action_idx, action in enumerate(interaction_sequence):
            # Generate reasoning for this action
            action_description = self._describe_action(action)

            reasoning_prompt = f"""{context}
{action_description}

Why was this action taken? What was the agent trying to achieve?"""

            with torch.no_grad():
                reasoning = self.reasoning_model.generate(
                    reasoning_prompt,
                    max_length=150,
                    temperature=0.7
                )

            annotated_record = {
                'action': action,
                'description': action_description,
                'reasoning': reasoning,
                'screenshot_before': action.get('screenshot_before'),
                'window_state': action.get('window_state')
            }

            annotated.append(annotated_record)
            context += f"\n{action_idx+1}. {action_description}"

        return annotated

    def _describe_action(self, action):
        """
        Convert action to natural language description.
        """
        action_type = action['type']

        if action_type == 'click':
            return f"Clicked at position ({action['x']}, {action['y']})"
        elif action_type == 'type':
            return f"Typed: '{action['text']}'"
        elif action_type == 'scroll':
            return f"Scrolled {action['direction']} by {action['amount']}"
        else:
            return str(action)

    def annotate_demonstration(self, session, goal_statement):
        """
        Full annotation of demonstration session.

        Args:
            session: Captured interaction session
            goal_statement: Natural language goal

        Returns:
            annotated_session: Complete annotated demonstration
        """
        annotated_interactions = self.generate_cot_for_interaction(
            session['interactions'],
            goal_statement
        )

        return {
            'session_id': session['session_id'],
            'app_name': session['app_name'],
            'goal': goal_statement,
            'interactions': annotated_interactions,
            'duration': session.get('duration'),
            'success': self._assess_success(annotated_interactions)
        }

    def _assess_success(self, annotated_interactions):
        """
        Determine if task was completed successfully.
        """
        # Check final state or explicit success signals
        return len(annotated_interactions) > 0
```

**Step 3: Build State-Action Pair Representation**

Convert annotated demonstrations to training format:

```python
# Pseudocode for state-action pair creation
class StateActionPairBuilder:
    def __init__(self, vision_encoder):
        super().__init__()
        self.vision_encoder = vision_encoder

    def build_training_pair(self, annotated_interaction):
        """
        Convert annotated interaction to state-action training pair.

        Args:
            annotated_interaction: Single interaction with reasoning

        Returns:
            training_pair: Structured state-action pair
        """
        # State: screenshot + context + goal
        screenshot = annotated_interaction['screenshot_before']
        screen_embedding = self.vision_encoder.encode(screenshot)

        state = {
            'visual': screen_embedding,
            'window_state': annotated_interaction['window_state'],
            'history': annotated_interaction.get('action_history', []),
            'goal': annotated_interaction.get('goal')
        }

        # Action: what the human did
        action = annotated_interaction['action']

        # Reasoning: why they did it
        reasoning = annotated_interaction['reasoning']

        return {
            'state': state,
            'action': action,
            'reasoning': reasoning,
            'description': annotated_interaction['description']
        }

    def build_dataset(self, annotated_sessions):
        """
        Convert multiple sessions to training dataset.

        Args:
            annotated_sessions: List of annotated demonstration sessions

        Returns:
            training_dataset: Ready-to-use training data
        """
        training_pairs = []

        for session in annotated_sessions:
            for interaction in session['interactions']:
                pair = self.build_training_pair(interaction)
                training_pairs.append(pair)

        return {
            'pairs': training_pairs,
            'num_pairs': len(training_pairs),
            'apps': set(s['app_name'] for s in annotated_sessions),
            'num_sessions': len(annotated_sessions)
        }
```

**Step 4: Implement Agent Training on Dataset**

Train computer-use agents:

```python
# Pseudocode for agent training
class ComputerUseAgentTrainer:
    def __init__(self, model, vision_encoder):
        super().__init__()
        self.model = model
        self.vision_encoder = vision_encoder

    def train_agent(self, training_dataset, num_epochs=3):
        """
        Train agent on state-action-reasoning data.

        Args:
            training_dataset: Built from demonstrations
            num_epochs: Training epochs

        Returns:
            trained_agent: Ready for deployment
        """
        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        for epoch in range(num_epochs):
            total_loss = 0

            for pair in training_dataset['pairs']:
                state = pair['state']
                action = pair['action']
                reasoning = pair['reasoning']

                # Encode state
                state_embedding = self._encode_state(state)

                # Forward pass: predict action and reasoning
                predicted_action, predicted_reasoning = self.model(state_embedding)

                # Compute loss
                action_loss = self._action_loss(predicted_action, action)
                reasoning_loss = self._reasoning_loss(predicted_reasoning, reasoning)

                total_loss_step = 0.7 * action_loss + 0.3 * reasoning_loss

                # Backward
                optimizer.zero_grad()
                total_loss_step.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += total_loss_step.item()

            print(f"Epoch {epoch+1}: Loss = {total_loss / len(training_dataset['pairs']):.4f}")

        return self.model

    def _encode_state(self, state):
        """
        Encode state representation for model input.
        """
        # Combine visual and contextual information
        visual = state['visual']
        context_text = str(state['window_state']) + "\n" + state.get('goal', '')

        # Encode text context
        context_embedding = self.model.tokenizer.encode(context_text)

        return {
            'visual': visual,
            'context': context_embedding
        }

    def _action_loss(self, predicted, ground_truth):
        """
        Loss for action prediction.
        """
        # Compare action distributions
        return F.cross_entropy(predicted, self._encode_action(ground_truth))

    def _reasoning_loss(self, predicted, ground_truth):
        """
        Loss for reasoning generation.
        """
        # Language modeling loss
        return F.cross_entropy(predicted, self.model.tokenizer.encode(ground_truth))

    def _encode_action(self, action):
        """
        Encode action for comparison.
        """
        action_type = action.get('type', 'unknown')
        action_mapping = {
            'click': 0,
            'type': 1,
            'scroll': 2,
            'key': 3
        }
        return torch.tensor(action_mapping.get(action_type, -1))

    def evaluate_agent(self, test_tasks, max_steps=50):
        """
        Evaluate agent on test tasks.

        Args:
            test_tasks: Tasks to perform
            max_steps: Maximum actions per task

        Returns:
            results: Performance metrics
        """
        success_count = 0
        total_steps = 0

        for task in test_tasks:
            # Initialize task
            state = self._prepare_task_state(task)

            # Run agent
            step_count = 0
            task_success = False

            while step_count < max_steps:
                # Agent decides action
                with torch.no_grad():
                    action, reasoning = self.model(self._encode_state(state))

                # Execute action
                new_state, task_done, success = self._execute_action(action, state, task)

                if task_done:
                    task_success = success
                    break

                state = new_state
                step_count += 1

            if task_success:
                success_count += 1

            total_steps += step_count

        return {
            'success_rate': success_count / len(test_tasks),
            'avg_steps': total_steps / len(test_tasks),
            'total_successes': success_count
        }

    def _prepare_task_state(self, task):
        """
        Prepare initial state for task.
        """
        return {
            'goal': task['instruction'],
            'visual': None,
            'window_state': {}
        }

    def _execute_action(self, action, state, task):
        """
        Execute agent action in environment.
        """
        # Simplified execution
        return state, False, False
```

### Practical Guidance

**Hyperparameters and Configuration**:
- CoT generation temperature: 0.7-0.9
- Agent training learning rate: 2e-5 to 5e-5
- Action/reasoning loss weight ratio: 0.7/0.3
- Maximum steps per task: 50-100
- Training epochs: 2-5

**When to Use OpenCUA**:
- Building computer-use agents for automation
- Scenarios with diverse applications (need broad coverage)
- Systems where interpretability through reasoning is valuable
- Applications requiring multi-OS support

**When NOT to Use**:
- Simple single-application automation (specialized tools better)
- Real-time systems with strict latency constraints
- Scenarios with limited demonstration data
- When screen complexity is very high

**Implementation Notes**:
- Reflective CoT critical for agent robustness and scaling
- Diverse demonstration dataset essential (3 OSes, 200+ apps)
- Vision encoding quality impacts action prediction
- Monitor success rates across different application domains
- Consider curriculum: start with simple tasks, increase complexity

### Reference

Paper: OpenCUA: Open Foundations for Computer-Use Agents
ArXiv: 2508.09123
Performance: OpenCUA-72B achieves 45.0% average success rate on OSWorld-Verified, state-of-the-art among open-source models
