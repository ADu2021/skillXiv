---
name: ifdecorator-instruction-following-rl
title: IFDecorator - Instruction Following RL with Verifiable Rewards
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: https://arxiv.org/abs/2508.04632
keywords: [reinforcement-learning, instruction-following, reward-verification, reward-hacking]
description: "Enhances RLVR through cooperative-adversarial flywheel, intent verification, and trap instructions. Detects reward hacking and improves training efficiency, achieving 87.43% on IFEval."
---

# IFDecorator: Instruction Following RL with Verifiable Rewards

## Core Concept

IFDecorator strengthens Reinforcement Learning with Verifiable Rewards (RLVR) by addressing two critical challenges: training inefficiency from inadequate difficulty assessment, and reward hacking where models exploit verification shortcuts without aligning to true user intent. The framework uses a cooperative-adversarial flywheel to progressively increase task difficulty, enforces alignment through intent checking, and detects shortcut exploitation through trap instructions.

## Architecture Overview

- **Cooperative-Adversarial Flywheel**: Co-evolves instructions and verifications to generate progressively challenging pairs
- **IntentCheck Module**: Verifies that model outputs align with actual user intent
- **Trip Wires**: Diagnostic system using trap instructions to detect reward hacking
- **Sample Efficiency**: Systematically increases task difficulty rather than random sampling
- **Robustness**: Detects and prevents exploitation of verification shortcuts

## Implementation Steps

### Step 1: Build the Cooperative-Adversarial Flywheel

Create a system that co-evolves instruction-verification pairs with increasing difficulty.

```python
class CooperativeAdversarialFlywheel:
    """
    Co-evolve instructions and verifications for progressive difficulty.
    """

    def __init__(self, generator_model, verifier_model, initial_difficulty=1.0):
        self.generator = generator_model
        self.verifier = verifier_model
        self.difficulty = initial_difficulty
        self.pairs = []

    def generate_instruction_pair(self, topic, difficulty_level):
        """
        Generate instruction-verification pair at target difficulty.

        Args:
            topic: Task topic or category
            difficulty_level: Target difficulty (1.0 = easy, 10.0 = hard)

        Returns:
            (instruction, verification) tuple
        """
        prompt = f"""
        Generate a challenging instruction for task: {topic}
        Difficulty level: {difficulty_level}/10.0

        The instruction should:
        1. Be clear and unambiguous
        2. Require non-trivial following to succeed
        3. Have measurable completion criteria

        Return as JSON with "instruction" and "completion_criteria" fields.
        """

        instruction_pair = self.generator.generate(prompt)

        return instruction_pair

    def generate_verification(self, instruction, completion_criteria):
        """
        Generate verifier function for instruction.

        Args:
            instruction: Task instruction
            completion_criteria: Criteria for successful completion

        Returns:
            Verifier function
        """
        prompt = f"""
        Create a verification function for:
        Instruction: {instruction}
        Criteria: {completion_criteria}

        Return Python code for a function that:
        - Takes model output as input
        - Returns True if instruction followed correctly
        - Returns False if any criteria not met

        Make the verification robust to minor variations.
        """

        verifier_code = self.generator.generate(prompt)

        return self._compile_verifier(verifier_code)

    def update_difficulty(self, model_success_rate):
        """
        Adapt difficulty based on model performance.

        Args:
            model_success_rate: Proportion of tasks successfully completed
        """
        # If model succeeds >80%, increase difficulty
        if model_success_rate > 0.8:
            self.difficulty = min(10.0, self.difficulty * 1.2)
        # If model succeeds <20%, decrease difficulty
        elif model_success_rate < 0.2:
            self.difficulty = max(1.0, self.difficulty / 1.2)

    def _compile_verifier(self, verifier_code):
        """Safely compile and return verifier function."""
        import types
        # Execute in restricted namespace
        exec_globals = {"True": True, "False": False}
        exec(verifier_code, exec_globals)
        return exec_globals.get("verify_instruction")
```

### Step 2: Implement IntentCheck Module

Create a module that ensures outputs align with true user intent, not just verification criteria.

```python
class IntentCheckModule:
    """
    Verify alignment with user intent beyond mechanical verification.
    """

    def __init__(self, intent_model, divergence_threshold=0.3):
        self.intent_model = intent_model
        self.divergence_threshold = divergence_threshold

    def check_intent_alignment(self, instruction, user_intent, model_output):
        """
        Check if model output aligns with true user intent.

        Args:
            instruction: Original instruction
            user_intent: Deep explanation of what user actually wants
            model_output: Model's generated response

        Returns:
            (aligned, alignment_score, reasoning)
        """
        # Extract intent from instruction
        prompt = f"""
        Analyze whether this output truly fulfills the user's intent:

        User Intent: {user_intent}
        Model Output: {model_output}

        Consider:
        1. Does output serve the underlying goal?
        2. Would the user be satisfied?
        3. Are there shortcuts that exploit verification but miss intent?

        Return JSON with:
        - "aligned": true/false
        - "score": 0.0-1.0
        - "reasoning": brief explanation
        """

        result = self.intent_model.generate(prompt)

        alignment = result["aligned"]
        score = result["score"]

        return alignment, score, result.get("reasoning", "")

    def detect_intent_divergence(self, outputs_batch, intents_batch):
        """
        Detect systematic divergence between outputs and intents.

        Args:
            outputs_batch: Batch of model outputs
            intents_batch: Corresponding user intents

        Returns:
            List of (output_idx, divergence_score) tuples
        """
        divergences = []

        for idx, (output, intent) in enumerate(zip(outputs_batch, intents_batch)):
            aligned, score, _ = self.check_intent_alignment("", intent, output)

            if score < self.divergence_threshold:
                divergences.append((idx, score))

        return divergences
```

### Step 3: Implement Trip Wires for Reward Hacking Detection

Create diagnostic traps that detect models exploiting shortcuts.

```python
class TripWireDetector:
    """
    Diagnostic system using trap instructions to detect reward hacking.
    """

    def __init__(self):
        self.trap_instructions = []
        self.hacking_indicators = {}

    def create_trap_instruction(self, base_instruction, exploit_path):
        """
        Create instruction that model should handle correctly.

        Args:
            base_instruction: Original instruction
            exploit_path: Potential shortcut/hack the model might use

        Returns:
            Trap instruction that exposes the shortcut
        """
        # Create instruction where the exploit would fail
        trap = f"""
        {base_instruction}

        IMPORTANT: The following will NOT work:
        - {exploit_path}

        You must solve this correctly without the shortcut.
        """

        self.trap_instructions.append({
            "base": base_instruction,
            "exploit": exploit_path,
            "trap": trap
        })

        return trap

    def test_for_hacking(self, model, instruction, expected_output):
        """
        Test if model is using exploits or solving genuinely.

        Args:
            model: Model to test
            instruction: Original instruction
            expected_output: Correct output

        Returns:
            (is_hacking, confidence, evidence)
        """
        # Generate output on original
        original_output = model.generate(instruction)

        # Generate output on trap version
        trap_instruction = self.create_trap_instruction(instruction, "unknown")
        trap_output = model.generate(trap_instruction)

        # Compare outputs
        if self._similar_patterns(original_output, trap_output):
            # Model likely using shortcut
            return True, 0.8, "Similar output on trap instruction"
        else:
            # Model adapted behavior
            return False, 0.8, "Different behavior on trap"

    def _similar_patterns(self, output1, output2):
        """Check if outputs follow similar patterns (hacking indicator)."""
        from difflib import SequenceMatcher

        ratio = SequenceMatcher(None, output1, output2).ratio()
        return ratio > 0.7

    def create_exploit_detection_set(self):
        """
        Create comprehensive set of trap instructions.

        Returns:
            Structured set of traps for model evaluation
        """
        common_exploits = [
            "verbosity without substance",
            "format compliance without content",
            "keyword stuffing",
            "logical shortcuts in reasoning",
            "copying from examples without adaptation"
        ]

        traps = []
        for exploit in common_exploits:
            trap = {
                "exploit_type": exploit,
                "trap_instruction": self._generate_trap_for_exploit(exploit),
                "success_indicator": self._get_success_indicator(exploit)
            }
            traps.append(trap)

        return traps

    def _generate_trap_for_exploit(self, exploit_type):
        """Generate trap for specific exploit type."""
        traps = {
            "verbosity": "Provide answer in single sentence",
            "format_compliance": "Answer must violate format constraints",
            "keyword_stuffing": "Answer cannot contain common keywords"
        }
        return traps.get(exploit_type, "")

    def _get_success_indicator(self, exploit_type):
        """Get indicator of successful mitigation."""
        return lambda output: len(output) < 100
```

### Step 4: Integrate into RLVR Training Loop

Combine all components into cohesive RLVR training system.

```python
def train_instruction_following_with_ifdecorator(
    model,
    initial_instructions,
    config
):
    """
    Train model with IFDecorator framework.

    Args:
        model: LLM to train
        initial_instructions: Starting instruction set
        config: Training configuration

    Returns:
        Trained model with improved instruction following
    """
    flywheel = CooperativeAdversarialFlywheel(
        model,
        verifier_model=load_verifier(),
        initial_difficulty=config.initial_difficulty
    )

    intent_checker = IntentCheckModule()
    trap_detector = TripWireDetector()

    # Training loop
    for episode in range(config.num_episodes):
        # 1. Generate progressive difficulty examples
        instruction, verification = flywheel.generate_instruction_pair(
            topic="instruction_following",
            difficulty_level=flywheel.difficulty
        )

        # 2. Get model response
        model_output = model.generate(instruction)

        # 3. Check verification
        verified = verification(model_output)

        # 4. Check intent alignment
        aligned, intent_score, _ = intent_checker.check_intent_alignment(
            instruction,
            user_intent="Genuine task completion",
            model_output=model_output
        )

        # 5. Detect reward hacking
        is_hacking, hack_confidence, _ = trap_detector.test_for_hacking(
            model,
            instruction,
            expected_output="correct"
        )

        # Compute reward
        reward = 0.0
        if verified:
            reward += config.verification_reward

        if aligned:
            reward += config.intent_reward

        if not is_hacking:
            reward += config.genuine_solving_reward
        else:
            reward -= config.hacking_penalty

        # 6. RL update
        model.update_from_reward(instruction, model_output, reward)

        # 7. Update difficulty
        if episode % config.difficulty_update_freq == 0:
            success_rate = calculate_recent_success_rate(model)
            flywheel.update_difficulty(success_rate)

    return model
```

## Practical Guidance

### When to Use IFDecorator

- **Instruction following benchmarks**: IFEval, MTEval where intent matters
- **Complex multi-step tasks**: Where shortcuts exist but don't solve genuine problems
- **Safety-critical systems**: Where reward hacking has real consequences
- **Dynamic task difficulty**: Where progressive training improves sample efficiency

### When NOT to Use IFDecorator

- **Simple verification tasks**: Where verification and intent perfectly align
- **Low-risk applications**: Reward hacking mitigation adds computational cost
- **Real-time training**: Trap instruction generation and testing add latency
- **Insufficient intent labels**: Framework requires explicit intent statements

### Hyperparameter Recommendations

- **Initial difficulty**: 2.0-3.0 on 1-10 scale
- **Verification reward**: 0.5 (baseline)
- **Intent reward**: 0.3-0.5 (emphasize intent)
- **Hacking penalty**: 1.0-2.0 (strong negative signal)
- **Difficulty update frequency**: Every 20-50 episodes

### Key Insights

The key innovation is explicit detection and prevention of reward hacking through trip wires. By creating instructions where common exploits fail, the system forces genuine problem-solving. The cooperative-adversarial flywheel ensures efficient training by focusing on task-specific difficulty, not random sampling.

## Reference

**IFDecorator: Instruction Following RL with Verifiable Rewards** (arXiv:2508.04632)

Introduces cooperative-adversarial flywheel, IntentCheck module, and trip wire detector system for robust RLVR. Achieves 87.43% on IFEval by preventing reward hacking while improving training efficiency.
