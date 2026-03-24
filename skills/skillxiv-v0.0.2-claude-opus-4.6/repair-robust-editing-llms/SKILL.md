---
name: repair-robust-editing-llms
title: "REPAIR: Robust Editing via Progressive Adaptive Intervention and Reintegration"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2510.01879"
keywords: [model editing, knowledge correction, lifelong learning, knowledge preservation, sequential edits]
description: "Perform robust, low-cost model updates via REPAIR: closed-loop feedback, dynamic memory management, and frequent knowledge fusion. Preserve non-target knowledge while achieving 10-30% improvements in editing accuracy. Enable sequential edits without catastrophic forgetting through strong locality guards and stable integration mechanisms."
---

# REPAIR: Robust Editing via Progressive Adaptive Intervention

## Core Concept

Model editing enables correcting errors and updating knowledge post-training, but naive approaches cause collateral damage (breaking other capabilities) and fail under sequential edits (conflicting updates). REPAIR uses closed-loop feedback, adaptive memory, and frequent knowledge fusion to enable stable, non-destructive lifelong editing.

## Architecture Overview

- **Closed-Loop Feedback**: Monitor edit success; iterate if needed rather than one-shot updates
- **Dynamic Memory Management**: Track edit context and affected knowledge regions
- **Knowledge Fusion**: Periodically integrate edit effects to prevent accumulated drift
- **Strong Locality Guards**: Constrains edits to affect only target knowledge
- **Sequential Stability**: Support multiple edits without interference

## Implementation Steps

### 1. Targeted Edit Specification

Define what to edit and what to preserve explicitly.

```python
class EditSpecification:
    def __init__(self, model, target_fact, desired_output, preservation_facts=None):
        """
        Specify an edit: change a specific fact while preserving others.

        Args:
            model: Target language model
            target_fact: Fact to edit (e.g., "Eiffel Tower is in France")
            desired_output: Correct answer
            preservation_facts: List of facts that MUST remain unchanged
        """
        self.model = model
        self.target_fact = target_fact
        self.desired_output = desired_output
        self.preservation_facts = preservation_facts or []

    def validate_edit(self):
        """
        Verify edit target makes sense and is editable.
        """
        # Check model currently has wrong knowledge
        current_output = self.model.generate(self.target_fact)

        if current_output == self.desired_output:
            print("⚠ Model already has correct knowledge")
            return False

        # Check preservation facts exist
        for fact in self.preservation_facts:
            fact_statement = fact['statement']
            fact_answer = self.model.generate(fact_statement)
            if fact_answer != fact['expected']:
                print(f"⚠ Preservation fact already incorrect: {fact_statement}")
                return False

        return True
```

### 2. Closed-Loop Feedback Mechanism

Monitor edit effectiveness and iterate if needed.

```python
class ClosedLoopEditor:
    def __init__(self, model, max_iterations=5):
        self.model = model
        self.max_iterations = max_iterations
        self.edit_history = []

    def edit_with_feedback(self, edit_spec, learning_rate=1e-5):
        """
        Iterative editing with feedback on success.
        """

        for iteration in range(self.max_iterations):
            print(f"Edit iteration {iteration + 1}/{self.max_iterations}")

            # Step 1: Identify neurons/parameters responsible for target fact
            target_neurons = self._identify_responsible_neurons(edit_spec.target_fact)

            # Step 2: Compute edit direction
            edit_direction = self._compute_edit_direction(
                edit_spec.target_fact,
                edit_spec.desired_output,
                target_neurons
            )

            # Step 3: Apply update
            self._apply_constrained_update(edit_direction, learning_rate)

            # Step 4: Evaluate success
            success_score = self._evaluate_edit(edit_spec)

            print(f"  Success score: {success_score:.2%}")

            # Step 5: Check preservation (side effects)
            preservation_loss = self._evaluate_preservation(edit_spec.preservation_facts)

            print(f"  Preservation loss: {preservation_loss:.4f}")

            # Record iteration
            self.edit_history.append({
                'iteration': iteration,
                'success': success_score,
                'preservation': preservation_loss,
                'learning_rate': learning_rate
            })

            # Early stopping: success threshold
            if success_score > 0.9 and preservation_loss < 0.05:
                print(f"✓ Edit successful at iteration {iteration + 1}")
                return True

            # Adapt learning rate
            if success_score < 0.3:
                learning_rate *= 0.5  # Reduce if overshooting

        print(f"✗ Edit incomplete after {self.max_iterations} iterations")
        return False

    def _identify_responsible_neurons(self, target_fact):
        """
        Find which neurons encode target fact via gradient analysis.
        """
        # Simplified: neurons with highest gradient w.r.t. target fact
        gradients = self.model.compute_gradients(target_fact)

        # Select top-k neurons
        top_k = 100
        responsible_idx = gradients.argsort()[-top_k:]

        return responsible_idx

    def _compute_edit_direction(self, target_fact, desired_output, target_neurons):
        """
        Compute direction to update neuron values.
        """

        current_output = self.model.generate(target_fact)

        # Difference vector: what needs to change
        # Simplified: use gradient of loss w.r.t. neuron outputs
        loss = self.model.compute_loss(target_fact, desired_output)
        edit_direction = torch.autograd.grad(loss, target_neurons, allow_unused=True)

        return edit_direction

    def _apply_constrained_update(self, edit_direction, learning_rate):
        """
        Apply update with locality constraints.
        """

        for param_idx, grad in enumerate(edit_direction):
            if grad is not None:
                # Update with learning rate
                param = self.model.get_parameter(param_idx)

                # Constraint: limit update magnitude (locality guard)
                max_update = 0.01 * torch.abs(param.data).mean()
                grad = torch.clamp(grad, -max_update, max_update)

                param.data -= learning_rate * grad

    def _evaluate_edit(self, edit_spec):
        """
        Measure how well edit succeeded.
        """

        output = self.model.generate(edit_spec.target_fact)
        success = 1.0 if output == edit_spec.desired_output else 0.0

        return success

    def _evaluate_preservation(self, preservation_facts):
        """
        Measure knowledge preservation (side effects).
        """

        total_loss = 0

        for fact in preservation_facts:
            output = self.model.generate(fact['statement'])
            loss = 0.0 if output == fact['expected'] else 1.0
            total_loss += loss

        return total_loss / max(len(preservation_facts), 1)
```

### 3. Dynamic Memory and Knowledge Fusion

Track edits and periodically fuse to prevent drift.

```python
class DynamicMemoryManager:
    def __init__(self, model, fusion_interval=10):
        self.model = model
        self.edits = []  # Track all edits
        self.edit_counter = 0
        self.fusion_interval = fusion_interval
        self.knowledge_buffer = {}  # Snapshot of model knowledge

    def record_edit(self, edit_spec, success):
        """
        Record completed edit in memory.
        """

        self.edits.append({
            'spec': edit_spec,
            'success': success,
            'timestamp': self.edit_counter
        })

        self.edit_counter += 1

        # Periodic knowledge fusion
        if self.edit_counter % self.fusion_interval == 0:
            self.fuse_edits()

    def fuse_edits(self):
        """
        Integrate accumulated edits to prevent drift.
        Resets baseline and re-applies edits cleanly.
        """

        print(f"Fusing {len(self.edits)} accumulated edits...")

        # Snapshot current state
        pre_fusion_state = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }

        # Reapply successful edits
        successful_edits = [e for e in self.edits if e['success']]

        for edit in successful_edits:
            # Reapply edit
            editor = ClosedLoopEditor(self.model, max_iterations=2)
            editor.edit_with_feedback(edit['spec'])

        # Verify no drift
        drift = self._compute_parameter_drift(pre_fusion_state)
        print(f"Post-fusion parameter drift: {drift:.4f}")

    def _compute_parameter_drift(self, reference_state):
        """
        Measure divergence from reference state.
        """

        total_drift = 0

        for (ref_name, ref_param), (name, param) in zip(
            reference_state.items(),
            self.model.named_parameters()
        ):
            if ref_name == name:
                drift = torch.norm(param.data - ref_param) / (torch.norm(ref_param) + 1e-8)
                total_drift += drift.item()

        return total_drift / len(reference_state)
```

### 4. Sequential Editing Without Catastrophic Forgetting

Enable multiple edits with conflict detection.

```python
class SequentialEditManager:
    def __init__(self, model):
        self.model = model
        self.memory_manager = DynamicMemoryManager(model)
        self.editor = ClosedLoopEditor(model)
        self.edit_graph = {}  # Track edit dependencies/conflicts

    def batch_edits(self, edit_specs, detect_conflicts=True):
        """
        Apply multiple edits sequentially with conflict management.
        """

        results = []

        for idx, edit_spec in enumerate(edit_specs):
            print(f"\nEdit {idx+1}/{len(edit_specs)}: {edit_spec.target_fact}")

            # Check for conflicts with prior edits
            if detect_conflicts:
                conflicts = self._detect_conflicts(edit_spec, results)
                if conflicts:
                    print(f"  ⚠ Potential conflicts detected: {conflicts}")

            # Apply edit
            success = self.editor.edit_with_feedback(edit_spec)
            results.append({'edit': edit_spec, 'success': success})

            # Record in memory
            self.memory_manager.record_edit(edit_spec, success)

            # Track in edit graph
            self.edit_graph[idx] = {
                'spec': edit_spec,
                'conflicts': conflicts if detect_conflicts else []
            }

        return results

    def _detect_conflicts(self, edit_spec, prior_edits):
        """
        Identify if new edit conflicts with prior edits.
        """

        conflicts = []

        for prior_result in prior_edits:
            prior_spec = prior_result['edit']

            # Simple conflict: edits affect overlapping facts
            if self._edits_overlap(edit_spec, prior_spec):
                conflicts.append(f"Overlaps with: {prior_spec.target_fact}")

        return conflicts

    def _edits_overlap(self, spec1, spec2):
        """
        Check if two edits potentially conflict.
        """

        # Overlap if target facts share knowledge regions
        overlap_score = self.model.compute_semantic_similarity(
            spec1.target_fact,
            spec2.target_fact
        )

        return overlap_score > 0.5
```

## Practical Guidance

**Edit Granularity**: Single, focused facts edit better than broad generalizations. "Eiffel Tower is in Paris" edits better than "All facts about Eiffel Tower."

**Preservation Facts**: Specify 3-5 related facts that must be preserved. More preservation facts slow editing but improve robustness.

**Learning Rate**: Start at 1e-5, reduce if overshooting. Adaptive rates based on success score work well.

**Fusion Interval**: Every 5-10 edits fuse to prevent accumulated drift. More frequent fusion is safer but slower.

## When to Use / When NOT to Use

**Use When**:
- Correcting factual errors post-deployment
- Updating knowledge as new information arrives
- Models need graceful degradation when edits fail
- Sequential edits on diverse facts

**NOT For**:
- Bulk retraining (dedicated fine-tuning more efficient)
- Adversarial robustness (edits don't improve worst-case)
- Tasks where strong consistency guarantees needed

## Reference

This skill synthesizes findings from "REPAIR: Robust Editing via Progressive Adaptive Intervention and Reintegration" (arXiv:2510.01879). Closed-loop feedback and knowledge fusion enable stable, low-cost editing.
