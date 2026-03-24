---
name: computer-using-world-model
title: "Computer-Using World Model"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.17365"
keywords: [world modeling, UI simulation, agent planning, desktop automation, think-then-act]
description: "Enable AI agents to safely explore action outcomes before execution by predicting UI state changes in desktop applications. Two-stage approach: first predict textual description of what changes, then synthesize visual representation of resulting screen. Allows agents to compare multiple candidate actions without risky trial-and-error, trained on Microsoft Office interactions (Word, Excel, PowerPoint)."
---

# Computer-Using World Model: Safe Desktop Automation through Outcome Prediction

Web and desktop automation agents face a critical challenge: many actions are irreversible and costly. Deleting a file, overwriting a spreadsheet cell, or closing an unsaved document cannot be undone, yet agents often learn through trial-and-error exploration. Unlike robotics where physical environments are forgiving, software environments demand careful planning before execution.

The standard approach—act first, observe consequences—is unsafe in software contexts where action consequences are immediate and permanent. A safer strategy is to enable agents to simulate action outcomes before committing to execution, supporting "think-then-act" decision-making that compares multiple candidate paths without risky exploration.

## Core Concept

The Computer-Using World Model (CUWM) predicts how desktop applications will change in response to user actions without actually executing them. The system operates in two stages:

1. **Textual Change Prediction**: Generate a natural-language description of what state changes will occur (e.g., "the cell A1 value will become 'Q4 Revenue'")
2. **Visual Synthesis**: Render a synthetic screenshot showing the predicted resulting state

This separation allows the model to focus on decision-relevant information (what changes) separately from appearance details (how it looks), improving generalization and prediction accuracy.

## Architecture Overview

- **State Encoder**: Encode current UI screenshot into a latent representation capturing semantic state (open documents, form field values, selection state)
- **Action Embedding**: Encode the candidate action (click coordinates, text input, keyboard shortcut) into semantic representation
- **Change Predictor**: Decode change description from encoded state + action using causal language modeling
- **Visual Renderer**: Synthesize predicted screenshot using base screenshot + change description, applying element-level modifications
- **Value Aggregator**: For multi-action planning, score predicted outcome against goal representation to guide action selection

## Implementation

Implement a two-stage predictor combining text generation and visual modification:

```python
def predict_ui_state_change(screenshot, action, model):
    """
    Predict what will change when action is taken.
    screenshot: PIL Image of current state
    action: dict with type, target, value (e.g., {'type': 'click', 'target': (x, y)})
    Returns: (change_description, predicted_screenshot)
    """
    # Encode current screenshot to latent state
    state_latent = model.encode_screenshot(screenshot)
    action_latent = model.embed_action(action)

    # Predict textual change
    change_description = model.predict_change(
        state_latent, action_latent, temperature=0.3
    )

    # Synthesize visual outcome
    predicted_screenshot = model.render_screenshot(
        screenshot, change_description, action
    )

    return change_description, predicted_screenshot
```

Implement multi-action planning by comparing outcomes:

```python
def plan_action_sequence(
    screenshot, goal, candidate_actions, model, num_lookahead=3
):
    """
    Evaluate multiple action candidates and select best.
    Avoids execution until final action selected.
    """
    outcomes = []

    for action in candidate_actions:
        change_desc, pred_screenshot = predict_ui_state_change(
            screenshot, action, model
        )

        # Score predicted outcome against goal
        outcome_score = model.score_against_goal(
            pred_screenshot, goal, change_desc
        )

        outcomes.append({
            'action': action,
            'prediction': change_desc,
            'screenshot': pred_screenshot,
            'score': outcome_score
        })

    # Rank by score and return best action
    best_outcome = max(outcomes, key=lambda x: x['score'])
    return best_outcome['action']
```

Implement training on observed (action, outcome) pairs from interaction traces:

```python
def train_world_model(
    interaction_traces, model, optimizer, num_epochs=10
):
    """
    Train on pairs of (screenshot_before, action, screenshot_after).
    interaction_traces: list of trajectories with screenshots and actions
    """
    for epoch in range(num_epochs):
        total_loss = 0

        for trajectory in interaction_traces:
            for step_idx in range(len(trajectory) - 1):
                before_screenshot = trajectory[step_idx]['screenshot']
                action = trajectory[step_idx]['action']
                after_screenshot = trajectory[step_idx + 1]['screenshot']

                # Predict change
                pred_change, pred_screenshot = predict_ui_state_change(
                    before_screenshot, action, model
                )

                # Compute losses
                text_loss = model.change_loss(
                    pred_change, trajectory[step_idx + 1]['change_description']
                )
                image_loss = model.visual_loss(pred_screenshot, after_screenshot)

                loss = text_loss + image_loss
                total_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Epoch {epoch}: avg loss = {total_loss / len(interaction_traces)}")
```

## Practical Guidance

| Parameter | Default | Guidance |
|---|---|---|
| Change prediction temperature | 0.3 | Lower (0.1–0.2) for deterministic predictions; higher for diversity |
| Visual synthesis method | Copy + modify | Use OCR-based element detection for text field updates |
| Action space | 50–100 candidates | Score top-k (k=5) actions to avoid redundant computation |
| Lookahead depth | 3 steps | Increase for planning-heavy tasks; limit for speed |

**When to use**: For autonomous desktop automation agents (RPA, form filling, document editing) where mistakes are costly or irreversible.

**When not to use**: For simple click-and-wait tasks where exploration risk is low; overhead of prediction may exceed benefit.

**Common pitfalls**:
- Predicting static page elements that don't change; filter predictions to focus on mutable state (form fields, selection, open dialogs)
- Overfitting to training applications; train on diverse themes (Word, Excel, PowerPoint) to improve generalization
- Not updating visual prediction when text prediction is uncertain; fall back to copy-and-slightly-modify when confidence is low

## Reference

CUWM enables "think-then-act" planning by allowing agents to evaluate action outcomes before execution. Training on Microsoft Office interactions demonstrates the approach scales to complex applications with multiple interacting components and rich state representations.
