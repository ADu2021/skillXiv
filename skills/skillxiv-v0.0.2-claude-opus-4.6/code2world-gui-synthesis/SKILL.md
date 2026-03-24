---
name: code2world-gui-synthesis
title: "Code2World: A GUI World Model via Renderable Code Generation"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2602.09856"
keywords: [GUI World Model, HTML Generation, Structured Prediction, Vision-Language, Deterministic Rendering]
description: "Predict GUI state evolution by generating HTML code rather than pixel images. Combines visual fidelity of pixel-based approaches with structural precision of code-based methods through deterministic rendering. Enables agents to evaluate action consequences and select best decisions before execution."
---

# Code2World: HTML-Based GUI World Modeling

Predicting GUI state changes requires balancing visual realism against structural controllability. Pixel-based models achieve realism but lack precision; text-only code lacks visual grounding. Code2World bridges this gap: VLMs generate HTML code for the next GUI state, then deterministically render it to visual images. This unifies the strengths of both approaches while enabling agents to anticipate action consequences with both visual and structural clarity.

## Core Concept

Standard pixel prediction: image + action → predicted next image. Realistic but spatially imprecise; hard for agents to verify if buttons appear in correct locations.

Code2World: image + action → predicted HTML code → render to image. Generates structured representation (HTML) that captures layout, text, and interaction patterns, then renders deterministically for visual output. Agents can parse HTML for precise interaction targets or render for visual context.

## Architecture Overview

- **Code Generation Stage**: VLM generates HTML code representing next GUI state
- **Deterministic Rendering**: Render HTML to image pixel-perfectly
- **Dual Representation**: Structured code (for parsing) + rendered image (for visual grounding)
- **Training Strategy**: Supervised fine-tuning (HTML generation) + RL (optimize for action consistency)
- **Action Evaluation**: Agents can sample multiple actions, render outcomes, and select best before execution

## Implementation

Implement HTML generation and rendering:

```python
from selenium import webdriver
from PIL import Image
import io
import torch
import torch.nn as nn

class Code2WorldModel(nn.Module):
    """VLM fine-tuned for HTML generation."""

    def __init__(self, base_vlm_model):
        super().__init__()
        self.vlm = base_vlm_model

    def predict_next_html(self, current_screenshot, action_description, temperature=0.7):
        """
        Generate HTML for next GUI state.
        Args:
            current_screenshot: PIL Image of current GUI
            action_description: str describing action to perform
        Returns:
            html_code: str with HTML markup for next state
        """
        # Prompt the VLM
        prompt = f"""Given this GUI screenshot and the action '{action_description}',
generate the HTML code for the next GUI state. Include all visible elements, their positions,
text content, and styling. Return valid HTML only.

Previous state screenshot: [provided]

Output HTML code for next state:
```html
"""

        # Generate HTML
        html_output = self.vlm.generate(
            current_screenshot,
            prompt,
            max_tokens=500,
            temperature=temperature
        )

        # Extract HTML from response
        html_code = self._extract_html(html_output)
        return html_code

    def _extract_html(self, text):
        """Extract HTML block from model output."""
        import re

        # Find content between ```html and ```
        match = re.search(r'```html\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1)
        return text

class HTMLRenderer:
    """Deterministically render HTML to images."""

    def __init__(self, viewport_width=1280, viewport_height=720):
        self.width = viewport_width
        self.height = viewport_height
        self.driver = self._init_selenium()

    def _init_selenium(self):
        """Initialize headless browser for rendering."""
        from selenium.webdriver.chrome.options import Options

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument(f"--window-size={self.width},{self.height}")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")

        driver = webdriver.Chrome(options=chrome_options)
        return driver

    def render_html_to_image(self, html_code):
        """Render HTML to PNG image."""
        # Create data URL
        data_url = f"data:text/html,{html_code}"

        # Navigate to URL
        self.driver.get(data_url)

        # Take screenshot
        screenshot = self.driver.get_screenshot_as_png()
        image = Image.open(io.BytesIO(screenshot))

        return image

    def close(self):
        self.driver.quit()

class Code2WorldWorldModel:
    """Full world model with HTML generation + rendering."""

    def __init__(self, vlm_model, viewport_size=(1280, 720)):
        self.code_gen = Code2WorldModel(vlm_model)
        self.renderer = HTMLRenderer(viewport_size[0], viewport_size[1])

    def predict_next_state(self, current_screenshot, action):
        """Predict next GUI state given action."""
        # Generate HTML
        html = self.code_gen.predict_next_html(current_screenshot, action)

        # Render to image
        next_screenshot = self.renderer.render_html_to_image(html)

        return next_screenshot, html

    def evaluate_actions(self, current_screenshot, candidate_actions, num_samples=3):
        """
        Evaluate multiple candidate actions and select best.
        Args:
            current_screenshot: PIL Image
            candidate_actions: List of action descriptions
            num_samples: Number of rollouts per action
        Returns:
            best_action: Str with highest expected reward
            predictions: Dict mapping actions to predicted screenshots
        """
        predictions = {}
        scores = {}

        for action in candidate_actions:
            rollout_images = []
            rollout_scores = []

            # Sample multiple predictions for uncertainty
            for _ in range(num_samples):
                next_img, html = self.predict_next_state(current_screenshot, action)
                rollout_images.append(next_img)

                # Score prediction (e.g., via reward model)
                score = self._score_prediction(next_img, action)
                rollout_scores.append(score)

            predictions[action] = rollout_images[0]  # Store best
            scores[action] = sum(rollout_scores) / len(rollout_scores)

        # Select best action
        best_action = max(scores, key=scores.get)

        return best_action, predictions, scores

    def _score_prediction(self, predicted_image, action):
        """Score quality of prediction (action following + visual fidelity)."""
        # Example scoring function
        # In practice, use trained reward model

        # Check for common visual artifacts
        artifact_penalty = 0.0
        # ... artifact detection logic ...

        # Score action consistency
        action_consistency = 0.5  # Placeholder
        # ... action evaluation logic ...

        return action_consistency - artifact_penalty
```

Integrate into training with RL:

```python
def code2world_training_step(code_gen, renderer, batch, optimizer):
    """Training step combining SFT + RL."""
    current_screenshots, actions, target_screenshots = batch

    # Generate HTML predictions
    predicted_htmls = []
    for i, (screenshot, action) in enumerate(zip(current_screenshots, actions)):
        html = code_gen.predict_next_html(screenshot, action)
        predicted_htmls.append(html)

    # Render predictions
    predicted_screenshots = []
    for html in predicted_htmls:
        img = renderer.render_html_to_image(html)
        predicted_screenshots.append(img)

    # Compute rewards
    # 1. Visual similarity (LPIPS or SSIM)
    visual_loss = F.mse_loss(
        torch.tensor(predicted_screenshots),
        torch.tensor(target_screenshots)
    )

    # 2. Action consistency (does predicted state reflect action?)
    action_consistency = evaluate_action_consistency(predicted_screenshots, actions)

    # Combined loss
    loss = 0.7 * visual_loss + 0.3 * (1.0 - action_consistency)

    # Update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

## Practical Guidance

| Component | Recommendation | Notes |
|-----------|-----------------|-------|
| HTML specification | HTML5 subset | Stick to standard features; complex CSS breaks rendering. |
| Viewport size | 1280×720 | Standard; adjust per target interface. |
| Rendering engine | Headless Chrome/Firefox | Consistent rendering; use same engine always. |
| Training data | 1K+ screenshots | Need diverse GUI states; pair with real trajectories. |
| RL weight | 30-50% | Balance supervised signal with action consistency. |

**When to Use**
- GUI agent needs to evaluate action consequences before executing
- Precise spatial reasoning matters (button locations, text regions)
- Generating diverse action rollouts for planning
- Interactive environments where agent makes sequential decisions

**When NOT to Use**
- Simple static page prediction (pixel models sufficient)
- Complex 3D or game interfaces (HTML inadequate)
- When rendering performance is critical (rendering adds latency)

**Common Pitfalls**
- HTML generation too verbose/invalid; use prompt engineering for concise valid code
- Rendering inconsistencies between training and deployment browsers; standardize
- Not handling dynamic content (JavaScript, animations); code2world assumes static output
- Over-fitting to training screenshots; augment data and regularize

## Reference

See https://arxiv.org/abs/2602.09856 for full architecture, including action evaluation, GUI agent baselines, and validation on real-world web navigation tasks.
