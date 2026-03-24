---
name: synthagent-web-adaptation
title: "Adapting Web Agents with Synthetic Supervision"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.06101"
keywords: [Web Automation, Synthetic Data, Domain Adaptation, Data Quality, LLM Agents]
description: "Adapt web agents to new domains through targeted synthetic data generation and quality-aware refinement—identifying and correcting hallucinations while preserving task consistency to enable efficient adaptation with minimal human supervision."
---

# Adapt Web Agents Through Quality-Focused Synthetic Data

Adapting language model agents to new websites typically requires collecting human demonstrations for each new domain. SynthAgent (Synthetic Supervision) automatically generates task examples through exploratory interactions with target websites, then applies quality-aware refinement to fix hallucinations and inconsistencies. The approach targets two critical failure modes: hallucinated tasks (tasks never tested against actual website) and noisy trajectories (incorrect action sequences).

By separating data generation from quality control, the system achieves efficient domain adaptation without expensive human labeling.

## Core Concept

SynthAgent implements a three-stage pipeline:

1. **Task Synthesis** - Explore website UI systematically to generate diverse task examples
2. **Conflict-Driven Refinement** - When task descriptions conflict with observed behavior, correct them
3. **Global Trajectory Refinement** - Post-hoc cleanup ensuring action sequences match tasks

This approach treats synthetic data generation as exploration with implicit labels, then applies targeted corrections only where conflicts appear—reducing false positives while fixing genuine errors.

## Architecture Overview

- **Exploratory Agent**: Navigates website discovering UI elements and interactions
- **Task Generator**: Creates task descriptions from observed interactions
- **Conflict Detector**: Identifies discrepancies between tasks and actual website behavior
- **Online Refinement**: Corrects tasks when conflicts detected
- **Trajectory Validator**: Verifies action sequences match task descriptions
- **Offline Polish**: Final refinement using global context

## Implementation Steps

**Step 1: Systematic Web Exploration and Task Synthesis**

Discover website structure and generate task examples through structured exploration.

```python
from typing import List, Dict, Set, Tuple
import json

class WebExplorer:
    """
    Systematically explores website and generates task examples.
    """

    def __init__(self, browser, base_url: str, max_pages: int = 100):
        """
        Args:
            browser: Selenium WebDriver or equivalent
            base_url: Starting URL
            max_pages: Maximum pages to explore
        """
        self.browser = browser
        self.base_url = base_url
        self.max_pages = max_pages

        self.explored_pages = set()
        self.discovered_tasks = []
        self.ui_elements = {}

    def explore_website(self) -> List[Dict]:
        """
        Systematically explore website and extract tasks.

        Returns:
            tasks: Generated task examples
        """
        to_explore = [self.base_url]
        visited_urls = set()

        while to_explore and len(visited_urls) < self.max_pages:
            url = to_explore.pop(0)

            if url in visited_urls:
                continue

            visited_urls.add(url)

            try:
                # Navigate to page
                self.browser.get(url)
                self._extract_page_info(url)

                # Find new links
                links = self._extract_links(url)
                for link in links:
                    if link not in visited_urls and len(visited_urls) < self.max_pages:
                        to_explore.append(link)

            except Exception as e:
                print(f"Error exploring {url}: {e}")
                continue

        return self.discovered_tasks

    def _extract_page_info(self, url: str):
        """Extract interactive elements and generate tasks."""
        page_source = self.browser.page_source
        current_url = self.browser.current_url

        # Find form elements
        forms = self.browser.find_elements("tag name", "form")
        for form in forms:
            task = self._form_to_task(form, current_url)
            if task:
                self.discovered_tasks.append(task)

        # Find clickable elements
        buttons = self.browser.find_elements("tag name", "button")
        for button in buttons:
            task = self._button_to_task(button, current_url)
            if task:
                self.discovered_tasks.append(task)

        # Find input patterns
        inputs = self.browser.find_elements("tag name", "input")
        for input_elem in inputs:
            task = self._input_to_task(input_elem, current_url)
            if task:
                self.discovered_tasks.append(task)

    def _form_to_task(self, form, url: str) -> Dict:
        """Convert form to task example."""
        try:
            form_id = form.get_attribute("id") or "form"
            labels = form.find_elements("tag name", "label")

            # Generate task description
            task_desc = f"Complete the form on {url}: "
            task_desc += ", ".join([l.text for l in labels[:3]])

            return {
                'type': 'form_fill',
                'description': task_desc,
                'url': url,
                'target': form_id,
                'elements': [l.text for l in labels]
            }
        except:
            return None

    def _button_to_task(self, button, url: str) -> Dict:
        """Convert button interaction to task."""
        try:
            button_text = button.text
            if len(button_text) < 50:  # Reasonable button text length
                return {
                    'type': 'click',
                    'description': f"Click '{button_text}' button",
                    'url': url,
                    'target': button_text
                }
        except:
            pass
        return None

    def _input_to_task(self, input_elem, url: str) -> Dict:
        """Convert input field to task."""
        try:
            input_type = input_elem.get_attribute("type")
            placeholder = input_elem.get_attribute("placeholder")
            name = input_elem.get_attribute("name")

            if placeholder:
                return {
                    'type': 'input',
                    'description': f"Enter {placeholder} in input field",
                    'url': url,
                    'input_type': input_type,
                    'placeholder': placeholder
                }
        except:
            pass
        return None

    def _extract_links(self, url: str) -> List[str]:
        """Extract navigable links from page."""
        links = []
        try:
            elements = self.browser.find_elements("tag name", "a")
            for elem in elements:
                href = elem.get_attribute("href")
                if href and href.startswith('http'):
                    links.append(href)
        except:
            pass
        return links
```

**Step 2: Conflict Detection and Online Refinement**

Identify discrepancies between generated tasks and actual website behavior.

```python
class ConflictDetector:
    """
    Detects conflicts between task descriptions and actual website behavior.
    """

    def __init__(self, browser):
        self.browser = browser

    def detect_conflicts(self, task: Dict) -> Tuple[bool, str]:
        """
        Check if task description matches actual website behavior.

        Args:
            task: Task example with description and target

        Returns:
            has_conflict: Whether conflict detected
            reason: Explanation if conflict found
        """
        try:
            # Navigate to task URL
            self.browser.get(task['url'])

            # Verify target element exists
            target = task['target']

            if task['type'] == 'form_fill':
                # Check form exists
                form = self.browser.find_element("id", target)
                if not form:
                    return True, f"Form {target} not found"

            elif task['type'] == 'click':
                # Check button exists and is visible
                buttons = self.browser.find_elements("tag name", "button")
                found = False
                for btn in buttons:
                    if btn.text == target:
                        found = True
                        if not btn.is_displayed():
                            return True, f"Button '{target}' not visible"
                        break

                if not found:
                    return True, f"Button '{target}' not found"

            elif task['type'] == 'input':
                # Check input exists
                inputs = self.browser.find_elements("tag name", "input")
                found = False
                for inp in inputs:
                    placeholder = inp.get_attribute("placeholder")
                    if placeholder == task.get('placeholder'):
                        found = True
                        break

                if not found:
                    return True, f"Input field not found"

            return False, ""

        except Exception as e:
            return True, str(e)

    def refine_task(self, task: Dict) -> Dict:
        """
        Correct task description to match actual website.

        Args:
            task: Original task

        Returns:
            refined_task: Corrected task
        """
        has_conflict, reason = self.detect_conflicts(task)

        if has_conflict:
            # Correct the task
            if "not found" in reason:
                # Task references non-existent element; remove or correct
                return None  # Discard unreliable task

            elif "not visible" in reason:
                # Element exists but hidden; mark as conditional
                task['conditional'] = True
                task['condition'] = "Element must be unhidden first"

        return task
```

**Step 3: Trajectory Validation and Refinement**

Verify that action sequences correspond to task descriptions.

```python
class TrajectoryValidator:
    """
    Validates and refines action trajectories against tasks.
    """

    def __init__(self, llm_api):
        self.llm = llm_api

    def validate_trajectory(self, task: Dict, trajectory: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if trajectory actually accomplishes task.

        Args:
            task: Task description
            trajectory: List of action descriptions

        Returns:
            is_valid: Whether trajectory achieves task
            corrected: Corrected action sequence if invalid
        """
        # Use LLM to reason about action sequence
        prompt = f"""Does this action sequence accomplish the task?

Task: {task['description']}

Actions:
{chr(10).join(trajectory)}

Respond with yes/no and explain."""

        response = self.llm.generate(prompt, max_tokens=200)

        is_valid = "yes" in response.lower()

        if not is_valid:
            # Ask LLM to correct trajectory
            correction_prompt = f"""The above trajectory doesn't achieve the task.
Generate corrected actions:

Task: {task['description']}

Corrected action sequence:"""

            corrected_str = self.llm.generate(correction_prompt, max_tokens=300)
            corrected = corrected_str.strip().split('\n')
        else:
            corrected = trajectory

        return is_valid, corrected

    def refine_trajectories(self, tasks_with_trajectories: List[Tuple[Dict, List[str]]],
                          batch_size: int = 10) -> List[Dict]:
        """
        Batch validate and refine trajectories.

        Args:
            tasks_with_trajectories: (task, trajectory) pairs
            batch_size: Batch processing size

        Returns:
            refined_examples: Validated task-trajectory pairs
        """
        refined = []

        for task, trajectory in tasks_with_trajectories:
            is_valid, corrected = self.validate_trajectory(task, trajectory)

            if is_valid:
                refined.append({
                    'task': task,
                    'trajectory': trajectory,
                    'validated': True
                })
            else:
                # Discard if correction is major (probable hallucination)
                if len(corrected) > len(trajectory) * 1.5:
                    continue  # Too different; discard

                refined.append({
                    'task': task,
                    'trajectory': corrected,
                    'validated': True,
                    'corrected': True
                })

        return refined
```

**Step 4: Integrated Adaptation Pipeline**

Combine exploration, refinement, and validation into domain adaptation pipeline.

```python
def adapt_agent_to_new_domain(base_agent, target_website_url: str,
                             browser, llm_api,
                             max_synthetic_examples: int = 100):
    """
    Adapt web agent to new domain using synthetic supervision.

    Args:
        base_agent: Pre-trained web agent
        target_website_url: URL of target website
        browser: Browser instance
        llm_api: LLM for reasoning
        max_synthetic_examples: Target number of examples

    Returns:
        adapted_agent: Fine-tuned agent for new domain
    """
    print("=== Synthetic Supervision Adaptation ===")

    # Step 1: Explore website and generate tasks
    print("Exploring website...")
    explorer = WebExplorer(browser, target_website_url)
    synthetic_tasks = explorer.explore_website()[:max_synthetic_examples]
    print(f"Generated {len(synthetic_tasks)} task candidates")

    # Step 2: Conflict detection and online refinement
    print("Detecting and correcting conflicts...")
    conflict_detector = ConflictDetector(browser)
    refined_tasks = []

    for task in synthetic_tasks:
        refined = conflict_detector.refine_task(task)
        if refined:
            refined_tasks.append(refined)

    print(f"Retained {len(refined_tasks)} after conflict removal")

    # Step 3: Generate trajectories and validate
    print("Generating and validating trajectories...")
    validator = TrajectoryValidator(llm_api)

    refined_examples = []
    for task in refined_tasks:
        # Generate trajectory from task
        trajectory = base_agent.predict_trajectory(task['description'])

        # Validate
        is_valid, corrected = validator.validate_trajectory(task, trajectory)

        if is_valid:
            refined_examples.append({
                'task': task['description'],
                'trajectory': corrected,
                'domain': 'new_domain'
            })

    print(f"Final training set: {len(refined_examples)} examples")

    # Step 4: Fine-tune agent on synthetic examples
    print("Fine-tuning agent...")
    import torch.optim as optim

    optimizer = optim.Adam(base_agent.parameters(), lr=1e-5)

    for epoch in range(3):
        total_loss = 0

        for example in refined_examples:
            # Forward pass
            logits = base_agent.forward(example['task'])

            # Supervise on trajectory
            trajectory_ids = base_agent.tokenize_trajectory(example['trajectory'])
            loss = compute_trajectory_loss(logits, trajectory_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss {total_loss / len(refined_examples):.4f}")

    return base_agent
```

## Practical Guidance

**When to Use SynthAgent:**
- Adapting web agents to new website domains
- Scenarios where human demonstration collection is expensive
- Situations where target website changes require re-adaptation

**When NOT to Use:**
- Domains requiring high-precision behaviors (synthetic data may be noisy)
- Websites with complex JavaScript rendering (difficult to explore)
- Tasks without clear success/failure signals

**Hyperparameters and Configuration:**
- Exploration depth: 100-500 pages (balance coverage with time)
- Conflict threshold: Discard if multiple conflicting elements found
- Trajectory length budget: 5-15 steps (longer sequences harder to learn)
- Refinement iterations: 1-2 passes over generated data

**Pitfalls to Avoid:**
1. **Stale website snapshots** - Website layout changes between exploration and use; re-validate periodically
2. **Over-filtering** - Discarding all tasks with minor conflicts removes valuable data
3. **LLM hallucination** - Trajectory generation may hallucinate actions; validate against actual website
4. **Domain shift** - Synthetic distribution may not match test distribution; monitor performance on real tasks

---

Reference: https://arxiv.org/abs/2511.06101
