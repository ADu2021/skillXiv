---
name: gui-360-desktop-agent-dataset
title: "GUI-360°: Comprehensive Dataset for Computer-Using Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.04307"
keywords: [Desktop Agents, GUI Grounding, Action Prediction, Multimodal Agents, Autonomous Agents]
description: "Enable training and evaluation of desktop computer-using agents through 1.2M action steps across diverse Windows applications, covering GUI grounding, screen parsing, and action prediction with hybrid GUI+API action space reflecting modern agent designs."
---

# Title: Build Comprehensive Training Data for Desktop Autonomous Agents

Desktop agents that autonomously complete computer tasks face unique challenges: grounding spatial coordinates, parsing complex UIs, predicting appropriate next actions. GUI-360° provides 1.2M executed action steps across thousands of real trajectories in Word, Excel, and PowerPoint, with full screenshots, accessibility metadata, task goals, reasoning traces, and both successful and failed action sequences. The dataset reveals that state-of-the-art models struggle with grounding and action prediction despite strong general capabilities.

The benchmark spans three canonical tasks: GUI grounding, screen parsing, and action prediction.

## Core Concept

**Comprehensive Desktop Agent Training and Evaluation**:
- **Large-Scale Data**: 1.2M action steps, 13,750 trajectories across three Office applications
- **Rich Annotations**: Screenshots + accessibility metadata + task descriptions + reasoning + failure cases
- **Hybrid Action Space**: GUI actions (click, type, drag) + app-specific APIs (table insertion, cell editing)
- **Three Evaluation Tasks**: Grounding (screen→coordinate), parsing (screen→UI elements), prediction (state→action)
- **Real-World Complexity**: Diverse task types, variable UI layouts, complex tool interactions

## Architecture Overview

- **Data Collection**: Instrumented Windows environment recording human demonstrations
- **Annotation Pipeline**: Automatic extraction of actions, coordinates, UI elements via accessibility APIs
- **Task Specification**: Goal descriptions with reasoning traces from human performers
- **Corpus Composition**: 41% Word, 31.6% Excel, 27.4% PowerPoint
- **Split Design**: Train/test maintaining consistent application distributions

## Implementation Steps

**1. Design Data Collection Pipeline**

Capture human demonstrations with full metadata.

```python
class DesktopAgentDataCollector:
    def __init__(self, recording_dir="./gui_recordings"):
        self.recording_dir = recording_dir
        self.current_trajectory = None

    def start_recording(self, task_description):
        """Begin recording a new trajectory"""
        self.current_trajectory = {
            'task': task_description,
            'steps': [],
            'screenshots': [],
            'metadata': {
                'app': None,
                'start_time': datetime.now(),
                'reasoning_notes': []
            }
        }

    def capture_step(self, action_type, coordinates=None, text=None, keystroke=None):
        """Record a single action step"""
        # Capture screenshot before action
        screenshot = self.screenshot_full_screen()

        # Get accessibility tree
        a11y_tree = self.get_accessibility_tree()

        # Record action
        step = {
            'action': {
                'type': action_type,  # 'click', 'type', 'drag', etc.
                'coordinates': coordinates,
                'text': text,
                'keystroke': keystroke
            },
            'screenshot': screenshot,
            'accessibility_tree': a11y_tree,
            'timestamp': datetime.now()
        }

        self.current_trajectory['steps'].append(step)
        return len(self.current_trajectory['steps'])

    def end_recording(self, task_completed, final_notes=""):
        """Finalize trajectory recording"""
        self.current_trajectory['metadata']['success'] = task_completed
        self.current_trajectory['metadata']['notes'] = final_notes
        self.current_trajectory['metadata']['end_time'] = datetime.now()

        # Save to disk
        trajectory_id = self.save_trajectory(self.current_trajectory)
        return trajectory_id

    def screenshot_full_screen(self):
        """Capture current screen state"""
        import pyautogui
        screenshot = pyautogui.screenshot()
        return np.array(screenshot)

    def get_accessibility_tree(self):
        """Extract UI element structure using accessibility APIs"""
        import pywinauto
        app = pywinauto.GetFocusedWindow()

        # Build tree of accessible elements
        a11y_tree = self.build_tree(app)
        return a11y_tree

    def build_tree(self, element, max_depth=5, current_depth=0):
        """Recursively build accessibility tree"""
        if current_depth > max_depth:
            return None

        tree = {
            'element': element.element_info.name,
            'role': element.element_info.control_type,
            'rectangle': element.rectangle().to_dict(),
            'is_enabled': element.is_enabled(),
            'is_visible': element.is_visible(),
            'children': []
        }

        # Get children if applicable
        try:
            for child in element.children():
                child_tree = self.build_tree(child, max_depth, current_depth + 1)
                if child_tree:
                    tree['children'].append(child_tree)
        except:
            pass

        return tree
```

**2. Implement Three Evaluation Tasks**

Structure benchmark around distinct technical challenges.

```python
class GUIAgentBenchmark:
    def __init__(self, dataset_path):
        self.trajectories = self.load_trajectories(dataset_path)

    def task1_gui_grounding(self, model):
        """Predict screen coordinates for target actions"""
        results = []

        for trajectory in self.trajectories:
            for step in trajectory['steps']:
                # Input: screenshot + action description
                # Output: coordinate prediction
                screenshot = step['screenshot']
                action_desc = step['action']['type']

                if step['action']['type'] == 'click':
                    true_coords = step['action']['coordinates']

                    # Model predicts coordinate
                    pred_coords = model.predict_click_location(screenshot, action_desc)

                    # Distance metric
                    distance = np.linalg.norm(np.array(pred_coords) - np.array(true_coords))

                    results.append({
                        'task': 'grounding',
                        'distance_error': distance,
                        'correct': distance < 50  # Within 50 pixels
                    })

        accuracy = np.mean([r['correct'] for r in results])
        return {'accuracy': accuracy, 'results': results}

    def task2_screen_parsing(self, model):
        """Predict UI elements and their properties"""
        results = []

        for trajectory in self.trajectories:
            for step in trajectory['steps']:
                screenshot = step['screenshot']
                true_a11y_tree = step['accessibility_tree']

                # Model parses screen
                pred_elements = model.parse_screen(screenshot)

                # Match predictions to ground truth
                # Metrics: precision, recall, F1 on element detection
                metrics = self.compute_parsing_metrics(pred_elements, true_a11y_tree)
                results.append(metrics)

        avg_f1 = np.mean([r['f1'] for r in results])
        return {'f1': avg_f1, 'results': results}

    def task3_action_prediction(self, model):
        """Predict next action given current state and goal"""
        results = []

        for trajectory in self.trajectories:
            for i in range(len(trajectory['steps']) - 1):
                current_step = trajectory['steps'][i]
                next_step = trajectory['steps'][i + 1]

                # Input: current screenshot + goal description
                screenshot = current_step['screenshot']
                goal = trajectory['task']

                # Ground truth next action
                true_action = next_step['action']

                # Model predicts action
                pred_action = model.predict_next_action(screenshot, goal)

                # Accuracy: action type matches
                action_correct = pred_action['type'] == true_action['type']

                results.append({
                    'action_correct': action_correct,
                    'action_type': true_action['type']
                })

        accuracy = np.mean([r['action_correct'] for r in results])
        accuracy_by_type = {}
        for action_type in set(r['action_type'] for r in results):
            type_results = [r for r in results if r['action_type'] == action_type]
            accuracy_by_type[action_type] = np.mean([r['action_correct'] for r in type_results])

        return {'accuracy': accuracy, 'by_type': accuracy_by_type, 'results': results}

    def compute_parsing_metrics(self, predicted, ground_truth):
        """Compare parsed UI elements to ground truth"""
        # Extract ground truth elements
        gt_elements = self.extract_elements(ground_truth)

        # Precision: predicted elements that match ground truth
        true_positives = 0
        for pred in predicted:
            for gt in gt_elements:
                if self.elements_match(pred, gt):
                    true_positives += 1
                    break

        precision = true_positives / len(predicted) if predicted else 0
        recall = true_positives / len(gt_elements) if gt_elements else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {'precision': precision, 'recall': recall, 'f1': f1}
```

**3. Implement Hybrid Action Execution**

Support both GUI and API-based actions.

```python
class HybridActionExecutor:
    def __init__(self, app_type='word'):
        self.app_type = app_type
        self.gui_executor = GUIActionExecutor()
        self.api_executor = APIActionExecutor(app_type)

    def execute_action(self, action_spec):
        """Execute action using appropriate method"""
        action_type = action_spec['type']

        if action_type in ['click', 'type', 'drag', 'scroll']:
            # GUI action
            return self.gui_executor.execute(action_spec)
        elif action_type in ['table_insert', 'cell_set', 'slide_modify']:
            # API action (app-specific)
            return self.api_executor.execute(action_spec)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

class GUIActionExecutor:
    def execute(self, action_spec):
        """Execute GUI action via mouse/keyboard"""
        import pyautogui
        action_type = action_spec['type']

        if action_type == 'click':
            pyautogui.click(action_spec['coordinates'])
        elif action_type == 'type':
            pyautogui.typewrite(action_spec['text'], interval=0.05)
        elif action_type == 'drag':
            start = action_spec['start_coords']
            end = action_spec['end_coords']
            pyautogui.moveTo(start)
            pyautogui.drag(end[0] - start[0], end[1] - start[1], duration=0.5)
        elif action_type == 'scroll':
            amount = action_spec['amount']
            pyautogui.scroll(amount)

        return {'success': True}

class APIActionExecutor:
    def __init__(self, app_type):
        self.app_type = app_type
        self.excel_api = ExcelAPI() if app_type == 'excel' else None
        self.word_api = WordAPI() if app_type == 'word' else None

    def execute(self, action_spec):
        """Execute action via application API"""
        if self.app_type == 'excel':
            return self.excel_api.execute(action_spec)
        elif self.app_type == 'word':
            return self.word_api.execute(action_spec)

class ExcelAPI:
    def execute(self, action_spec):
        """Execute Excel-specific actions"""
        from openpyxl import load_workbook

        action_type = action_spec['type']

        if action_type == 'cell_set':
            wb = load_workbook(action_spec['file'])
            ws = wb[action_spec['sheet']]
            ws[action_spec['cell']] = action_spec['value']
            wb.save(action_spec['file'])

        elif action_type == 'auto_fill':
            ws = self.get_worksheet(action_spec)
            start_cell = action_spec['start']
            fill_range = action_spec['range']
            # Implementation details...

        return {'success': True}
```

## Practical Guidance

**When to Use**:
- Training desktop/office automation agents
- Evaluating grounding, parsing, action prediction capabilities
- Developing systems for document/data processing automation

**Hyperparameters**:
- Screenshot resolution: 1920×1080 (standardize environments)
- Action timeout: 5s per action
- Max trajectory length: 50-100 steps (typical task size)

**When NOT to Use**:
- Mobile or web application agents (different UI paradigms)
- Custom enterprise applications with unique toolbars
- Real-time system interaction (latency-sensitive)

**Pitfalls**:
- **Accessibility tree incompleteness**: Some custom UI elements not exposed via APIs; supplement with vision
- **Coordinate brittleness**: Screen resolution changes invalidate coordinates; use relative positions
- **Task ambiguity**: Unclear task descriptions harm learning; use detailed goal specifications

**Integration Strategy**: Use as pre-training data for agents. Fine-tune on domain-specific applications after. Combine with synthetic data augmentation for scaling.

## Reference

arXiv: https://arxiv.org/abs/2511.04307
