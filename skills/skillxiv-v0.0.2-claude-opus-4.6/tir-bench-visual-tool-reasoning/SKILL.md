---
name: tir-bench-visual-tool-reasoning
title: "TIR-Bench: Agentic Thinking-with-Images Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.01833"
keywords: [Visual Reasoning, Tool Use, Multimodal Reasoning, Benchmark, Image Manipulation]
description: "Evaluate multimodal models on agentic visual reasoning through 13 diverse tasks requiring novel tool use for image processing and manipulation in chain-of-thought, revealing that strong performance requires genuine thinking-with-images capabilities."
---

# Title: Test Genuine Visual Reasoning With Interactive Tool Use

Existing visual benchmarks test pattern recognition but not reasoning. TIR-Bench evaluates whether models can act as agents: generating, manipulating, and reasoning over images as part of solving problems. The benchmark includes 13 diverse tasks (color VQA, symbolic reasoning, math, maze solving, etc.) requiring tool creation and deployment—highlighting gaps in current approaches and demonstrating that agentic fine-tuning substantially outperforms direct supervised approaches.

The key insight: visual reasoning is more than recognition—it requires agency.

## Core Concept

**Agentic Visual Reasoning Evaluation**:
- **13 Diverse Tasks**: Spanning perception, reasoning, and tool-dependent operations
- **Tool Requirements**: Enhancement, geometric, programmatic, drawing—models must create and apply tools
- **Two Reasoning Modes**: Direct vs. agentic fine-tuning evaluation
- **1,215 Examples**: 665 multiple-choice + 550 free-form questions
- **Universal Challenge**: Even top models (o3-TU: 46%, non-agentic: 29%) struggle

## Architecture Overview

- **Task Categories**: Vision-centric (puzzles, mazes) + perception (VQA, reading)
- **Evaluation Metrics**: Accuracy for MC + IOU for grounding
- **Model Types Tested**: 22 models spanning open-source and proprietary
- **Fine-Tuning Approaches**: Direct SFT vs. agentic (tool-use enabled)
- **Benchmark Suite**: Tool API definitions + grading functions + visualization tools

## Implementation Steps

**1. Define Task Types and Tool Requirements**

Structure benchmark around tool-use patterns.

```python
class TIRBenchTaskTypes:
    # Vision-centric: require visual reasoning
    VISION_TASKS = {
        'eyeballing_puzzles': {
            'description': 'Estimate relative sizes/distances',
            'tools': ['measurement', 'highlighting', 'comparison'],
            'metric': 'accuracy'
        },
        'visual_puzzles': {
            'description': 'Find patterns in visual layouts',
            'tools': ['annotation', 'tracing', 'comparison'],
            'metric': 'accuracy'
        },
        'maze_solving': {
            'description': 'Navigate mazes',
            'tools': ['path_drawing', 'direction_marking'],
            'metric': 'accuracy'
        },
        'low_light_vqa': {
            'description': 'VQA in dark/low-contrast images',
            'tools': ['enhancement', 'contrast_adjustment'],
            'metric': 'accuracy'
        },
        'jigsaw': {
            'description': 'Complete jigsaw puzzles',
            'tools': ['piece_selection', 'placement_verification'],
            'metric': 'iou'
        },
        'spot_difference': {
            'description': 'Find differences between images',
            'tools': ['highlighting', 'comparison', 'cropping'],
            'metric': 'iou'
        },
        'rotation_game': {
            'description': 'Identify rotation angles',
            'tools': ['rotation', 'angle_measurement'],
            'metric': 'accuracy'
        }
    }

    # Perception-centric: still require tool manipulation
    PERCEPTION_TASKS = {
        'color_vqa': {
            'description': 'Identify colors in images',
            'tools': ['color_picking', 'pixel_analysis'],
            'metric': 'accuracy'
        },
        'proportion_vqa': {
            'description': 'Estimate proportions/ratios',
            'tools': ['measurement', 'calculation'],
            'metric': 'accuracy'
        },
        'symbolic_reasoning': {
            'description': 'Interpret symbols and sequences',
            'tools': ['annotation', 'logic_tracing'],
            'metric': 'accuracy'
        },
        'instrument_reading': {
            'description': 'Read measurements from instruments',
            'tools': ['zooming', 'value_extraction'],
            'metric': 'accuracy'
        },
        'math_problems': {
            'description': 'Solve math from visual format',
            'tools': ['equation_markup', 'calculation'],
            'metric': 'accuracy'
        },
        'word_search': {
            'description': 'Find words in grids',
            'tools': ['highlighting', 'search_marking'],
            'metric': 'accuracy'
        },
        'rotated_ocr': {
            'description': 'Read text at angles',
            'tools': ['rotation', 'ocr_enhancement'],
            'metric': 'accuracy'
        }
    }

    ALL_TASKS = {**VISION_TASKS, **PERCEPTION_TASKS}
```

**2. Implement Tool API**

Define operations models can perform on images.

```python
class VisualToolAPI:
    def __init__(self, image):
        self.image = image
        self.history = []

    def enhance_image(self, brightness=0, contrast=1.0):
        """Adjust brightness and contrast"""
        enhanced = self.image.copy()
        enhanced = enhanced * contrast + brightness
        self.history.append({'tool': 'enhance', 'brightness': brightness, 'contrast': contrast})
        return enhanced

    def highlight_region(self, bbox, color='red', alpha=0.3):
        """Draw attention to region"""
        highlighted = self.image.copy()
        x1, y1, x2, y2 = bbox
        highlighted[y1:y2, x1:x2] = (
            highlighted[y1:y2, x1:x2] * (1 - alpha) +
            np.array(RGB(color)) * alpha
        )
        self.history.append({'tool': 'highlight', 'bbox': bbox})
        return highlighted

    def crop(self, bbox):
        """Extract image region"""
        x1, y1, x2, y2 = bbox
        cropped = self.image[y1:y2, x1:x2]
        self.history.append({'tool': 'crop', 'bbox': bbox})
        return cropped

    def rotate(self, angle):
        """Rotate image"""
        from scipy import ndimage
        rotated = ndimage.rotate(self.image, angle)
        self.history.append({'tool': 'rotate', 'angle': angle})
        return rotated

    def draw_path(self, points, color='blue', thickness=2):
        """Draw path through image"""
        path_image = self.image.copy()
        for i in range(len(points) - 1):
            cv2.line(path_image, points[i], points[i+1], RGB(color), thickness)
        self.history.append({'tool': 'draw_path', 'points': points})
        return path_image

    def measure_distance(self, point1, point2):
        """Compute pixel distance"""
        dist = np.linalg.norm(np.array(point2) - np.array(point1))
        return dist

    def ocr_region(self, bbox):
        """Extract text from region"""
        x1, y1, x2, y2 = bbox
        region = self.image[y1:y2, x1:x2]
        text = pytesseract.image_to_string(region)
        return text
```

**3. Evaluate Model Performance**

Test models with and without tool access.

```python
class TIRBenchEvaluator:
    def __init__(self, models, task_set='all'):
        self.models = models
        self.tasks = TIRBenchTaskTypes.ALL_TASKS if task_set == 'all' else TIRBenchTaskTypes.get(task_set)

    def evaluate_model(self, model):
        """Evaluate model on all tasks"""
        results = {}

        for task_name, task_config in self.tasks.items():
            # Load task examples
            examples = self.load_task_examples(task_name)

            accuracies = []
            for example in examples[:10]:  # Evaluate on subset
                # Test with tool access
                output = model.solve_with_tools(
                    example['image'],
                    example['question'],
                    available_tools=task_config['tools']
                )

                # Grade answer
                if task_config['metric'] == 'accuracy':
                    correct = output == example['answer']
                else:  # IOU for grounding
                    correct = compute_iou(output, example['answer']) > 0.5

                accuracies.append(correct)

            results[task_name] = np.mean(accuracies)

        return results

    def compare_direct_vs_agentic(self, model_name):
        """Compare direct answering vs. tool-use approaches"""
        results = {'direct': {}, 'agentic': {}}

        for task_name in self.tasks:
            examples = self.load_task_examples(task_name)

            # Direct approach: answer without tools
            direct_acc = self._evaluate_direct(model_name, examples, task_name)
            results['direct'][task_name] = direct_acc

            # Agentic approach: use tools
            agentic_acc = self._evaluate_agentic(model_name, examples, task_name)
            results['agentic'][task_name] = agentic_acc

        return results

    def _evaluate_direct(self, model_name, examples, task_name):
        """Evaluate without tool use"""
        correct = 0
        for example in examples:
            output = GPT_API.call(
                model_name,
                f"Answer this question: {example['question']} based on this image.",
                image=example['image']
            )
            if output == example['answer']:
                correct += 1
        return correct / len(examples)

    def _evaluate_agentic(self, model_name, examples, task_name):
        """Evaluate with tool-use capability"""
        correct = 0
        for example in examples:
            # Enable tool use through function calling
            tools = TIRBenchTaskTypes.ALL_TASKS[task_name]['tools']
            output = GPT_API.call_with_tools(
                model_name,
                f"Solve this problem: {example['question']}",
                tools=self._get_tool_definitions(tools),
                image=example['image']
            )
            if output == example['answer']:
                correct += 1
        return correct / len(examples)
```

## Practical Guidance

**When to Use**:
- Evaluating multimodal models on reasoning capabilities
- Testing tool-use and agentic abilities
- Benchmarking visual chain-of-thought systems

**Dataset Split**:
- 665 multiple-choice questions (easier)
- 550 free-form questions (harder)
- All zero-shot evaluation

**When NOT to Use**:
- Models without image understanding
- Systems lacking tool calling APIs
- Purely text-based evaluation scenarios

**Key Findings**:
- State-of-the-art (o3-TU with code interpreter): 46% accuracy
- Agentic models substantially outperform direct approaches
- Non-agentic models plateau at ~29% accuracy

## Reference

arXiv: https://arxiv.org/abs/2511.01833
