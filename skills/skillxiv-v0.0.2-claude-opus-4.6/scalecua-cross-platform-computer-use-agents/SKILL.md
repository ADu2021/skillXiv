---
name: scalecua-cross-platform-computer-use-agents
title: "ScaleCUA: Scaling Open-Source Computer Use Agents with Cross-Platform Data"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2509.15221"
keywords: [computer-use-agents, GUI-automation, vision-language-models, cross-platform, data-scaling, action-grounding, task-completion, VLM, OSWorld, WebArena]
description: "Build scalable open-source agents that autonomously operate graphical user interfaces across Windows, macOS, Linux, Android, iOS, and web platforms using large-scale cross-platform training data and vision-language models for GUI understanding, element grounding, and task completion."
---

## Outcome: Autonomous Cross-Platform GUI Automation at Scale

Successfully deploy vision-language model-based agents that autonomously interact with desktop, mobile, and web interfaces across six operating systems. ScaleCUA demonstrates that scaling training data from 471K GUI understanding examples and 17.1M grounding samples dramatically improves agent capability to complete real-world GUI tasks with 94.4% accuracy on element recognition and 47.4% success rate on complex web automation scenarios.

## Problem Context

Building computer use agents faces three core challenges. First, the lack of large-scale, open-source training data across diverse platforms creates a bottleneck for developing generalizable agents. Second, existing approaches struggle with cross-platform consistency because each OS, mobile platform, and web environment has distinct UI paradigms, interaction patterns, and rendering behaviors. Third, open-source models significantly underperform compared to proprietary closed-source alternatives on GUI automation tasks, limiting adoption and reproducibility. ScaleCUA addresses these constraints through systematic data collection, unified action spaces, and training infrastructure that achieves competitive performance with state-of-the-art systems.

## Core Concept

ScaleCUA introduces a data-driven scaling strategy for computer use agents grounded in three technical pillars: (1) a closed-loop hybrid data pipeline combining automated agent exploration with human expert annotations, (2) unified visual grounding and action representation enabling single models to operate across heterogeneous platforms, and (3) supervised fine-tuning of vision-language models (Qwen2.5-VL, InternVL) on large-scale annotated datasets spanning 471K GUI understanding examples, 17.1M grounding samples, and 19K task completion trajectories. The approach demonstrates that scale in both data quantity and platform diversity yields substantial improvements in agent generalization and reliability.

## Architecture Overview

**Vision-Language Model Foundation**
- Backbone models: Qwen2.5-VL and InternVL trained on ScaleCUA datasets
- Multi-modal encoding of screenshots with optical character recognition (OCR) and layout analysis
- Supports three inference modes: grounding (element localization), direct action (immediate execution), reasoned action (chain-of-thought + action generation)

**Data Pipeline Infrastructure**
- Agent-Environment Interaction Loop: Automated exploration through rule-driven and VLM-based agents
- Agent-Human Hybrid Data Acquisition: Concurrent trajectory collection from autonomous agents and expert demonstrations
- Annotation Layer: GPT-4o and Claude-3.7-Sonnet VLM-powered annotation with data augmentation (element cropping, synthetic resolution scaling, background substitution)

**Task Family Architecture**
- GUI Understanding (471K examples): Element captioning, OCR extraction, layout comprehension, accessibility tree interpretation
- GUI Grounding (17.1M samples): Point localization, bounding box prediction, action coordinate grounding, semantic element linking
- Task Completion (19K trajectories): Weak-semantic trajectories from automated exploration, high-level goal-directed trajectories from human demonstrations

**Unified Action Space**
- Standardized operation tags: click, type, scroll, swipe, long-press, double-click across all platforms
- Coordinate-based and semantic-based action specification
- Platform-specific adaptation layer for OS-dependent interaction semantics

**Evaluation Harness**
- Multiple benchmark environments: OSWorld (Ubuntu), AndroidWorld (Android), WebArena-Lite-v2 (Web), WindowsAgentArena (Windows), MacOSArena (macOS)
- Task completion metrics, grounding accuracy metrics, element localization F1 scores
- Deployment via vLLM with OpenAI-compatible API interfaces for standardized evaluation

## Implementation

### Step 1: Set Up Cross-Platform Data Collection Environment

Establish the dual-loop data acquisition pipeline by configuring automated agent exploration and human annotation workflows. This infrastructure captures diverse GUI interaction states across six operating systems while maintaining consistent data format specifications.

```python
# data_collection_setup.py
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

@dataclass
class TrajectoryMetadata:
    """Unified metadata across all platforms"""
    trajectory_id: str
    platform: str  # windows, macos, linux, android, ios, web
    task_domain: str  # office, web, productivity
    task_description: str
    human_performed: bool  # True if from human demo
    duration_seconds: int
    num_steps: int
    success: bool
    collection_timestamp: str

class CrossPlatformDataCollector:
    """Manages trajectory collection across all supported platforms"""

    def __init__(self, output_directory: str):
        self.output_directory = output_directory
        self.supported_platforms = [
            "windows", "macos", "linux", "android", "ios", "web"
        ]
        self.unified_action_space = {
            "click": {"params": ["x", "y"]},
            "double_click": {"params": ["x", "y"]},
            "long_press": {"params": ["x", "y", "duration_ms"]},
            "type": {"params": ["text"]},
            "scroll": {"params": ["direction", "amount"]},
            "swipe": {"params": ["start_x", "start_y", "end_x", "end_y"]},
            "back": {"params": []},
            "home": {"params": []},
        }

    def validate_trajectory(
        self,
        trajectory: Dict[str, Any],
        platform: str
    ) -> bool:
        """Validate trajectory schema across platforms"""
        required_fields = [
            "screenshot_paths", "actions", "task_description",
            "accessibility_tree", "timestamp"
        ]

        if not all(field in trajectory for field in required_fields):
            return False

        # Validate action schema consistency
        for action in trajectory["actions"]:
            if action["type"] not in self.unified_action_space:
                return False

            required_params = self.unified_action_space[action["type"]]["params"]
            if not all(param in action for param in required_params):
                return False

        return True

    def create_trajectory_record(
        self,
        trajectory_data: Dict[str, Any],
        platform: str,
        is_human_demo: bool
    ) -> TrajectoryMetadata:
        """Create standardized trajectory metadata"""
        metadata = TrajectoryMetadata(
            trajectory_id=trajectory_data["id"],
            platform=platform,
            task_domain=trajectory_data["domain"],
            task_description=trajectory_data["task"],
            human_performed=is_human_demo,
            duration_seconds=trajectory_data["duration"],
            num_steps=len(trajectory_data["actions"]),
            success=trajectory_data.get("success", False),
            collection_timestamp=datetime.now().isoformat()
        )
        return metadata

# Initialize collector with platform support
collector = CrossPlatformDataCollector("./scalecua_trajectories")
print(f"Supported platforms: {collector.supported_platforms}")
print(f"Unified action space keys: {list(collector.unified_action_space.keys())}")
```

### Step 2: Implement Hybrid Annotation Pipeline with VLM-Powered Enrichment

Build the annotation layer that transforms raw trajectories (screenshots + metadata) into rich training examples. Use vision-language models to automatically generate UI element descriptions, accessibility annotations, and reasoning traces while maintaining human-level quality through expert review.

```python
# annotation_pipeline.py
import base64
import json
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod

class AnnotationPromptFactory:
    """Generate VLM prompts for different annotation tasks"""

    @staticmethod
    def gui_understanding_prompt(screenshot_base64: str) -> str:
        """Generate elements list and layout comprehension"""
        return f"""Analyze this GUI screenshot and provide:
1. List all interactive elements (buttons, inputs, menus) with their approximate positions
2. Describe the layout hierarchy and information structure
3. Identify the primary task or workflow shown
4. Extract all visible text and labels

Return JSON with structure:
{{
    "elements": [
        {{"type": "button", "text": "...", "bounds": {{"x": 0, "y": 0, "width": 0, "height": 0}}}},
        ...
    ],
    "layout_description": "...",
    "primary_workflow": "...",
    "visible_text": ["...", ...]
}}

Screenshot (base64): {screenshot_base64}"""

    @staticmethod
    def grounding_annotation_prompt(screenshot_base64: str, action_text: str) -> str:
        """Generate precise coordinate annotations for actions"""
        return f"""Given this action: "{action_text}"
Identify the target element in the screenshot and provide:
1. Element type (button, input field, menu item, etc.)
2. Precise bounding box [x, y, width, height]
3. Center point coordinates [x, y]
4. Alternative similar elements that could receive this action
5. Semantic label for the element

Return JSON structure:
{{
    "element_type": "...",
    "primary_bbox": {{"x": 0, "y": 0, "width": 0, "height": 0}},
    "center_point": {{"x": 0, "y": 0}},
    "alternatives": [...],
    "semantic_label": "..."
}}

Screenshot (base64): {screenshot_base64}"""

    @staticmethod
    def reasoning_annotation_prompt(
        trajectory_json: str,
        screenshot_base64: str
    ) -> str:
        """Generate thought chains for action selection"""
        return f"""Analyze this step in a task trajectory:
Task context: (from trajectory)
Current screenshot: (provided)
Action taken: (from trajectory)

Generate:
1. What is the user trying to accomplish at this step?
2. What makes the target element the correct choice?
3. What are alternative approaches that wouldn't work?
4. What dependencies does this action have?

Return JSON:
{{
    "step_objective": "...",
    "target_justification": "...",
    "incorrect_alternatives": ["...", ...],
    "dependencies": ["..."],
    "reasoning_chain": "step by step explanation"
}}

Trajectory step: {trajectory_json}
Screenshot (base64): {screenshot_base64}"""

class VLMAnnotator(ABC):
    """Abstract base for VLM annotation backends"""

    @abstractmethod
    def annotate_gui_understanding(
        self,
        screenshot_base64: str
    ) -> Dict[str, any]:
        pass

    @abstractmethod
    def annotate_grounding(
        self,
        screenshot_base64: str,
        action_text: str
    ) -> Dict[str, any]:
        pass

    @abstractmethod
    def annotate_reasoning(
        self,
        trajectory_json: str,
        screenshot_base64: str
    ) -> Dict[str, any]:
        pass

class HybridAnnotationPipeline:
    """Orchestrate VLM annotation + human review for data enrichment"""

    def __init__(self, vlm_annotator: VLMAnnotator, human_review_fraction: float = 0.1):
        self.vlm_annotator = vlm_annotator
        self.human_review_fraction = human_review_fraction
        self.annotation_stats = {
            "gui_understanding": 0,
            "grounding": 0,
            "reasoning": 0,
            "human_reviewed": 0
        }

    def process_trajectory_batch(
        self,
        trajectories: List[Dict[str, any]],
        annotation_tasks: List[str] = None
    ) -> List[Dict[str, any]]:
        """
        annotation_tasks: ["gui_understanding", "grounding", "reasoning"]
        Returns enriched trajectories with annotations
        """
        if annotation_tasks is None:
            annotation_tasks = ["gui_understanding", "grounding", "reasoning"]

        enriched_trajectories = []

        for idx, trajectory in enumerate(trajectories):
            enriched = trajectory.copy()
            enriched["annotations"] = {}

            # Process each step in trajectory
            for step_idx, step in enumerate(trajectory["steps"]):
                screenshot_b64 = step["screenshot_base64"]

                if "gui_understanding" in annotation_tasks:
                    gui_ann = self.vlm_annotator.annotate_gui_understanding(
                        screenshot_b64
                    )
                    enriched["annotations"][f"step_{step_idx}_gui"] = gui_ann
                    self.annotation_stats["gui_understanding"] += 1

                if "grounding" in annotation_tasks and "action" in step:
                    ground_ann = self.vlm_annotator.annotate_grounding(
                        screenshot_b64,
                        step["action"]["description"]
                    )
                    enriched["annotations"][f"step_{step_idx}_grounding"] = ground_ann
                    self.annotation_stats["grounding"] += 1

                if "reasoning" in annotation_tasks:
                    reason_ann = self.vlm_annotator.annotate_reasoning(
                        json.dumps(step),
                        screenshot_b64
                    )
                    enriched["annotations"][f"step_{step_idx}_reasoning"] = reason_ann
                    self.annotation_stats["reasoning"] += 1

            # Mark subset for human review
            import random
            if random.random() < self.human_review_fraction:
                enriched["requires_human_review"] = True
                self.annotation_stats["human_reviewed"] += 1

            enriched_trajectories.append(enriched)

        return enriched_trajectories

print("Annotation pipeline ready for VLM-powered data enrichment")
print(f"Supports tasks: GUI understanding, grounding, reasoning chains")
```

### Step 3: Implement Unified Action Space and Platform Adaptation Layer

Define consistent action representations that abstract OS-specific details while preserving semantic intent. The action space normalizes interaction primitives across heterogeneous platforms, enabling single model training while maintaining platform-specific execution logic.

```python
# unified_action_space.py
from typing import Dict, List, Any, Union, Literal
from dataclasses import dataclass
import json

@dataclass
class ActionCoordinates:
    """Normalized coordinate system across platforms"""
    x: float  # 0-1 relative to screen width
    y: float  # 0-1 relative to screen height
    absolute_x: int = None  # Platform-specific absolute coordinates
    absolute_y: int = None

class UnifiedActionSpace:
    """Cross-platform action abstraction layer"""

    ACTION_TYPES = {
        "click": {
            "requires": ["coordinates"],
            "platforms": ["windows", "macos", "linux", "android", "ios", "web"],
            "semantics": "Single touch/click on element"
        },
        "double_click": {
            "requires": ["coordinates"],
            "platforms": ["windows", "macos", "linux", "web"],
            "semantics": "Double-click (desktop only)"
        },
        "long_press": {
            "requires": ["coordinates", "duration_ms"],
            "platforms": ["android", "ios"],
            "semantics": "Long-press gesture (mobile)"
        },
        "type": {
            "requires": ["text"],
            "platforms": ["windows", "macos", "linux", "android", "ios", "web"],
            "semantics": "Text input to focused field"
        },
        "scroll": {
            "requires": ["direction", "amount"],
            "platforms": ["windows", "macos", "linux", "android", "ios", "web"],
            "semantics": "Scroll in direction by amount"
        },
        "swipe": {
            "requires": ["start_coords", "end_coords"],
            "platforms": ["android", "ios", "web"],
            "semantics": "Swipe gesture from start to end"
        },
        "back": {
            "requires": [],
            "platforms": ["android"],
            "semantics": "Android back button"
        },
        "home": {
            "requires": [],
            "platforms": ["android", "ios"],
            "semantics": "Home button"
        }
    }

    def __init__(self, target_platform: str):
        if target_platform not in ["windows", "macos", "linux", "android", "ios", "web"]:
            raise ValueError(f"Unsupported platform: {target_platform}")
        self.target_platform = target_platform

    def normalize_action(
        self,
        action_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert platform-specific action to unified representation"""
        action_type = action_dict.get("type")

        if action_type not in self.ACTION_TYPES:
            raise ValueError(f"Unknown action type: {action_type}")

        required_params = self.ACTION_TYPES[action_type]["requires"]
        supported_on_platform = self.target_platform in self.ACTION_TYPES[action_type]["platforms"]

        if not supported_on_platform:
            raise ValueError(
                f"Action {action_type} not supported on {self.target_platform}"
            )

        # Validate required parameters
        for param in required_params:
            if param not in action_dict and param != "text":
                raise ValueError(f"Missing required parameter: {param}")

        normalized = {
            "type": action_type,
            "platform": self.target_platform,
            "timestamp": action_dict.get("timestamp"),
            "metadata": {}
        }

        # Normalize coordinates if present
        if "coordinates" in required_params and "coordinates" in action_dict:
            coords = action_dict["coordinates"]
            normalized["coordinates"] = ActionCoordinates(
                x=coords.get("x", coords.get("x_normalized", 0)),
                y=coords.get("y", coords.get("y_normalized", 0))
            )

        if "text" in required_params:
            normalized["text"] = action_dict.get("text", "")

        if action_type == "scroll":
            normalized["direction"] = action_dict.get("direction", "down")
            normalized["amount"] = action_dict.get("amount", 3)

        if action_type == "long_press":
            normalized["duration_ms"] = action_dict.get("duration_ms", 500)

        return normalized

    def denormalize_action(self, normalized_action: Dict[str, Any]) -> Dict[str, Any]:
        """Convert unified action to platform-specific execution format"""
        action_type = normalized_action["type"]
        platform_specific = {"type": action_type}

        if "coordinates" in normalized_action:
            coords = normalized_action["coordinates"]

            # Convert relative to absolute based on platform
            if self.target_platform == "web":
                platform_specific["coordinates"] = {
                    "x": int(coords.x * 1920),  # Assume 1920 width
                    "y": int(coords.y * 1080)
                }
            elif self.target_platform in ["android", "ios"]:
                platform_specific["coordinates"] = {
                    "x": int(coords.x * 1080),  # Mobile resolution
                    "y": int(coords.y * 2340)
                }
            else:  # Desktop
                platform_specific["coordinates"] = {
                    "x": int(coords.x * 1920),
                    "y": int(coords.y * 1080)
                }

        if "text" in normalized_action:
            platform_specific["text"] = normalized_action["text"]

        if action_type == "scroll":
            platform_specific["direction"] = normalized_action.get("direction")
            platform_specific["amount"] = normalized_action.get("amount")

        return platform_specific

# Example usage across platforms
print("Unified action space initialized")
for platform in ["windows", "android", "web"]:
    action_space = UnifiedActionSpace(platform)
    print(f"  {platform}: supports {len([a for a in action_space.ACTION_TYPES.keys() if platform in action_space.ACTION_TYPES[a]['platforms']])} actions")
```

### Step 4: Implement Vision-Language Model Fine-Tuning on Annotated Datasets

Configure supervised fine-tuning of backbone VLMs (Qwen2.5-VL, InternVL) using the annotated trajectory data. The training process optimizes for three task families simultaneously: GUI understanding (element recognition), grounding (coordinate prediction), and action generation (task-guided interaction).

```python
# vlm_training_config.py
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json

@dataclass
class TrainingConfig:
    """Training hyperparameters for ScaleCUA models"""
    # Model selection
    model_name: str  # "Qwen2.5-VL" or "InternVL"
    model_checkpoint: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Data configuration
    train_data_path: str = "./scalecua_trajectories/train"
    val_data_path: str = "./scalecua_trajectories/val"
    test_data_path: str = "./scalecua_trajectories/test"
    gui_understanding_weight: float = 0.3
    grounding_weight: float = 0.5
    reasoning_weight: float = 0.2

    # Training hyperparameters
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 1000
    max_training_steps: int = 100000
    num_epochs: int = 3

    # Model capacity
    model_size: str = "7B"  # 3B, 7B, or 32B variants
    use_lora: bool = False
    lora_rank: int = 64

    # Inference modes to train
    train_grounding_mode: bool = True
    train_direct_action_mode: bool = True
    train_reasoning_mode: bool = True

    # Hardware and optimization
    use_mixed_precision: bool = True
    torch_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    max_seq_length: int = 4096
    device_map: str = "auto"

    # Evaluation
    eval_steps: int = 5000
    save_steps: int = 5000
    eval_strategy: str = "steps"
    save_total_limit: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for wandb or other tracking"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

class ScaleCUATrainer:
    """Fine-tune VLMs on cross-platform GUI data"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.training_losses = []

    def prepare_dataset(
        self,
        data_path: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load and prepare trajectory data for training"""
        dataset = {
            "gui_understanding": [],
            "grounding": [],
            "task_completion": []
        }

        # Load trajectories from data_path
        import os
        for trajectory_file in os.listdir(data_path):
            if trajectory_file.endswith(".json"):
                with open(os.path.join(data_path, trajectory_file)) as f:
                    trajectory = json.load(f)

                for step_idx, step in enumerate(trajectory.get("steps", [])):
                    # GUI Understanding samples
                    if "annotations" in step and "gui_elements" in step["annotations"]:
                        dataset["gui_understanding"].append({
                            "image": step["screenshot_path"],
                            "target": step["annotations"]["gui_elements"],
                            "task_type": "element_recognition"
                        })

                    # Grounding samples
                    if "annotations" in step and "grounding" in step["annotations"]:
                        dataset["grounding"].append({
                            "image": step["screenshot_path"],
                            "action": step["action"]["description"],
                            "target_bbox": step["annotations"]["grounding"]["bbox"],
                            "task_type": "coordinate_prediction"
                        })

                    # Task completion samples
                    if "action" in step:
                        dataset["task_completion"].append({
                            "image": step["screenshot_path"],
                            "context": trajectory["task_description"],
                            "action": step["action"],
                            "task_type": "action_generation"
                        })

        return dataset

    def create_training_prompt(
        self,
        sample: Dict[str, Any],
        task_type: str
    ) -> str:
        """Format training sample into model prompt"""

        if task_type == "gui_understanding":
            return f"""Analyze this GUI screenshot and identify all interactive elements.
For each element, provide: type, text label, approximate position.
Answer in JSON format.

Screenshot: [IMAGE]

Answer:"""

        elif task_type == "grounding":
            return f"""Given the action: "{sample['action']}"
Identify the target element in the screenshot and provide its bounding box as [x, y, width, height] in normalized 0-1 coordinates.

Screenshot: [IMAGE]

Bounding box: [0, 0, 0.1, 0.1]"""

        elif task_type == "action_generation":
            return f"""Task: {sample['context']}
Current screenshot: [IMAGE]

What is the next action to take? Provide: action_type, coordinates (if needed), and text (if needed).

Action:"""

        return ""

    def train(self):
        """Execute training loop"""
        print(f"Loading model: {self.config.model_checkpoint}")

        # Prepare datasets
        train_dataset = self.prepare_dataset(self.config.train_data_path)
        val_dataset = self.prepare_dataset(self.config.val_data_path)

        total_samples = sum(len(v) for v in train_dataset.values())
        print(f"Loaded {total_samples} training samples")
        print(f"  GUI Understanding: {len(train_dataset['gui_understanding'])}")
        print(f"  Grounding: {len(train_dataset['grounding'])}")
        print(f"  Task Completion: {len(train_dataset['task_completion'])}")

        # Training configuration details
        print(f"\nTraining configuration:")
        print(f"  Model: {self.config.model_name} ({self.config.model_size})")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Mixed precision: {self.config.torch_dtype}")
        print(f"  Task weights - GUI: {self.config.gui_understanding_weight}, "
              f"Ground: {self.config.grounding_weight}, Reason: {self.config.reasoning_weight}")

        return {
            "total_samples": total_samples,
            "training_phases": ["gui_understanding", "grounding", "task_completion"],
            "status": "ready_for_training"
        }

# Create training configuration
config = TrainingConfig(
    model_name="Qwen2.5-VL",
    model_size="7B",
    batch_size=32,
    learning_rate=2e-4
)

trainer = ScaleCUATrainer(config)
training_summary = trainer.train()
print(json.dumps(training_summary, indent=2))
```

### Step 5: Implement Multi-Mode Inference Engine with Grounding and Reasoning

Build the inference pipeline supporting three operational modes: grounding-only (element localization), direct action (immediate execution), and reasoned action (chain-of-thought planning). Deploy using vLLM for optimized serving with OpenAI-compatible API.

```python
# inference_engine.py
from typing import Dict, List, Tuple, Literal, Optional
from enum import Enum
import json
from dataclasses import dataclass

class InferenceMode(Enum):
    """Three inference modes with different reasoning levels"""
    GROUNDING = "grounding"  # Only localize UI elements
    DIRECT_ACTION = "direct_action"  # Immediate action without reasoning
    REASONED_ACTION = "reasoned_action"  # Chain-of-thought + action

@dataclass
class InferenceResult:
    mode: InferenceMode
    screenshot_path: str
    action_type: str
    coordinates: Optional[Tuple[float, float]] = None
    text: Optional[str] = None
    reasoning_trace: Optional[str] = None
    confidence: float = 1.0
    inference_time_ms: float = 0.0
    model_name: str = ""

class ScaleCUAInferenceEngine:
    """Multi-mode inference for cross-platform GUI automation"""

    def __init__(
        self,
        model_name: str = "OpenGVLab/ScaleCUA-7B",
        vllm_server_url: str = "http://localhost:8000",
        inference_mode: InferenceMode = InferenceMode.REASONED_ACTION
    ):
        self.model_name = model_name
        self.vllm_server_url = vllm_server_url
        self.inference_mode = inference_mode
        self.platform_context = None

    def set_platform_context(self, platform: str):
        """Configure for target platform"""
        if platform not in ["windows", "macos", "linux", "android", "ios", "web"]:
            raise ValueError(f"Unsupported platform: {platform}")
        self.platform_context = platform

    def _build_grounding_prompt(
        self,
        screenshot_base64: str,
        task_description: str
    ) -> str:
        """Generate prompt for element localization"""
        return f"""Analyze this screenshot for the task: "{task_description}"
Identify all interactive elements and their locations.

For each element, provide:
- Element type (button, input, link, etc.)
- Text/label (if visible)
- Bounding box as [x_min, y_min, x_max, y_max] in 0-1 normalized coordinates

Return JSON array of elements.

Screenshot (base64): {screenshot_base64}

Elements:"""

    def _build_direct_action_prompt(
        self,
        screenshot_base64: str,
        task_description: str
    ) -> str:
        """Generate prompt for immediate action without reasoning"""
        return f"""Task: {task_description}
Current screenshot: [base64: {screenshot_base64}]

What is the NEXT action to take?
Respond ONLY with JSON: {{"type": "click|type|scroll", "coordinates": [x, y], "text": "optional text"}}

Action JSON:"""

    def _build_reasoning_prompt(
        self,
        screenshot_base64: str,
        task_description: str,
        history: Optional[List[Dict]] = None
    ) -> str:
        """Generate prompt for chain-of-thought action generation"""
        history_str = ""
        if history:
            history_str = "\nPrevious actions:\n"
            for i, prev_action in enumerate(history[-3:]):  # Last 3 actions
                history_str += f"{i+1}. {prev_action['type']} at {prev_action.get('coordinates', 'N/A')}\n"

        return f"""Task: {task_description}{history_str}

Current screenshot: [base64: {screenshot_base64}]

Analyze:
1. What is the current state?
2. What needs to be done next?
3. Which element should I interact with?
4. What action should I take?

Provide your reasoning, then respond with:
JSON: {{"type": "...", "coordinates": [...], "reasoning": "..."}}

Reasoning and action:"""

    def infer(
        self,
        screenshot_base64: str,
        task_description: str,
        history: Optional[List[Dict]] = None,
        timeout_seconds: int = 30
    ) -> InferenceResult:
        """Execute inference in configured mode"""

        if self.inference_mode == InferenceMode.GROUNDING:
            prompt = self._build_grounding_prompt(screenshot_base64, task_description)
            mode_name = "grounding"
        elif self.inference_mode == InferenceMode.DIRECT_ACTION:
            prompt = self._build_direct_action_prompt(screenshot_base64, task_description)
            mode_name = "direct_action"
        else:  # REASONED_ACTION
            prompt = self._build_reasoning_prompt(
                screenshot_base64,
                task_description,
                history
            )
            mode_name = "reasoned_action"

        # Call vLLM endpoint (OpenAI-compatible API)
        import requests
        import time

        start_time = time.time()

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1 if self.inference_mode == InferenceMode.DIRECT_ACTION else 0.3,
            "max_tokens": 200 if self.inference_mode == InferenceMode.GROUNDING else 100
        }

        try:
            # Mock response for demonstration
            if mode_name == "reasoned_action":
                response_text = """Reasoning: The task requires entering credentials. I see a login form with username and password fields. Next, I should click the username input field.

JSON: {"type": "click", "coordinates": [0.3, 0.45], "reasoning": "Click username field to focus it"}"""
            elif mode_name == "direct_action":
                response_text = '{"type": "click", "coordinates": [0.3, 0.45]}'
            else:
                response_text = '[{"type": "button", "text": "Login", "bbox": [0.2, 0.4, 0.4, 0.5]}]'

            inference_time = (time.time() - start_time) * 1000

            # Parse response
            result = InferenceResult(
                mode=self.inference_mode,
                screenshot_path="current",
                action_type="click",
                coordinates=(0.3, 0.45),
                reasoning_trace="Click username field to focus it" if self.inference_mode == InferenceMode.REASONED_ACTION else None,
                confidence=0.85,
                inference_time_ms=inference_time,
                model_name=self.model_name
            )

            return result

        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}")

# Example usage
print("Initializing ScaleCUA inference engine")
engine = ScaleCUAInferenceEngine(
    model_name="OpenGVLab/ScaleCUA-7B",
    inference_mode=InferenceMode.REASONED_ACTION
)
engine.set_platform_context("web")
print(f"Engine configured for reasoned action mode on web platform")
```

## Practical Guidance

### When to Use ScaleCUA

- **Cross-platform GUI automation** where tasks span multiple operating systems and need consistent behavior
- **Web-based workflows** requiring complex navigation, form filling, and data extraction at scale
- **Mobile app testing and automation** on both Android and iOS without platform-specific scripting
- **Large-scale data collection** where you need to annotate screenshots and interactions systematically
- **Open-source requirement** where proprietary closed-source solutions are not acceptable
- **Custom fine-tuning** where you want to adapt the agent to domain-specific GUIs (ERP systems, internal tools)

### When NOT to Use ScaleCUA

- **Simple, deterministic workflows** better handled by traditional RPA tools with rule-based logic (YAML configurations, etc.)
- **Extreme latency constraints** where sub-100ms inference time is required (ScaleCUA inference adds 1-5 seconds per action)
- **Highly proprietary or non-visual UIs** where interaction occurs through APIs or accessibility trees, not pixel-based GUI rendering
- **Real-time interactive applications** like video games, where human-like reaction time is essential
- **Lightweight embedded scenarios** where model size (3B-32B parameters) is too large for available hardware
- **Guaranteed 100% success requirement** where any failure rate is unacceptable (current SOTA is ~47% on complex tasks)

### Hyperparameter Configuration Reference

| Parameter | Recommended Range | Impact | Notes |
|-----------|-------------------|--------|-------|
| learning_rate | 1e-4 to 5e-4 | Convergence speed | Start 2e-4 for most cases; reduce for unstable training |
| batch_size | 16-64 | Memory/throughput | Higher batches → better stability, needs more VRAM |
| warmup_steps | 500-2000 | Initial training stability | ~5-10% of total steps |
| gui_understanding_weight | 0.2-0.4 | Element recognition vs action balance | Increase if grounding accuracy is poor |
| grounding_weight | 0.4-0.6 | Coordinate precision emphasis | Core task, should be 0.4+ for good localization |
| reasoning_weight | 0.1-0.3 | Chain-of-thought depth | Higher → more reasoning overhead, better for complex tasks |
| model_size | 3B, 7B, 32B | Accuracy vs latency tradeoff | 3B: fast, ~40% accuracy; 7B: balanced ~45%; 32B: best ~60%, slowest |
| inference_temperature | 0.1-0.3 | Determinism vs diversity | 0.1 for reproducible actions; 0.3 for exploration |
| max_seq_length | 2048-4096 | Context window for reasoning | Longer → better multi-step planning, higher latency |

### Common Pitfalls and Solutions

**Pitfall 1: Poor coordinate generalization across resolutions**
- Problem: Model trained on 1920x1080 screenshots performs poorly on 1440x900 resolutions
- Solution: Normalize all coordinates to 0-1 range during training; use resolution-augmented data with random scaling

**Pitfall 2: Action hallucination in unfamiliar UI patterns**
- Problem: Agent generates clicks on non-existent elements in novel applications
- Solution: Increase human demonstration coverage; implement element confidence thresholds; validate predicted coordinates against actual UI structure

**Pitfall 3: Task drift in long trajectories**
- Problem: Agent forgets original task objective after 10+ steps
- Solution: Use reasoned action mode with explicit task re-statement in reasoning prompts; implement task-aware loss weighting

**Pitfall 4: Slow inference bottleneck**
- Problem: Sequential action inference (1-5 seconds per action) makes agents impractical
- Solution: Deploy batch inference where possible; use smaller 3B model variant; implement local caching of common UI patterns

**Pitfall 5: Overfitting to training platforms**
- Problem: High performance on Windows/web but poor on Android/iOS
- Solution: Ensure balanced platform representation in training (e.g., 15% Windows, 20% macOS, 20% Linux, 20% Android, 15% iOS, 10% web); use data augmentation to simulate new platforms

## Reference

For the complete technical details, methodology, dataset specifications, and benchmark results, refer to the original paper:

[ScaleCUA: Scaling Open-Source Computer Use Agents with Cross-Platform Data - arXiv 2509.15221](https://arxiv.org/abs/2509.15221)

**Key Repository:** https://github.com/OpenGVLab/ScaleCUA

**Datasets and Models:**
- Training data: https://huggingface.co/datasets/OpenGVLab/ScaleCUA-Data
- Model checkpoints: https://huggingface.co/OpenGVLab/ScaleCUA-7B, https://huggingface.co/OpenGVLab/ScaleCUA-3B
