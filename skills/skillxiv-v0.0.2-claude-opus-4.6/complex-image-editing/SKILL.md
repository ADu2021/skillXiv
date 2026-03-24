---
name: complex-image-editing
title: "Beyond Simple Edits: X-Planner for Complex Instruction-Based Image Editing"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.05259"
keywords: [Image Editing, Instruction Following, Multimodal Planning, Mask Generation, Object Localization]
description: "Decompose complex image editing instructions into simpler sub-tasks with automatically generated control guidance. Handles multi-object edits, preserves identity of surrounding regions, and eliminates manual mask creation."
---

# X-Planner: Planning-Based Image Editing from Complex Instructions

Editing images based on complex instructions requires more than direct pixel manipulation. When a user says "make the building taller and the sky more dramatic," the system must understand that these are two separate edits targeting different objects, generate precise boundaries for each, and apply appropriate transformations without bleeding into adjacent regions. X-Planner solves this by decomposing complex instructions into manageable sub-tasks, automatically generating the masks and control signals that guide editing models.

The core challenge is that complex instructions are indirectly specified and often target multiple objects. Current approaches either require users to manually provide masks or fail when identity preservation matters—editing one object corrupts its surroundings.

## Core Concept

X-Planner operates as a three-stage pipeline that separates planning from execution:

1. **Instruction decomposition**: Parse the complex instruction into simpler, atomic sub-instructions
2. **Mask generation**: For each sub-instruction, generate precise segmentation masks tailored to the edit type
3. **Bounding box prediction**: For insertion tasks, predict spatial locations for new objects

By treating masking as a learned task conditioned on edit type, the system generates tighter masks for texture edits and dilated masks for shape changes—each adapted to the specific editing goal.

## Architecture Overview

- **MLLM instruction parser**: Analyzes complex instructions and produces structured sub-tasks with edit types
- **Edit-specific mask generator**: Creates customized segmentation masks based on edit type (replacement, style change, insertion, etc.)
- **Spatial predictor**: For insertions, predicts bounding boxes where new objects should appear
- **Compatible editing backend**: Works with existing models (UltraEdit, InstructPix2Pix, etc.)
- **Iterative refinement**: Applies sub-instructions sequentially, each building on previous edits

## Implementation

Start by analyzing a complex instruction and decomposing it into sub-tasks:

```python
from xplanner.decomposer import InstructionDecomposer
from xplanner.masker import MaskGenerator

decomposer = InstructionDecomposer(model="gpt-4-vision")

# Complex instruction that targets multiple objects implicitly
instruction = "Make the car red, remove the traffic cone, and brighten the road"

# Decompose into atomic sub-instructions
sub_tasks = decomposer.decompose(
    instruction=instruction,
    image=image
)

# Output:
# [
#   {"text": "change the car color to red", "target": "car", "type": "color_change"},
#   {"text": "remove the traffic cone", "target": "traffic_cone", "type": "deletion"},
#   {"text": "brighten the road surface", "target": "road", "type": "lighting_change"}
# ]
```

For each sub-task, generate a specialized mask conditioned on the edit type:

```python
masker = MaskGenerator()

for sub_task in sub_tasks:
    edit_type = sub_task["type"]

    # Generate mask adapted to edit type
    mask = masker.generate_mask(
        image=image,
        target_description=sub_task["text"],
        edit_type=edit_type,
        # Different masks for different edits:
        # - "texture" or "color_change": tight mask (exact object)
        # - "shape" or "size": dilated mask (include context)
        # - "deletion": precise boundary
        # - "global": full image mask
    )

    # Validate mask covers the target
    assert masker.validate_coverage(mask, sub_task["target"])

    sub_task["mask"] = mask
```

For insertion tasks, predict bounding boxes since existing detectors can't hallucinate objects not in the original image:

```python
from xplanner.spatial import BoundingBoxPredictor

predictor = BoundingBoxPredictor()

insertion_tasks = [t for t in sub_tasks if t["type"] == "insertion"]

for task in insertion_tasks:
    # Predict where new object should appear
    bbox = predictor.predict(
        image=image,
        instruction=task["text"],
        context_objects=get_visible_objects(image)
    )

    # Bbox provides spatial guidance to editing model
    task["bbox"] = bbox
```

Apply the sub-tasks iteratively using a compatible editing model:

```python
from xplanner.executor import ImageEditor

editor = ImageEditor(backend="ultarEdit")  # or InstructPix2Pix

result_image = image.copy()

# Apply sub-tasks sequentially
for i, sub_task in enumerate(sub_tasks):
    # Get mask and optional spatial guidance
    mask = sub_task["mask"]
    bbox = sub_task.get("bbox", None)

    # Edit using specified mask and guidance
    result_image = editor.edit(
        image=result_image,
        instruction=sub_task["text"],
        mask=mask,
        spatial_guidance=bbox,
        preserve_identity=True  # Keep regions outside mask unchanged
    )

    # Validate edit quality
    assert editor.validate_quality(result_image, result_image_prev)

return result_image
```

## Practical Guidance

### When to Use X-Planner

Use this approach for:
- Complex, multi-object editing instructions
- Scenarios where identity preservation is critical
- User instructions with ambiguous or indirect language
- Cases where manual masks would be tedious or error-prone
- Applications requiring iterative refinement of edits

### When NOT to Use

Avoid X-Planner for:
- Simple, single-object edits (direct approaches are faster)
- Fully structured instructions already decomposed by users
- Style transfer or artistic transformations (doesn't require decomposition)
- Real-time editing requiring immediate feedback
- Highly specialized editing domains with custom models

### Edit-Type Mask Strategies

| Edit Type | Mask Strategy | Example |
|-----------|---------------|---------|
| Color change | Tight mask (exact object boundary) | "Make the car blue" |
| Shape change | Dilated mask (object + buffer) | "Make the building taller" |
| Style transfer | Full region mask | "Make the road surface glossy" |
| Deletion | Precise boundary | "Remove the traffic cone" |
| Insertion | Bounding box guidance | "Add a tree near the building" |
| Global edit | Full image mask | "Brighten the entire scene" |

### Key Hyperparameters

| Parameter | Typical Range | Guidance |
|-----------|---------------|----------|
| Mask dilation | 0-30 pixels | Larger for shape edits, smaller for color |
| Confidence threshold | 0.5-0.9 | Higher = more selective masks |
| Iteration count | 1-5 steps | More iterations for complex edits, but slower |
| Model backbone | GPT-4V, Claude | Larger models decompose better |

### Common Pitfalls

1. **Over-decomposing**: Not every instruction needs splitting. Keep sub-tasks atomic but not granular.
2. **Ignoring mask quality**: A good mask is 80% of the editing success. Validate carefully.
3. **Forgetting spatial context**: When inserting objects, ensure they appear in physically plausible locations.
4. **Sequential error accumulation**: Each edit can degrade the image. Monitor quality after each step.
5. **Missing identity preservation**: Ensure masks don't bleed into adjacent objects, or explicitly dilate for shape changes.

### Validation Checklist

- [ ] Each sub-instruction is atomic and independent
- [ ] Masks cover intended targets completely
- [ ] Masks don't overlap with protected regions
- [ ] Inserted objects have valid bounding boxes
- [ ] Edit sequence respects dependencies
- [ ] Final image preserves original identity outside edited regions

## Reference

"Beyond Simple Edits: X-Planner for Complex Instruction-Based Image Editing" - [arXiv:2507.05259](https://arxiv.org/abs/2507.05259)
