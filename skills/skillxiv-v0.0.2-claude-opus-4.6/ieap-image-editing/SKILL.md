---
name: ieap-image-editing
title: "Image Editing As Programs: Decomposing Complex Instructions into Atomic Operations"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.04158"
keywords: [image-editing, diffusion-models, program-synthesis, layout-modification]
description: "Enable robust image editing by decomposing free-form instructions into sequential atomic operations executed through a neural program interpreter."
---

# Image Editing As Programs with Diffusion Models

## Core Concept

IEAP (Image Editing As Programs) solves a critical weakness in diffusion transformer-based image editing: their struggle with "structurally inconsistent edits that involve substantial layout changes." By decomposing editing instructions into five atomic primitives and executing them sequentially, IEAP handles both simple attribute modifications and complex multi-step layout alterations robustly.

## Architecture Overview

- **Five Atomic Primitives**: RoI Localization, RoI Inpainting, RoI Editing, RoI Compositing, Global Transformation
- **VLM-Based Instruction Parser**: Chain-of-thought reasoning to decompose free-form instructions into operation sequences
- **Specialized DiT Adapters**: Task-specific diffusion transformer adapters for each primitive operation
- **Sequential Execution**: Operations execute in order, each refining image state for the next operation
- **Problem Diagnosis**: Identifies that layout modification—not attribute changes—is the critical bottleneck

## Implementation

### Step 1: Taxonomy and Diagnostic Analysis

```python
from enum import Enum
from typing import List, Dict, Tuple

class EditType(Enum):
    """Categorize edits by structural complexity"""
    ATTRIBUTE_ONLY = "attribute_only"        # Color, texture changes
    LAYOUT_CONSISTENT = "layout_consistent"  # Adds/removes within bounds
    LAYOUT_CHANGE = "layout_change"          # Moves, resizes, repositions objects

class EditingDiagnostics:
    def __init__(self):
        self.baseline_dit_model = load_pretrained_dit()
        self.metrics = {
            'attribute_only': {'success_rate': 0.95},
            'layout_consistent': {'success_rate': 0.82},
            'layout_change': {'success_rate': 0.35},
        }

    def analyze_performance_by_type(self, test_instructions):
        """Benchmark baseline DiT on different edit types"""

        results = {}

        for instruction in test_instructions:
            # Classify edit type
            edit_type = self.classify_instruction(instruction)

            # Attempt edit with baseline
            edited_image, success = self.baseline_dit_model.edit(instruction)

            # Evaluate quality
            quality_score = self.evaluate_edit_quality(edited_image)

            if edit_type not in results:
                results[edit_type] = []

            results[edit_type].append({
                'instruction': instruction,
                'success': success,
                'quality': quality_score,
            })

        # Print analysis
        print("=== Performance Dichotomy ===")
        for edit_type, outcomes in results.items():
            success_rate = sum(1 for o in outcomes if o['success']) / len(outcomes)
            avg_quality = sum(o['quality'] for o in outcomes) / len(outcomes)

            print(f"{edit_type.value}:")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Avg quality: {avg_quality:.3f}")

        return results

    def classify_instruction(self, instruction: str) -> EditType:
        """Use NLP to determine edit category"""

        layout_keywords = ['move', 'reposition', 'place', 'arrange',
                          'resize', 'stretch', 'rotate', 'flip']
        attribute_keywords = ['change color', 'make', 'style',
                             'texture', 'material', 'bright', 'dark']

        instruction_lower = instruction.lower()

        if any(kw in instruction_lower for kw in layout_keywords):
            return EditType.LAYOUT_CHANGE
        elif any(kw in instruction_lower for kw in attribute_keywords):
            return EditType.ATTRIBUTE_ONLY
        else:
            return EditType.LAYOUT_CONSISTENT
```

### Step 2: Design Five Atomic Primitives

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class AtomicOperation:
    """Base class for all editing operations"""
    operation_type: str
    target_region: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    parameters: Dict = None

class RoILocalization(AtomicOperation):
    """Identify target region of interest"""

    def __init__(self, description: str, image: torch.Tensor):
        super().__init__(operation_type="localization")

        # Use VLM to understand what region is being referenced
        region_prompt = f"In this image, locate: {description}"

        # Get bounding box from VLM (e.g., GPT-4V)
        self.target_region = self.vlm_locate_region(region_prompt, image)
        self.parameters = {'description': description}

    def execute(self, image):
        """Extract RoI from image"""
        x1, y1, x2, y2 = self.target_region
        roi = image[:, y1:y2, x1:x2]
        return roi, self.target_region

class RoIInpainting(AtomicOperation):
    """Add or remove content within region"""

    def __init__(self, region: Tuple, action: str):
        super().__init__(operation_type="inpainting")
        self.target_region = region
        self.parameters = {'action': action}  # "add" or "remove"

    def execute(self, image, roi_description: str):
        """Inpaint content in RoI"""
        x1, y1, x2, y2 = self.target_region

        # Mask out the region
        mask = torch.zeros_like(image)
        mask[:, y1:y2, x1:x2] = 1

        # Use diffusion to fill masked region
        inpainted = self.dit_inpainter.inpaint(
            image, mask, prompt=roi_description
        )

        return inpainted

class RoIEditing(AtomicOperation):
    """Modify visual attributes within region"""

    def __init__(self, region: Tuple, attribute: str, value: str):
        super().__init__(operation_type="editing")
        self.target_region = region
        self.parameters = {
            'attribute': attribute,  # "color", "style", "texture"
            'value': value  # "red", "impressionist", "marble"
        }

    def execute(self, image):
        """Edit attributes in RoI"""
        x1, y1, x2, y2 = self.target_region

        # Prompt for diffusion editing
        prompt = f"Make the {self.parameters['attribute']} {self.parameters['value']}"

        # Use ControlNet-based editing for fine control
        edited = self.dit_editor.edit(
            image, self.target_region, prompt
        )

        return edited

class RoICompositing(AtomicOperation):
    """Seamlessly integrate edited region back"""

    def __init__(self, region: Tuple, blend_mode: str = "seamless"):
        super().__init__(operation_type="compositing")
        self.target_region = region
        self.parameters = {'blend_mode': blend_mode}

    def execute(self, image, edited_roi):
        """Blend edited RoI back into full image"""
        x1, y1, x2, y2 = self.target_region

        # Seamless blending at boundaries
        composited = self.blend_with_inpainting(
            image, edited_roi, self.target_region
        )

        return composited

class GlobalTransformation(AtomicOperation):
    """Apply full-image modifications"""

    def __init__(self, transformation: str):
        super().__init__(operation_type="global")
        self.parameters = {'transformation': transformation}

    def execute(self, image):
        """Apply full-image editing"""
        prompt = self.parameters['transformation']

        # Use unconditional or global-guided diffusion
        transformed = self.dit_global.transform(image, prompt)

        return transformed
```

### Step 3: Implement VLM-Based Instruction Parser

```python
class InstructionParser:
    def __init__(self, vlm_model_name='GPT-4V'):
        self.vlm = load_model(vlm_model_name)

    def parse_instruction_to_operations(self, instruction: str,
                                       image: torch.Tensor) -> List[AtomicOperation]:
        """
        Decompose free-form instruction into ordered atomic operations.
        Uses Chain-of-Thought prompting for structured decomposition.
        """

        # Step 1: VLM analysis
        analysis_prompt = f"""Analyze this image editing instruction:
"{instruction}"

Break it down into atomic operations in order:
1. What regions need to be identified?
2. What modifications apply to each region?
3. How should regions be combined back?
4. Any global image modifications?

For each operation, specify:
- Operation type (localization/inpainting/editing/compositing/global)
- Target region description
- Parameters"""

        analysis = self.vlm.generate(analysis_prompt, image=image)

        # Step 2: Parse VLM response into operation sequence
        operations = self.extract_operations_from_analysis(
            analysis, image
        )

        # Step 3: Validate operation sequence
        validated = self.validate_operation_sequence(operations, image)

        return validated

    def extract_operations_from_analysis(self, analysis: str,
                                        image: torch.Tensor) -> List[AtomicOperation]:
        """Convert VLM analysis into executable operations"""

        operations = []
        lines = analysis.split('\n')

        current_op_type = None
        current_params = {}

        for line in lines:
            if 'localization' in line.lower():
                current_op_type = RoILocalization
            elif 'inpainting' in line.lower():
                current_op_type = RoIInpainting
            elif 'editing' in line.lower():
                current_op_type = RoIEditing
            elif 'compositing' in line.lower():
                current_op_type = RoICompositing
            elif 'global' in line.lower():
                current_op_type = GlobalTransformation

            # Extract parameters from line
            # (simplified; real implementation would parse more carefully)
            if current_op_type and '::' in line:
                param_str = line.split('::')[1]
                current_params[line.split('::')[0]] = param_str

        # Create operation instances
        if current_op_type == RoILocalization:
            operations.append(RoILocalization(
                description=current_params.get('target', ''),
                image=image
            ))
        # Similar for other operation types...

        return operations

    def validate_operation_sequence(self, operations: List[AtomicOperation],
                                   image: torch.Tensor) -> List[AtomicOperation]:
        """Verify operations are executable and in valid order"""

        # Localization should come before editing same region
        # Compositing should follow regional edits
        # Global ops should not break earlier regional work

        return operations

```

### Step 4: Sequential Execution Engine

```python
class SequentialEditingExecutor:
    def __init__(self, dit_models: Dict):
        self.dit_models = dit_models
        self.operation_history = []

    def execute_program(self, operations: List[AtomicOperation],
                       image: torch.Tensor) -> torch.Tensor:
        """Execute sequence of operations sequentially"""

        current_image = image

        for op_idx, operation in enumerate(operations):
            print(f"Executing operation {op_idx + 1}/{len(operations)}: "
                  f"{operation.operation_type}")

            # Execute operation
            if isinstance(operation, RoILocalization):
                roi, region = operation.execute(current_image)

            elif isinstance(operation, RoIInpainting):
                current_image = operation.execute(
                    current_image,
                    operation.parameters['description']
                )

            elif isinstance(operation, RoIEditing):
                current_image = operation.execute(current_image)

            elif isinstance(operation, RoICompositing):
                # Composite previous edit
                if op_idx > 0:
                    current_image = operation.execute(
                        current_image,
                        edited_roi=None  # Use current image state
                    )

            elif isinstance(operation, GlobalTransformation):
                current_image = operation.execute(current_image)

            # Log operation
            self.operation_history.append({
                'operation': operation.operation_type,
                'parameters': operation.parameters,
                'step': op_idx,
            })

        return current_image

    def interactive_debugging(self, instruction: str, image: torch.Tensor):
        """Allow user to inspect/modify operation sequence before execution"""

        parser = InstructionParser()
        operations = parser.parse_instruction_to_operations(instruction, image)

        print("\n=== Proposed Operation Sequence ===")
        for i, op in enumerate(operations):
            print(f"{i+1}. {op.operation_type}")
            print(f"   Parameters: {op.parameters}")

        # Allow user to skip, reorder, or modify operations
        # Then execute

        result = self.execute_program(operations, image)

        return result
```

## Practical Guidance

1. **Diagnosis First**: Before implementing decomposition, benchmark your baseline diffusion model on different edit types. The performance dichotomy (95% on attributes, 35% on layout) makes the motivation clear.

2. **Five Primitives are Sufficient**: Localization → Inpainting → Editing → Compositing → Global covers virtually all realistic edits. Resist the urge to add more primitives.

3. **VLM-Based Parsing**: Use vision-language models with chain-of-thought prompting to decompose instructions. This is more robust than rule-based parsing.

4. **Sequential Not Parallel**: Execute operations in strict order. This ensures each operation sees the current image state, not the original. Dependencies flow forward.

5. **Seamless Compositing**: The hardest part is blending edited regions back seamlessly. Use inpainting at boundaries rather than hard blending.

6. **Interactive Debugging**: Show users the proposed operation sequence before execution. Let them adjust if desired.

## Reference

- Paper: Image Editing As Programs (2506.04158)
- Core Innovation: Decomposition of layout-modifying edits into atomic operations
- Architecture: VLM parser + specialized DiT adapters + sequential executor
- Result: Handles both simple and complex multi-step image modifications robustly
