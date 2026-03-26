---
name: vp-vla-visual-prompting-robotics
title: "VP-VLA: Visual Prompting as Spatial Reasoning Interface for Embodied Control"
version: 0.0.3
engine: skillxiv-v0.0.3-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2603.22003"
keywords: [Visual Prompting, VLA, Robotics, Spatial Grounding, Dual-System Reasoning]
description: "Replace monolithic VLA single-pathway decision-making with a decomposed System-2/System-1 architecture where a pretrained VLM planner identifies targets as visual prompts (crosshairs, bounding boxes) and a VLA controller executes on grounded observations, improving success rates by 5-8% on manipulation tasks. Use when spatial precision is critical, multi-step reasoning is needed, and you have access to pretrained segmentation and vision-language models."
category: "Component Innovation"
---

## What This Skill Does

Decompose a monolithic VLA architecture into a reasoning-execution pipeline: a pretrained VLM (System 2) decomposes tasks and identifies target objects with explicit visual prompts, while a VLA controller (System 1) receives these grounded prompts as image overlays, improving spatial reasoning and manipulation success.

## The Component Swap

**Old component:** Single end-to-end VLA pathway where visual encoding, spatial reasoning, and action prediction happen jointly in a black-box manner.

```python
# Traditional monolithic VLA
class TraditionalVLA(nn.Module):
    def forward(self, observation, instruction):
        # Single forward pass: encode vision, encode language, predict action
        vision_feat = self.vision_encoder(observation)
        lang_feat = self.language_encoder(instruction)
        action = self.action_head(torch.cat([vision_feat, lang_feat], dim=-1))
        return action
```

**New component:** Dual-system architecture with explicit spatial prompting interface.

```python
# VP-VLA: decomposed System 2 (planning) and System 1 (control)
class VPVLA(nn.Module):
    def __init__(self, vlm_planner, vla_controller, segmenter):
        super().__init__()
        self.vlm_planner = vlm_planner      # Qwen3-VL or similar
        self.vla_controller = vla_controller  # Pretrained VLA
        self.segmenter = segmenter           # SAM3 for object detection

    def forward(self, observation, instruction):
        # System 2: VLM decomposes task and identifies target
        task_plan = self.vlm_planner.generate(
            observation,
            prompt=f"Task: {instruction}. What object? Where?"
        )
        target_obj = self.parse_target_from_plan(task_plan)

        # Segmentation: get target location and mask
        target_mask = self.segmenter.segment(observation, target_obj)
        bbox = self.get_bounding_box(target_mask)

        # Visual prompting: overlay explicit spatial cues
        prompted_obs = self.draw_prompts(
            observation,
            crosshair=bbox.center,
            bounding_box=bbox,
            color=(255, 0, 0)  # Red for target
        )

        # System 1: VLA executes on grounded observation
        action = self.vla_controller(prompted_obs, instruction)
        return action

    def draw_prompts(self, obs, crosshair, bounding_box, color):
        """Overlay visual prompts: crosshairs at target center, box outline."""
        overlay = obs.clone()
        # Draw crosshair at center
        overlay = cv2.drawMarker(
            overlay,
            tuple(map(int, crosshair)),
            color,
            cv2.MARKER_CROSS,
            20,
            2
        )
        # Draw bounding box
        overlay = cv2.rectangle(overlay, tuple(map(int, bounding_box.topleft)),
                               tuple(map(int, bounding_box.bottomright)), color, 2)
        return overlay
```

The key swap is from implicit spatial reasoning (VLA must learn to parse instructions into coordinates) to explicit visual prompting (target location is directly marked in observation space), enabling the controller to focus on motion execution rather than spatial inference.

## Performance Impact

**Robocasa-GR1 Tabletop Manipulation:**
- Success rate: 48.8% → 53.8% (**+5.0 percentage points**, +10.2% relative)

**SimplerEnv Benchmark:**
- Success rate: 50.0% → 58.3% (**+8.3 percentage points**, +16.6% relative)

**Real-world object placement:**
- In-distribution accuracy: 80% → 87.5% (**+7.5 percentage points**)
- Out-of-distribution accuracy: 63.3% → 85% (**+21.7 percentage points**, major improvement on unseen objects)

**Fine-grained task (egg placement):**
- Success rate: 70.63% → 91.25% (**+20.6 percentage points**, critical for contact-sensitive tasks)

## When to Use

- Tabletop manipulation tasks requiring spatial precision
- Multi-step tasks where intermediate goals benefit from explicit grounding
- When novel objects appear (visual prompts transfer better than learned spatial heuristics)
- Tasks involving pick-and-place with specific target locations
- When you have access to pretrained VLM (Qwen3-VL, LLaVA) and segmentation models (SAM3)

## When NOT to Use

- Simple point-reach tasks (single-system VLA is sufficient)
- Tasks where visual prompts may occlude critical environment features
- If segmentation model is unreliable for your object classes
- When continuous replanning overhead is unacceptable (VLM inference cost)
- Tasks where the instruction already specifies exact coordinates

## Implementation Checklist

**1. Component integration:**
- Load pretrained VLM: `vlm = AutoModel.from_pretrained('qwen3-vl')`
- Load pretrained VLA controller: your baseline VLA checkpoint
- Load segmenter: `segmenter = SAM3(checkpoint_path)`

**2. Spatial prompting pipeline:**
```python
# Minimal swap: wrap existing VLA with planning layer
vp_vla = VPVLA(
    vlm_planner=vlm,
    vla_controller=your_vla_model,
    segmenter=SAM3()
)
# Forward pass now includes visual prompting
action = vp_vla(observation, instruction)
```

**3. Trigger mechanism:**
- Use event-driven replanning: recompute prompts on gripper state changes (gripper open→close)
- Avoids continuous VLM overhead while maintaining spatial accuracy
- Typical: 2-3 planning steps per task execution

**4. Verification:**
- Measure success rate on Robocasa-GR1 or SimplerEnv benchmark
- Compare: baseline VLA (0% prompts) vs. your implementation
- Ablate: remove visual prompts to confirm +5-8% delta

**5. Hyperparameter tuning if needed:**
- VLM prompt template: "Task: {instruction}. Identify target object and location." works well
- Crosshair size: 20-40 pixels depending on image resolution
- Bounding box color: red/green/blue; cyan for visibility
- Segmentation confidence threshold: 0.5-0.8 (adjust for false positives)

**6. Known issues:**
- VLM hallucinations: segmenter may fail to find spurious objects mentioned in plan; validate segmentation output
- Prompt visibility: colored overlays may fade in low-light; consider image contrast enhancement
- OOD objects: SAM3 generalizes well but may struggle on transparent/reflective surfaces
- GPU memory: VLM + segmenter + VLA = large VRAM footprint (>20GB for joint inference)

## Related Work

This builds on System 1/System 2 decomposition (Kahneman) and vision-language grounding (SAM, LLaVA). The visual prompting interface is inspired by human spatial communication (pointing, marking regions) and relates to instruction-following architectures that explicitly ground language in spatial coordinates (VoxPoser, RoboHelper).
