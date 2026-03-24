---
name: spatiallm-3d-scenes
title: "SpatialLM: Training LLMs for Structured Indoor Modeling"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.07491"
keywords: [3d-understanding, point-clouds, indoor-scenes, vision-language, code-generation]
description: "Train multimodal LLMs to parse 3D point clouds and generate executable Python code representing structured indoor scene layouts with walls, doors, and objects."
---

# SpatialLM: Training LLMs for Structured Indoor Modeling

## Core Concept

SpatialLM enables LLMs to understand 3D spatial environments by generating structured Python code that represents indoor scenes. Rather than using specialized geometric networks, it leverages standard multimodal LLM architecture fine-tuned from open-source models, combining point cloud encoders with language models to output human-interpretable, editable scene descriptions. This approach is extensible to new object classes without code modifications and achieves state-of-the-art layout estimation by exploiting LLMs' pre-trained coding capabilities.

## Architecture Overview

- **Point Cloud Encoder**: Transformer-based (Sonata/PTv3) converting point clouds to feature embeddings
- **Encoder Output Compression**: Reduces point cloud tokens from N to K where K << N (query-based selection)
- **MLP Projector**: Two-layer alignment module bridging point cloud features to LLM embedding space
- **Fine-tuned LLM**: Qwen2.5-0.5B or similar for code generation
- **Python Code Output**: Structured representation of walls, doors, windows, and bounding boxes
- **Single-Stage Training**: End-to-end training with all components differentiable

## Implementation

### Step 1: Set Up Point Cloud Encoder

Implement or integrate a point cloud transformer encoder:

```python
import torch
import torch.nn as nn
from torch_points3d.models.segmentation import KPCONV
import numpy as np

class PointCloudEncoder(nn.Module):
    """
    Encode point clouds to feature embeddings using transformer approach.
    Converts N points to K query tokens (K << N).
    """

    def __init__(self, input_dim=3, output_dim=768, num_queries=64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_queries = num_queries

        # Feature extraction
        self.feature_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, output_dim)
        )

        # Transformer encoder (can use Sonata or PointTransformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # Query tokens for compression
        self.queries = nn.Parameter(torch.randn(num_queries, output_dim))
        nn.init.normal_(self.queries, std=0.02)

        # Query-key cross-attention
        self.cross_attention = nn.MultiheadAttention(
            output_dim, 8, batch_first=True
        )

    def forward(self, points, colors=None):
        """
        Args:
            points: (batch_size, num_points, 3) - xyz coordinates
            colors: (batch_size, num_points, 3) - optional RGB

        Returns:
            features: (batch_size, num_queries, output_dim)
        """
        batch_size, num_points, _ = points.shape

        # Combine position and color if available
        if colors is not None:
            point_features = torch.cat([points, colors], dim=-1)
        else:
            point_features = points

        # MLP feature extraction
        features = self.feature_mlp(point_features)  # (B, N, D)

        # Transformer encoding
        encoded = self.transformer(features)  # (B, N, D)

        # Compress to queries via cross-attention
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # (B, K, D)
        compressed, _ = self.cross_attention(
            queries, encoded, encoded,
            key_padding_mask=None
        )

        return compressed  # (B, K, D)
```

### Step 2: Create MLP Projector and LLM Integration

Build the bridge between point cloud embeddings and LLM:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

class SpatialLMModel(nn.Module):
    """Full SpatialLM model: encoder -> projector -> LLM"""

    def __init__(self, pc_encoder_dim=768, llm_name="Qwen/Qwen2.5-0.5B-Instruct"):
        super().__init__()

        # Point cloud encoder
        self.pc_encoder = PointCloudEncoder(output_dim=pc_encoder_dim)

        # MLP Projector: align point cloud features to LLM embedding space
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        llm_hidden_dim = self.llm.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(pc_encoder_dim, llm_hidden_dim),
            nn.GELU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)

    def forward(self, points, instruction_ids, colors=None):
        """
        Args:
            points: (batch_size, num_points, 3)
            instruction_ids: (batch_size, instruction_len) - text tokens
            colors: (batch_size, num_points, 3) - optional

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # Encode point cloud
        pc_features = self.pc_encoder(points, colors)  # (B, K, D_pc)

        # Project to LLM space
        pc_embeddings = self.projector(pc_features)  # (B, K, D_llm)

        # Get text embeddings
        text_embeddings = self.llm.get_input_embeddings()(instruction_ids)

        # Concatenate: [instruction_tokens, point_cloud_tokens]
        combined_embeddings = torch.cat(
            [text_embeddings, pc_embeddings], dim=1
        )  # (B, T+K, D_llm)

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=combined_embeddings,
            output_hidden_states=True
        )

        return outputs.logits
```

### Step 3: Implement Code Generation for Scene Description

Create decoder that generates Python code from spatial understanding:

```python
class SceneCodeGenerator:
    """Generate executable Python code for scene layout"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def generate_code(self, model, points, instruction="Describe this indoor scene"):
        """Generate Python code representing the scene"""

        instruction_ids = self.tokenizer.encode(instruction, return_tensors='pt')

        # Forward pass
        with torch.no_grad():
            outputs = model(points, instruction_ids)
            logits = outputs.logits

        # Generate tokens greedily (or use beam search)
        generated_ids = torch.argmax(logits, dim=-1)

        # Decode to code
        code = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return code

    def postprocess_code(self, code_str):
        """Clean up generated code"""
        # Remove explanatory text, keep only code blocks
        if "```python" in code_str:
            code_str = code_str.split("```python")[1].split("```")[0]

        return code_str.strip()

    def parse_scene_from_code(self, code_str):
        """Execute code and extract scene structure"""
        namespace = {}
        exec(code_str, namespace)

        scene = {
            'walls': namespace.get('walls', []),
            'doors': namespace.get('doors', []),
            'windows': namespace.get('windows', []),
            'objects': namespace.get('objects', [])
        }

        return scene
```

### Step 4: Training Pipeline

Implement end-to-end training on synthetic indoor scene dataset:

```python
def train_spatiallm(model, dataloader, optimizer, epochs=10):
    """Train SpatialLM end-to-end"""

    for epoch in range(epochs):
        total_loss = 0

        for batch in dataloader:
            points = batch['points'].cuda()
            colors = batch['colors'].cuda() if 'colors' in batch else None
            instruction_ids = batch['instruction_ids'].cuda()
            target_code_ids = batch['target_code_ids'].cuda()

            optimizer.zero_grad()

            # Forward pass
            logits = model(points, instruction_ids, colors)

            # Compute loss
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_code_ids.reshape(-1),
                ignore_index=-100
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

def create_training_batch(scene_data, tokenizer):
    """
    Create training batch from synthetic scene data.

    Example scene code:
    walls = [
        {'start': [0, 0], 'end': [10, 0], 'height': 3.0},
        {'start': [10, 0], 'end': [10, 8], 'height': 3.0}
    ]
    doors = [
        {'position': [5, 0], 'width': 0.9, 'height': 2.1}
    ]
    objects = [
        {'class': 'chair', 'bbox': [4, 3, 1, 1]},
        {'class': 'table', 'bbox': [5, 4, 2, 1]}
    ]
    """

    points = scene_data['point_cloud']
    colors = scene_data['colors']
    code = scene_data['code']
    instruction = "Generate Python code for this indoor layout:"

    instruction_ids = tokenizer.encode(instruction, return_tensors='pt')
    target_ids = tokenizer.encode(code, return_tensors='pt')

    return {
        'points': torch.tensor(points),
        'colors': torch.tensor(colors),
        'instruction_ids': instruction_ids,
        'target_code_ids': target_ids
    }
```

### Step 5: Evaluation Metrics

Implement evaluation for layout estimation quality:

```python
class LayoutEvaluator:
    """Evaluate scene layout extraction accuracy"""

    def __init__(self):
        pass

    def compute_layout_metrics(self, predicted_code, ground_truth_code):
        """Compare predicted vs. ground truth layouts"""

        pred_scene = execute_code(predicted_code)
        gt_scene = execute_code(ground_truth_code)

        # Metrics for walls
        wall_iou = self.compute_iou(pred_scene['walls'], gt_scene['walls'])

        # Metrics for objects
        object_ap = self.compute_average_precision(
            pred_scene['objects'], gt_scene['objects']
        )

        return {
            'wall_iou': wall_iou,
            'object_ap': object_ap,
            'overall_score': (wall_iou + object_ap) / 2
        }

    def compute_iou(self, pred_elements, gt_elements):
        """Intersection over Union for geometric elements"""
        if len(gt_elements) == 0:
            return 0.0

        ious = []
        for pred in pred_elements:
            best_iou = 0.0
            for gt in gt_elements:
                iou = self.element_iou(pred, gt)
                best_iou = max(best_iou, iou)
            ious.append(best_iou)

        return np.mean(ious) if ious else 0.0

    def compute_average_precision(self, pred_objects, gt_objects):
        """Compute AP for object detection within scenes"""
        # Standard COCO-style AP computation
        pass
```

## Practical Guidance

- **Point Cloud Preprocessing**: Normalize coordinates to [-1, 1] range; handle sparse/dense point clouds
- **Query Compression**: Use 32-64 queries typical; adjust based on scene complexity
- **LLM Choice**: Smaller models (0.5-7B) work well; fine-tune on multiple downstream tasks
- **Code Structure**: Define consistent Python API for scene representation (walls, doors, objects)
- **Synthetic Data**: 12k+ scenes needed; use professional CAD sources for diversity
- **Single-Stage Training**: More efficient than multi-stage; all components learn jointly
- **Extensibility**: New object classes only require code changes, not retraining embeddings
- **Downstream Tasks**: Layout estimation, object placement, navigation planning

## Reference

- Point cloud transformers (Sonata, PTv3) are superior to voxel/mapping approaches for geometric detail
- Code generation enables interpretability and editability compared to geometric regression
- Query-based compression reduces computation while preserving spatial information
- LLM pre-training on code provides strong inductive bias for structured scene generation
