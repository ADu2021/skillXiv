---
name: canvas-to-image-composition
title: "Canvas-to-Image: Compositional Image Generation with Multimodal Controls"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.21691"
keywords: [Compositional Image Generation, Multi-Task Diffusion, Spatial Control, Multi-Person Generation]
description: "Generate images with unified control over identity, spatial position, pose, and layout: encode diverse control modalities (spatial canvas, pose canvas, box canvas) into single RGB image, train diffusion model jointly across all control types, and enable flexible multi-modal composition at inference without task-specific fine-tuning."
---

# Canvas-to-Image: Unified Compositional Image Generation

Current image generation systems handle individual control types separately (identity control, spatial guidance, pose constraints), forcing practitioners to chain multiple models or accept reduced control flexibility. This skill demonstrates Canvas-to-Image, which consolidates diverse guidance signals into a unified visual canvas—a single RGB image encoding all control information—enabling diffusion transformers to learn compositional generation jointly across control modalities.

The core innovation is treating multimodal controls as encodable canvas variants, enabling flexible composition without retraining during inference.

## Core Concept

Canvas-to-Image implements unified compositional control through:

1. **Multi-Task Canvas Representation**: Encodes different control types into RGB canvas variants (spatial, pose, box)
2. **Joint Training Architecture**: Single diffusion transformer trained across all canvas types simultaneously
3. **Emergent Generalization**: Model learns to compose multiple controls together despite single-control training samples
4. **Constant Computational Cost**: Adding more controls doesn't increase inference cost

## Architecture Overview

- **Canvas Encoder**: Converts control canvases to visual tokens
- **Vision-Language Model**: Encodes text and visual controls jointly
- **Diffusion Transformer**: Generates images conditioned on canvas + text
- **Multi-Task Loss**: Single objective covering all control types
- **Flexible Canvas Composition**: Supports mixing control types at inference

## Implementation Steps

The system converts controls to canvas format and trains with unified architecture.

**1. Implement Canvas Encoding for Different Control Types**

Create methods to encode each control type into RGB canvas.

```python
class CanvasEncoder:
    """
    Converts different control types into unified RGB canvas representation.
    Supports spatial placement, pose guidance, and bounding box constraints.
    """
    def __init__(self, canvas_size=(512, 512)):
        self.canvas_size = canvas_size

    def encode_spatial_canvas(self, subjects: List[Dict], image_height=512, image_width=512) -> np.ndarray:
        """
        Create spatial canvas with segmented subject cutouts at desired locations.
        Args:
            subjects: List of dicts with 'image' (PIL Image), 'x', 'y' (position)
            image_height, image_width: Canvas dimensions
        Returns:
            canvas: (height, width, 3) RGB canvas with placed subjects
        """
        canvas = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255

        for subject in subjects:
            subject_img = subject['image']
            x_norm = subject['x']  # Normalized [0, 1]
            y_norm = subject['y']

            # Convert to pixel coordinates
            x_pixel = int(x_norm * image_width)
            y_pixel = int(y_norm * image_height)

            # Resize subject to fit
            subject_width = int(image_width * 0.25)  # 25% of canvas
            subject_height = int(subject_img.height * subject_width / subject_img.width)

            subject_resized = subject_img.resize((subject_width, subject_height))

            # Place on canvas
            x_end = min(x_pixel + subject_width, image_width)
            y_end = min(y_pixel + subject_height, image_height)

            x_start = max(0, x_pixel)
            y_start = max(0, y_pixel)

            canvas[y_start:y_end, x_start:x_end] = np.array(subject_resized)

        return canvas.astype(np.float32) / 255.0

    def encode_pose_canvas(self, poses: List[Dict], image_height=512, image_width=512) -> np.ndarray:
        """
        Create pose canvas with skeleton overlays.
        Args:
            poses: List of dicts with 'keypoints' (e.g., from OpenPose)
            image_height, image_width: Canvas dimensions
        Returns:
            canvas: (height, width, 3) RGB canvas with pose overlays
        """
        canvas = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255

        for pose in poses:
            keypoints = pose['keypoints']  # List of (x, y, confidence)

            # Draw skeleton
            skeleton_pairs = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Right arm
                (0, 5), (5, 6), (6, 7),  # Left arm
                (0, 8), (8, 9), (9, 10),  # Right leg
                (0, 11), (11, 12), (12, 13)  # Left leg
            ]

            for start_idx, end_idx in skeleton_pairs:
                if start_idx < len(keypoints) and end_idx < len(keypoints):
                    start = keypoints[start_idx]
                    end = keypoints[end_idx]

                    # Convert to canvas coordinates
                    x1 = int(start[0] * image_width)
                    y1 = int(start[1] * image_height)
                    x2 = int(end[0] * image_width)
                    y2 = int(end[1] * image_height)

                    # Draw line with semi-transparency
                    cv2.line(canvas, (x1, y1), (x2, y2), (100, 150, 200), 2)

            # Draw joints
            for kpt in keypoints:
                x = int(kpt[0] * image_width)
                y = int(kpt[1] * image_height)
                cv2.circle(canvas, (x, y), 3, (0, 255, 0), -1)

        return canvas.astype(np.float32) / 255.0

    def encode_box_canvas(self, boxes: List[Dict], image_height=512, image_width=512) -> np.ndarray:
        """
        Create box canvas with bounding boxes and text annotations.
        Args:
            boxes: List of dicts with 'bbox' (x1, y1, x2, y2), 'label' (str)
            image_height, image_width: Canvas dimensions
        Returns:
            canvas: (height, width, 3) RGB canvas with boxes
        """
        canvas = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        for idx, box in enumerate(boxes):
            bbox = box['bbox']  # Normalized [0, 1]
            label = box['label']

            # Convert to pixel coordinates
            x1 = int(bbox[0] * image_width)
            y1 = int(bbox[1] * image_height)
            x2 = int(bbox[2] * image_width)
            y2 = int(bbox[3] * image_height)

            color = colors[idx % len(colors)]

            # Draw box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # Draw label
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(canvas, label, (x1, y1 - 5), font, 0.5, color, 1)

        return canvas.astype(np.float32) / 255.0

    def merge_canvases(self, canvases: List[np.ndarray], weights: List[float] = None) -> np.ndarray:
        """
        Merge multiple canvas types into single control canvas.
        Enables simultaneous multi-modal control.
        Args:
            canvases: List of canvas arrays
            weights: Blending weights
        Returns:
            merged: Blended canvas
        """
        if weights is None:
            weights = [1.0 / len(canvases)] * len(canvases)

        merged = np.zeros_like(canvases[0])

        for canvas, weight in zip(canvases, weights):
            merged += canvas * weight

        return np.clip(merged, 0, 1)
```

**2. Build Vision-Language Encoder for Canvas**

Create encoder processing both canvas and text conditions.

```python
class CanvasVisionLanguageEncoder(torch.nn.Module):
    """
    Encodes canvas control images and text prompts jointly.
    """
    def __init__(self, hidden_dim=768, text_vocab_size=30522):
        super().__init__()

        # Canvas encoder: ResNet for image processing
        self.canvas_encoder = torchvision.models.resnet50(pretrained=True)
        self.canvas_proj = torch.nn.Linear(2048, hidden_dim)

        # Text encoder: BERT
        self.text_encoder = AutoModel.from_pretrained('bert-base-uncased')

        # Fusion layer
        self.fusion = torch.nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )

        self.output_projection = torch.nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, canvas: torch.Tensor, text_input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode canvas and text jointly.
        Args:
            canvas: (batch, 3, height, width) RGB canvas
            text_input_ids: (batch, text_len) tokenized text
        Returns:
            condition_embeddings: (batch, hidden_dim) unified condition
        """
        # Encode canvas
        canvas_features = self.canvas_encoder(canvas)  # (batch, 2048, 7, 7)
        canvas_features = torch.nn.functional.adaptive_avg_pool2d(canvas_features, 1)  # (batch, 2048, 1, 1)
        canvas_features = canvas_features.squeeze(-1).squeeze(-1)  # (batch, 2048)
        canvas_embeddings = self.canvas_proj(canvas_features)  # (batch, hidden_dim)

        # Encode text
        text_output = self.text_encoder(text_input_ids)
        text_embeddings = text_output.last_hidden_state.mean(dim=1)  # (batch, hidden_dim)

        # Fuse via cross-attention
        canvas_expanded = canvas_embeddings.unsqueeze(1)  # (batch, 1, hidden_dim)
        text_expanded = text_embeddings.unsqueeze(1)  # (batch, 1, hidden_dim)

        fused, _ = self.fusion(canvas_expanded, text_expanded, text_expanded)

        # Combine
        combined = torch.cat([fused, text_expanded], dim=-1).squeeze(1)
        conditions = self.output_projection(combined)

        return conditions
```

**3. Implement Diffusion Transformer with Multi-Task Training**

Build diffusion model handling all canvas types in unified training.

```python
class CanvasConditionedDiffusionTransformer(torch.nn.Module):
    """
    Diffusion transformer generating images conditioned on canvas + text.
    Single model trained across all control types (spatial, pose, box).
    """
    def __init__(self, hidden_dim=768, num_layers=24):
        super().__init__()

        # Canvas-language encoder
        self.condition_encoder = CanvasVisionLanguageEncoder(hidden_dim=hidden_dim)

        # Diffusion UNet with transformer backbone
        self.diffusion_transformer = DiffusionTransformer(
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

    def forward(
        self,
        x_t: torch.Tensor,
        timestep: int,
        canvas: torch.Tensor,
        text_input_ids: torch.Tensor,
        canvas_type: str = None
    ) -> torch.Tensor:
        """
        Predict noise for diffusion step.
        Args:
            x_t: (batch, channels, height, width) noisy image
            timestep: Diffusion timestep
            canvas: (batch, 3, height, width) control canvas
            text_input_ids: (batch, text_len) text tokens
            canvas_type: 'spatial', 'pose', 'box', or None (mixed)
        Returns:
            noise_prediction: Predicted noise
        """
        # Encode conditions
        conditions = self.condition_encoder(canvas, text_input_ids)

        # Add timestep embedding
        t_emb = get_timestep_embedding(timestep, hidden_dim=conditions.shape[-1])

        # Fuse conditions with timestep
        conditions = conditions + t_emb

        # Predict noise
        noise_pred = self.diffusion_transformer(x_t, conditions, timestep)

        return noise_pred
```

**4. Implement Multi-Task Training Loss**

Create unified loss across all canvas types.

```python
def canvas_diffusion_loss(
    model,
    x_0: torch.Tensor,
    canvas: torch.Tensor,
    text_input_ids: torch.Tensor,
    canvas_type: str,
    timesteps: torch.Tensor,
    noise: torch.Tensor
):
    """
    Compute diffusion loss for canvas-conditioned generation.
    Single loss function handles all canvas types.
    Args:
        model: Diffusion transformer
        x_0: (batch, channels, height, width) clean images
        canvas: Control canvas
        text_input_ids: Text conditions
        canvas_type: Type of canvas used (for weighting)
        timesteps: Sampled diffusion timesteps
        noise: Ground truth noise
    Returns:
        loss: Scalar loss
    """
    batch_size = x_0.shape[0]

    # Add noise to images at sampled timesteps
    x_t = x_0 * np.sqrt(1 - timesteps.view(-1, 1, 1, 1)) + noise * np.sqrt(timesteps.view(-1, 1, 1, 1))

    # Predict noise
    noise_pred = model(x_t, timesteps, canvas, text_input_ids, canvas_type)

    # Diffusion loss: L2 between predicted and ground truth noise
    diffusion_loss = torch.nn.functional.mse_loss(noise_pred, noise)

    # Task-specific weighting (encourage all types equally)
    canvas_type_weight = {
        'spatial': 1.0,
        'pose': 1.0,
        'box': 1.0,
        None: 1.0  # Mixed type
    }

    weighted_loss = diffusion_loss * canvas_type_weight.get(canvas_type, 1.0)

    return weighted_loss
```

**5. Training with Multi-Task Sampling**

Train single model across all canvas types.

```python
def train_canvas_diffusion(
    model,
    train_dataloader,
    optimizer,
    num_epochs=100
):
    """
    Train canvas-conditioned diffusion model.
    Samples uniformly across all canvas types during training.
    """
    canvas_encoder = CanvasEncoder()

    for epoch in range(num_epochs):
        for batch in train_dataloader:
            images = batch['image']
            text = batch['text']
            text_tokens = tokenize_text(text)

            batch_size = images.shape[0]

            # Randomly sample canvas type for this batch
            canvas_types = ['spatial', 'pose', 'box']
            canvas_type = np.random.choice(canvas_types)

            # Encode canvas based on control signals
            if canvas_type == 'spatial':
                canvases = []
                for subject_data in batch.get('subjects', []):
                    canvas = canvas_encoder.encode_spatial_canvas(subject_data)
                    canvases.append(canvas)

            elif canvas_type == 'pose':
                canvases = []
                for pose_data in batch.get('poses', []):
                    canvas = canvas_encoder.encode_pose_canvas(pose_data)
                    canvases.append(canvas)

            else:  # 'box'
                canvases = []
                for box_data in batch.get('boxes', []):
                    canvas = canvas_encoder.encode_box_canvas(box_data)
                    canvases.append(canvas)

            canvases = torch.from_numpy(np.stack(canvases)).to(images.device)

            # Sample timesteps
            timesteps = torch.randint(1, 1000, (batch_size,)).to(images.device)

            # Sample noise
            noise = torch.randn_like(images)

            # Compute loss
            loss = canvas_diffusion_loss(
                model, images, canvases, text_tokens, canvas_type, timesteps, noise
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
```

**6. Inference with Flexible Canvas Composition**

Generate images using unified canvas interface.

```python
def generate_with_canvas(
    model,
    canvas: np.ndarray,
    text_prompt: str,
    num_steps=50,
    guidance_scale=7.5,
    device='cuda'
):
    """
    Generate image with canvas control.
    Supports any canvas type or composition without fine-tuning.
    Args:
        model: Trained diffusion model
        canvas: (height, width, 3) control canvas
        text_prompt: Generation prompt
        num_steps: Diffusion steps
        guidance_scale: Classifier-free guidance scale
        device: Torch device
    Returns:
        generated_image: Generated image tensor
    """
    # Convert to tensors
    canvas_tensor = torch.from_numpy(canvas).unsqueeze(0).to(device)
    text_tokens = tokenize_text(text_prompt).unsqueeze(0).to(device)

    # Initialize random noise
    x_t = torch.randn(1, 3, 512, 512).to(device)

    # Diffusion loop (backward through timesteps)
    for t in range(num_steps - 1, 0, -1):
        # Predict noise with guidance
        noise_pred = model(x_t, t, canvas_tensor, text_tokens)

        # Unconditional noise for classifier-free guidance
        noise_uncond = model(x_t, t, torch.ones_like(canvas_tensor), torch.zeros_like(text_tokens))

        # Apply guidance
        noise_guided = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

        # Denoise step
        alpha = get_alpha_t(t)
        x_t = (x_t - np.sqrt(1 - alpha) * noise_guided) / np.sqrt(alpha)

        # Add noise for next step
        if t > 0:
            x_t = x_t + np.sqrt((1 - alpha) / alpha) * torch.randn_like(x_t)

    # Decode to image
    generated_image = x_t

    return generated_image
```

## Practical Guidance

**When to Use Canvas-to-Image:**
- Projects requiring simultaneous multi-type control (pose + spatial + text)
- Need flexible composition without retraining
- Applications where constant computational cost is important
- Scenarios with diverse control requirements (user-friendly systems)

**When NOT to Use:**
- Single-type control (use specialized models for maximum quality)
- Scenarios where control types conflict significantly
- Real-time generation where diffusion steps are bottleneck

**Key Design Choices:**
- Canvas encoding must be visually clear and learnable
- Keep canvas size matching generated image size (reduces complexity)
- Single model across types enables emergent generalization

**Training Data Tips:**
- Collect training data with ALL control types (even if underrepresented)
- Augment underrepresented canvas types synthetically
- Balance canvas type distribution during training

**Multi-Modal Composition:**
When combining control types at inference, blend canvases by weighted averaging:
```python
merged = 0.5 * spatial_canvas + 0.3 * pose_canvas + 0.2 * box_canvas
```

## Reference

Research paper: https://arxiv.org/abs/2511.21691
