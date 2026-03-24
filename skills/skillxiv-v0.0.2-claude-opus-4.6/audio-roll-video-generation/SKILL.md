---
name: audio-roll-video-generation
title: "Seeing Voices: Generating A-Roll Video from Audio with Mirage"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.08279"
keywords: [audio-to-video, A-roll generation, multimodal synthesis, speech video]
description: "Generate realistic video footage of people from audio input using a unified self-attention framework, producing convincing speaker performances without domain-specific restrictions."
---

# Seeing Voices: Audio-to-Video Generation

## Core Concept

Mirage is an audio-to-video foundation model that generates realistic video from audio input, specializing in A-roll generation—footage of people delivering performances based on speech. The key innovation is using a unified self-attention architecture applicable across diverse scenarios rather than audio-specific or speech-restricted design choices.

## Architecture Overview

- **Unified self-attention framework**: General architecture applicable to multiple scenarios
- **Audio-visual alignment**: Conditions video generation on speech-containing audio
- **No domain restrictions**: Avoids specialized modules for speech or appearance
- **Generalist approach**: Better performance than audio-specialized alternatives
- **End-to-end trainable**: Single model for flexible audio-to-video synthesis

## Implementation

### Step 1: Design Audio Encoder

Extract meaningful features from speech audio:

```python
class AudioEncoder(torch.nn.Module):
    def __init__(self, sample_rate: int = 16000,
                 hidden_dim: int = 512):
        super().__init__()
        self.sample_rate = sample_rate
        self.hidden_dim = hidden_dim

        # Mel-spectrogram extractor
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )

        # Spectrogram encoder (temporal CNN)
        self.spec_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Conv1d(256, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(512)
        )

        # Transformer for temporal context
        self.transformer = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=4
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to feature sequence."""

        # Convert to mel-spectrogram
        mel = self.mel_spec(audio)  # (batch, 128, time)

        # Encode spectrogram
        spec_features = self.spec_encoder(mel)  # (batch, 512)

        # Reshape for transformer
        spec_features = spec_features.transpose(1, 2)

        # Apply temporal transformer
        audio_features = self.transformer(spec_features)

        return audio_features  # (batch, seq_len, hidden_dim)
```

### Step 2: Build Unified Video Generator

Create self-attention based video generation model:

```python
class UnifiedAudioVideoGenerator(torch.nn.Module):
    def __init__(self, hidden_dim: int = 512,
                 num_frames: int = 120,
                 frame_height: int = 512,
                 frame_width: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames

        # Audio encoder
        self.audio_encoder = AudioEncoder(hidden_dim=hidden_dim)

        # Unified attention mechanism
        self.cross_attention = torch.nn.MultiheadAttention(
            hidden_dim,
            num_heads=8,
            batch_first=True
        )

        # Frame decoder with self-attention
        self.frame_decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            ),
            num_layers=6
        )

        # Frame prediction head
        self.frame_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 3 * frame_height * frame_width)
        )

        # Learnable frame embeddings
        self.frame_queries = torch.nn.Parameter(
            torch.randn(num_frames, hidden_dim)
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Generate video frames from audio."""

        # Encode audio
        audio_features = self.audio_encoder(audio)

        # Initialize frame queries
        frame_queries = self.frame_queries.unsqueeze(0).expand(
            audio.shape[0], -1, -1
        )

        # Cross-attention: align frames with audio
        attended_frames, _ = self.cross_attention(
            frame_queries,
            audio_features,
            audio_features
        )

        # Decode frames with self-attention
        frame_features = self.frame_decoder(
            attended_frames,
            attended_frames
        )

        # Predict frame pixels
        frame_logits = self.frame_head(frame_features)

        # Reshape to (batch, num_frames, 3, height, width)
        frames = frame_logits.reshape(
            audio.shape[0],
            self.num_frames,
            3,
            512,
            512
        )

        # Normalize to [0, 1]
        frames = torch.sigmoid(frames)

        return frames
```

### Step 3: Implement Training with Audio-Visual Alignment

Train on speech-video pairs with alignment loss:

```python
class AudioVideoTrainer:
    def __init__(self, generator: UnifiedAudioVideoGenerator):
        self.generator = generator
        self.optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=1e-4
        )

    def compute_alignment_loss(self,
                              audio_features: torch.Tensor,
                              video_features: torch.Tensor
                              ) -> torch.Tensor:
        """Loss enforcing audio-video alignment."""

        # Temporal alignment: match feature trends
        audio_diff = audio_features[:, 1:] - audio_features[:, :-1]
        video_diff = video_features[:, 1:] - video_features[:, :-1]

        alignment_loss = torch.nn.functional.mse_loss(
            audio_diff,
            video_diff
        )

        return alignment_loss

    def compute_reconstruction_loss(self,
                                   generated_frames: torch.Tensor,
                                   real_frames: torch.Tensor
                                   ) -> torch.Tensor:
        """L2 loss on frame reconstruction."""

        return torch.nn.functional.mse_loss(
            generated_frames,
            real_frames
        )

    def compute_perceptual_loss(self,
                               generated_frames: torch.Tensor,
                               real_frames: torch.Tensor,
                               pretrained_vgg: torch.nn.Module
                               ) -> torch.Tensor:
        """Perceptual loss using pretrained features."""

        gen_features = pretrained_vgg(generated_frames.reshape(
            -1, 3, 512, 512
        ))
        real_features = pretrained_vgg(real_frames.reshape(
            -1, 3, 512, 512
        ))

        return torch.nn.functional.mse_loss(
            gen_features,
            real_features
        )

    def train_step(self, audio_batch: torch.Tensor,
                  video_batch: torch.Tensor,
                  pretrained_vgg: torch.nn.Module
                  ) -> dict:
        """Single training step."""

        # Generate video from audio
        generated_frames = self.generator(audio_batch)

        # Encode audio for alignment loss
        audio_features = self.generator.audio_encoder(audio_batch)

        # Extract video features from generated frames
        video_features = pretrained_vgg(
            generated_frames.reshape(-1, 3, 512, 512)
        )[:generated_frames.shape[0]]

        # Compute losses
        recon_loss = self.compute_reconstruction_loss(
            generated_frames,
            video_batch
        )

        align_loss = self.compute_alignment_loss(
            audio_features,
            video_features
        )

        percept_loss = self.compute_perceptual_loss(
            generated_frames,
            video_batch,
            pretrained_vgg
        )

        # Combined loss
        total_loss = (recon_loss +
                     0.5 * align_loss +
                     0.1 * percept_loss)

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "reconstruction": recon_loss.item(),
            "alignment": align_loss.item(),
            "perceptual": percept_loss.item(),
            "total": total_loss.item()
        }
```

### Step 4: Inference with Audio Conditioning

Generate video from speech at inference time:

```python
def generate_video_from_speech(model: UnifiedAudioVideoGenerator,
                               audio_path: str,
                               output_path: str,
                               fps: int = 30):
    """Generate speaking video from audio file."""

    # Load audio
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)

    # Pad or truncate to model input
    target_samples = 16000 * 4  # 4 seconds
    if audio.shape[1] < target_samples:
        audio = torch.nn.functional.pad(
            audio,
            (0, target_samples - audio.shape[1])
        )
    else:
        audio = audio[:, :target_samples]

    # Generate video
    with torch.no_grad():
        audio = audio.unsqueeze(0)
        frames = model(audio)

    # Convert to numpy
    frames_np = frames[0].cpu().numpy()
    frames_np = (frames_np * 255).astype(np.uint8)
    frames_np = np.transpose(frames_np, (0, 2, 3, 1))

    # Save video
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps,
                            (512, 512))

    for frame in frames_np:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Video saved to {output_path}")
```

## Practical Guidance

**Unified Architecture**: Avoid audio-specific modules (e.g., speech-only decoders). General self-attention works better across diverse content.

**No Domain Restrictions**: Don't hard-code constraints on speakers or appearances. Let the model learn from data.

**Audio-Visual Alignment**: The alignment loss between audio and video feature gradients is crucial for synchronization. Use temporal derivatives for strong signal.

**Perceptual Loss**: Pretrained VGG features significantly improve quality over pixel-level L2 loss alone.

**When to Apply**: Use Mirage when generating realistic videos from speech, creating multimodal content with text-to-speech integration, or building interactive avatar systems.

## Reference

Mirage demonstrates that maintaining architectural generality doesn't compromise output quality in audio-to-video synthesis. The unified self-attention framework outperforms specialized designs by leveraging broader inductive biases. Key insight: alignment losses between audio and visual features provide the necessary constraint without domain-specific modules.
