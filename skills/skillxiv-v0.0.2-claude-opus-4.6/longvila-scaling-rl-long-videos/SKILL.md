---
name: longvila-scaling-rl-long-videos
title: "Scaling RL to Long Videos"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.07966"
keywords: [Long-Form Video, Reinforcement Learning, Reasoning, Multi-Modal, Sequence Parallelism]
description: "Train vision-language models on hour-long videos using a two-stage pipeline combining supervised fine-tuning with GRPO, reaching 71% accuracy on VideoMME while supporting 8,192 frames through efficient multi-modal sequence parallelism."
---

# LongVILA-R1: Extending Video LLMs to Hour-Long Reasoning Tasks

Standard video understanding models process clips of seconds, not entire movies or events. LongVILA-R1 addresses this through three components: a 104K question-answer dataset with chain-of-thought reasoning annotations, a training pipeline that starts with supervised fine-tuning then applies reinforcement learning, and a novel infrastructure innovation (MR-SP) that parallelizes both video encoding and prefilling to handle thousands of frames efficiently.

The key design choice is two-stage training: first establish reasoning capabilities via supervised examples, then push performance higher through RL rewards while remaining sample-efficient. Infrastructure innovations avoid the memory nightmare of processing hour-long videos on standard hardware.

## Core Concept

Long video reasoning is fundamentally different from short-clip understanding. Models must maintain temporal context over hours, track relationships across distant frames, and answer questions requiring holistic narrative understanding. LongVILA-R1 tackles this in three stages: (1) data engineers create a diverse reasoning dataset with temporal, spatial, goal-oriented, and plot-focused questions; (2) the model learns from these examples via supervised fine-tuning, establishing baseline capabilities; (3) GRPO RL refines answers using outcome-based rewards, pushing toward higher accuracy without requiring more annotations.

Infrastructure parallelism (MR-SP) is the practical enabler: by splitting the video encoding and LLM prefilling stages across GPUs, the approach achieves 2.1× speedup, making hour-long video training tractable.

## Architecture Overview

- **Base Model**: 7B and 1.5B parameter LLaVA-style vision-language models
- **Video Encoding**: Vision transformer processes frame tokens, supporting up to 8,192 frames
- **Multi-Modal Sequence Parallelism (MR-SP)**: Distributes video encoding across GPUs, prefills LLM layer-by-layer in parallel
- **Stage 1 Training**: Supervised fine-tuning on 36K filtered chain-of-thought examples
- **Stage 2 Training**: GRPO on 68K challenging + 102K open-source examples
- **LongVideo-Reason Dataset**: 104K QA pairs across sports, games, vlogs with reasoning annotations

## Implementation

### Step 1: Prepare Long Video Dataset with Reasoning Annotations

Construct a dataset of long videos paired with reasoning questions. Use video captioning and LLM generation to create diverse question types:

```python
import json
from datasets import Dataset
from typing import List, Dict

def create_long_video_dataset(video_paths: List[str],
                             video_captions: List[str]) -> Dataset:
    """
    Create dataset of long videos with reasoning questions.
    Input: list of video paths and their captions from NVILA-8B.
    Output: dataset with questions of 4 types: temporal, spatial, goal, narrative.
    """
    examples = []

    for video_path, caption in zip(video_paths, video_captions):
        # Use LLM (DeepSeek-R1) to generate diverse questions
        # Example prompt structure
        prompt = f"""Video caption: {caption}

Generate 4 reasoning questions about this video:
1. Temporal: What sequence of events occurs?
2. Goal/Purpose: What is the character/object trying to achieve?
3. Spatial: Where are the key objects/people located?
4. Plot/Narrative: How does the scene develop?

Provide questions and their answers based on typical video content."""

        # Call LLM to generate questions
        generated = call_llm(prompt, model="deepseek-r1-671b")

        # Parse generated questions and answers
        questions = parse_qa_from_llm_output(generated)

        for question_type, question, answer in questions:
            examples.append({
                "video_path": video_path,
                "question": question,
                "question_type": question_type,
                "answer": answer,
                "caption": caption
            })

    return Dataset.from_dict({
        "video_path": [ex["video_path"] for ex in examples],
        "question": [ex["question"] for ex in examples],
        "answer": [ex["answer"] for ex in examples],
        "question_type": [ex["question_type"] for ex in examples]
    })

def call_llm(prompt: str, model: str) -> str:
    """Call DeepSeek-R1 or equivalent to generate reasoning questions."""
    # Placeholder; in practice use API or local deployment
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def parse_qa_from_llm_output(output: str) -> List[tuple]:
    """Extract Q&A from LLM structured output."""
    # Parse LLM output into (question_type, question, answer) tuples
    lines = output.strip().split('\n')
    results = []
    for line in lines:
        if ':' in line:
            parts = line.split(':', 1)
            results.append(("generic", parts[0], parts[1]))
    return results
```

### Step 2: Supervised Fine-Tuning Stage

Train on 36K filtered examples with chain-of-thought reasoning to establish baseline capabilities:

```python
import torch
from transformers import AutoModel, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig

class VideoQAModel(torch.nn.Module):
    def __init__(self, model_name="qwen/qwen-vl-7b"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Optional: use LoRA for efficiency
        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        self.model = get_peft_model(self.model, lora_config)

    def forward(self, video_frames, question, answer=None):
        """
        Process long video and generate reasoning chain + answer.
        video_frames: [num_frames, 3, 224, 224]
        question: string
        answer: string (for training)
        """
        # Encode video frames
        vision_outputs = self.model.visual_encoder(video_frames)

        # Create prompt with chain-of-thought instruction
        prompt = f"Video understanding task.\nQuestion: {question}\nReasoning:\n"

        # Generate chain-of-thought reasoning
        reasoning_output = self.model.generate(
            vision_outputs,
            prompt,
            max_length=512,
            do_sample=False
        )

        if answer is not None:
            # Training: compute loss
            full_text = reasoning_output + f" Answer: {answer}"
            loss = compute_language_modeling_loss(self.model, full_text)
            return loss
        else:
            # Inference: return reasoning and answer
            return reasoning_output

def sft_training_loop(model, train_dataset, num_epochs=2):
    """
    Supervised fine-tuning on 36K examples.
    """
    training_args = TrainingArguments(
        output_dir="./models/longvila-sft",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=500,
        logging_steps=100,
        save_strategy="epoch",
        bf16=True  # Use bfloat16 on A100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda batch: default_data_collator(batch)
    )

    trainer.train()
    return model
```

### Step 3: GRPO Reinforcement Learning Stage

Refine using GRPO with outcome-based rewards on 68K challenging examples:

```python
import torch.nn.functional as F

def compute_grpo_loss(model, video_frames, question, ground_truth_answer):
    """
    Group Relative Policy Optimization: compare outputs within a batch group.
    Reward based on answer correctness and reasoning quality.
    """
    batch_size = 4
    num_samples_per_prompt = 4

    all_outputs = []
    all_rewards = []

    # Generate multiple samples per input
    for sample_id in range(num_samples_per_prompt):
        output = model.generate(
            video_frames,
            question,
            max_length=512,
            temperature=0.7 + 0.1 * sample_id  # Vary temperature
        )
        all_outputs.append(output)

        # Extract final answer from reasoning chain
        final_answer = extract_final_answer(output)

        # Reward 1: Accuracy - is answer correct?
        accuracy = 1.0 if final_answer == ground_truth_answer else 0.0

        # Reward 2: Reasoning quality - does output show work?
        reasoning_quality = 1.0 if "because" in output.lower() else 0.5

        # Combine rewards
        reward = 0.8 * accuracy + 0.2 * reasoning_quality
        all_rewards.append(reward)

    # Group relative loss: compare within the group
    all_rewards = torch.tensor(all_rewards, device=model.device)
    mean_reward = all_rewards.mean()
    advantages = all_rewards - mean_reward

    # Policy gradient with advantage weighting
    log_probs = compute_log_probs(model, all_outputs, question)
    loss = -(log_probs * advantages).mean()

    # KL divergence penalty (stay close to SFT model)
    kl_loss = compute_kl_divergence(model, sft_model)
    total_loss = loss + 0.04 * kl_loss

    return total_loss

def grpo_training_loop(model, sft_model, train_dataset, num_epochs=1):
    """
    GRPO refinement on 68K challenging + 102K open-source examples.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_dataset)
    )

    for epoch in range(num_epochs):
        for batch in train_dataset:
            video_frames = batch["video"]
            question = batch["question"]
            answer = batch["answer"]

            loss = compute_grpo_loss(
                model, video_frames, question, answer
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return model

def extract_final_answer(reasoning_text: str) -> str:
    """Extract final answer from chain-of-thought reasoning."""
    lines = reasoning_text.split('\n')
    for line in reversed(lines):
        if 'answer' in line.lower():
            return line.split(':')[-1].strip()
    return reasoning_text.split()[-1]
```

### Step 4: Inference with Multi-Modal Sequence Parallelism

During inference, distribute video encoding across GPUs and parallelize LLM prefilling:

```python
def inference_with_mrsp(model, video_path, question,
                       num_gpus=8, device_ids=None):
    """
    Inference using Multi-modal Reinforcement Sequence Parallelism.
    Distributes video encoding and LLM prefilling across GPUs.

    Achieves ~2.1x speedup for hour-long videos (8,192 frames).
    """
    if device_ids is None:
        device_ids = list(range(num_gpus))

    # Stage 1: Distribute video encoding across GPUs
    video_frames = load_video_frames(video_path)  # [8192, 3, 224, 224]
    chunk_size = len(video_frames) // num_gpus

    vision_outputs_per_gpu = []
    for gpu_id in device_ids:
        start_idx = gpu_id * chunk_size
        end_idx = (gpu_id + 1) * chunk_size
        video_chunk = video_frames[start_idx:end_idx]

        with torch.device(f"cuda:{gpu_id}"):
            chunk_output = model.visual_encoder(video_chunk)
            vision_outputs_per_gpu.append(chunk_output)

    # Gather outputs on primary GPU
    all_vision_outputs = torch.cat(vision_outputs_per_gpu, dim=0)

    # Stage 2: Parallelize LLM prefilling
    # Prefix tree structure: different prefixes processed simultaneously
    prompt = f"Question: {question}\nAnswer:"
    prompt_tokens = tokenizer(prompt)["input_ids"]

    # Run LLM with distributed prefilling
    output_tokens = model.generate(
        input_ids=prompt_tokens,
        vision_outputs=all_vision_outputs,
        max_length=256,
        do_sample=False,
        num_return_sequences=1
    )

    answer = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    return answer

def load_video_frames(video_path, num_frames=8192):
    """Load and sample frames from hour-long video."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Uniformly sample num_frames
    frame_indices = torch.linspace(
        0, total_frames - 1, num_frames
    ).long()

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx.item())
        ret, frame = cap.read()
        if ret:
            # Resize to 224x224
            frame = cv2.resize(frame, (224, 224))
            frames.append(torch.from_numpy(frame).float() / 255.0)

    cap.release()
    return torch.stack(frames)
```

## Practical Guidance

| Hyperparameter | Recommended Value | Notes |
|---|---|---|
| SFT Dataset Size | 36K examples | High-quality, filtered examples |
| GRPO Dataset Size | 68K challenging + 102K open-source | Diverse for robustness |
| Max Frames Supported | 8,192 frames | ~40-50 minutes of video |
| SFT Learning Rate | 2e-4 | Standard for instruction tuning |
| GRPO Learning Rate | 2e-5 | Conservative for RL stability |
| KL Beta | 0.04 | Prevents divergence from SFT |
| Accuracy Reward Weight | 0.8 | Prioritize correctness |
| Reasoning Quality Weight | 0.2 | Secondary emphasis |
| Gradient Accumulation | 8 steps | Effective batch size ~32 |
| GPU Setup | 8 A100s or equivalent | Multi-modal sequence parallelism needs multiple GPUs |
| Temperature Variation | 0.7-1.0 for GRPO | Encourage diverse outputs |

**When to use LongVILA-R1:**
- Long-form video understanding (movies, sports games, tutorials)
- Reasoning tasks requiring temporal understanding across hours
- Scenarios where RL can improve upon supervised baselines
- Applications with sufficient GPU resources (multi-GPU setups)

**When NOT to use LongVILA-R1:**
- Short video clips (< 1 minute) where standard models suffice
- Single-GPU deployments (sequence parallelism requires multiple GPUs)
- Real-time inference (preprocessing and multi-GPU coordination adds latency)
- Data-scarce domains without ability to create reasoning annotations

**Common pitfalls:**
- Not filtering SFT examples, leading to learned poor reasoning patterns
- KL beta too high, preventing RL from improving over SFT baseline
- Temperature too low in GRPO, reducing output diversity
- Not balancing accuracy and reasoning rewards, overfitting to answering without explanation
- Video sampling not matching temporal patterns (sports vs narrative vs tutorial)
- GPU communication overhead negating parallelism gains with small batch sizes

## Reference

Long, Y., Yu, M., Shen, S., & Chen, W. (2025). Scaling RL to Long Videos. arXiv:2507.07966. https://arxiv.org/abs/2507.07966
