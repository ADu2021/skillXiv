---
name: mathflow-visual-mathematical-reasoning
title: "MathFlow: Enhancing the Perceptual Flow of MLLMs for Visual Mathematical Problems"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2503.16549"
keywords: [Multimodal Learning, Mathematical Reasoning, Vision-Language Models, Visual Perception, Problem Decomposition]
description: "Improve mathematical problem-solving in multimodal models by decoupling visual perception from inference reasoning. A two-stage pipeline extracts essential visual information and reasoned properties before passing enriched text to inference models, dramatically improving accuracy on visual math problems."
---

## Core Concept

MathFlow addresses a fundamental bottleneck in visual mathematical reasoning: most multimodal large language models (MLLMs) "rely more on reading text than seeing diagrams." The solution separates problem-solving into two independent stages—perception and inference—each optimized for its specific task. A specialized perception model first extracts visual information, then a separate inference model performs mathematical reasoning on enriched text representations.

## Architecture Overview

The MathFlow system decomposes visual math problem-solving into distinct stages:

- **FlowVerse Benchmark Framework**: Categorizes problem information into four semantic components—Descriptive Information (scene setup), Essential Information (critical visual elements), Reasoned Property (derived observations), and Only Question (the actual query)—enabling systematic evaluation across six problem versions
- **Perception Stage (MathFlow-P-7B)**: A specialized 7B parameter model trained to extract critical visual information from diagrams and convert it to high-quality text representations
- **Inference Stage**: Any inference model (GPT-4V, DeepSeek-r1, Gemini-1.5-pro) uses the enriched textual input for mathematical reasoning without needing to "see" the diagram

## Implementation

### FlowVerse Benchmark Construction

The FlowVerse benchmark systematically evaluates MLLMs by generating six problem versions that vary information availability. This enables measuring how models leverage different information types and where they struggle.

```python
from dataclasses import dataclass
from typing import List, Dict
import json

@dataclass
class MathProblem:
    """Structure for visual mathematical problems with information decomposition."""
    descriptive_info: str  # Scene/context description
    essential_info: str    # Critical visual elements from diagram
    reasoned_property: str # Derived observations/properties
    question: str          # The actual question asked
    image_path: str        # Reference image path
    ground_truth: str      # Correct answer

def generate_flowverse_versions(problem: MathProblem) -> Dict[str, str]:
    """Generate six problem versions to evaluate different information types."""
    versions = {
        "v1_image_only": problem.question,  # Image + question only
        "v2_descriptive": (
            f"{problem.descriptive_info}\n{problem.question}"
        ),  # With scene description
        "v3_essential": (
            f"{problem.essential_info}\n{problem.question}"
        ),  # With essential visual info
        "v4_reasoned": (
            f"{problem.essential_info}\n{problem.reasoned_property}\n"
            f"{problem.question}"
        ),  # With derived properties
        "v5_full": (
            f"{problem.descriptive_info}\n{problem.essential_info}\n"
            f"{problem.reasoned_property}\n{problem.question}"
        ),  # All information
        "v6_cot_enhanced": (
            f"{problem.descriptive_info}\n{problem.essential_info}\n"
            f"{problem.reasoned_property}\nStep-by-step reasoning:\n"
            f"{problem.question}"
        ),  # With CoT scaffold
    }
    return versions
```

### MathFlow-P Perception Model Training

The perception stage uses multi-task pretraining to extract two critical information types. A 3:1 data ratio between Essential Information (EI) and Reasoned Property (RP) tasks leverages larger EI datasets while focusing RP training on the more challenging extraction task.

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch.optim as optim

class MathFlowPerceptionModel(nn.Module):
    """Two-task perception model for visual information extraction."""

    def __init__(self, base_model="Qwen2-VL-7B"):
        super().__init__()
        self.vision_encoder = AutoModel.from_pretrained(base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        # Task-specific heads
        self.ei_head = nn.Linear(4096, 768)  # Essential Info head
        self.rp_head = nn.Linear(4096, 768)  # Reasoned Property head
        self.ei_decoder = nn.Linear(768, 50257)  # Token decoder for EI
        self.rp_decoder = nn.Linear(768, 50257)  # Token decoder for RP

    def forward(self, image, task_type="ei"):
        """Extract visual information based on task type."""
        # Encode image to visual features
        visual_features = self.vision_encoder(image)

        if task_type == "ei":
            # Extract essential information (critical visual elements)
            ei_embedding = self.ei_head(visual_features)
            ei_tokens = self.ei_decoder(ei_embedding)
            return ei_tokens
        elif task_type == "rp":
            # Extract reasoned property (derived observations)
            rp_embedding = self.rp_head(visual_features)
            rp_tokens = self.rp_decoder(rp_embedding)
            return rp_tokens

def train_perception_model(
    model, train_loader_ei, train_loader_rp,
    ei_ratio=0.75, epochs=10, lr=1e-5
):
    """Train MathFlow-P with 3:1 EI:RP data ratio."""
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0

        # EI pretraining (75% of batches)
        for batch_idx, (images, ei_targets) in enumerate(train_loader_ei):
            if batch_idx / len(train_loader_ei) > ei_ratio:
                break

            ei_predictions = model(images, task_type="ei")
            loss = criterion(ei_predictions, ei_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # RP pretraining (25% of batches)
        for batch_idx, (images, rp_targets) in enumerate(train_loader_rp):
            if batch_idx / len(train_loader_rp) > (1 - ei_ratio):
                break

            rp_predictions = model(images, task_type="rp")
            loss = criterion(rp_predictions, rp_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

    return model
```

### Supervised Fine-Tuning for Accuracy

After multi-task pretraining, fine-tune the perception model on the MathFlow-SFT dataset specifically designed to improve accuracy on real math problems. This stage adapts the model to the target domain and task distributions.

```python
def supervised_finetune_perception(
    pretrained_model, sft_dataset, batch_size=16,
    num_epochs=5, lr=5e-6
):
    """Fine-tune perception model for accuracy on real problems."""
    model = pretrained_model
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataloader = torch.utils.data.DataLoader(
        sft_dataset, batch_size=batch_size, shuffle=True
    )

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images, ei_texts, rp_texts, questions in dataloader:
            # Extract EI and RP from visual input
            ei_predictions = model(images, task_type="ei")
            rp_predictions = model(images, task_type="rp")

            # Tokenize ground truth texts
            ei_targets = model.tokenizer(
                ei_texts, return_tensors="pt", padding=True
            )["input_ids"]
            rp_targets = model.tokenizer(
                rp_texts, return_tensors="pt", padding=True
            )["input_ids"]

            # Combined loss
            loss_ei = criterion(ei_predictions, ei_targets)
            loss_rp = criterion(rp_predictions, rp_targets)
            loss = loss_ei + loss_rp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"FT Epoch {epoch}: Loss = {epoch_loss / len(dataloader):.4f}")

    return model
```

### MathFlow Inference Pipeline

During inference, use the trained perception model to extract rich textual descriptions from problem images, then pass the enriched text to any inference model for mathematical reasoning. This decoupling allows leveraging state-of-the-art reasoning models.

```python
def mathflow_inference(
    perception_model, inference_model,
    problem_image, problem_question, use_cot=True
):
    """Apply MathFlow pipeline: perception + inference."""

    # Stage 1: Extract visual information through perception model
    with torch.no_grad():
        ei_tokens = perception_model(problem_image, task_type="ei")
        rp_tokens = perception_model(problem_image, task_type="rp")

    # Decode to text
    essential_info = decode_tokens_to_text(ei_tokens)
    reasoned_property = decode_tokens_to_text(rp_tokens)

    # Stage 2: Construct enriched problem text
    if use_cot:
        enriched_prompt = f"""
Essential Visual Information:
{essential_info}

Derived Properties:
{reasoned_property}

Problem:
{problem_question}

Solve this step-by-step:
"""
    else:
        enriched_prompt = f"""
{essential_info}
{reasoned_property}
{problem_question}
"""

    # Stage 3: Perform inference with state-of-the-art model
    # (GPT-4V, DeepSeek-r1, Gemini-1.5-pro, etc.)
    response = inference_model.generate(enriched_prompt, max_tokens=500)

    return response

def evaluate_on_flowverse(
    perception_model, inference_model,
    flowverse_benchmark, num_samples=2000
):
    """Evaluate MathFlow across six problem versions."""
    results = {}

    for version_name in ["v1", "v2", "v3", "v4", "v5", "v6"]:
        correct = 0

        for problem in flowverse_benchmark[:num_samples]:
            # Generate answer with specific version
            answer = mathflow_inference(
                perception_model, inference_model,
                problem["image"], problem[version_name]
            )

            if check_answer_correctness(answer, problem["ground_truth"]):
                correct += 1

        accuracy = correct / num_samples
        results[version_name] = accuracy
        print(f"{version_name}: {accuracy:.1%}")

    return results
```

## Practical Guidance

**When to use MathFlow:**
- You have visual math problems (geometry, diagrams, charts) where perception is a bottleneck
- You want to leverage advanced reasoning models (GPT-4, Gemini, DeepSeek) without expensive fine-tuning
- You have limited training data for your specific math domain
- You need interpretable extraction of visual information before reasoning

**When NOT to use:**
- Problems are purely text-based without visual diagrams
- Your inference model already has strong visual understanding (some cutting-edge multimodal models)
- You need real-time single-pass inference (two-stage pipeline adds latency)
- Your training data is extremely small (< 10,000 problem-image pairs)

**Hyperparameter tuning:**
- **EI:RP ratio**: 3:1 ratio shown optimal; adjust to 2:1 if RP extraction is critical for your domain
- **Pretraining LR**: 1e-5 balances convergence and stability; reduce to 5e-6 if training diverges
- **Fine-tuning LR**: 5e-6 is conservative; can increase to 1e-5 if validation metrics plateau
- **Batch size**: Use 16-32 for perception model (constrained by image resolution); scale down on limited VRAM

**Common pitfalls:**
- **Mismatch between EI extraction and question type**: Geometry problems need precise spatial info, while physics problems need physical property extraction; customize task definitions per domain
- **Insufficient SFT data**: 130,000 SFT samples were critical; with < 50,000, overfitting is likely
- **Inference model selection**: Pairing weak reasoning models with excellent perception extraction still yields poor results; match reasoning capability to problem difficulty
- **Token imbalance**: Very long essential info extractions can exceed context limits; implement truncation strategies

## Reference

- **Perception architecture**: Vision encoder (Qwen2-VL) + dual task heads inspired by multi-task learning frameworks
- **Data sources**: MAVIS and Geo170k for EI pretraining (650,000 image-text pairs), custom MathFlow-RP dataset (130,000 samples), evaluation on MathVerse and custom FlowVerse benchmark
- **Evaluation metrics**: Accuracy on FlowVerse (6-version evaluation), MathVerse testmini baseline comparison, ablation analysis of information contribution
- **Key findings**: MLLMs heavily weight text over visual information; RP (reasoned properties) contribute significantly to performance; enhanced perception is a critical bottleneck in mathematical reasoning
- **Related work**: MAVIS (math-aware visual instruction tuning), Geo170k (geometric reasoning), CoT methods for visual reasoning
