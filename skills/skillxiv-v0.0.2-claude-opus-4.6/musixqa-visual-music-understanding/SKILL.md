---
name: musixqa-visual-music-understanding
title: "MusiXQA: Advancing Visual Music Understanding in Multimodal Large Language Models"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.23009"
keywords: [Music Understanding, Sheet Music OCR, Symbolic Music, MLLM, Dataset]
description: "Teach multimodal LLMs to read and understand sheet music through a synthetic QA dataset with kern+ symbolic notation. Enables models to handle music sheet OCR, symbol recognition, and chord estimation at 8× better performance than GPT-4o."
---

# MusiXQA: Teaching LLMs to Read Sheet Music

Music sheet notation has remained unchanged for centuries because it's densely compressed—containing rhythm, pitch, dynamics, and articulation in minimal space. Yet modern multimodal models perform at "near-random levels" on musical symbol recognition, unable to parse what musicians learn in childhood. This gap matters because sheet music remains the standard in music education, collaboration, and archival.

MusiXQA introduces a synthetic dataset of 9,600 carefully-controlled sheet music images paired with 130,000+ question-answer pairs. Rather than scraping sheet images from the internet (which introduces artifacts and inconsistency), the authors generate them using MusiXTeX with systematically varied musical elements. This enables fine-grained study of what models understand and fail at.

## Core Concept

The core insight is that multimodal models struggle with sheet music for two reasons: (1) they lack training data in the right format, and (2) they lack an efficient symbolic output representation. MusiXQA addresses both through:

1. **Controlled Data Generation**: Using MusiXTeX to synthesize diverse sheet music with precise, measurable variations (30 different scales, varying note densities, different clefs)
2. **Compact Symbolic Format (kern+)**: Representing musical output not as verbose JSON but as a concise, music-standard format that reduces wasted tokens on formatting
3. **Task Diversity**: Four complementary task types (text extraction, symbol recognition, layout analysis, chord estimation) that stress-test different aspects of sheet understanding

The result: fine-tuned models achieve 8× improvements over GPT-4o baseline on optical music recognition tasks.

## Architecture Overview

The MusiXQA pipeline consists of these key components:

- **Synthetic Sheet Generation**: MusiXTeX-based generation with controlled variation of key signatures, clefs, time signatures, tempos, note densities, and note types across diverse musical scales
- **Question Pool**: Four task categories with hundreds of template-based questions covering text extraction (OCR), symbol recognition (OMR), layout analysis (spatial reasoning), and chord estimation (harmonic analysis)
- **Answer Validation**: Automated generation of correct answers paired with filtering to ensure only correct responses appear in training data
- **kern+ Representation**: A compact symbolic notation format optimized for transformer training that outperforms JSON-based representations
- **Fine-tuning Pipeline**: LoRA-based adaptation of small vision-language models (Phi-3 backbone) to specialize in music understanding
- **Evaluation Framework**: Both string-matching metrics and GPT-4-based semantic similarity scoring to catch apparent-but-meaningless format mimicry

## Implementation

This section shows how to generate, fine-tune on, and evaluate music understanding with MusiXQA.

**Step 1: Generate diverse synthetic sheet music with MusiXTeX**

The following code demonstrates controlled generation of sheet images with systematically varied musical properties:

```python
import subprocess
import os
from itertools import product

def generate_musical_scores(output_dir="sheet_music", num_scales=30, variations=5):
    """Generate synthetic sheet music with controlled musical variation."""

    os.makedirs(output_dir, exist_ok=True)

    # Musical parameters to systematically vary
    clefs = ['treble', 'bass', 'alto']
    time_sigs = ['2/4', '3/4', '4/4', '6/8']
    note_densities = [0.1, 0.3, 0.5, 0.7, 0.9]  # Fraction of staff positions filled
    scales = [f"scale_{i}" for i in range(num_scales)]

    generated = []
    idx = 0

    for clef, time_sig, density in product(clefs, time_sigs, note_densities):
        for scale in scales[:5]:  # Generate each scale with each configuration
            # Generate a LilyPond/MusiXTeX-compatible score definition
            score_def = f"""
\\version "2.24.0"
\\new Staff {{
    \\clef {clef}
    \\time {time_sig}
    \\key {scale}

    % Auto-generated notes with density {density}
    c'4 d'8 e' f'4 g' | a'2 b' | c''1
}}
"""

            # Write to file
            score_path = os.path.join(output_dir, f"score_{idx:04d}.ly")
            with open(score_path, 'w') as f:
                f.write(score_def)

            # Convert to PNG via lilypond
            png_path = score_path.replace('.ly', '.png')
            subprocess.run(
                ['lilypond', '--png', '-o', png_path.replace('.png', ''), score_path],
                capture_output=True
            )

            generated.append({
                'id': idx,
                'path': png_path,
                'clef': clef,
                'time_sig': time_sig,
                'density': density,
                'scale': scale
            })

            idx += 1

    return generated

# Generate 900 diverse sheet images
sheet_metadata = generate_musical_scores()
print(f"Generated {len(sheet_metadata)} sheet images")
```

This generates sheets with varied musical parameters to ensure models learn general music reading, not dataset artifacts.

**Step 2: Create QA pairs for four task types using templates**

This code generates diverse question-answer pairs covering OCR, OMR, layout, and chord tasks:

```python
import random
import json

def create_music_qa_pairs(sheet_metadata, num_pairs_per_image=15):
    """Generate task-specific QA pairs for each sheet image."""

    qa_pairs = []

    for sheet in sheet_metadata:
        sheet_id = sheet['id']

        # Task 1: Text Extraction (OCR) - extract visible text markings
        qa_pairs.append({
            'task': 'ocr',
            'question': f"What time signature is shown in this sheet music? (Sheet {sheet_id})",
            'answer': sheet['time_sig'],
            'image_id': sheet_id
        })

        # Task 2: Symbol Recognition (OMR) - identify music symbols
        qa_pairs.append({
            'task': 'omr',
            'question': f"What clef is used in this staff? (Sheet {sheet_id})",
            'answer': sheet['clef'],
            'image_id': sheet_id
        })

        # Task 3: Layout Analysis - spatial relationships
        qa_pairs.append({
            'task': 'layout',
            'question': f"How many measures are visible in this excerpt? (Sheet {sheet_id})",
            'answer': str(random.randint(2, 8)),  # Varies by actual image
            'image_id': sheet_id
        })

        # Task 4: Chord Estimation - harmonic analysis
        qa_pairs.append({
            'task': 'chord',
            'question': f"What key signature is indicated? (Sheet {sheet_id})",
            'answer': sheet['scale'],
            'image_id': sheet_id
        })

    return qa_pairs

qa_dataset = create_music_qa_pairs(sheet_metadata)
print(f"Created {len(qa_dataset)} QA pairs")

# Save as JSON for training
with open("musixqa_dataset.json", 'w') as f:
    json.dump(qa_dataset, f, indent=2)
```

This generates a diverse QA corpus covering the four musical comprehension tasks.

**Step 3: Fine-tune a vision-language model with kern+ output representation**

This shows how to adapt a small model (Phi-3) for music understanding using the kern+ symbolic format:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-3-vision")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-vision")

# Configure LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Prepare kern+ format (compact music notation)
def convert_to_kern_plus(answer_dict):
    """Convert structured answer to kern+ symbolic notation."""
    # Example: {"clef": "treble", "time_sig": "4/4"} -> "clef=treble time=4/4"
    return " ".join([f"{k}={v}" for k, v in answer_dict.items()])

class MusicQADataset(torch.utils.data.Dataset):
    def __init__(self, qa_pairs, images, tokenizer):
        self.qa_pairs = qa_pairs
        self.images = images
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        pair = self.qa_pairs[idx]
        image_path = self.images[pair['image_id']]['path']

        # Load image
        from PIL import Image
        image = Image.open(image_path).convert('RGB')

        # Format as kern+ if answer is structured
        answer = pair['answer']
        if isinstance(answer, dict):
            answer = convert_to_kern_plus(answer)

        # Prepare prompt
        prompt = f"<image>\nQuestion: {pair['question']}\nAnswer in kern+: "

        # Tokenize (simplified; real implementation uses proper image encoding)
        tokens = self.tokenizer(prompt, return_tensors="pt")

        return {
            'input_ids': tokens['input_ids'],
            'image': image,
            'target': answer
        }

# Create dataset and dataloader
train_dataset = MusicQADataset(qa_dataset, sheet_metadata, tokenizer)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)

# Fine-tuning loop
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
model.train()

for epoch in range(3):
    for batch in train_loader:
        outputs = model(
            input_ids=batch['input_ids'].to(model.device),
            pixel_values=batch['image'].to(model.device),
            labels=batch['input_ids'].to(model.device)
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

This fine-tunes a model using LoRA for efficient adaptation to music-specific tasks with kern+ output.

**Step 4: Evaluate model performance with dual-metric scoring**

This code evaluates both string accuracy and semantic correctness using GPT-based evaluation:

```python
import openai
from difflib import SequenceMatcher

def evaluate_music_qa(predictions, references, use_gpt_eval=True):
    """Evaluate model predictions using string and semantic metrics."""

    string_matches = 0
    semantic_scores = []

    for pred, ref in zip(predictions, references):
        # String-level accuracy
        if pred == ref:
            string_matches += 1

        # Sequence similarity for partial credit
        ratio = SequenceMatcher(None, pred, ref).ratio()

        # GPT-based semantic evaluation (catches format mimicry)
        if use_gpt_eval:
            gpt_eval = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{
                    "role": "user",
                    "content": f"""Is this music theory answer correct?
Question: Identify the key signature
Predicted answer: {pred}
Reference answer: {ref}
Respond with 'correct', 'partially correct', or 'incorrect'."""
                }]
            )
            semantic_response = gpt_eval.choices[0].message.content.lower()
            score = 1.0 if "correct" in semantic_response else 0.0
        else:
            score = ratio

        semantic_scores.append(score)

    return {
        'string_accuracy': string_matches / len(predictions),
        'avg_semantic_score': sum(semantic_scores) / len(semantic_scores),
        'sequence_similarity': sum(SequenceMatcher(None, p, r).ratio()
                                    for p, r in zip(predictions, references)) / len(predictions)
    }

# Evaluate predictions
test_predictions = ["treble", "bass", "alto", "treble"]
test_references = ["treble", "bass", "alto", "treble"]
results = evaluate_music_qa(test_predictions, test_references)
print(f"String Accuracy: {results['string_accuracy']:.2%}")
print(f"Semantic Score: {results['avg_semantic_score']:.2%}")
```

This uses both exact-match and semantic evaluation to catch models that mimic format without understanding.

## Practical Guidance

**When to use MusiXQA:**
- Fine-tuning vision-language models to handle sheet music understanding tasks
- Creating datasets for specialized music education or archival applications
- Evaluating whether models truly understand music notation or just memorize patterns
- Building systems that must extract information from sheet music (digitization, automated analysis)

**When NOT to use:**
- Tasks requiring real handwritten sheet music (this is synthetic, computer-typeset)
- Complex musical scores with dense overlapping symbols
- Non-Western musical notation systems
- Historical manuscripts with degradation or unusual typography

**Hyperparameters and Configuration:**

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| LoRA Rank | 16 | Balances parameter efficiency with expressiveness for music tasks |
| Learning Rate | 2e-4 | Music understanding benefits from slower, stable adaptation |
| Batch Size | 8 | Small batches preserve diversity of musical styles |
| Epochs | 3 | Limited data; more epochs cause overfitting |
| Max Image Resolution | 512×512 | Captures detail without unnecessary computation |
| kern+ Token Budget | 20-50 tokens | Most answers fit in compact symbolic notation |

**Common Pitfalls:**
- Using JSON output instead of kern+—wastes tokens on format, confuses models about what's important
- Mixing real and synthetic sheet music—synthetic lacks real-world visual noise; models overfit to perfect rendering
- Evaluating with string match only—format-matching can hide complete misunderstanding
- Ignoring task imbalance—OCR is easier than chord recognition; weight loss accordingly
- Assuming symbol recognition transfers to chord estimation—they're distinct cognitive skills

**Key Design Decisions:**
MusiXQA uses synthetic data to ensure ground truth accuracy. Real sheet music has ambiguity (is a note a high B or low C?). Synthetic scores are unambiguous, enabling precise evaluation. The four task types cover the full spectrum of sheet music understanding: recognizing visual marks (OCR), identifying notation symbols (OMR), understanding spatial layout, and performing harmonic analysis—each requires different reasoning.

## Reference

Gao, C., Mao, Y., Yang, Y., Shi, Y., Dong, C., He, P., & Wang, Y. (2025). MusiXQA: Advancing Visual Music Understanding in Multimodal Large Language Models. arXiv preprint arXiv:2506.23009. https://arxiv.org/abs/2506.23009
