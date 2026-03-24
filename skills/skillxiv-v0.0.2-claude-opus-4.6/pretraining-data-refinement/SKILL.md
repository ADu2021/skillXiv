---
name: pretraining-data-refinement
title: "RefineX: Learning to Refine Pre-training Data at Scale from Expert-Guided Programs"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.03253"
keywords: [Data Refinement, Pretraining Efficiency, Distillation, Program Extraction, Text Cleaning]
description: "Improve pretraining efficiency by refining noisy data through expert-guided programs: learn to generate deletion operations that clean documents, achieving 2.6-7.2% performance gains with fewer training tokens."
---

# RefineX: Scalable Data Refinement via Learned Edit Programs

Pretraining data quality directly impacts model performance, but refining large-scale corpora is expensive. RefineX proposes learning what to delete: an expert LLM generates refined versions of documents, the system extracts minimal edit operations (deletions only), and a compact model learns to apply those operations at scale. This approach achieves 2.6%-7.2% improvements on downstream tasks using significantly fewer training tokens, with the advantage that edit programs are interpretable and don't introduce hallucinated content.

The key insight is that refinement doesn't require complex edits—deletion suffices for many quality issues. Removing duplicates, irrelevant sections, noisy content, and low-quality text often improves downstream performance more than inserting or rewriting. By constraining edits to deletions and learning programmatic operations, RefineX avoids the hallucination risks of generative refinement while remaining computationally efficient.

## Core Concept

RefineX operates in three phases. First, an expert LLM (Qwen2.5-72B) refines documents by removing low-quality content, emphasizing deletions over rewrites. Second, the system identifies minimal deletion operations needed to transform original text into refined versions using edit distance algorithms. Third, a compact student model (0.6B parameters) learns to generate three functions—`remove_lines()`, `remove_str()`, and `keep_all()`—which execute efficiently. The student model applies these operations to the full pretraining corpus at scale.

This three-stage approach balances quality (expert refinement), efficiency (compact student model), and interpretability (explicit deletion operations). The result is cleaner pretraining data with measurable improvements across multiple downstream tasks.

## Architecture Overview

The system comprises three stages:

- **End-to-End Refinement**: Expert LLM generates cleaned versions emphasizing deletion-based operations, producing high-quality supervision
- **Program Extraction**: Minimum edit distance algorithms identify minimal deletion operations transforming original to refined text, discarding insertions/replacements
- **Model Distillation & Execution**: Compact 0.6B model learns to predict removal operations, which execute as simple string manipulations at scale

## Implementation

Start with the program extraction engine using edit distance:

```python
from difflib import SequenceMatcher
from typing import List, Tuple, Dict
import re

class ProgramExtractor:
    """
    Extract minimal deletion operations from original to refined text.

    Uses edit distance to identify what lines/spans to delete,
    producing interpretable, hallucination-free refinement programs.
    """

    def __init__(self):
        self.operations = []

    def extract_line_deletions(self, original: str, refined: str) -> List[Tuple[int, int]]:
        """
        Identify lines deleted from original to get refined.

        Returns list of (start_line, end_line) tuples marking deleted ranges.
        """
        original_lines = original.split('\n')
        refined_lines = refined.split('\n')

        # Use sequence matcher to find longest common subsequence
        matcher = SequenceMatcher(None, original_lines, refined_lines)
        matching_blocks = matcher.get_matching_blocks()

        # Identify gaps as deletions
        deletions = []
        last_orig_end = 0

        for block in matching_blocks:
            orig_start, refined_start, size = block.a, block.b, block.size

            # Gap in original = deleted lines
            if orig_start > last_orig_end:
                deletions.append((last_orig_end, orig_start))

            last_orig_end = orig_start + size

        # Check for deletion at end
        if last_orig_end < len(original_lines):
            deletions.append((last_orig_end, len(original_lines)))

        return deletions

    def extract_substring_deletions(self, original: str, refined: str,
                                    max_span_length: int = 100) -> List[Dict]:
        """
        Extract substrings (spans) deleted from original to get refined.

        More granular than line-level; identifies specific text spans to remove.
        """
        deletions = []

        # Find longest common subsequences
        matcher = SequenceMatcher(None, original, refined)
        matching_blocks = matcher.get_matching_blocks()

        last_orig_end = 0
        for block in matching_blocks:
            orig_start, refined_start, size = block.a, block.b, block.size

            # Span between last match and this match = deletion
            if orig_start > last_orig_end:
                deleted_text = original[last_orig_end:orig_start]
                if len(deleted_text) <= max_span_length:
                    deletions.append({
                        'span_start': last_orig_end,
                        'span_end': orig_start,
                        'text': deleted_text
                    })

            last_orig_end = orig_start + size

        # Check end
        if last_orig_end < len(original):
            deleted_text = original[last_orig_end:]
            if len(deleted_text) <= max_span_length:
                deletions.append({
                    'span_start': last_orig_end,
                    'span_end': len(original),
                    'text': deleted_text
                })

        return deletions

    def extract_program(self, original: str, refined: str) -> Dict:
        """
        Extract complete refinement program (set of deletion operations).

        Returns dict with operation types and targets that reproduce
        the expert refinement.
        """
        line_deletions = self.extract_line_deletions(original, refined)
        substring_deletions = self.extract_substring_deletions(original, refined)

        # Decide whether to use line-level or substring-level operations
        if len(line_deletions) <= 5:
            # Simple line deletions are common
            operations = [{
                'type': 'remove_lines',
                'ranges': line_deletions
            }]
        else:
            # More complex; use substring deletions
            operations = [{
                'type': 'remove_substrings',
                'spans': substring_deletions
            }]

        return {
            'original_length': len(original),
            'refined_length': len(refined),
            'compression_ratio': len(refined) / (len(original) + 1e-8),
            'operations': operations
        }
```

Implement the student model learning to predict operations:

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class RefinementOperationPredictor(nn.Module):
    """
    Learn to predict text refinement operations from document.

    Student model trained to generate remove_lines() and remove_str()
    function calls that reproduce expert refinement at scale.
    """

    def __init__(self, model_name: str = "gpt2-medium"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model_name = model_name

        # Special tokens for operations
        self.remove_lines_token = "<REMOVE_LINES>"
        self.remove_str_token = "<REMOVE_STR>"
        self.keep_all_token = "<KEEP_ALL>"

        # Add special tokens to vocab
        self.tokenizer.add_tokens([
            self.remove_lines_token,
            self.remove_str_token,
            self.keep_all_token
        ])
        self.model.resize_token_embeddings(len(self.tokenizer))

    def generate_program(self, text: str, max_new_tokens: int = 256) -> str:
        """
        Generate refinement program (sequence of remove operations).

        Returns string like:
        "remove_lines([0, 5], [10, 12]) remove_str('spam', 'ads')"
        """
        prompt = f"""Analyze the text and generate removal operations to clean it:

Text: {text[:500]}

Output removal operations using functions:
- remove_lines(start, end): remove lines from start to end
- remove_str(substring): remove all occurrences of substring
- keep_all(): do not remove anything

Program:"""

        inputs = self.tokenizer(prompt, return_tensors="pt")
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=False
        )

        program = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return program

    def parse_program(self, program_str: str) -> List[Dict]:
        """
        Parse generated program string into executable operations.

        Converts "remove_lines(0, 5)" into {'type': 'remove_lines', 'args': [0, 5]}
        """
        operations = []

        # Simple regex parsing
        line_pattern = r'remove_lines\((\d+),\s*(\d+)\)'
        for match in re.finditer(line_pattern, program_str):
            start, end = int(match.group(1)), int(match.group(2))
            operations.append({
                'type': 'remove_lines',
                'start': start,
                'end': end
            })

        str_pattern = r"remove_str\('([^']*)'\)"
        for match in re.finditer(str_pattern, program_str):
            text = match.group(1)
            operations.append({
                'type': 'remove_str',
                'text': text
            })

        if 'keep_all' in program_str:
            operations.append({'type': 'keep_all'})

        return operations
```

Implement the refinement executor that applies operations at scale:

```python
class RefinementExecutor:
    """
    Execute refinement programs generated by student model on documents.

    Applies learned operations to scale text cleaning to full corpus.
    """

    def __init__(self, predictor: RefinementOperationPredictor):
        self.predictor = predictor

    def execute_remove_lines(self, text: str, start: int, end: int) -> str:
        """Remove lines from start to end index."""
        lines = text.split('\n')
        if start < 0 or end > len(lines):
            return text

        refined_lines = lines[:start] + lines[end:]
        return '\n'.join(refined_lines)

    def execute_remove_str(self, text: str, substring: str) -> str:
        """Remove all occurrences of substring."""
        return text.replace(substring, '')

    def refine_document(self, text: str) -> str:
        """
        Generate and execute refinement program for document.

        Returns refined text with same meaning but cleaned content.
        """
        # Generate program
        program_str = self.predictor.generate_program(text)

        # Parse operations
        operations = self.predictor.parse_program(program_str)

        # Execute operations
        refined = text
        for op in operations:
            if op['type'] == 'keep_all':
                # No refinement needed
                break
            elif op['type'] == 'remove_lines':
                refined = self.execute_remove_lines(
                    refined, op['start'], op['end']
                )
            elif op['type'] == 'remove_str':
                refined = self.execute_remove_str(refined, op['text'])

        return refined

    def refine_corpus(self, documents: List[str],
                     batch_size: int = 32) -> List[str]:
        """
        Refine large corpus of documents in batches.

        Applies learned refinement operations at scale.
        """
        refined_docs = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            for doc in batch:
                refined = self.refine_document(doc)
                refined_docs.append(refined)

            if (i + batch_size) % 100 == 0:
                print(f"Refined {i + batch_size}/{len(documents)} documents")

        return refined_docs
```

Implement the training pipeline:

```python
class RefineXTrainer:
    """
    Train student model to predict refinement operations.

    Uses expert-generated refinement programs as supervision.
    """

    def __init__(self, predictor: RefinementOperationPredictor,
                 extractor: ProgramExtractor):
        self.predictor = predictor
        self.extractor = extractor
        self.optimizer = torch.optim.Adam(
            predictor.model.parameters(), lr=1e-4
        )

    def train_step(self, original_texts: List[str],
                   refined_texts: List[str]) -> float:
        """
        Train on one batch of original->refined pairs.

        Student learns to generate operations that reproduce expert refinement.
        """
        total_loss = 0.0

        for original, refined in zip(original_texts, refined_texts):
            # Extract program from expert refinement
            program_dict = self.extractor.extract_program(original, refined)
            program_str = self._program_dict_to_str(program_dict)

            # Prepare training data
            prompt = f"Refine this text:\n{original}\n\nProgram:"
            target = f" {program_str}"

            # Tokenize
            inputs = self.predictor.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            )
            targets = self.predictor.tokenizer(
                prompt + target, return_tensors="pt", truncation=True, max_length=512
            )

            # Forward pass
            outputs = self.predictor.model(
                input_ids=targets['input_ids'],
                labels=targets['input_ids']
            )
            loss = outputs.loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(original_texts)

    def _program_dict_to_str(self, program_dict: Dict) -> str:
        """Convert program dict to executable string."""
        ops = program_dict['operations']
        if not ops:
            return "keep_all()"

        parts = []
        for op in ops:
            if op['type'] == 'remove_lines':
                for start, end in op['ranges']:
                    parts.append(f"remove_lines({start}, {end})")
            elif op['type'] == 'remove_substrings':
                for span in op['spans']:
                    text = span['text'].replace("'", "\\'")
                    parts.append(f"remove_str('{text}')")

        return ' '.join(parts) if parts else "keep_all()"
```

## Practical Guidance

**Hyperparameter Table:**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Student model size | 0.6B | 0.3B-2.7B | 0.6B balance quality and speed; 2.7B for top quality |
| Expert LLM | Qwen2.5-72B | - | Larger experts produce better supervision |
| Training epochs | 3 | 1-5 | Usually quick convergence on refinement task |
| Batch size | 32 | 8-128 | Larger batches more stable |
| Learning rate | 1e-4 | 1e-5 to 1e-3 | Conservative; refinement task is stable |
| Max deletion span | 100 chars | 50-500 | Longer spans risk over-deletion |

**When to Use:**
- You have large-scale pretraining data with quality issues (duplicates, noise, irrelevant content)
- You want to improve downstream task performance with clean data
- You can afford expert LLM cost for generating supervision
- You want interpretable refinement (explicit deletions, no hallucinations)
- You need to refine multiple large corpora efficiently

**When NOT to Use:**
- Your data is already high quality; refinement has diminishing returns
- You need data enrichment rather than cleaning (synthesis, rewriting)
- You can't afford expert LLM calls for supervision generation
- You're working with small datasets where per-example cost dominates
- Your domain requires complex edits beyond simple deletion

**Common Pitfalls:**
- **Expert refinement quality**: If expert LLM produces poor refinement, student learns noise. Validate expert outputs before training.
- **Over-deletion**: Aggressive deletion operations can remove useful context. Tune max span length and review operations.
- **Operation specificity**: Substring deletions too specific may not generalize (e.g., removing unique phrases not seen in test data). Prefer line-level operations.
- **Hallucination in student**: If student model hallucinates new operations, validate before applying to corpus. Conservative templates help.
- **Compression ratio variance**: Different documents compress differently. Monitor median compression and flag outliers.
- **Training data leakage**: If refinement training data overlaps corpus, model may learn spurious patterns. Keep separate.

## Reference

Authors (2025). RefineX: Learning to Refine Pre-training Data at Scale from Expert-Guided Programs. arXiv preprint arXiv:2507.03253. https://arxiv.org/abs/2507.03253
