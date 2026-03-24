---
name: llm-finetuning-dataset-synthesis
title: "Easy Dataset: A Unified and Extensible Framework for Synthesizing LLM Fine-Tuning Data"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2507.04009"
keywords: [Dataset Synthesis, Document Processing, QA Generation, Fine-tuning Data, Domain Adaptation]
description: "Automatically generate high-quality domain-specific fine-tuning datasets from raw documents using adaptive processing and persona-driven synthesis, preserving general capabilities while specializing models."
---

# Easy Dataset: Automated Fine-Tuning Data Synthesis from Documents

Building high-quality fine-tuning datasets from domain-specific documents typically requires significant manual effort. This work introduces a unified framework that automates the entire pipeline: processing heterogeneous document formats (PDF, DOCX, Markdown, text) with format-aware extraction, then synthesizing diverse question-answer pairs through a persona-driven approach. The system creates stylistically varied training examples that improve domain performance while maintaining general language understanding.

The key insight is that diversity in fine-tuning data comes from both content variation and stylistic variation. Rather than generating a single Q&A pair per passage, the framework creates multiple personas (different audiences and genres) that condition answer generation, producing diverse acceptable answers to the same question. This diversity improves generalization and reduces overfitting to specific formulations.

## Core Concept

Easy Dataset operates on two principles: (1) adaptive document processing that respects format structure while handling heterogeneity robustly, and (2) persona-driven synthesis that generates diverse training examples from single passages. Instead of manual annotation, domain experts specify documents, and the system extracts information reliably (handling complex PDFs with vision-language models when needed) and synthesizes high-quality training examples.

The persona-driven approach combats a subtle problem with synthetic data: if all examples follow identical patterns, fine-tuned models may overfit to those patterns. By varying the audience (technical vs lay, formal vs casual) and genre (instruction manual, FAQ, tutorial), the framework ensures models learn robust generalizations rather than superficial correlations.

## Architecture Overview

The system comprises two main components:

- **Adaptive Document Processing Pipeline**: Handles multi-format inputs (PDF, DOCX, Markdown, plain text) with specialized extractors, leveraging vision-language models for complex PDFs, followed by hybrid chunking that balances semantic structure with flexible user control
- **Persona-Driven Data Synthesis**: Automatically generates diverse persona pairs, conditions question and answer generation on personas, and produces semantically grounded yet stylistically varied training pairs

Supporting infrastructure includes a graphical configuration interface, human-in-the-loop refinement for intermediate outputs, flexible model integration (API and local), and export to multiple formats including LlamaFactory integration.

## Implementation

Start with document extraction from heterogeneous formats:

```python
import json
from typing import List, Dict, Literal
from pathlib import Path
import PyPDF2
from docx import Document

class DocumentExtractor:
    """
    Extract text from multiple document formats with format-specific handling.

    Supports PDF (with vision-language fallback for complex layouts),
    DOCX, Markdown, and plain text, producing normalized content.
    """

    def __init__(self, use_vision_model_for_pdf: bool = True):
        self.use_vision_model = use_vision_model_for_pdf
        if self.use_vision_model:
            # Initialize VLM for complex PDFs
            from transformers import AutoProcessor, AutoModelForVision2Seq
            self.vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL")
            self.vlm_model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen-VL")

    def extract_pdf(self, pdf_path: str, use_vision: bool = False) -> str:
        """
        Extract text from PDF with optional vision-language fallback.

        For standard PDFs, uses text layer. For complex layouts (scanned,
        image-heavy), falls back to VLM-based extraction if enabled.
        """
        text_content = []

        try:
            # Try standard text extraction
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(f"--- Page {page_num + 1} ---\n{text}")
                    elif use_vision and self.use_vision_model:
                        # Fallback to vision model for this page
                        from pdf2image import convert_from_path
                        images = convert_from_path(pdf_path, first_page=page_num+1,
                                                   last_page=page_num+1)
                        if images:
                            vlm_text = self._extract_via_vlm(images[0])
                            text_content.append(f"--- Page {page_num + 1} (VLM) ---\n{vlm_text}")

        except Exception as e:
            print(f"Error extracting PDF: {e}")
            if use_vision and self.use_vision_model:
                return self._extract_pdf_via_vision(pdf_path)

        return "\n".join(text_content)

    def extract_docx(self, docx_path: str) -> str:
        """Extract text from DOCX preserving paragraph structure."""
        doc = Document(docx_path)
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        return "\n".join(paragraphs)

    def extract_text(self, file_path: str) -> str:
        """Automatically detect format and extract."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == '.pdf':
            return self.extract_pdf(file_path, use_vision=self.use_vision_model)
        elif suffix == '.docx':
            return self.extract_docx(file_path)
        elif suffix in ['.md', '.txt']:
            return path.read_text(encoding='utf-8')
        else:
            raise ValueError(f"Unsupported format: {suffix}")

    def _extract_via_vlm(self, image):
        """Use vision-language model to extract text from image."""
        # Placeholder for VLM extraction
        return "VLM-extracted content"
```

Implement adaptive chunking that respects document structure:

```python
from typing import Tuple

class HybridChunker:
    """
    Split documents into chunks respecting semantic boundaries.

    Attempts to chunk by paragraphs/sections, falling back to token-based
    splitting if semantic chunks don't fit size constraints.
    """

    def __init__(self, target_chunk_size: int = 512,
                 overlap_size: int = 128,
                 tokenizer_name: str = "gpt2"):
        self.target_size = target_chunk_size
        self.overlap_size = overlap_size
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))

    def chunk_document(self, text: str) -> List[str]:
        """
        Chunk document with semantic and size awareness.

        Tries to split by paragraphs/sections, then within paragraphs
        by sentences, ensuring chunks fit target size.
        """
        chunks = []
        paragraphs = text.split('\n\n')

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            if para_tokens <= self.target_size:
                # Paragraph fits; add as-is
                if chunks and (self.count_tokens(chunks[-1]) +
                              para_tokens < self.target_size):
                    # Append to previous chunk if there's room
                    chunks[-1] += "\n\n" + para
                else:
                    chunks.append(para)
            else:
                # Paragraph too large; split by sentences
                sentences = para.replace('! ', '!|').replace('? ', '?|').split('|')
                current_chunk = ""

                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence)
                    current_tokens = self.count_tokens(current_chunk)

                    if current_tokens + sentence_tokens <= self.target_size:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence

                if current_chunk:
                    chunks.append(current_chunk.strip())

        return [c for c in chunks if c.strip()]
```

Implement persona-driven synthesis:

```python
from dataclasses import dataclass

@dataclass
class Persona:
    """Represents audience and genre for answer variation."""
    genre: str  # "tutorial", "faq", "manual", "blog"
    audience: str  # "beginner", "expert", "general"
    formality: str  # "formal", "casual"

class PersonaDrivenSynthesizer:
    """
    Generate diverse QA pairs by conditioning on personas.

    Creates multiple personas from document domain, then generates
    questions and answers appropriate to each persona.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-2-70b-chat"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_personas(self, domain: str, num_personas: int = 4) -> List[Persona]:
        """
        Auto-generate diverse personas for a domain.

        Samples from genre and audience combinations appropriate to domain.
        """
        genres = ["tutorial", "faq", "manual", "case_study"]
        audiences = ["beginner", "intermediate", "expert"]
        formalities = ["formal", "casual"]

        personas = []
        for i in range(num_personas):
            persona = Persona(
                genre=genres[i % len(genres)],
                audience=audiences[i % len(audiences)],
                formality=formalities[i % len(formalities)]
            )
            personas.append(persona)
        return personas

    def generate_qa_pair(self, passage: str, persona: Persona) -> Tuple[str, str]:
        """
        Generate question and answer conditioned on persona.

        Creates a question appropriate to the persona's audience level,
        then an answer matching the persona's genre and formality.
        """
        question_prompt = f"""Based on this passage, generate a {persona.audience}-level question suitable for a {persona.genre}.
Passage: {passage}
Question:"""

        answer_prompt = f"""Answer the following question in a {persona.formality} {persona.genre} style for a {persona.audience} audience.
Question: {{question}}
Passage: {passage}
Answer:"""

        # Generate question
        inputs = self.tokenizer(question_prompt, return_tensors="pt")
        question_ids = self.model.generate(**inputs, max_length=100, temperature=0.7)
        question = self.tokenizer.decode(question_ids[0], skip_special_tokens=True)

        # Generate answer conditioned on question
        answer_input = answer_prompt.format(question=question)
        inputs = self.tokenizer(answer_input, return_tensors="pt")
        answer_ids = self.model.generate(**inputs, max_length=300, temperature=0.7)
        answer = self.tokenizer.decode(answer_ids[0], skip_special_tokens=True)

        return question, answer

    def synthesize_dataset(self, chunks: List[str], num_qa_per_chunk: int = 3) -> List[Dict]:
        """
        Generate complete fine-tuning dataset from document chunks.

        Produces multiple diverse QA pairs per chunk by varying personas,
        creating stylistic variation while maintaining semantic grounding.
        """
        dataset = []
        personas = self.generate_personas(domain="general", num_personas=num_qa_per_chunk)

        for chunk_id, chunk in enumerate(chunks):
            for persona_idx, persona in enumerate(personas):
                try:
                    question, answer = self.generate_qa_pair(chunk, persona)
                    dataset.append({
                        'id': f"{chunk_id}_{persona_idx}",
                        'instruction': question,
                        'input': '',
                        'output': answer,
                        'chunk_id': chunk_id,
                        'persona': {
                            'genre': persona.genre,
                            'audience': persona.audience,
                            'formality': persona.formality
                        }
                    })
                except Exception as e:
                    print(f"Error generating QA pair: {e}")
                    continue

        return dataset
```

Integrate the pipeline:

```python
class EasyDatasetFramework:
    """
    End-to-end framework for LLM fine-tuning dataset synthesis.

    Orchestrates document extraction, chunking, persona generation,
    and QA synthesis into a unified pipeline.
    """

    def __init__(self, output_format: Literal["json", "jsonl", "csv"] = "jsonl"):
        self.extractor = DocumentExtractor(use_vision_model_for_pdf=True)
        self.chunker = HybridChunker(target_chunk_size=512)
        self.synthesizer = PersonaDrivenSynthesizer()
        self.output_format = output_format

    def process_documents(self, file_paths: List[str],
                         output_path: str) -> Dict:
        """
        Process multiple documents and synthesize fine-tuning dataset.

        Returns metadata about the generated dataset and saves to disk.
        """
        all_datasets = []

        for file_path in file_paths:
            print(f"Processing {file_path}...")

            # Extract text
            text = self.extractor.extract_text(file_path)

            # Chunk document
            chunks = self.chunker.chunk_document(text)
            print(f"  Generated {len(chunks)} chunks")

            # Synthesize QA pairs
            qa_dataset = self.synthesizer.synthesize_dataset(chunks)
            print(f"  Generated {len(qa_dataset)} QA pairs")

            all_datasets.extend(qa_dataset)

        # Export dataset
        self._export(all_datasets, output_path)

        return {
            'total_examples': len(all_datasets),
            'output_path': output_path,
            'format': self.output_format
        }

    def _export(self, dataset: List[Dict], output_path: str):
        """Export dataset to specified format."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if self.output_format == "jsonl":
            with open(output_path, 'w') as f:
                for item in dataset:
                    f.write(json.dumps(item) + '\n')
        elif self.output_format == "json":
            with open(output_path, 'w') as f:
                json.dump(dataset, f, indent=2)
        elif self.output_format == "csv":
            import pandas as pd
            df = pd.DataFrame(dataset)
            df.to_csv(output_path, index=False)
```

## Practical Guidance

**Hyperparameter Table:**

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Chunk size (tokens) | 512 | 256-1024 | Larger = more context; smaller = more specific |
| Overlap size | 128 | 0-256 | Prevents information loss at chunk boundaries |
| Num personas | 3 | 1-8 | More personas = higher diversity; diminishing returns ~5 |
| QA pairs per chunk | 3 | 1-5 | Trade-off between dataset size and diversity |
| MLM probability (if used) | 0.15 | 0.1-0.3 | For domain-specific pretraining within synthesis |

**When to Use:**
- You have domain-specific documents (financial reports, technical manuals, legal texts) and want to fine-tune models
- You need to create training data without manual annotation
- You want to preserve general capabilities while specializing for a domain
- You have heterogeneous document formats (PDFs, Word docs, markdown)
- You need human-in-the-loop refinement before using synthetic data

**When NOT to Use:**
- Your documents are already well-labeled with high-quality Q&A
- You need very high-precision responses for safety-critical domains
- You have very small documents where chunking loses critical context
- Your domain requires specialized jargon not in model training (may hallucinate)
- You need sub-linear scaling—generating synthetic data has overhead

**Common Pitfalls:**
- **Low-quality document extraction**: Poor PDF extraction ruins downstream synthesis. Always validate extraction quality before generating QA.
- **Persona mismatch**: Personas mismatched to actual use cases produce unusable answers. Validate persona choices on sample documents.
- **Loss of structure in chunking**: Large chunks lose semantic structure; small chunks lose context. Tune chunk size for your domain.
- **Synthetic data overfitting**: If all examples are too similar (weak persona variation), fine-tuned models overfit. Enforce persona diversity.
- **No validation**: Synthetic data quality varies. Always manually review sample QA pairs before using the full dataset.
- **Forgetting to deduplicate**: Persona variation may create duplicate questions. Dedup before fine-tuning.

## Reference

Authors (2025). Easy Dataset: A Unified and Extensible Framework for Synthesizing LLM Fine-Tuning Data. arXiv preprint arXiv:2507.04009. https://arxiv.org/abs/2507.04009
