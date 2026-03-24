---
name: essential-web-taxonomy
title: "Essential-Web v1.0: 24T tokens of organized web data"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2506.14111"
keywords: [dataset, web-data, taxonomy, annotation, pretraining]
description: "24-trillion-token web dataset with 12-category taxonomy labels enabling efficient curation of specialized datasets through SQL filtering and small annotation models."
---

# Essential-Web v1.0: 24T tokens of organized web data

## Core Concept

Essential-Web v1.0 is a large-scale, taxonomy-annotated web corpus of 24 trillion tokens where each document receives categorical labels across twelve dimensions (topic, format, complexity, quality). A lightweight 0.5B parameter annotation model (EAI-Distill-0.5b) achieves performance within 3% of 32B models. Simple SQL filtering on taxonomy labels yields competitive domain-specific datasets, demonstrating practical utility for efficient data curation at scale.

## Architecture Overview

- **12-Category Taxonomy**: Covers topic (STEM, code, news, etc.), format (article, code, list, etc.), complexity (beginner, advanced), quality (high, medium, low)
- **Efficient Annotation Model**: EAI-Distill-0.5b trained for classification, matching large-model performance at 0.5B scale
- **Document-Level Labels**: Every document in 24T-token corpus receives categorical assignments
- **SQL-Based Curation**: Query taxonomy labels to construct specialized datasets for specific domains
- **Quality Metrics**: Annotator agreement validated against large reference models

## Implementation

### Step 1: Define Taxonomy Schema

Design the 12-category taxonomy for document classification:

```python
import json

class WebTaxonomy:
    """
    Defines taxonomy for document annotation across multiple dimensions.
    """
    TAXONOMY = {
        'topic': {
            'STEM': 'Science, technology, engineering, mathematics content',
            'CODE': 'Programming and software development',
            'NEWS': 'News and current events',
            'BUSINESS': 'Business and economics',
            'CREATIVE': 'Creative writing and entertainment',
            'SOCIAL': 'Social and humanities content',
            'OTHER': 'Miscellaneous topics'
        },
        'format': {
            'ARTICLE': 'News or blog article format',
            'CODE': 'Source code or code snippet',
            'LIST': 'Lists, tables, or structured data',
            'DIALOGUE': 'Conversational or question-answer format',
            'REFERENCE': 'Reference material or documentation',
            'CREATIVE': 'Story, poem, or creative text',
            'OTHER': 'Other formats'
        },
        'complexity': {
            'BEGINNER': 'Introductory or basic content',
            'INTERMEDIATE': 'Moderate difficulty level',
            'ADVANCED': 'Expert or specialized content'
        },
        'quality': {
            'HIGH': 'High-quality, well-written content',
            'MEDIUM': 'Reasonable quality content',
            'LOW': 'Poor quality or spam-like content'
        }
    }

    @classmethod
    def to_prompt(cls):
        """Generate prompt for annotation model"""
        prompt = "Classify the document across these dimensions:\n\n"

        for dimension, categories in cls.TAXONOMY.items():
            prompt += f"{dimension.upper()}:\n"
            for cat, desc in categories.items():
                prompt += f"  - {cat}: {desc}\n"

        return prompt

    @classmethod
    def schema_sql(cls):
        """Generate SQL schema for storing labels"""
        return """
        CREATE TABLE documents (
            doc_id VARCHAR PRIMARY KEY,
            content TEXT,
            topic VARCHAR,
            format VARCHAR,
            complexity VARCHAR,
            quality VARCHAR,
            timestamp DATETIME
        );
        """
```

### Step 2: Implement Efficient Annotation Model

Build lightweight classifier matching large-model performance:

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class AnnotationModel(nn.Module):
    """
    Efficient 0.5B parameter annotation model.
    Smaller than 32B reference but matches performance within 3%.
    """
    def __init__(self, base_model='bert-base', hidden_dim=512):
        super().__init__()

        # Use lightweight base model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.encoder = AutoModel.from_pretrained(base_model)

        # Freeze encoder weights for efficiency
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Classification heads
        self.topic_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 7)  # 7 topic categories
        )

        self.format_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 7)  # 7 format categories
        )

        self.complexity_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # 3 complexity levels
        )

        self.quality_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # 3 quality levels
        )

    def forward(self, documents):
        """
        Args:
            documents: list of document strings

        Returns:
            predictions: dict of category predictions
        """
        # Tokenize
        inputs = self.tokenizer(
            documents,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )

        # Encode
        with torch.no_grad():
            outputs = self.encoder(**inputs)

        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token

        # Classify across dimensions
        topic_logits = self.topic_head(pooled)
        format_logits = self.format_head(pooled)
        complexity_logits = self.complexity_head(pooled)
        quality_logits = self.quality_head(pooled)

        return {
            'topic': torch.argmax(topic_logits, dim=1),
            'format': torch.argmax(format_logits, dim=1),
            'complexity': torch.argmax(complexity_logits, dim=1),
            'quality': torch.argmax(quality_logits, dim=1)
        }
```

### Step 3: Annotate Documents at Scale

Process corpus and assign taxonomy labels:

```python
def annotate_corpus(documents, annotation_model, batch_size=32):
    """
    Annotate large document corpus with taxonomy labels.

    Args:
        documents: list of document strings
        annotation_model: AnnotationModel instance
        batch_size: batch size for inference

    Returns:
        annotations: list of label dictionaries
    """
    annotations = []
    num_batches = (len(documents) + batch_size - 1) // batch_size

    annotation_model.eval()

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(documents))

        batch_docs = documents[start_idx:end_idx]

        # Get predictions
        predictions = annotation_model(batch_docs)

        # Convert to category names
        taxonomy = WebTaxonomy.TAXONOMY

        for j, doc_idx in enumerate(range(start_idx, end_idx)):
            topic_idx = predictions['topic'][j].item()
            format_idx = predictions['format'][j].item()
            complexity_idx = predictions['complexity'][j].item()
            quality_idx = predictions['quality'][j].item()

            topic_cats = list(taxonomy['topic'].keys())
            format_cats = list(taxonomy['format'].keys())
            complexity_cats = list(taxonomy['complexity'].keys())
            quality_cats = list(taxonomy['quality'].keys())

            annotation = {
                'doc_id': doc_idx,
                'topic': topic_cats[topic_idx],
                'format': format_cats[format_idx],
                'complexity': complexity_cats[complexity_idx],
                'quality': quality_cats[quality_idx]
            }

            annotations.append(annotation)

    return annotations
```

### Step 4: SQL-Based Dataset Curation

Query taxonomy labels to curate specialized datasets:

```python
import sqlite3

class DatasetCurator:
    """
    Curates domain-specific datasets using SQL queries on taxonomy.
    """
    def __init__(self, db_path='essential_web.db'):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def curate_code_dataset(self, quality_threshold='HIGH'):
        """
        Create code-focused dataset.
        """
        query = f"""
        SELECT doc_id, content FROM documents
        WHERE format = 'CODE' AND quality = '{quality_threshold}'
        ORDER BY doc_id;
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def curate_stem_dataset(self, min_complexity='INTERMEDIATE',
                           quality_threshold='HIGH'):
        """
        Create STEM-focused dataset with complexity filter.
        """
        complexity_order = {'BEGINNER': 0, 'INTERMEDIATE': 1, 'ADVANCED': 2}
        min_level = complexity_order[min_complexity]

        query = f"""
        SELECT doc_id, content FROM documents
        WHERE topic = 'STEM'
        AND complexity IN ('INTERMEDIATE', 'ADVANCED')
        AND quality = '{quality_threshold}'
        ORDER BY doc_id;
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_stats(self):
        """Get dataset statistics by category"""
        stats = {}

        for dimension in ['topic', 'format', 'complexity', 'quality']:
            query = f"""
            SELECT {dimension}, COUNT(*) as count
            FROM documents GROUP BY {dimension};
            """
            self.cursor.execute(query)
            stats[dimension] = dict(self.cursor.fetchall())

        return stats

    def close(self):
        self.conn.close()
```

### Step 5: Validate Annotation Quality

Compare against reference models for agreement:

```python
def validate_annotations(documents, annotations, reference_model):
    """
    Validate annotation quality against larger reference model.

    Args:
        documents: list of document strings
        annotations: list of annotation dicts from EAI-Distill-0.5b
        reference_model: larger model for validation

    Returns:
        agreement_scores: dict of agreement rates per dimension
    """
    reference_predictions = reference_model(documents)

    taxonomy = WebTaxonomy.TAXONOMY
    dimensions = ['topic', 'format', 'complexity', 'quality']

    agreement_scores = {}

    for dim in dimensions:
        matches = 0
        total = len(documents)

        cats = list(taxonomy[dim].keys())

        for i in range(total):
            annotation_cat = annotations[i][dim]
            reference_idx = reference_predictions[dim][i].item()
            reference_cat = cats[reference_idx]

            if annotation_cat == reference_cat:
                matches += 1

        agreement = matches / total
        agreement_scores[dim] = agreement

    # Report overall agreement
    avg_agreement = sum(agreement_scores.values()) / len(agreement_scores)
    print(f"Average agreement with reference: {avg_agreement:.1%}")

    return agreement_scores
```

## Practical Guidance

- **Taxonomy Design**: Iteratively refine categories based on corpus analysis; ensure coverage of major content types
- **Model Selection**: Start with BERT-base or DistilBERT as encoder base; experiment with quantization for further efficiency
- **Batch Processing**: Process in batches of 32-64 documents for GPU efficiency; stream results to database
- **Quality Validation**: Sample 5-10% of corpus for human agreement validation; target >97% agreement with reference
- **Storage**: Use SQLite for smaller corpora, PostgreSQL/BigQuery for 24T-token scale
- **Filtering Strategies**: Start with high-quality documents (quality='HIGH'); gradually expand as needed
- **Downstream Use**: Share curated datasets on HuggingFace; enable reproducible dataset construction

## Reference

Paper: arXiv:2506.14111
Key metrics: EAI-Distill-0.5b achieves 3% difference from Qwen2.5-32B; web code +14.3%, STEM +24.5% improvements
Dataset: 24T tokens with 12-category taxonomy labels
Related work: Data curation, web scraping, dataset annotation, knowledge distillation
