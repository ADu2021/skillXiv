---
name: text-video-alignment-evaluation
title: "ETVA: Evaluation of Text-to-Video Alignment via Fine-grained Question Generation and Answering"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2503.16867"
keywords: [Text-to-Video Generation, Evaluation Metrics, Question Generation, Video Understanding, Benchmark]
description: "Evaluate text-to-video alignment through fine-grained semantic understanding via multi-agent question generation and knowledge-augmented answering. Generate 12,000 atomic yes/no questions from 2,000 prompts across 10 evaluation categories, achieving 58.47 correlation with human judgment."
---

## Core Concept

ETVA addresses a fundamental limitation in text-to-video (T2V) generation evaluation: existing metrics (CLIP score, FVD) fail to capture semantic alignment at fine-grained levels. Instead of crude similarity scores, ETVA simulates human annotation by decomposing natural language prompts into atomic yes/no questions covering semantics (objects, attributes, relationships), spatial-temporal properties (layout, motion), and physics (dynamics, physics simulation). A multi-agent system generates questions systematically, then a knowledge-augmented LLM answers them by reasoning through video content.

## Architecture Overview

The ETVA evaluation framework consists of two integrated systems:

- **Question Generation (Multi-Agent)**: Element Extractor identifies semantic components (entities, attributes, relationships); Graph Builder constructs scene graphs representing dependencies; Graph Traverser systematically explores graphs to generate yes/no questions in dependency order
- **Question Answering (Knowledge-Augmented)**: An auxiliary LLM retrieves relevant commonsense knowledge (physics, spatial reasoning); a video LLM performs three-step analysis (video understanding, reflection with knowledge, conclusive answer)
- **ETVABench Evaluation Dataset**: 2,000 diverse prompts generating 12,000 atomic questions across 10 evaluation categories (existence, action, material, spatial, number, shape, color, camera, physics, other)

## Implementation

### Multi-Agent Question Generation System

The question generation pipeline extracts semantic elements from prompts and systematically generates atomic questions that probe different aspects of video content.

```python
from dataclasses import dataclass
from typing import List, Dict, Set
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

@dataclass
class SemanticElement:
    """Represents a semantic element extracted from prompt."""
    type: str  # "entity", "attribute", "relationship", "action"
    value: str
    dependencies: Set[str] = None

class ElementExtractor:
    """Extract entities, attributes, and relationships from text prompts."""

    def __init__(self, model_name="gpt-4"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def extract_elements(self, prompt: str) -> List[SemanticElement]:
        """Identify all semantic components in prompt."""
        extraction_prompt = f"""
Extract semantic elements from this prompt:
"{prompt}"

Identify:
1. Entities (objects, people, animals)
2. Attributes (colors, sizes, materials, states)
3. Actions (verbs, motions)
4. Relationships (spatial, temporal)

Format each as: [type] [value] [dependencies]
"""
        # Use LLM to extract elements
        elements = []
        extraction_result = self._call_llm(extraction_prompt)

        # Parse extraction results
        for line in extraction_result.split('\n'):
            if line.strip():
                parts = line.split('[')
                if len(parts) >= 3:
                    element_type = parts[1].strip(']').strip()
                    value = parts[2].split(']')[0].strip()
                    element = SemanticElement(
                        type=element_type,
                        value=value,
                        dependencies=set()
                    )
                    elements.append(element)

        return elements

class SceneGraphBuilder:
    """Construct hierarchical scene graph from semantic elements."""

    def __init__(self):
        self.graph = {}

    def build_graph(self, elements: List[SemanticElement]) -> Dict:
        """Create directed acyclic graph of semantic dependencies."""
        # Initialize nodes
        for elem in elements:
            self.graph[elem.value] = {
                'type': elem.type,
                'dependencies': [],
                'dependents': []
            }

        # Identify dependencies between elements
        # (e.g., "red cube" depends on "red" and "cube")
        dependency_prompt = f"""
Given these elements: {[e.value for e in elements]}

Identify dependencies (which elements are prerequisites for others).
Format: [dependent] <- [prerequisite]
"""
        dependency_result = self._call_llm(dependency_prompt)

        # Parse and update graph
        for line in dependency_result.split('\n'):
            if '<-' in line:
                parts = line.split('<-')
                dependent = parts[0].strip()
                prerequisite = parts[1].strip()

                if dependent in self.graph and prerequisite in self.graph:
                    self.graph[dependent]['dependencies'].append(prerequisite)
                    self.graph[prerequisite]['dependents'].append(dependent)

        return self.graph

class GraphTraverser:
    """Systematically traverse scene graph to generate atomic questions."""

    def __init__(self, scene_graph: Dict):
        self.scene_graph = scene_graph
        self.visited = set()

    def generate_questions(self) -> List[str]:
        """Generate yes/no questions following dependency order."""
        questions = []

        # Topological sort to respect dependencies
        sorted_elements = self._topological_sort()

        for element in sorted_elements:
            node = self.scene_graph[element]

            # Generate questions based on element type
            if node['type'] == 'entity':
                questions.append(
                    f"Does the video contain a {element}?"
                )
            elif node['type'] == 'attribute':
                questions.append(
                    f"Is something {element} in the video?"
                )
                # Relationship questions with dependencies
                for dependent in node['dependents']:
                    questions.append(
                        f"Is the {dependent} {element}?"
                    )
            elif node['type'] == 'action':
                questions.append(
                    f"Does something {element} in the video?"
                )
            elif node['type'] == 'spatial':
                questions.append(
                    f"Is there a {element} spatial arrangement?"
                )

        return questions

    def _topological_sort(self) -> List[str]:
        """Sort elements by dependencies using DFS."""
        visited = set()
        sorted_list = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)

            for dep in self.scene_graph[node]['dependencies']:
                dfs(dep)

            sorted_list.append(node)

        for node in self.scene_graph.keys():
            dfs(node)

        return sorted_list

def generate_questions_for_prompt(prompt: str) -> List[str]:
    """Complete pipeline: extract, graph, traverse."""
    # Extract semantic elements
    extractor = ElementExtractor()
    elements = extractor.extract_elements(prompt)

    # Build scene graph
    graph_builder = SceneGraphBuilder()
    scene_graph = graph_builder.build_graph(elements)

    # Traverse and generate questions
    traverser = GraphTraverser(scene_graph)
    questions = traverser.generate_questions()

    return questions
```

### Knowledge-Augmented Question Answering

The QA stage uses an auxiliary LLM to retrieve commonsense knowledge (especially critical for physics questions), then has a video LLM perform multi-step reasoning through the video.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class KnowledgeAugmentedQA:
    """Answer yes/no questions about video with external knowledge."""

    def __init__(
        self,
        knowledge_model_name="Qwen2.5-72B",
        video_model_name="Qwen2-VL-72B"
    ):
        self.knowledge_model = AutoModelForCausalLM.from_pretrained(
            knowledge_model_name
        )
        self.video_model = AutoModelForCausalLM.from_pretrained(
            video_model_name
        )
        self.knowledge_tokenizer = AutoTokenizer.from_pretrained(
            knowledge_model_name
        )
        self.video_tokenizer = AutoTokenizer.from_pretrained(
            video_model_name
        )
        self.knowledge_cache = {}

    def retrieve_knowledge(self, question: str) -> str:
        """Retrieve commonsense knowledge for context."""
        # Check cache first
        if question in self.knowledge_cache:
            return self.knowledge_cache[question]

        # Retrieve knowledge for specific domains
        knowledge_prompt = f"""
Question: {question}

Provide relevant background knowledge or physical principles that help answer this question.
Keep response concise (2-3 sentences).
"""

        knowledge = self._call_knowledge_model(knowledge_prompt)
        self.knowledge_cache[question] = knowledge

        return knowledge

    def answer_question_multistage(
        self, video, question: str
    ) -> Dict[str, any]:
        """Three-stage reasoning: understand, reflect, answer."""

        # Stage 1: Video Understanding
        understanding_prompt = f"""
Watch this video and describe what you see:
[VIDEO_PLACEHOLDER]

Focus on: objects, actions, spatial layout, temporal progression.
Keep description under 100 words.
"""
        video_understanding = self._call_video_model(
            video, understanding_prompt
        )

        # Stage 2: Contextual Reflection with Knowledge
        knowledge = self.retrieve_knowledge(question)

        reflection_prompt = f"""
Video content: {video_understanding}

Background knowledge: {knowledge}

Question: {question}

What aspects of the video are relevant to answering this question?
"""
        reflection = self._call_video_model(video, reflection_prompt)

        # Stage 3: Conclusive Answer
        answer_prompt = f"""
Video: {video_understanding}
Knowledge: {knowledge}
Reflection: {reflection}

Question: {question}

Answer YES or NO. Provide brief justification (1-2 sentences).
"""
        answer_result = self._call_video_model(video, answer_prompt)

        # Parse answer
        answer_lower = answer_result.lower()
        is_yes = "yes" in answer_lower or answer_lower.startswith("yes")

        return {
            'question': question,
            'answer': "YES" if is_yes else "NO",
            'confidence': self._extract_confidence(answer_result),
            'reasoning': answer_result,
            'video_understanding': video_understanding,
            'knowledge': knowledge,
            'reflection': reflection
        }

    def _call_knowledge_model(self, prompt: str) -> str:
        """Query knowledge model."""
        inputs = self.knowledge_tokenizer(prompt, return_tensors="pt")
        outputs = self.knowledge_model.generate(
            **inputs, max_length=150, temperature=0.7
        )
        return self.knowledge_tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

    def _call_video_model(
        self, video: torch.Tensor, prompt: str
    ) -> str:
        """Query video understanding model."""
        # In practice: properly encode video frames
        inputs = self.video_tokenizer(prompt, return_tensors="pt")
        outputs = self.video_model.generate(
            **inputs, max_length=200, temperature=0.7
        )
        return self.video_tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from model response."""
        # Simple heuristic: count certainty markers
        certainty_markers = ["definitely", "clearly", "certainly", "no doubt"]
        uncertainty_markers = ["might", "seems", "appears", "uncertain"]

        certainty_count = sum(
            1 for marker in certainty_markers
            if marker in response.lower()
        )
        uncertainty_count = sum(
            1 for marker in uncertainty_markers
            if marker in response.lower()
        )

        confidence = (certainty_count - uncertainty_count) / max(
            certainty_count + uncertainty_count, 1
        )
        return max(0.0, min(1.0, 0.5 + confidence * 0.5))

def evaluate_video_qa(
    video, generated_video, questions: List[str]
) -> Dict:
    """Evaluate how well generated video answers atomic questions."""
    qa_system = KnowledgeAugmentedQA()

    results = {
        'question_answers': [],
        'accuracy': 0.0,
        'category_performance': {}
    }

    correct_count = 0
    for question in questions:
        # Get answer for both reference and generated video
        ref_answer = qa_system.answer_question_multistage(
            video, question
        )
        gen_answer = qa_system.answer_question_multistage(
            generated_video, question
        )

        # Check alignment
        aligned = (ref_answer['answer'] == gen_answer['answer'])
        correct_count += int(aligned)

        results['question_answers'].append({
            'question': question,
            'reference_answer': ref_answer['answer'],
            'generated_answer': gen_answer['answer'],
            'aligned': aligned
        })

    results['accuracy'] = correct_count / len(questions)

    return results
```

### ETVABench Benchmark Construction and Evaluation

Build a comprehensive benchmark with diverse prompts covering 10 evaluation categories, then evaluate T2V models systematically.

```python
from enum import Enum

class EvaluationCategory(Enum):
    """10 evaluation categories for text-to-video assessment."""
    EXISTENCE = "existence"  # Does object exist?
    ACTION = "action"  # Is action performed?
    MATERIAL = "material"  # Object material properties?
    SPATIAL = "spatial"  # Spatial arrangement?
    NUMBER = "number"  # Quantity of objects?
    SHAPE = "shape"  # Shape properties?
    COLOR = "color"  # Color attributes?
    CAMERA = "camera"  # Camera control/movement?
    PHYSICS = "physics"  # Physics simulation?
    OTHER = "other"  # Other semantic properties

class ETVABench:
    """Text-to-video alignment benchmark with 2000 prompts."""

    def __init__(self):
        self.prompts = []
        self.questions_by_prompt = {}
        self.category_distribution = {}

    def construct_benchmark(self, num_prompts: int = 2000):
        """Build benchmark with diverse prompts across categories."""
        # Load or generate diverse prompts
        prompts_per_category = num_prompts // len(EvaluationCategory)

        for category in EvaluationCategory:
            category_prompts = self._generate_prompts_for_category(
                category, prompts_per_category
            )
            self.prompts.extend(category_prompts)

            # Generate questions for each prompt
            for prompt in category_prompts:
                questions = generate_questions_for_prompt(prompt)
                self.questions_by_prompt[prompt] = questions

            self.category_distribution[category.value] = len(
                category_prompts
            )

    def _generate_prompts_for_category(
        self, category: EvaluationCategory, count: int
    ) -> List[str]:
        """Generate diverse prompts for evaluation category."""
        prompts = []

        if category == EvaluationCategory.EXISTENCE:
            templates = [
                "A {object} in a {setting}",
                "{object} doing {action}",
                "{object} with {attribute} {property}",
            ]
        elif category == EvaluationCategory.ACTION:
            templates = [
                "{object} {action} in {setting}",
                "{object} slowly {action}",
                "{object} quickly {action}",
            ]
        elif category == EvaluationCategory.PHYSICS:
            templates = [
                "{object} falling under gravity in {setting}",
                "{object} floating in {environment}",
                "{object} bouncing on {surface}",
            ]

        # Expand templates with variations
        for template in templates[:count]:
            prompt = template.replace(
                "{object}", "cat"
            ).replace(
                "{action}", "running"
            ).replace(
                "{setting}", "a park"
            )
            prompts.append(prompt)

        return prompts

def evaluate_t2v_models(
    t2v_models: Dict[str, any],
    benchmark: ETVABench
) -> Dict[str, Dict]:
    """Evaluate multiple T2V models on ETVABench."""
    evaluation_results = {}

    for model_name, model in t2v_models.items():
        print(f"\nEvaluating {model_name}...")
        model_results = {
            'overall_accuracy': 0.0,
            'category_accuracy': {},
            'temporal_accuracy': 0.0,
            'physics_accuracy': 0.0
        }

        category_accuracies = {}
        all_accuracies = []

        for prompt in benchmark.prompts:
            # Generate video
            generated_video = model.generate(prompt)

            # Get questions for this prompt
            questions = benchmark.questions_by_prompt[prompt]

            # Evaluate alignment
            qa_results = evaluate_video_qa(
                None, generated_video, questions
            )

            all_accuracies.append(qa_results['accuracy'])

            # Track by category
            category = categorize_prompt(prompt)
            if category not in category_accuracies:
                category_accuracies[category] = []
            category_accuracies[category].append(qa_results['accuracy'])

        # Aggregate results
        model_results['overall_accuracy'] = (
            sum(all_accuracies) / len(all_accuracies)
        )

        for category, accuracies in category_accuracies.items():
            model_results['category_accuracy'][category] = (
                sum(accuracies) / len(accuracies)
            )

        evaluation_results[model_name] = model_results

    return evaluation_results

def categorize_prompt(prompt: str) -> str:
    """Categorize prompt into evaluation category."""
    prompt_lower = prompt.lower()

    if any(word in prompt_lower for word in ["exist", "contain", "has"]):
        return "existence"
    elif any(word in prompt_lower for word in ["falling", "bouncing", "gravity", "float"]):
        return "physics"
    elif any(word in prompt_lower for word in ["camera", "pan", "zoom", "move"]):
        return "camera"
    else:
        return "other"
```

## Practical Guidance

**When to use ETVA:**
- You're developing or evaluating T2V models and need fine-grained semantic alignment assessment
- You want to identify specific weaknesses (e.g., physics simulation, camera control) in generated videos
- You need evaluation that correlates well with human judgment (Spearman's ρ = 58.47 vs. VideoScore's 31.0)
- You're building a benchmark for reproducible T2V evaluation across multiple models

**When NOT to use:**
- You need real-time evaluation (multi-agent QA + video understanding is computationally expensive)
- Your videos are extremely short (< 2 seconds) where atomic question answering is unreliable
- You need to evaluate non-semantic aspects (technical quality, compression artifacts)
- Budget is extremely limited (requires multiple LLM API calls per video)

**Hyperparameter and design choices:**
- **Number of atomic questions**: 6 questions per prompt typical; increase to 10 for complex scenes, reduce to 3 for simple objects
- **Knowledge model**: Qwen2.5-72B recommended for physics/reasoning; GPT-4 acceptable if budget allows
- **Video LLM**: Qwen2-VL-72B for open-source; GPT-4V or Gemini for closed-source (may improve accuracy)
- **Question categories**: 10 core categories defined; customize based on evaluation priorities
- **Multi-stage reasoning steps**: 3 stages (understand, reflect, answer) optimal; can reduce to 2 for speed

**Common pitfalls:**
- **Insufficient knowledge augmentation**: Skipping the knowledge retrieval stage causes physics questions to fail; always retrieve domain-specific knowledge
- **Weak question generation**: Vanilla LLM prompting generates redundant questions; use multi-agent graph traversal for systematic coverage
- **Evaluation category imbalance**: If physics represents only 10% of prompts, physics limitations aren't well-detected; balance categories for comprehensive assessment
- **Video understanding failures**: If video LLM struggles to understand generated video content, all downstream QA fails; validate video LLM separately
- **Yes/no answer ambiguity**: Models may give nuanced answers; parse answers strictly for binary yes/no to avoid ambiguity

## Reference

- **Architecture**: Multi-agent system (Element Extractor, Graph Builder, Graph Traverser) + Knowledge-augmented QA (Qwen2.5-72B, Qwen2-VL-72B)
- **Benchmark**: ETVABench with 2,000 prompts generating 12,000 atomic questions across 10 categories
- **Evaluation metrics**: Spearman's ρ correlation with human judgment (58.47 overall), ablation studies showing 34.1% improvement from multi-agent QA, 21.93% from knowledge augmentation, 6.5-11.2% from multi-stage reasoning
- **Key findings**: All 15 evaluated T2V models struggle with temporal dynamics (physics max 0.600, camera max 0.474); static attribute generation is stronger than spatiotemporal reasoning
- **Evaluation dataset**: Manually deconstructed CoT steps (FlowVerse-CoT-E) more robust than automatic step extraction
- **Related work**: CLIP-based metrics, FVD (Fréchet Video Distance), VideoScore baseline evaluation methods
