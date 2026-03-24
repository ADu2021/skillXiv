---
name: thinking-with-video-multimodal-reasoning
title: "Thinking with Video: Video Generation as Multimodal Reasoning"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.04570"
keywords: [Video Generation, Multimodal Reasoning, Thinking Mechanisms, Problem Solving, Dynamic Representation]
description: "Leverage video generation models as unified multimodal reasoning engines that bridge text and vision by embedding reasoning as dynamic visual processes, enabling models to tackle both spatial puzzles and mathematical problems within a single temporal framework."
---

# Title: Use Video Generation as a Universal Reasoning Modality

Text-centric reasoning (thinking with text) excels at logic but struggles with spatial problems. Image-centric reasoning (thinking with images) handles space but lacks temporal dynamics. Video reasoning bridges both: generate videos showing problem-solving steps, with text embedded in frames. This unifies multimodal understanding within a single continuous temporal process.

The key insight is that video naturally represents processes—drawing, moving, transforming—that neither text nor static images capture well.

## Core Concept

**Video as Unified Reasoning Substrate**:
- **Dynamic Representation**: Video captures continuous transformations over time
- **Text Embedding**: Embed reasoning text within video frames (e.g., solving equations step-by-step)
- **Spatial Reasoning**: Show spatial manipulations explicitly (moving, rotating, drawing)
- **Temporal Reasoning**: Natural representation of sequences and processes
- **Single Model**: One video generation model handles diverse reasoning tasks

## Architecture Overview

- **Video Generation Model**: Pre-trained (Sora-2 or similar) for high-fidelity generation
- **Task-Specific Prompting**: Frame problems as video generation tasks
- **Text-in-Video Encoding**: Embed reasoning steps as video content
- **Evaluation**: Both spatial (puzzle accuracy) and mathematical (answer correctness)

## Implementation Steps

**1. Formulate Problem as Video Generation Task**

Convert different problem types into video generation prompts.

```python
class VideoReasoningPromptFormatting:
    @staticmethod
    def spatial_puzzle_to_video(puzzle_description):
        """Convert spatial puzzle to video generation prompt"""
        prompt = f"""Generate a video showing the solution to this puzzle:
        {puzzle_description}

        The video should:
        1. Start with the puzzle state
        2. Show step-by-step manipulations
        3. End with the solution
        4. Include visual indicators (arrows, highlights) showing the transformations
        """
        return prompt

    @staticmethod
    def math_problem_to_video(problem_text):
        """Convert math problem to video generation prompt"""
        prompt = f"""Generate a video showing the step-by-step solution to this math problem:
        {problem_text}

        The video should:
        1. Display the problem at the beginning
        2. Show each reasoning step on screen
        3. Include handwritten/typeset equations
        4. Highlight key transitions between steps
        5. Display the final answer prominently
        """
        return prompt

    @staticmethod
    def eyeballing_puzzle_to_video(puzzle_image):
        """Convert visual puzzle to video generation prompt"""
        prompt = f"""Generate a video showing how to solve this eyeballing puzzle:
        [Image shown]

        The video should:
        1. Show the original puzzle
        2. Demonstrate measurements or comparisons
        3. Show the discovered relationship
        4. End with the answer
        """
        return prompt
```

**2. Generate Videos and Extract Answers**

Use video generation model as reasoning engine.

```python
class VideoReasoningEngine:
    def __init__(self, video_model, answer_extractor):
        self.video_model = video_model  # Sora-2 or similar
        self.answer_extractor = answer_extractor

    def solve_spatial_problem(self, puzzle_description):
        # Convert problem to video prompt
        prompt = VideoReasoningPromptFormatting.spatial_puzzle_to_video(puzzle_description)

        # Generate video showing solution
        video = self.video_model.generate(
            prompt=prompt,
            resolution="1280x720",
            num_frames=24,  # ~1 second at 24fps
            duration=1.0
        )

        # Extract answer from final frame
        final_frame = video.frames[-1]
        answer = self.answer_extractor.extract_from_image(final_frame)

        return answer, video

    def solve_math_problem(self, problem_text):
        # Convert problem to video prompt
        prompt = VideoReasoningPromptFormatting.math_problem_to_video(problem_text)

        # Generate video showing step-by-step solution
        video = self.video_model.generate(
            prompt=prompt,
            resolution="1280x720",
            num_frames=120,  # ~5 seconds for more complex problems
            duration=5.0
        )

        # Extract reasoning and answer from video frames
        # Parse text that appears in frames
        reasoning_steps = self.extract_reasoning_from_frames(video.frames)
        final_answer = reasoning_steps[-1]

        return final_answer, video, reasoning_steps

    def extract_reasoning_from_frames(self, frames):
        # Use OCR to extract text from frames
        steps = []
        for frame in frames:
            text = pytesseract.image_to_string(frame)
            if text.strip():
                steps.append(text)
        return steps
```

**3. Implement Answer Extraction**

Parse answers from generated video frames.

```python
class VideoAnswerExtractor:
    def __init__(self):
        self.ocr_engine = pytesseract  # Or cloud-based OCR
        self.object_detector = ObjectDetector()  # For spatial answers

    def extract_from_image(self, frame):
        # Try multiple extraction strategies
        # Strategy 1: OCR
        text = self.ocr_engine.image_to_string(frame)
        if text and any(c.isdigit() for c in text):
            return text

        # Strategy 2: Object detection for spatial problems
        objects = self.object_detector.detect(frame)
        if objects:
            # Extract spatial relationship
            return self.describe_spatial_relationship(objects)

        # Strategy 3: Color/shape analysis
        return self.analyze_visual_properties(frame)

    def extract_answer_from_video(self, video, problem_type):
        """Extract answer from entire video sequence"""
        if problem_type == 'math':
            # Parse all text frames for equations
            frames_with_text = []
            for frame in video.frames:
                text = self.ocr_engine.image_to_string(frame)
                if text:
                    frames_with_text.append(text)
            final_line = frames_with_text[-1] if frames_with_text else ""
            # Extract numeric answer
            return self.extract_number(final_line)

        elif problem_type == 'spatial':
            # Use final frame(s) showing solution
            final_frame = video.frames[-1]
            return self.extract_from_image(final_frame)

    def extract_number(self, text):
        # Extract numeric answer from text
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return numbers[-1] if numbers else None
```

**4. Benchmark on VideoThinkBench**

Evaluate reasoning across different problem types.

```python
class VideoReasoningBenchmark:
    def __init__(self, engine, answer_extractor):
        self.engine = engine
        self.extractor = answer_extractor

    def evaluate_spatial_tasks(self, spatial_problems):
        """Evaluate spatial reasoning (puzzles, mazes, etc.)"""
        accuracies = []
        for problem in spatial_problems:
            predicted_answer, video = self.engine.solve_spatial_problem(problem['description'])
            correct = predicted_answer == problem['ground_truth']
            accuracies.append(correct)
        return np.mean(accuracies)

    def evaluate_math_tasks(self, math_problems):
        """Evaluate mathematical reasoning"""
        accuracies = []
        for problem in math_problems:
            predicted_answer, video, steps = self.engine.solve_math_problem(
                problem['problem_text']
            )
            correct = self.check_answer_equivalence(
                predicted_answer, problem['ground_truth']
            )
            accuracies.append(correct)
        return np.mean(accuracies)

    def check_answer_equivalence(self, predicted, ground_truth):
        # Handle different answer formats
        try:
            pred_val = float(predicted)
            truth_val = float(ground_truth)
            return abs(pred_val - truth_val) < 0.01
        except:
            return predicted.strip() == ground_truth.strip()

    def run_full_benchmark(self, benchmark_dataset):
        """Run complete evaluation"""
        results = {
            'spatial': self.evaluate_spatial_tasks(benchmark_dataset['spatial']),
            'math': self.evaluate_math_tasks(benchmark_dataset['math']),
            'vision_centric': self.evaluate_spatial_tasks(benchmark_dataset['vision_centric']),
            'text_centric': self.evaluate_math_tasks(benchmark_dataset['text_centric'])
        }
        return results
```

## Practical Guidance

**When to Use**:
- Reasoning tasks combining spatial and textual elements
- Problem-solving that benefits from visual/temporal explanation
- Applications where reasoning process visibility is valuable
- Educational contexts (video explanations)

**Hyperparameters**:
- num_frames: 24-120 (more frames for complex reasoning)
- duration: 1-5 seconds (longer for detailed steps)
- resolution: 1280x720 (balance quality and inference cost)

**When NOT to Use**:
- Real-time reasoning (video generation is slow)
- Closed-book contexts (answer visible in generated video violates some setups)
- High-frequency reasoning (too expensive per problem)

**Pitfalls**:
- **Answer leakage**: Generated video may show answer too early; control frame ordering
- **OCR errors**: Extracting text from video frames is fragile; use robust OCR
- **Incoherent videos**: Video models can generate unrealistic visualizations; validate plausibility

**Integration Strategy**: Use as auxiliary reasoning modality alongside text. For hybrid tasks, generate both text reasoning and video reasoning, then ensemble answers.

## Reference

arXiv: https://arxiv.org/abs/2511.04570
