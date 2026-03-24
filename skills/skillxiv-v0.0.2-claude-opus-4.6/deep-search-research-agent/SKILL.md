---
name: deep-search-research-agent
title: "DeepSearchQA: Bridging the Comprehensiveness Gap for Deep Research Agents"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2601.20975"
keywords: [research-agent, information-retrieval, question-answering, deep-search, comprehensiveness]
description: "Build research agents that systematically search for comprehensive answers to complex questions by maintaining search state, iterating on queries, and validating answer completeness. Implement strategies for identifying knowledge gaps and conducting follow-up searches to ensure thorough coverage of topics."
---

## Problem

Research agents often provide incomplete answers to complex questions because they stop searching too early or fail to identify gaps in their knowledge. A single query rarely captures all relevant information, and agents need strategies to recognize when they have insufficient information and conduct targeted follow-up searches.

## Solution

Implement a DeepSearchQA framework where agents:

1. **Maintain Search State**: Track what has been searched, what answers have been found, and what gaps remain
2. **Identify Gaps**: Use gap detection to recognize incomplete coverage of question aspects
3. **Iterate on Queries**: Generate targeted follow-up searches based on identified gaps
4. **Validate Comprehensiveness**: Check that answers cover multiple perspectives and dimensions of the question
5. **Stop Strategically**: Determine when additional searches provide diminishing returns

## When to Use

- Building research assistants for complex multi-faceted questions
- Question-answering systems requiring comprehensive coverage
- Literature review and synthesis tasks
- Competitive analysis and market research automation
- Policy analysis and decision-support systems

## When NOT to Use

- Simple factual lookup questions (single search suffices)
- Time-sensitive applications (iterative searching adds latency)
- Narrow technical queries with definitive answers
- Constrained search budgets or API rate limits

## Implementation

### Step 1: Build Search State Management

Track what has been discovered and what gaps remain.

```python
class SearchState:
    """Maintain comprehensive search state"""

    def __init__(self, query):
        self.original_query = query
        self.search_history = []
        self.discovered_answers = []
        self.known_gaps = set()
        self.relevant_subtopics = set()
        self.coverage_map = {}

    def record_search(self, query, results):
        """Log search execution and results"""
        search_record = {
            "query": query,
            "num_results": len(results),
            "timestamp": datetime.now(),
            "results_summary": self.summarize_results(results)
        }
        self.search_history.append(search_record)

        # Extract answers and update coverage
        for result in results:
            answer = self.extract_answer(result)
            if answer not in self.discovered_answers:
                self.discovered_answers.append(answer)

    def update_coverage_map(self):
        """Map which question dimensions are covered"""
        self.coverage_map = {
            "definition": self.covers_aspect("what_is", self.discovered_answers),
            "how": self.covers_aspect("how_does", self.discovered_answers),
            "why": self.covers_aspect("why", self.discovered_answers),
            "examples": self.covers_aspect("examples", self.discovered_answers),
            "exceptions": self.covers_aspect("limitations", self.discovered_answers),
            "recent_developments": self.covers_aspect("recent", self.discovered_answers)
        }

    def identify_gaps(self):
        """Find aspects of question with insufficient coverage"""
        self.update_coverage_map()
        gaps = [aspect for aspect, covered in self.coverage_map.items() if not covered]
        self.known_gaps = set(gaps)
        return gaps

    def get_uncovered_subtopics(self):
        """Extract entities and concepts not yet fully explored"""
        all_entities = self.extract_all_entities(self.discovered_answers)
        explored = {e for e in all_entities if self.has_sufficient_coverage(e)}
        return all_entities - explored
```

### Step 2: Implement Gap-Driven Query Generation

Create follow-up queries based on identified gaps.

```python
def generate_follow_up_queries(state):
    """
    Generate targeted follow-up searches to address gaps
    """
    gaps = state.identify_gaps()
    uncovered_subtopics = state.get_uncovered_subtopics()

    follow_ups = []

    # For each gap, create targeted query
    for gap in gaps:
        if gap == "definition":
            query = f"What is {state.original_query}? Definition and meaning"
        elif gap == "how":
            query = f"How does {state.original_query} work? Mechanism and process"
        elif gap == "why":
            query = f"Why {state.original_query}? Reasons and motivations"
        elif gap == "examples":
            query = f"{state.original_query} examples use cases real world applications"
        elif gap == "exceptions":
            query = f"{state.original_query} limitations edge cases exceptions"
        elif gap == "recent_developments":
            query = f"{state.original_query} recent advances 2024 2025"

        follow_ups.append({
            "query": query,
            "gap_addressed": gap,
            "priority": self.compute_gap_priority(gap, state)
        })

    # Explore uncovered subtopics
    for subtopic in list(uncovered_subtopics)[:3]:  # Top 3 subtopics
        query = f"{state.original_query} {subtopic} detailed explanation"
        follow_ups.append({
            "query": query,
            "gap_addressed": f"subtopic_{subtopic}",
            "priority": 0.5
        })

    # Sort by priority (descending)
    follow_ups.sort(key=lambda x: x["priority"], reverse=True)
    return follow_ups
```

### Step 3: Build Comprehensiveness Validator

Evaluate when answer coverage is sufficient.

```python
class ComprehensivenessValidator:
    """Determine if search has yielded comprehensive answers"""

    def __init__(self, target_coverage=0.85):
        self.target_coverage = target_coverage

    def compute_coverage_score(self, state):
        """
        Score how comprehensively the answer covers the question.
        Considers: breadth of dimensions, depth per dimension, diversity of sources
        """
        state.update_coverage_map()

        # Dimension coverage: what % of question aspects are addressed
        dimensions = state.coverage_map.values()
        dimension_score = sum(dimensions) / len(dimensions)

        # Depth score: how many answers/perspectives per dimension
        answer_count = len(state.discovered_answers)
        depth_score = min(answer_count / 5, 1.0)  # Normalize to 5+ answers

        # Diversity score: sources/perspectives vary
        source_diversity = self.measure_source_diversity(state.discovered_answers)
        diversity_score = source_diversity

        # Composite score: weighted combination
        coverage_score = (
            0.4 * dimension_score +
            0.35 * depth_score +
            0.25 * diversity_score
        )

        return coverage_score

    def should_continue_searching(self, state, search_count):
        """
        Determine if more searches are needed
        """
        coverage = self.compute_coverage_score(state)
        gaps = state.identify_gaps()

        # Stop if coverage is high enough
        if coverage >= self.target_coverage:
            return False

        # Stop if too many searches already
        if search_count > 10:
            return False

        # Stop if diminishing returns (last 2 searches added <5% new info)
        if len(state.search_history) >= 3:
            recent_gain = self.compute_recent_information_gain(state, window=2)
            if recent_gain < 0.05:
                return False

        return True

    def compute_recent_information_gain(self, state, window=2):
        """Measure how much new information recent searches added"""
        if len(state.search_history) < window:
            return 1.0

        recent_answers = set()
        for record in state.search_history[-window:]:
            recent_answers.update(record["results_summary"])

        total_answers = set(state.discovered_answers)
        gain = len(recent_answers) / max(len(total_answers), 1)
        return gain
```

### Step 4: Orchestrate the Deep Search Loop

Implement the main agent search iteration.

```python
def deep_search_agent(question, max_searches=10):
    """
    Main agent loop: search -> evaluate -> identify gaps -> search again
    """
    state = SearchState(question)
    validator = ComprehensivenessValidator(target_coverage=0.8)

    search_count = 0

    while search_count < max_searches:
        if search_count == 0:
            # Initial search with original question
            query = question
        else:
            # Generate gap-driven follow-up queries
            follow_ups = generate_follow_up_queries(state)
            if not follow_ups:
                break

            query = follow_ups[0]["query"]

        # Execute search
        results = execute_search(query)
        state.record_search(query, results)
        search_count += 1

        # Check if comprehensive coverage achieved
        coverage = validator.compute_coverage_score(state)

        print(f"Search {search_count}: Coverage={coverage:.2%}, Gaps={len(state.known_gaps)}")

        # Decide whether to continue
        if not validator.should_continue_searching(state, search_count):
            break

    # Synthesize final answer
    final_answer = synthesize_answer(state.discovered_answers, state.coverage_map)

    return {
        "answer": final_answer,
        "coverage_score": validator.compute_coverage_score(state),
        "searches_conducted": search_count,
        "answer_count": len(state.discovered_answers),
        "dimensions_covered": sum(state.coverage_map.values())
    }
```

### Step 5: Answer Synthesis and Validation

Combine discovered answers into coherent comprehensive response.

```python
def synthesize_answer(discovered_answers, coverage_map):
    """
    Organize discovered answers by dimension (what, how, why, examples, etc.)
    """
    synthesis = {
        "summary": "",
        "dimensions": {}
    }

    # Organize by dimension
    for dimension, covered in coverage_map.items():
        if covered:
            relevant_answers = [
                a for a in discovered_answers
                if categorizes_as_dimension(a, dimension)
            ]
            synthesis["dimensions"][dimension] = {
                "answers": relevant_answers,
                "synthesized": synthesize_dimension(relevant_answers)
            }

    # Create overview
    synthesis["summary"] = create_overview(synthesis["dimensions"])

    return synthesis
```

## Key Strategies

- **Multi-Dimensional Coverage**: Ensure answers cover what, how, why, examples, limitations, recent work
- **Iterative Refinement**: Use gap detection to guide follow-up searches
- **Diminishing Returns Detection**: Stop searching when new information becomes sparse
- **Diversity Over Quantity**: Prioritize varied sources and perspectives over raw answer count

## Success Indicators

- Coverage score >= 0.8 across multiple dimensions
- Answer includes definitions, mechanisms, applications, and limitations
- Multiple independent sources support key claims
- Recent developments and current state of topic included

## References

- arXiv:2601.20975: DeepSearchQA framework for comprehensive research
- Designed for question-answering agents requiring thorough coverage
