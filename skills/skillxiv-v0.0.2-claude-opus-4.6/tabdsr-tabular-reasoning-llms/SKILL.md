---
name: tabdsr-tabular-reasoning-llms
title: "TabDSR: Decompose, Sanitize, Reason for Tabular Data"
version: 0.0.2
engine: skillxiv-v0.0.2-claude-opus-4.6
license: MIT
url: "https://arxiv.org/abs/2511.02219"
keywords: [Tabular QA, Table Understanding, Program Synthesis, Numerical Reasoning, Data Cleaning]
description: "Improve language model performance on complex numerical reasoning over tables through three-stage pipeline: decompose questions into sub-questions, sanitize and clean table data, then generate executable Python code for precise computation."
---

# Title: Enable Accurate Tabular Reasoning Through Decomposition and Code Generation

Language models struggle with numerical reasoning over tables because they hallucinate calculations and misalign rows/columns. TabDSR tackles this with a three-stage pipeline: (1) decompose complex questions into simpler sub-queries, (2) sanitize table data (fix headers, standardize formats, remove noise), and (3) use code generation to compute answers precisely. Each agent specializes in one task, enabling modular improvement.

The key insight is using code as the reasoning substrate for numerical operations.

## Core Concept

**Three-Stage Tabular Reasoning Pipeline**:
- **Query Decomposer**: Break complex questions into tractable sub-questions
- **Table Sanitizer**: Clean structural and content issues, standardize formats
- **Program-of-Thoughts (PoT) Reasoner**: Generate Python code for precise computation
- **Modular Design**: Each component can be improved independently
- **Evaluation**: New CalTab151 benchmark prevents data leakage

## Architecture Overview

- **Query Decomposer Agent**: LLM-based question analyzer using linguistic cues
- **Table Sanitizer Agent**: Multi-step cleaning (header reconstruction, value standardization)
- **PoT Reasoner**: Code generation using Pandas DataFrames
- **Execution Environment**: Sandboxed Python for safe code execution
- **Dataset**: CalTab151 with multi-hop questions and human verification

## Implementation Steps

**1. Implement Query Decomposer**

Analyze questions to identify sub-problems.

```python
class QueryDecomposer:
    def __init__(self, llm_model):
        self.llm = llm_model

    def analyze_query_complexity(self, query):
        """Determine if question needs decomposition"""
        # Check for linguistic patterns indicating multi-step reasoning
        complex_patterns = [
            'compare', 'how many', 'what percentage', 'ratio',
            'more than', 'less than', 'between', 'and then'
        ]

        is_complex = any(pattern in query.lower() for pattern in complex_patterns)
        return is_complex

    def decompose_query(self, query, table_context):
        """Break complex query into sub-questions"""
        if not self.analyze_query_complexity(query):
            return [query]  # Already simple

        # Use LLM to propose decomposition
        decompose_prompt = f"""Break this question into simpler sub-questions:
        Question: {query}
        Table context: {self.summarize_table(table_context)}

        Sub-questions (numbered):"""

        decomposition = self.llm.generate(decompose_prompt)
        sub_questions = self.parse_subquestions(decomposition)
        return sub_questions

    def parse_subquestions(self, text):
        """Extract numbered sub-questions from LLM output"""
        import re
        matches = re.findall(r'\d+\.\s*([^0-9]+?)(?=\d+\.|$)', text)
        return [m.strip() for m in matches]

    def summarize_table(self, table):
        """Provide table summary for context"""
        return f"Columns: {list(table.columns)}, Rows: {len(table)}"
```

**2. Implement Table Sanitizer**

Clean structural and content issues in tables.

```python
class TableSanitizer:
    def __init__(self):
        self.null_indicators = ['N/A', 'NA', 'null', 'None', '-', '?']

    def sanitize_headers(self, table):
        """Fix and standardize column headers"""
        # Issue 1: Multi-level headers (merge into single level)
        if isinstance(table.columns, pd.MultiIndex):
            table.columns = ['_'.join(col).strip() for col in table.columns]

        # Issue 2: Inconsistent naming
        table.columns = [col.lower().strip().replace(' ', '_') for col in table.columns]

        # Issue 3: Empty headers
        table.columns = [f"col_{i}" if col == "" else col for i, col in enumerate(table.columns)]

        return table

    def sanitize_content(self, table):
        """Clean cell values"""
        # Convert to appropriate types
        for col in table.columns:
            # Try numeric conversion
            try:
                table[col] = pd.to_numeric(table[col], errors='coerce')
            except:
                # Keep as string
                table[col] = table[col].astype(str)

            # Replace null indicators
            for null_ind in self.null_indicators:
                table[col] = table[col].replace(null_ind, None)

        # Remove fully null rows/columns
        table = table.dropna(how='all')
        table = table.dropna(axis=1, how='all')

        return table

    def reconstruct_segmented_tables(self, table):
        """Handle tables split across multiple sections"""
        # Detect horizontal breaks (rows with all None)
        null_rows = table.isnull().all(axis=1).values
        break_indices = np.where(null_rows)[0]

        if len(break_indices) > 0:
            # Split and process each segment
            segments = []
            start = 0
            for break_idx in break_indices:
                segment = table.iloc[start:break_idx]
                segments.append(segment)
                start = break_idx + 1
            segments.append(table.iloc[start:])

            # Return largest or most complete segment
            return max(segments, key=lambda x: x.shape[0])

        return table

    def sanitize(self, table):
        """Full sanitization pipeline"""
        table = self.sanitize_headers(table)
        table = self.sanitize_content(table)
        table = self.reconstruct_segmented_tables(table)
        return table
```

**3. Implement Program-of-Thoughts Reasoner**

Generate executable Python code for computation.

```python
class ProgramOfThoughtsReasoner:
    def __init__(self, code_llm, execution_env):
        self.code_llm = code_llm
        self.exec_env = execution_env

    def generate_program(self, question, table_df):
        """Generate Python code to answer question"""
        # Provide table schema
        schema = self.get_table_schema(table_df)

        code_prompt = f"""Write Python code to answer this question:
        Question: {question}

        Table schema:
        {schema}

        Available variable: 'df' is a pandas DataFrame with the table.

        Code (complete and executable):
        """

        code = self.code_llm.generate(code_prompt, max_tokens=500)
        return code

    def get_table_schema(self, df):
        """Describe table structure and dtypes"""
        schema = f"Columns: {list(df.columns)}\n"
        schema += f"Dtypes: {dict(df.dtypes)}\n"
        schema += f"Shape: {df.shape}\n"
        schema += f"Sample row: {df.iloc[0].to_dict()}"
        return schema

    def execute_program(self, code, df):
        """Execute Python code in sandbox"""
        # Create execution context
        local_context = {'df': df, 'pd': pd, 'np': np}

        try:
            exec(code, {}, local_context)

            # Extract answer (look for 'answer' or 'result' variable)
            if 'answer' in local_context:
                return local_context['answer']
            elif 'result' in local_context:
                return local_context['result']
            else:
                # Return last assigned non-df variable
                return self.extract_final_value(local_context)

        except Exception as e:
            return f"Execution error: {str(e)}"

    def extract_final_value(self, context):
        """Extract final computed value from context"""
        # Get variables that were created/modified
        non_builtins = {k: v for k, v in context.items() if not k.startswith('_')}
        # Find numeric or string results
        for k, v in sorted(non_builtins.items(), reverse=True):
            if isinstance(v, (int, float, str)):
                return v
        return None

    def reason(self, question, table_df):
        """Full reasoning pipeline"""
        code = self.generate_program(question, table_df)
        answer = self.execute_program(code, table_df)
        return answer, code
```

**4. Integrate Three Stages**

Combine pipeline into unified interface.

```python
class TabDSRPipeline:
    def __init__(self, decomposer, sanitizer, reasoner):
        self.decomposer = decomposer
        self.sanitizer = sanitizer
        self.reasoner = reasoner

    def answer_question(self, question, table):
        """Answer tabular question using three-stage pipeline"""
        # Stage 1: Decompose
        sub_questions = self.decomposer.decompose_query(question, table)

        # Stage 2: Sanitize
        clean_table = self.sanitizer.sanitize(table)

        # Stage 3: Reason
        if len(sub_questions) == 1:
            # Simple question
            answer, code = self.reasoner.reason(question, clean_table)
        else:
            # Complex question: answer sub-questions, then combine
            sub_answers = []
            for sub_q in sub_questions:
                sub_answer, _ = self.reasoner.reason(sub_q, clean_table)
                sub_answers.append(sub_answer)

            # Combine sub-answers (simple merge for now)
            answer = self.combine_answers(question, sub_answers, clean_table)
            code = "# See sub-questions above"

        return {
            'answer': answer,
            'code': code,
            'sub_questions': sub_questions,
            'cleaned_table': clean_table
        }

    def combine_answers(self, original_question, sub_answers, table):
        """Merge sub-question answers into final result"""
        # For complex multi-step reasoning, generate code to combine results
        combine_prompt = f"""Combine these sub-answers to answer the original question:
        Original: {original_question}
        Sub-answers: {sub_answers}

        Final answer:"""

        combined = self.reasoner.code_llm.generate(combine_prompt)
        return combined
```

## Practical Guidance

**When to Use**:
- Numerical QA over business/scientific tables
- Tasks with complex multi-hop reasoning
- Settings where accuracy is critical (hallucinations unacceptable)

**Hyperparameters**:
- decomposition_threshold: Complexity level triggering decomposition
- null_replacement: How to handle missing values
- code_max_tokens: 500-1000 depending on problem complexity

**When NOT to Use**:
- Simple lookup questions (single cell retrieval)
- Tables with highly irregular structure
- Domains where code generation fails (expert knowledge needed)

**Pitfalls**:
- **Code generation hallucination**: Model may generate syntactically valid but logically wrong code; validate with intermediate results
- **Over-decomposition**: Some questions don't benefit from decomposition; check impact on results
- **Table cleaning over-aggressiveness**: Over-sanitizing can remove important nuance

**Integration Strategy**: Use as fallback for numerical questions. Combine with semantic matching for hybrid QA systems.

## Reference

arXiv: https://arxiv.org/abs/2511.02219
