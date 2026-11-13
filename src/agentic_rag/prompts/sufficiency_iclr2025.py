"""ICLR 2025 Paper-Aligned Sufficient Context Evaluation Prompt.

Methodological Reference:
"Sufficient Context: A New Lens on Retrieval Augmented Generation Systems"
Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, Cyrus Rashtchian
ICLR 2025 — Official Repository: https://github.com/hljoren/sufficientcontext

Key Methodological Elements:
1. Strictly 1-shot demonstration using the verified Roald Dahl / British Railways Board example.
2. Two-phase diagnostic reasoning (Chain-of-Thought):
   a. Output step-by-step questions (assumptions, calculations).
   b. Answer each question step by step.
3. Explicit temporal grounding via `<TIMESTAMP>`.
4. Output format: `### EXPLANATION (Text)` followed by `### JSON` containing `{"Sufficient Context": 0|1}`.
5. Zero Oracle Leakage: The evaluator receives only the question, retrieved context (references),
   and timestamp. Planned required facts, candidate labels, and gold annotations are strictly excluded.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

ICLR2025_PROMPT_VERSION = "iclr2025_oneshot"
DEFAULT_BENCHMARK_TIMESTAMP = "2024-01-01"

# Official 1-shot demonstration from Joren et al. (ICLR 2025) / hljoren/sufficientcontext
ONE_SHOT_DEMONSTRATION = """EXAMPLE:
### QUESTION
In which year did the publisher of Roald Dahl's Guide to Railway Safety cease to exist?

### References
Roald Dahl's Guide to Railway Safety was published in 1991 by the British Railways Board. The British Railways Board had asked Roald Dahl to write the text of the booklet, and Quentin Blake to illustrate it, to help young people enjoy using the railways safely. The British Railways Board (BRB) was a nationalised industry in the United Kingdom that operated from 1963 to 2001. Until 1997 it was responsible for most railway services in Great Britain, trading under the brand name British Railways and, from 1965, British Rail. It did not operate railways in Northern Ireland, where railways were the responsibility of the Government of Northern Ireland.

### EXPLANATION
The context mentions that Roald Dahl's Guide to Railway Safety was published by the British Railways Board. It also states that the British Railways Board operated from 1963 to 2001, meaning the year it ceased to exist was 2001. Therefore, the context does provide a precise answer to the question.

### JSON
{"Sufficient Context": 1}"""


ICLR2025_INSTRUCTIONS = """You are an expert LLM evaluator that excels at evaluating a QUESTION and REFERENCES. 
Consider the following criteria:
Sufficient Context: 1 IF the CONTEXT is sufficient to infer the answer to the question and 0 IF the CONTEXT cannot be used to infer the answer to the question

Assume the queries have timestamp <TIMESTAMP>. 

First, output a list of step-by-step questions that would be used to arrive at a label for the criteria. Make sure to include questions about assumptions implicit in the QUESTION. Include questions about any mathematical calculations or arithmetic that would be required. 

Next, answer each of the questions. Make sure to work step by step through any required mathematical calculations or arithmetic. 

Finally, use these answers to evaluate the criteria. 

Output the ### EXPLANATION (Text). Then, use the EXPLANATION to output the ### EVALUATION (JSON)"""


def build_iclr2025_prompt(
    question: str,
    references: str,
    timestamp: str = DEFAULT_BENCHMARK_TIMESTAMP,
) -> str:
    """Construct the official ICLR 2025 1-shot Chain-of-Thought prompt.

    Args:
        question: The user query to evaluate.
        references: The retrieved context passages / references.
        timestamp: Temporal anchor date string for time-sensitive queries.

    Returns:
        The formatted complete prompt ready for LLM generation.
    """
    instructions = ICLR2025_INSTRUCTIONS.replace("<TIMESTAMP>", timestamp.strip())

    prompt_parts = [
        instructions,
        "",
        ONE_SHOT_DEMONSTRATION,
        "",
        "### QUESTION",
        question.strip(),
        "",
        "### References",
        references.strip(),
        "",
        "### EXPLANATION",
    ]
    return "\n".join(prompt_parts)
