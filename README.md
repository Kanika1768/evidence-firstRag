# EvidenceFirst RAG

EvidenceFirst RAG is a retrieval-augmented generation system focused on
**context sufficiency, adaptive recovery, grounded answering, and
evidence traceability**.

The project is inspired by the paper:

> Sufficient Context: A New Lens on Retrieval Augmented Generation
> Systems\
> ICLR 2025

## Overview

Traditional RAG systems may retrieve documents and generate an answer
even when the retrieved context does not contain enough information to
answer the question reliably.

EvidenceFirst RAG introduces a sufficiency-aware workflow:

``` text
Documents
    |
    v
Document Loading & Chunking
    |
    v
Retrieval
    |
    v
Context Sufficiency Evaluation
    |
    +-------------------+
    |                   |
 Sufficient?         Insufficient
    |                   |
    v                   v
Generate             Reformulate
Answer               & Retrieve
    |                   |
    |               Re-evaluate
    |                   |
    +---------+---------+
              |
              v
       Grounded Answer
       + Evidence/Citations
```

## Key Features

-   Context sufficiency evaluation
-   Evidence-grounded answering
-   Adaptive recovery when retrieved context is insufficient
-   Evidence provenance and citations
-   Selective answering and abstention
-   Multi-document retrieval
-   PDF, TXT, and Markdown document support
-   Interactive Streamlit interface
-   Deterministic local execution without requiring an API key

## How the System Works

1.  Documents are loaded and divided into chunks with document and page
    metadata.
2.  The user's question is decomposed into required facts.
3.  Relevant evidence is retrieved from the local document collection.
4.  The retrieved context is evaluated for sufficiency.
5.  If sufficient evidence is available, a grounded answer is generated.
6.  If evidence is insufficient, the system generates a targeted
    recovery query and performs another retrieval iteration.
7.  The final answer includes evidence provenance and citations.

## Project Structure

``` text
evidence-first-rag/
|
+-- data/
|   +-- demo_documents/
|
+-- examples/
|   +-- run_generic_evidencefirst.py
|   +-- streamlit_app.py
|
+-- src/
|   +-- agentic_rag/
|       +-- adapters/
|       +-- chunking.py
|       +-- document_index.py
|       +-- document_loader.py
|       +-- generic_pipeline.py
|       +-- general_planner.py
|       +-- orchestrator.py
|       +-- sufficiency.py
|
+-- tests/
+-- docs/
+-- pyproject.toml
+-- README.md
```

## Running the Demo

Create and activate the virtual environment:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project:

``` bash
pip install -e .
```

Run the command-line demo:

``` bash
python examples/run_generic_evidencefirst.py
```

Run the interactive Streamlit interface:

``` bash
streamlit run examples/streamlit_app.py
```

The Streamlit interface allows a user to upload a PDF, TXT, or Markdown
document and ask questions about its contents.

## Example

For a question such as:

``` text
What is NovaTech's annual revenue?
```

the system can:

1.  Retrieve available evidence.
2.  Check whether the retrieved context is sufficient.
3.  Detect missing evidence.
4.  Perform a recovery retrieval step.
5.  Re-evaluate the context.
6.  Generate an answer grounded in the retrieved evidence.

## Evaluation Plan

The project evaluates whether explicit context-sufficiency checking can
reduce unsupported answers in RAG systems.

The planned comparison includes:

-   **Baseline RAG:** retrieve once and answer.
-   **Sufficiency-aware RAG:** answer only when sufficient evidence is
    available.
-   **EvidenceFirst RAG:** sufficiency checking with adaptive recovery
    and grounded evidence verification.

Planned metrics include:

-   Answer correctness
-   Groundedness
-   Unsupported-answer rate
-   Coverage
-   Recovery success
-   Iteration count

## Research Basis

The system implements the sufficient-context principle described in the
ICLR 2025 paper:

**Sufficient Context: A New Lens on Retrieval Augmented Generation
Systems**

Official paper:

[ICLR 2025
paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html)

Official paper repository:

[GitHub repository](https://github.com/hljoren/sufficientcontext)

## Project Scope

This project implements the sufficient-context principle from the paper
and extends the workflow with adaptive recovery, evidence provenance,
grounded answering, and an interactive document interface.
It is **not intended to be an exact reproduction of the paper's complete
experimental setup**.

## Testing

The implementation includes automated tests covering the core RAG,
sufficiency, retrieval, orchestration, and evaluation components.

Run:

``` bash
python -m pytest -q
```

