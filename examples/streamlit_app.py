import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_rag.chunking import chunk_documents
from agentic_rag.contracts import Corpus
from agentic_rag.document_index import build_local_index
from agentic_rag.document_loader import load_documents
from agentic_rag.generic_pipeline import build_generic_components
from agentic_rag.orchestrator import (
    AgenticRAGOrchestrator,
    OrchestratorConfig,
)


st.set_page_config(
    page_title="EvidenceFirst RAG",
    page_icon="🔎",
    layout="wide",
)

st.title("EvidenceFirst RAG")
st.caption(
    "Context sufficiency • Adaptive recovery • Grounded answers"
)

st.markdown(
    """
Upload a document, ask a question, and watch the system determine
whether the retrieved evidence is sufficient before answering.
"""
)


def build_corpora(documents):
    corpora = []

    for document in documents:
        corpus_id = Path(document.document_name).stem

        corpora.append(
            Corpus(
                id=corpus_id,
                description=(
                    f"{document.document_name} "
                    f"{document.text[:500]}"
                ),
            )
        )

    return tuple(corpora)


def build_orchestrator(documents):
    retriever = build_local_index(
        documents,
        chunk_size=800,
        overlap=120,
        per_query_limit=5,
    )

    components = build_generic_components()

    return (
        AgenticRAGOrchestrator(
            planner=components["planner"],
            rewriter=components["rewriter"],
            retriever=retriever,
            drafter=components["drafter"],
            judge=components["judge"],
            synthesizer=components["synthesizer"],
            config=OrchestratorConfig(
                max_iterations=2,
            ),
        ),
        build_corpora(documents),
    )


uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt", "md"],
)

if uploaded_file is not None:
    st.success(f"Loaded: {uploaded_file.name}")

    temp_dir = ROOT / ".streamlit_uploads"
    temp_dir.mkdir(exist_ok=True)

    file_path = temp_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getvalue())

    try:
        documents = load_documents(file_path.parent)

        documents = tuple(
            document
            for document in documents
            if document.document_name == uploaded_file.name
        )

        if not documents:
            st.error("Could not read the uploaded document.")
            st.stop()

        st.info(
            f"Document loaded successfully • "
            f"{len(documents)} page(s)"
        )

        question = st.text_input(
            "Ask a question",
            placeholder="e.g. What is the main contribution of this paper?",
        )

        if st.button("Ask", type="primary") and question.strip():

            with st.spinner("Retrieving evidence and checking sufficiency..."):
                orchestrator, corpora = build_orchestrator(documents)

                result = orchestrator.run(
                    question.strip(),
                    corpora,
                )

            answer = result.answer

            st.divider()

            st.subheader("Final Answer")

            if answer.answer:
                st.write(answer.answer)
            else:
                st.warning("No grounded answer is available.")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Status",
                    str(answer.status).replace("AnswerStatus.", ""),
                )

            with col2:
                st.metric(
                    "Sufficiency",
                    f"{answer.sufficiency_score:.2f}",
                )

            with col3:
                st.metric(
                    "Iterations",
                    len(result.iterations),
                )

            if answer.citations:
                st.subheader("Evidence / Citations")

                for citation in answer.citations:
                    st.markdown(f"**Claim:** {citation.claim}")
                    st.caption(
                        "Source snippets: "
                        + ", ".join(citation.snippet_ids)
                    )

            st.subheader("EvidenceFirst Trace")

            for trace in result.iterations:
                assessment = trace.assessment

                status = str(assessment.status).replace(
                    "ContextStatus.",
                    "",
                )

                with st.expander(
                    f"Iteration {trace.iteration + 1} — "
                    f"{status} — score "
                    f"{assessment.sufficiency_score:.2f}",
                    expanded=True,
                ):
                    st.write("**Queries**")

                    for query in trace.subqueries:
                        st.code(query.query)

                    if assessment.missing_facts:
                        st.write("**Missing facts**")
                        for fact in assessment.missing_facts:
                            st.write(f"- {fact}")

                    if assessment.feedback_queries:
                        st.write("**Recovery queries**")

                        for feedback in assessment.feedback_queries:
                            st.code(feedback.query)

            st.subheader("Retrieved Evidence")

            for snippet in result.snippets:
                with st.expander(
                    f"{snippet.document_id} "
                    f"• score {snippet.score:.2f}"
                ):
                    st.write(snippet.text)

    except Exception as exc:
        st.error(f"Could not process the document: {exc}")

else:
    st.info("Upload a PDF, TXT, or Markdown document to begin.")
