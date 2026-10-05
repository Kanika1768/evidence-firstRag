"""EvidenceFirst RAG: Interactive Streamlit Application.

Implements complete auditable visualization:
- Document upload (PDF, TXT, MD) + pre-indexed demo corpus
- One-click buttons for the 3 Canonical Demo Scenarios:
  1. Sufficient context (Direct answer)
  2. Recovery success (Incomplete context -> focused recovery -> verified answer)
  3. Grounded abstention (Unanswerable query -> safe abstention)
- Full step-by-step trace:
  - Retrieved Top Chunks (Document Name, Page, Section, Chunk ID)
  - Sufficiency Evaluation (Probability, Missing Facts, Diligent Reader Rationale)
  - Adaptive Recovery Trace (Triggered, Reformulated Query, Post-Recovery State)
  - Grounded Answer with Claim Attribution & Citations
  - Claim Entailment Verification Report
  - Final Decision & Monitored Latency
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agentic_rag.adapters.llm_client import create_llm_client
from agentic_rag.adapters.structured_sufficiency import LLMSufficiencyAutoRater
from evidencefirst_rag.chunking import Chunk, chunk_document_pages
from evidencefirst_rag.ingestion import load_corpus, load_document
from evidencefirst_rag.pipeline import EvidenceFirstPipeline, PipelineTrace
from evidencefirst_rag.retrieval import HybridRetriever
from evidencefirst_rag.sufficiency import ContextSufficiencyEvaluator

st.set_page_config(
    page_title="EvidenceFirst RAG Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("EvidenceFirst RAG: Context Sufficiency & Adaptive Recovery")


@st.cache_resource
def get_default_chunks() -> list[Chunk]:
    """Load or generate default demo chunks from data/demo_documents."""
    processed_json = ROOT / "data" / "processed" / "chunks.json"
    if processed_json.exists():
        with open(processed_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [Chunk.from_dict(c) for c in raw]

    # Ingest fallback
    pages = load_corpus(ROOT / "data" / "demo_documents")
    return chunk_document_pages(pages, target_tokens=550, overlap_tokens=80)


chunks = get_default_chunks()

# Sidebar: Corpus Management & Documents
with st.sidebar:
    st.header("Document Corpus")
    st.info(f"Active Index: **{len(chunks)} chunks** indexed across demo documents.")

    uploaded_files = st.file_uploader(
        "Upload additional PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        upload_dir = ROOT / ".streamlit_uploads"
        upload_dir.mkdir(exist_ok=True)
        new_pages = []
        for uf in uploaded_files:
            dest = upload_dir / uf.name
            dest.write_bytes(uf.getvalue())
            pages = load_document(dest)
            new_pages.extend(pages)
        if new_pages:
            new_chunks = chunk_document_pages(new_pages, target_tokens=550, overlap_tokens=80)
            chunks = list(chunks) + new_chunks
            st.success(f"Added {len(new_chunks)} chunks from uploaded files!")

    with st.expander("Browse Indexed Documents", expanded=False):
        unique_docs = sorted(list({c.document_name for c in chunks}))
        for d in unique_docs:
            doc_chunks = [c for c in chunks if c.document_name == d]
            st.markdown(f"**{d}** ({len(doc_chunks)} chunks)")

    st.divider()
    st.markdown("### Architecture Pipeline")
    st.markdown(
        """
        1. **Hybrid Retrieval**: Dense (FAISS) + BM25 $\\to$ RRF ($k=60$)
        2. **Reranker**: Cross-Encoder $\\to$ Top 6 chunks
        3. **Sufficiency Judge**: Diligent Reader standard
        4. **Adaptive Recovery**: Targeted query (max 1 pass)
        5. **Grounded Generation**: Claim-level attribution
        6. **Citation Verifier**: Entailment support check
        """
    )

# Quick Demo Scenario Buttons
st.markdown("#### Quick Scenarios")
col_s1, col_s2, col_s3 = st.columns(3)

default_q = ""
if col_s1.button("Scenario 1: Sufficient (Single-Pass)", use_container_width=True):
    default_q = "What does NovaTech develop?"
if col_s2.button("Scenario 2: Recovery (Multi-Hop)", use_container_width=True):
    default_q = "Who founded NovaTech and when was it founded?"
if col_s3.button("Scenario 3: Abstention (Unanswerable)", use_container_width=True):
    default_q = "Who is the Chief Financial Officer of NovaTech?"

# User Question Input
question_input = st.text_input(
    "Enter query for EvidenceFirst RAG:",
    value=default_q,
    placeholder="e.g. What does NovaTech develop? or ask about uploaded documents",
)

if question_input.strip():
    llm_client, mode_desc = create_llm_client()
    if llm_client is not None:
        autorater = LLMSufficiencyAutoRater(llm_client=llm_client)
        autorater.mode_description = mode_desc
        evaluator = ContextSufficiencyEvaluator(autorater=autorater, fallback_to_heuristic=True)
    else:
        evaluator = ContextSufficiencyEvaluator(fallback_to_heuristic=True)
    pipeline = EvidenceFirstPipeline(chunks=chunks, evaluator=evaluator)

    with st.spinner("Executing EvidenceFirst RAG workflow..."):
        trace: PipelineTrace = pipeline.run(question_input.strip())

    st.divider()

    # 1. Final Decision & Answer Banner
    if trace.final_decision == "ANSWERED":
        st.success(f"### Decision: ANSWERED (Latency: {trace.latency_ms:.2f} ms)")
        st.markdown(f"**Answer:** {trace.final_answer}")
    else:
        st.warning(f"### Decision: ABSTAINED (Latency: {trace.latency_ms:.2f} ms)")
        st.markdown(f"**System Output:** *{trace.final_answer}*")
        st.caption("Context lacked verified evidence under the Diligent Reader standard; hallucination prevented.")

    # Key Metrics Columns
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Sufficiency Probability",
            f"{trace.sufficiency_result.probability * 100:.1f}%" if trace.sufficiency_result else "0.0%",
        )
    with m2:
        st.metric("Final Evidence Chunks", len(trace.final_evidence_chunks))
    with m3:
        st.metric(
            "Recovery Triggered",
            "Yes (1 Pass)" if (trace.recovery_trace and trace.recovery_trace.triggered) else "No (Skipped)",
        )
    with m4:
        st.metric(
            "Citation Support",
            "100% Verified" if (trace.verification_report and trace.verification_report.all_supported) else "Guarded / None",
        )

    # Tabs for Auditable Trace Steps
    tab_suff, tab_recov, tab_claims, tab_chunks = st.tabs(
        ["1. Sufficiency Evaluation", "2. Adaptive Recovery", "3. Claims & Citations", "4. Top Retrieved Chunks"]
    )

    with tab_suff:
        st.subheader("Context Sufficiency Evaluation")
        if trace.sufficiency_result:
            suff = trace.sufficiency_result
            status_color = "green" if suff.sufficient else "red"
            st.markdown(f"**Status:** :{status_color}[{'SUFFICIENT' if suff.sufficient else 'INSUFFICIENT'}]")
            mode = getattr(suff, "evaluator_mode", "Deterministic fallback")
            st.caption(f"Evaluator Mode: **{mode}**")
            st.markdown(f"**Probability Score:** `{suff.probability:.4f}`")
            st.markdown(f"**Diligent Reader Rationale:** {suff.rationale}")
            if suff.missing_information:
                st.markdown("**Identified Missing Facts:**")
                for mf in suff.missing_information:
                    st.markdown(f"- ⚠️ `{mf}`")
            else:
                st.markdown("✅ No missing facts identified. Context is complete.")

    with tab_recov:
        st.subheader("Adaptive Recovery Controller")
        if trace.recovery_trace and trace.recovery_trace.triggered:
            rec = trace.recovery_trace
            st.markdown(f"**Reformulated Targeted Query:** `{rec.reformulated_query}`")
            st.markdown(f"**Original Missing Information:** {rec.original_missing_facts}")
            st.markdown(f"**Newly Recovered Chunks:** {len(rec.recovered_chunks)}")
            st.markdown(
                f"**Post-Recovery Sufficiency:** "
                f":{'green' if rec.post_recovery_sufficiency else 'red'}[{rec.post_recovery_sufficiency}]"
            )
        else:
            st.info("Recovery was NOT triggered because initial retrieval was sufficient or query has no routed target.")

    with tab_claims:
        st.subheader("Grounded Generation & Citation Verification")
        if trace.answer_record and trace.answer_record.claims:
            for idx, claim in enumerate(trace.answer_record.claims, start=1):
                st.markdown(f"**Claim {idx}:** {claim.text}")
                st.caption(f"Cited Chunks: {', '.join(claim.source_chunk_ids)}")
            if trace.verification_report:
                verif = trace.verification_report
                st.markdown(f"**Verifier Entailment Score:** `{verif.overall_support_score:.2f}`")
                st.markdown(f"**All Claims Supported:** `{verif.all_supported}`")
        else:
            st.caption("No positive claims asserted (system selectively abstained).")

    with tab_chunks:
        st.subheader("Retrieved Chunks Provenance")
        for idx, chunk in enumerate(trace.final_evidence_chunks, start=1):
            with st.expander(f"Chunk #{idx}: [{chunk.document_name}, Page {chunk.page}, ID: {chunk.chunk_id}]"):
                st.markdown(f"**Section:** `{chunk.section}` | **Tokens:** `{chunk.token_count}`")
                st.text(chunk.text)
