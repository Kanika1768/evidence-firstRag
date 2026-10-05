"""Typed Structured Sufficiency Result Adapter for EvidenceFirst RAG.

Provides validation, serialization, repair-parsing fallback, and bidirectional
conversion between LLM structured outputs and internal ContextAssessment contracts.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from agentic_rag.contracts import ContextAssessment, ContextStatus


@dataclass
class StructuredSufficiencyResult:
    """Typed representation of a context sufficiency evaluation result.

    Adheres strictly to the Diligent Reader standard schema:
    - sufficient: True if context contains all required facts to definitively answer.
    - probability: Calibrated float confidence score in [0.0, 1.0].
    - missing_information: List of missing facts or claims (empty if sufficient).
    - rationale: Justification for the judgment.
    """

    sufficient: bool
    probability: float
    missing_information: list[str] = field(default_factory=list)
    rationale: str = ""
    evaluator_mode: str = "Deterministic fallback"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate field types and constraints."""
        if not isinstance(self.sufficient, bool):
            raise TypeError(f"sufficient must be a bool, got {type(self.sufficient).__name__}")

        if not isinstance(self.probability, (int, float)):
            raise TypeError(f"probability must be a float, got {type(self.probability).__name__}")
        self.probability = float(self.probability)
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(f"probability must be between 0.0 and 1.0, got {self.probability}")

        if not isinstance(self.missing_information, (list, tuple)):
            raise TypeError(
                f"missing_information must be a list/tuple of strings, got {type(self.missing_information).__name__}"
            )
        self.missing_information = [str(item).strip() for item in self.missing_information if str(item).strip()]

        if not isinstance(self.rationale, str):
            raise TypeError(f"rationale must be a string, got {type(self.rationale).__name__}")
        self.rationale = self.rationale.strip()
        if not self.rationale:
            # Provide sensible default rationale if empty
            if self.sufficient:
                self.rationale = "Retrieved context contains all necessary facts to answer the question."
            else:
                missing_str = ", ".join(self.missing_information) if self.missing_information else "required facts"
                self.rationale = f"Context lacks necessary evidence for: {missing_str}."

    def to_dict(self) -> dict[str, Any]:
        """Convert to Python dictionary."""
        d: dict[str, Any] = {
            "sufficient": self.sufficient,
            "probability": round(self.probability, 4),
            "missing_information": list(self.missing_information),
            "rationale": self.rationale,
        }
        if hasattr(self, "evaluator_mode") and self.evaluator_mode:
            d["evaluator_mode"] = self.evaluator_mode
        return d

    def to_json(self, indent: int | None = None) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StructuredSufficiencyResult:
        """Construct and validate from dictionary."""
        sufficient_val = data.get("sufficient")
        if isinstance(sufficient_val, str):
            sufficient = sufficient_val.strip().lower() in ("true", "1", "yes", "sufficient")
        else:
            sufficient = bool(sufficient_val)

        prob_val = data.get("probability")
        if prob_val is None:
            probability = 1.0 if sufficient else 0.0
        else:
            try:
                probability = float(prob_val)
            except (ValueError, TypeError):
                probability = 1.0 if sufficient else 0.0
        probability = max(0.0, min(1.0, probability))

        missing_raw = data.get("missing_information") or data.get("missing_facts") or []
        if isinstance(missing_raw, (str, bytes)):
            missing_info = [str(missing_raw).strip()] if str(missing_raw).strip() else []
        elif isinstance(missing_raw, (list, tuple)):
            missing_info = [str(item).strip() for item in missing_raw if str(item).strip()]
        else:
            missing_info = []

        rationale_raw = data.get("rationale") or data.get("reason") or ""
        rationale = str(rationale_raw).strip()
        evaluator_mode = str(data.get("evaluator_mode", "Deterministic fallback"))

        return cls(
            sufficient=sufficient,
            probability=probability,
            missing_information=missing_info,
            rationale=rationale,
            evaluator_mode=evaluator_mode,
        )

    @classmethod
    def from_json(cls, raw_text: str) -> StructuredSufficiencyResult:
        """Parse from JSON string with robust multi-stage repair fallback."""
        if not raw_text or not raw_text.strip():
            return cls(
                sufficient=False,
                probability=0.0,
                missing_information=["Empty model response"],
                rationale="Empty model response received.",
            )

        text = raw_text.strip()

        # Step 1: Strip markdown code block wrappers
        markdown_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if markdown_match:
            text = markdown_match.group(1).strip()

        # Step 2: Try direct standard JSON parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return cls.from_dict(parsed)
        except json.JSONDecodeError:
            pass

        # Step 3: Extract outer JSON object substring { ... }
        brace_match = re.search(r"(\{[\s\S]*\})", text)
        if brace_match:
            candidate = brace_match.group(1)
            # Remove trailing commas before } or ]
            cleaned = re.sub(r",\s*([\}\]])", r"\1", candidate)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return cls.from_dict(parsed)
            except json.JSONDecodeError:
                pass

            # Replace single quotes with double quotes
            try:
                sq_cleaned = re.sub(r"'([^']*)'", r'"\1"', cleaned)
                parsed = json.loads(sq_cleaned)
                if isinstance(parsed, dict):
                    return cls.from_dict(parsed)
            except json.JSONDecodeError:
                pass

        # Step 4: Regex-based field extraction heuristic fallback
        suff_match = re.search(r'"?sufficient"?\s*:\s*(true|false)', text, re.IGNORECASE)
        if suff_match:
            try:
                sufficient = suff_match.group(1).lower() == "true"

                prob_match = re.search(r'"?probability"?\s*:\s*([0-9]*\.?[0-9]+)', text)
                probability = float(prob_match.group(1)) if prob_match else (1.0 if sufficient else 0.0)
                probability = max(0.0, min(1.0, probability))

                rationale_match = re.search(r'"?rationale"?\s*:\s*"([^"]+)"', text, re.IGNORECASE)
                rationale = rationale_match.group(1) if rationale_match else "Extracted via regex fallback."

                missing_info: list[str] = []
                missing_match = re.search(r'"?missing_information"?\s*:\s*\[([\s\S]*?)\]', text, re.IGNORECASE)
                if missing_match:
                    items = re.findall(r'"([^"]+)"', missing_match.group(1))
                    missing_info = [it.strip() for it in items if it.strip()]

                return cls(
                    sufficient=sufficient,
                    probability=probability,
                    missing_information=missing_info,
                    rationale=rationale,
                )
            except Exception:
                pass

        # Step 5: Safe final fallback
        return cls(
            sufficient=False,
            probability=0.0,
            missing_information=["Unparseable response format"],
            rationale=f"Failed to parse LLM sufficiency response: {raw_text[:120]}...",
        )

    def to_context_assessment(self) -> ContextAssessment:
        """Convert this structured result into the internal ContextAssessment contract."""
        status = ContextStatus.SUFFICIENT if self.sufficient else ContextStatus.INSUFFICIENT
        return ContextAssessment(
            status=status,
            sufficiency_score=self.probability,
            missing_facts=tuple(self.missing_information),
            reason=self.rationale,
        )

    @classmethod
    def from_context_assessment(cls, assessment: ContextAssessment) -> StructuredSufficiencyResult:
        """Construct a StructuredSufficiencyResult from a ContextAssessment instance."""
        status_val = (
            assessment.status.value
            if hasattr(assessment.status, "value")
            else str(assessment.status).lower()
        )
        sufficient = status_val == ContextStatus.SUFFICIENT.value or status_val == "sufficient"
        probability = float(assessment.sufficiency_score)
        probability = max(0.0, min(1.0, probability))

        return cls(
            sufficient=sufficient,
            probability=probability,
            missing_information=list(assessment.missing_facts),
            rationale=assessment.reason or "",
        )


class LLMSufficiencyAutoRater:
    """Pluggable LLM-based Sufficient Context AutoRater.

    Evaluates whether retrieved evidence is strictly sufficient for a diligent reader
    to construct a definitive answer using ONLY the provided context under the
    Diligent Reader standard (Joren et al., ICLR 2025).

    Accepts:
        question: User query string
        evidence: Sequence of Chunk/Snippet objects, sequence of strings, or raw text string
        required_facts: Optional sequence of decomposed required fact strings

    Returns:
        StructuredSufficiencyResult containing:
            - sufficient: bool
            - probability: float in [0.0, 1.0]
            - missing_information: list of str
            - rationale: str
        Also provides .to_json() / .to_dict() / evaluate_json() for validated serialization.
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        prompt_version: str = "v1.0",
        fallback_judge: Any | None = None,
        system_prompt: str | None = None,
        catch_exceptions: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_version = prompt_version
        self.fallback_judge = fallback_judge
        self.system_prompt = system_prompt
        self.catch_exceptions = catch_exceptions

    @staticmethod
    def format_evidence(evidence: Any) -> str:
        """Format chunks, snippets, strings, or raw text into a coherent context string."""
        if not evidence:
            return ""
        if isinstance(evidence, str):
            return evidence.strip()
        if isinstance(evidence, (list, tuple)):
            parts: list[str] = []
            for idx, item in enumerate(evidence, start=1):
                if hasattr(item, "text"):
                    doc_name = getattr(item, "document_name", getattr(item, "document_id", f"Document_{idx}"))
                    page = getattr(item, "page", None)
                    page_str = f", Page {page}" if page is not None else ""
                    parts.append(f"[{doc_name}{page_str}]: {item.text}")
                elif isinstance(item, Mapping):
                    text = item.get("text", str(item))
                    doc_name = item.get("document_name", item.get("document_id", f"Document_{idx}"))
                    parts.append(f"[{doc_name}]: {text}")
                elif isinstance(item, str):
                    parts.append(item.strip())
                else:
                    parts.append(str(item))
            return "\n\n".join(p for p in parts if p)
        return str(evidence).strip()

    def _call_llm(self, prompt: str, question: str, context_str: str) -> Any:
        if self.llm_client is None:
            raise ValueError("No LLM client configured on LLMSufficiencyAutoRater")

        if callable(self.llm_client):
            return self.llm_client(prompt)
        if hasattr(self.llm_client, "complete_json"):
            return self.llm_client.complete_json(
                task="sufficiency_judge",
                prompt=prompt,
                payload={"question": question, "context": context_str},
            )
        if hasattr(self.llm_client, "complete"):
            return self.llm_client.complete(prompt)
        if hasattr(self.llm_client, "generate"):
            return self.llm_client.generate(prompt)
        raise TypeError(f"Unsupported llm_client type: {type(self.llm_client).__name__}")

    def _invoke_fallback(self, question: str, evidence: Any) -> StructuredSufficiencyResult:
        if self.fallback_judge is None:
            raise RuntimeError("Fallback requested but no fallback judge is configured.")

        if hasattr(self.fallback_judge, "evaluate"):
            chunks = self._ensure_chunks(evidence)
            return self.fallback_judge.evaluate(question, chunks)

        if hasattr(self.fallback_judge, "assess"):
            from agentic_rag.contracts import (
                Claim,
                Corpus,
                DraftAnswer,
                FactPriority,
                RequiredFact,
                RetrievalPlan,
                Route,
                Snippet,
            )
            from agentic_rag.general_planner import GeneralPlanner, _meaningful_terms

            chunks = self._ensure_chunks(evidence)
            unique_doc_ids = sorted({getattr(c, "document_id", "doc") for c in chunks}) or ["doc"]
            corpora = tuple(Corpus(id=doc_id, description=f"Document {doc_id}") for doc_id in unique_doc_ids)

            planner = GeneralPlanner()
            plan = planner.plan(question, corpora)
            if not plan.required_facts:
                fact = RequiredFact(
                    id="f1",
                    description=question,
                    priority=FactPriority.MUST,
                    metadata={"required_terms": _meaningful_terms(question)},
                )
                route = Route(
                    fact_id="f1",
                    candidate_corpus_ids=tuple(unique_doc_ids),
                    reason="Default route for question",
                )
                plan = RetrievalPlan(question=question, required_facts=(fact,), routes=(route,))

            snippets = tuple(
                Snippet(
                    id=getattr(c, "chunk_id", f"c_{idx}"),
                    corpus_id=getattr(c, "document_id", "doc"),
                    document_id=getattr(c, "document_id", "doc"),
                    text=getattr(c, "text", str(c)),
                )
                for idx, c in enumerate(chunks)
            )
            claims = tuple(
                Claim(text=getattr(c, "text", str(c)), snippet_ids=(getattr(c, "chunk_id", f"c_{idx}"),))
                for idx, c in enumerate(chunks)
            )
            draft = DraftAnswer(text="\n\n".join(getattr(c, "text", str(c)) for c in chunks), claims=claims)

            assessment = self.fallback_judge.assess(question, plan, snippets, draft)
            res = StructuredSufficiencyResult.from_context_assessment(assessment)
            res.validate()
            return res

        if callable(self.fallback_judge):
            res = self.fallback_judge(question, evidence)
            if isinstance(res, StructuredSufficiencyResult):
                return res
            if isinstance(res, Mapping):
                return StructuredSufficiencyResult.from_dict(res)
            if isinstance(res, str):
                return StructuredSufficiencyResult.from_json(res)

        raise TypeError(f"Unsupported fallback_judge type: {type(self.fallback_judge).__name__}")

    @staticmethod
    def _ensure_chunks(evidence: Any) -> list[Any]:
        if not evidence:
            return []
        if isinstance(evidence, str):
            from evidencefirst_rag.chunking import Chunk

            return [
                Chunk(
                    chunk_id="c_fallback_0",
                    document_id="doc_fallback",
                    document_name="doc_fallback",
                    page=1,
                    section="General",
                    text=evidence,
                    token_count=len(evidence.split()),
                    char_count=len(evidence),
                )
            ]
        if isinstance(evidence, (list, tuple)):
            chunks = []
            for idx, item in enumerate(evidence):
                if hasattr(item, "text"):
                    chunks.append(item)
                else:
                    from evidencefirst_rag.chunking import Chunk

                    text = str(item)
                    chunks.append(
                        Chunk(
                            chunk_id=f"c_fallback_{idx}",
                            document_id="doc_fallback",
                            document_name="doc_fallback",
                            page=1,
                            section="General",
                            text=text,
                            token_count=len(text.split()),
                            char_count=len(text),
                        )
                    )
            return chunks
        return []

    def evaluate(
        self,
        question: str,
        evidence: Sequence[Any] | str,
        required_facts: Sequence[str] | None = None,
    ) -> StructuredSufficiencyResult:
        """Evaluate sufficiency of retrieved evidence for the question."""
        context_str = self.format_evidence(evidence)
        if not context_str.strip():
            return StructuredSufficiencyResult(
                sufficient=False,
                probability=0.0,
                missing_information=[question],
                rationale="No evidence was provided.",
            )

        if self.llm_client is None:
            if self.fallback_judge is not None:
                return self._invoke_fallback(question, evidence)
            return StructuredSufficiencyResult(
                sufficient=False,
                probability=0.0,
                missing_information=["No LLM client configured"],
                rationale="No LLM client provided and no fallback judge configured.",
            )

        from agentic_rag.prompts.sufficiency_v1 import (
            build_sufficiency_prompt,
            build_sufficiency_user_prompt,
        )

        full_prompt = build_sufficiency_prompt(question, context_str, required_facts)
        if self.system_prompt:
            user_part = build_sufficiency_user_prompt(question, context_str, required_facts)
            full_prompt = f"{self.system_prompt.strip()}\n\n{'=' * 40}\n\n{user_part}"

        try:
            raw_output = self._call_llm(full_prompt, question, context_str)
        except Exception as exc:
            if self.fallback_judge is not None:
                return self._invoke_fallback(question, evidence)
            if not self.catch_exceptions:
                raise
            return StructuredSufficiencyResult(
                sufficient=False,
                probability=0.0,
                missing_information=[f"LLM call failed: {str(exc)}"],
                rationale=f"LLM invocation encountered an error: {str(exc)}",
            )

        # Parse LLM response
        try:
            if isinstance(raw_output, StructuredSufficiencyResult):
                res = raw_output
            elif isinstance(raw_output, Mapping):
                res = StructuredSufficiencyResult.from_dict(raw_output)
            elif isinstance(raw_output, str):
                res = StructuredSufficiencyResult.from_json(raw_output)
            else:
                raise ValueError(f"Unexpected output type from LLM: {type(raw_output).__name__}")
            res.evaluator_mode = getattr(self, "mode_description", "LLM AutoRater")
            res.validate()
            return res
        except Exception:
            if self.fallback_judge is not None:
                return self._invoke_fallback(question, evidence)
            if isinstance(raw_output, str):
                repaired = StructuredSufficiencyResult.from_json(raw_output)
                repaired.evaluator_mode = getattr(self, "mode_description", "LLM AutoRater")
                return repaired
            raise

    def evaluate_to_json(
        self,
        question: str,
        evidence: Sequence[Any] | str,
        required_facts: Sequence[str] | None = None,
        indent: int | None = None,
    ) -> str:
        """Evaluate sufficiency and return validated JSON string."""
        res = self.evaluate(question, evidence, required_facts=required_facts)
        return res.to_json(indent=indent)

    def evaluate_json(
        self,
        question: str,
        evidence: Sequence[Any] | str,
        required_facts: Sequence[str] | None = None,
        indent: int | None = None,
    ) -> str:
        """Alias for evaluate_to_json."""
        return self.evaluate_to_json(question, evidence, required_facts=required_facts, indent=indent)

    def evaluate_dict(
        self,
        question: str,
        evidence: Sequence[Any] | str,
        required_facts: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Evaluate sufficiency and return validated Python dictionary."""
        return self.evaluate(question, evidence, required_facts=required_facts).to_dict()

    def __call__(
        self,
        question: str,
        evidence: Sequence[Any] | str,
        required_facts: Sequence[str] | None = None,
    ) -> StructuredSufficiencyResult:
        """Callable invocation shortcut."""
        return self.evaluate(question, evidence, required_facts=required_facts)
