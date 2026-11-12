"""Human Validation Infrastructure for EvidenceFirst RAG Sufficiency Benchmark.

Implements reproducible workflows for human validation:
1. Exporting blinded reviewer-friendly packages (CSV and JSONL).
2. Validating reviewer submissions against the Diligent Reader schema.
3. Calculating inter-annotator agreement (Percent Agreement, Cohen's Kappa, category breakdown).
4. Adjudication workflow promoting verified consensus to gold labels while preserving candidate labels.
5. Reporting human validation progress and pending status.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

VALID_SUFFICIENCY_LABELS = frozenset({"SUFFICIENT", "INSUFFICIENT"})


@dataclass(frozen=True)
class ReviewerAnnotation:
    """A single human review record."""

    id: str
    decision: str
    rationale: str
    annotator_id: str
    timestamp: str = ""

    def validate(self) -> None:
        if not self.id or not str(self.id).strip():
            raise ValueError("Annotation missing record ID")
        norm_decision = self.decision.strip().upper()
        if norm_decision not in VALID_SUFFICIENCY_LABELS:
            raise ValueError(
                f"Record {self.id}: invalid decision '{self.decision}'. "
                f"Must be one of {sorted(VALID_SUFFICIENCY_LABELS)}"
            )
        if not self.annotator_id or not str(self.annotator_id).strip():
            raise ValueError(f"Record {self.id}: annotator_id must not be empty")
        if not self.rationale or not str(self.rationale).strip():
            raise ValueError(f"Record {self.id}: rationale must not be empty")


@dataclass(frozen=True)
class InterAnnotatorAgreementReport:
    """Inter-annotator agreement metrics between two independent reviewers."""

    total_records: int
    agreed_count: int
    disagreed_count: int
    observed_agreement: float
    expected_agreement: float
    cohens_kappa: float
    confusion_matrix: dict[str, dict[str, int]]
    category_agreement: dict[str, dict[str, Any]]
    discrepancy_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "agreed_count": self.agreed_count,
            "disagreed_count": self.disagreed_count,
            "observed_agreement_pct": round(self.observed_agreement * 100.0, 2),
            "expected_agreement_pct": round(self.expected_agreement * 100.0, 2),
            "cohens_kappa": round(self.cohens_kappa, 4),
            "confusion_matrix": self.confusion_matrix,
            "category_agreement": self.category_agreement,
            "discrepancy_ids": self.discrepancy_ids,
        }


def export_blinded_annotation_package(
    benchmark_path: Path,
    output_dir: Path,
    package_name: str = "human_validation_package",
) -> tuple[Path, Path]:
    """Export benchmark instances into reviewer-friendly blinded CSV and JSONL formats.

    Blinding guarantee:
    - candidate_label, silver_annotation, curator_rationale, and model confidence are strictly omitted.
    - Exposes only task metadata, question, and context.
    - Adds empty fields for human_decision, human_rationale, annotator_id, and timestamp.
    """
    benchmark_path = Path(benchmark_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    jsonl_path = output_dir / f"{package_name}.jsonl"
    csv_path = output_dir / f"{package_name}.csv"

    blinded_records: list[dict[str, Any]] = []
    for r in records:
        blinded_records.append(
            {
                "id": r["id"],
                "source_dataset": r.get("source_dataset", ""),
                "reasoning_type": r.get("reasoning_type", ""),
                "original_question": r.get("original_question", ""),
                "context": r.get("context", ""),
                "human_decision": "",
                "human_rationale": "",
                "annotator_id": "",
                "annotation_timestamp": "",
            }
        )

    # Write JSONL
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for br in blinded_records:
            f.write(json.dumps(br, ensure_ascii=False) + "\n")

    # Write CSV
    fieldnames = [
        "id",
        "source_dataset",
        "reasoning_type",
        "original_question",
        "context",
        "human_decision",
        "human_rationale",
        "annotator_id",
        "annotation_timestamp",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(blinded_records)

    return jsonl_path, csv_path


def load_and_validate_reviews(path: Path) -> dict[str, ReviewerAnnotation]:
    """Load reviews from JSONL or CSV and validate schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Review file not found: {path}")

    reviews: dict[str, ReviewerAnnotation] = {}
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec_id = row.get("id") or row.get("record_id") or ""
                decision = (row.get("human_decision") or row.get("decision") or "").strip().upper()
                rationale = (row.get("human_rationale") or row.get("rationale") or "").strip()
                annotator_id = (row.get("annotator_id") or "").strip()
                ann = ReviewerAnnotation(
                    id=rec_id,
                    decision=decision,
                    rationale=rationale,
                    annotator_id=annotator_id,
                    timestamp=row.get("annotation_timestamp", ""),
                )
                ann.validate()
                reviews[ann.id] = ann
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rec_id = d.get("id") or d.get("record_id") or ""
                decision = (d.get("human_decision") or d.get("decision") or "").strip().upper()
                rationale = (d.get("human_rationale") or d.get("rationale") or "").strip()
                annotator_id = (d.get("annotator_id") or "").strip()
                ann = ReviewerAnnotation(
                    id=rec_id,
                    decision=decision,
                    rationale=rationale,
                    annotator_id=annotator_id,
                    timestamp=d.get("annotation_timestamp", ""),
                )
                ann.validate()
                reviews[ann.id] = ann

    return reviews


def calculate_inter_annotator_agreement(
    reviews_a: Mapping[str, ReviewerAnnotation],
    reviews_b: Mapping[str, ReviewerAnnotation],
    benchmark_records: Sequence[Mapping[str, Any]] | None = None,
) -> InterAnnotatorAgreementReport:
    """Compute Cohen's Kappa and percentage agreement between two independent reviewers."""
    common_ids = sorted(set(reviews_a.keys()) & set(reviews_b.keys()))
    if not common_ids:
        raise ValueError("No common record IDs between the two review sets")

    n = len(common_ids)
    labels = sorted(list(VALID_SUFFICIENCY_LABELS))
    # Confusion matrix: rows = A, cols = B
    matrix: dict[str, dict[str, int]] = {l1: {l2: 0 for l2 in labels} for l1 in labels}

    agreed = 0
    discrepancies: list[str] = []

    for rid in common_ids:
        dec_a = reviews_a[rid].decision.upper()
        dec_b = reviews_b[rid].decision.upper()
        matrix[dec_a][dec_b] += 1
        if dec_a == dec_b:
            agreed += 1
        else:
            discrepancies.append(rid)

    p_o = agreed / n if n > 0 else 0.0

    # Marginals for chance agreement (Cohen's Kappa)
    p_e = 0.0
    for label in labels:
        p_a = sum(matrix[label][other] for other in labels) / n
        p_b = sum(matrix[other][label] for other in labels) / n
        p_e += p_a * p_b

    kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) > 0.0 else 1.0

    # Breakdown by category if benchmark records supplied
    cat_agreement: dict[str, dict[str, Any]] = {}
    if benchmark_records:
        rec_by_id = {r["id"]: r for r in benchmark_records}
        cat_counts: dict[str, list[bool]] = {}
        for rid in common_ids:
            cat = rec_by_id.get(rid, {}).get("reasoning_type", "unknown")
            is_match = (reviews_a[rid].decision.upper() == reviews_b[rid].decision.upper())
            cat_counts.setdefault(cat, []).append(is_match)
        for cat, matches in sorted(cat_counts.items()):
            c_tot = len(matches)
            c_agr = sum(1 for m in matches if m)
            cat_agreement[cat] = {
                "total": c_tot,
                "agreed": c_agr,
                "agreement_pct": round((c_agr / c_tot) * 100.0, 2) if c_tot else 0.0,
            }

    return InterAnnotatorAgreementReport(
        total_records=n,
        agreed_count=agreed,
        disagreed_count=len(discrepancies),
        observed_agreement=p_o,
        expected_agreement=p_e,
        cohens_kappa=kappa,
        confusion_matrix=matrix,
        category_agreement=cat_agreement,
        discrepancy_ids=discrepancies,
    )


def adjudicate_and_promote_gold(
    benchmark_path: Path,
    adjudicated_reviews: Mapping[str, str],
    output_path: Path | None = None,
) -> int:
    """Update benchmark records with verified gold labels from human adjudication.

    Preserves candidate_label and all provenance fields intact.
    Returns the count of promoted gold records.
    """
    benchmark_path = Path(benchmark_path)
    output_path = Path(output_path or benchmark_path)

    records: list[dict[str, Any]] = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    promoted_count = 0
    for r in records:
        rid = r["id"]
        if rid in adjudicated_reviews:
            decision = adjudicated_reviews[rid].strip().upper()
            if decision in VALID_SUFFICIENCY_LABELS:
                r["gold_label"] = decision
                r["annotator_id"] = "HUMAN_ADJUDICATED"
                promoted_count += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return promoted_count


def generate_validation_status_summary(benchmark_path: Path) -> dict[str, Any]:
    """Inspect gold validation progress on the benchmark without modifying files."""
    benchmark_path = Path(benchmark_path)
    records: list[dict[str, Any]] = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    total = len(records)
    pending = sum(1 for r in records if r.get("gold_label") == "PENDING")
    annotated = total - pending

    gold_counts: dict[str, int] = {}
    for r in records:
        lbl = r.get("gold_label", "PENDING")
        gold_counts[lbl] = gold_counts.get(lbl, 0) + 1

    return {
        "benchmark_file": str(benchmark_path),
        "total_instances": total,
        "verified_human_gold_count": annotated,
        "pending_human_validation_count": pending,
        "validation_completion_pct": round((annotated / total) * 100.0, 2) if total else 0.0,
        "gold_label_distribution": gold_counts,
        "status": "PENDING HUMAN REVIEW" if pending > 0 else "FULLY VALIDATED",
    }
