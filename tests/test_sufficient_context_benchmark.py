"""Unit tests for the EvidenceFirst Independent Sufficient-Context Benchmark."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

from prepare_sufficient_context_benchmark import (  # noqa: E402
    EXPECTED_CONTEXT_SOURCE_TYPE,
    LEGACY_DRAFT_PATH,
    REQUIRED_FIELDS,
    VALID_GOLD_LABELS,
    VALID_LABELS,
    VALID_SOURCES,
    get_verified_provenance_benchmark_records,
    validate_records,
)
from run_sufficiency_benchmark import (  # noqa: E402
    compute_binary_metrics,
    load_benchmark,
    run_benchmark,
)


class SufficientContextBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.benchmark_path = ROOT / "data" / "evaluation" / "benchmark.jsonl"
        self.template_path = ROOT / "data" / "evaluation" / "annotation_template.jsonl"
        self.annotations_path = ROOT / "data" / "evaluation" / "independent_llm_annotations.jsonl"
        self.human_subset_path = ROOT / "data" / "evaluation" / "human_verification_subset.jsonl"
        self.human_guide_path = ROOT / "data" / "evaluation" / "HUMAN_ANNOTATION_GUIDE.md"

    def test_benchmark_file_exists_and_loads_exact_count(self) -> None:
        self.assertTrue(self.benchmark_path.exists(), "benchmark.jsonl must exist")
        records = load_benchmark(self.benchmark_path)
        self.assertEqual(80, len(records), "Benchmark must contain exactly 80 records")

    def test_benchmark_source_distribution_balanced(self) -> None:
        records = load_benchmark(self.benchmark_path)
        source_counts: dict[str, int] = {}
        for r in records:
            source = r["source_dataset"]
            source_counts[source] = source_counts.get(source, 0) + 1

        expected = {
            "PopQA": 20,
            "FreshQA": 20,
            "Natural Questions": 20,
            "EntityQuestions": 20,
        }
        self.assertEqual(expected, source_counts, "Must have exactly 20 records per source dataset")

    def test_benchmark_schema_integrity(self) -> None:
        records = load_benchmark(self.benchmark_path)
        seen_ids: set[str] = set()

        for idx, record in enumerate(records, start=1):
            for field, expected_type in REQUIRED_FIELDS.items():
                self.assertIn(field, record, f"Record #{idx} missing field '{field}'")
                type_name = (
                    " | ".join(t.__name__ for t in expected_type)
                    if isinstance(expected_type, tuple)
                    else expected_type.__name__
                )
                self.assertIsInstance(
                    record[field],
                    expected_type,
                    f"Record #{idx} field '{field}' should be {type_name}",
                )

            # ID uniqueness
            rec_id = record["id"]
            self.assertNotIn(rec_id, seen_ids, f"Duplicate ID: {rec_id}")
            seen_ids.add(rec_id)

            # Value validation
            self.assertIn(record["source_dataset"], VALID_SOURCES)
            self.assertIn(record["candidate_label"], VALID_LABELS)
            self.assertIn(record["gold_label"], VALID_GOLD_LABELS)

            # Provenance fields
            if record["source_dataset"] in {"PopQA", "FreshQA"}:
                self.assertIsNotNone(record["source_record_id"], f"Record #{idx} should have upstream source_record_id")
                self.assertTrue(record["source_record_id"].strip())
            elif record["source_dataset"] in {"Natural Questions", "EntityQuestions"}:
                self.assertIsNone(
                    record["source_record_id"],
                    f"Record #{idx} must have source_record_id=null because upstream lacks IDs",
                )

            self.assertIn("source_locator", record, f"Record #{idx} missing source_locator")
            self.assertTrue(record["source_locator"].strip())
            self.assertEqual(EXPECTED_CONTEXT_SOURCE_TYPE, record["context_source_type"])

            # Non-empty strings
            self.assertTrue(record["original_question"].strip())
            self.assertTrue(record["context"].strip())
            self.assertTrue(record["rationale"].strip())
            self.assertTrue(record["source_qa_answer"].strip())

            # Non-empty facts list
            self.assertGreater(len(record["required_facts"]), 0)
            for fact in record["required_facts"]:
                self.assertIsInstance(fact, str)
                self.assertTrue(fact.strip())

    def test_legacy_draft_candidates_preserved(self) -> None:
        self.assertTrue(LEGACY_DRAFT_PATH.exists(), "Legacy candidate draft must be preserved")
        with open(LEGACY_DRAFT_PATH, "r", encoding="utf-8") as f:
            legacy = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(80, len(legacy), "Legacy candidate draft must contain 80 records")

    def test_gold_labels_are_pending_until_human_annotated(self) -> None:
        records = load_benchmark(self.benchmark_path)
        # All records in the unannotated benchmark must have gold_label='PENDING'
        pending_records = [r for r in records if r["gold_label"] == "PENDING"]
        self.assertEqual(
            80,
            len(pending_records),
            "Unannotated benchmark must maintain gold_label='PENDING' to prevent artificial gold claims",
        )

        # Candidate labels must be balanced
        candidate_labels = [r["candidate_label"] for r in records]
        self.assertEqual(40, candidate_labels.count("SUFFICIENT"))
        self.assertEqual(40, candidate_labels.count("INSUFFICIENT"))

    def test_annotation_template_structure(self) -> None:
        self.assertTrue(self.template_path.exists(), "annotation_template.jsonl must exist")
        with open(self.template_path, "r", encoding="utf-8") as f:
            template_records = [json.loads(line) for line in f if line.strip()]

        self.assertEqual(80, len(template_records))
        for r in template_records:
            self.assertIn("source_record_id", r)
            self.assertIn("source_locator", r)
            if r["source_dataset"] in {"PopQA", "FreshQA"}:
                self.assertIsNotNone(r["source_record_id"])
            else:
                self.assertIsNone(r["source_record_id"])
            self.assertTrue(r["source_locator"].strip())
            self.assertIn("context_source_type", r)
            self.assertIn("curator_candidate_label", r)
            self.assertIn("curator_rationale", r)
            self.assertIn("gold_label", r)
            self.assertIn("annotator_id", r)
            self.assertIn("annotation_timestamp", r)
            self.assertIn("annotator_notes", r)
            self.assertEqual("PENDING", r["gold_label"])

    def test_binary_metrics_computation(self) -> None:
        # Perfect agreement
        perfect = compute_binary_metrics(
            ["SUFFICIENT", "INSUFFICIENT", "SUFFICIENT"],
            ["SUFFICIENT", "INSUFFICIENT", "SUFFICIENT"],
        )
        self.assertEqual(1.0, perfect["accuracy"])
        self.assertEqual(1.0, perfect["precision"])
        self.assertEqual(1.0, perfect["recall"])
        self.assertEqual(1.0, perfect["f1"])
        self.assertEqual(2.0, perfect["tp"])
        self.assertEqual(1.0, perfect["tn"])
        self.assertEqual(0.0, perfect["fp"])
        self.assertEqual(0.0, perfect["fn"])

        # Known confusion matrix: TP=1, TN=1, FP=1, FN=1
        mixed = compute_binary_metrics(
            ["SUFFICIENT", "SUFFICIENT", "INSUFFICIENT", "INSUFFICIENT"],
            ["SUFFICIENT", "INSUFFICIENT", "SUFFICIENT", "INSUFFICIENT"],
        )
        self.assertEqual(0.5, mixed["accuracy"])
        self.assertEqual(0.5, mixed["precision"])
        self.assertEqual(0.5, mixed["recall"])
        self.assertEqual(0.5, mixed["f1"])
        self.assertEqual(1.0, mixed["tp"])
        self.assertEqual(1.0, mixed["tn"])
        self.assertEqual(1.0, mixed["fp"])
        self.assertEqual(1.0, mixed["fn"])

        # Zero-division safety (no positives predicted)
        zero_pred = compute_binary_metrics(
            ["SUFFICIENT", "SUFFICIENT"],
            ["INSUFFICIENT", "INSUFFICIENT"],
        )
        self.assertEqual(0.0, zero_pred["accuracy"])
        self.assertEqual(0.0, zero_pred["precision"])
        self.assertEqual(0.0, zero_pred["recall"])
        self.assertEqual(0.0, zero_pred["f1"])

        # Length mismatch raises error
        with self.assertRaises(ValueError):
            compute_binary_metrics(["SUFFICIENT"], ["SUFFICIENT", "INSUFFICIENT"])

    def test_validation_detects_malformed_records(self) -> None:
        # 1. Missing required field
        bad_record = {"id": "test-1", "source_dataset": "PopQA"}
        val = validate_records([bad_record])
        self.assertFalse(val["valid"])
        self.assertTrue(any("missing required field" in err for err in val["errors"]))

        # 2. Duplicate ID
        records = get_verified_provenance_benchmark_records()[:2]
        duped = [dict(records[0]), dict(records[0])]
        val_dup = validate_records(duped)
        self.assertFalse(val_dup["valid"])
        self.assertTrue(any("Duplicate id" in err for err in val_dup["errors"]))

        # 3. Invalid label
        invalid_lbl = dict(records[0])
        invalid_lbl["candidate_label"] = "MAYBE"
        val_lbl = validate_records([invalid_lbl])
        self.assertFalse(val_lbl["valid"])
        self.assertTrue(any("invalid candidate_label" in err for err in val_lbl["errors"]))

        # 4. NQ with non-null source_record_id
        nq_record = [r for r in get_verified_provenance_benchmark_records() if r["source_dataset"] == "Natural Questions"][0]
        bad_nq = dict(nq_record)
        bad_nq["source_record_id"] = "fake-upstream-id"
        val_nq = validate_records([bad_nq])
        self.assertFalse(val_nq["valid"])
        self.assertTrue(any("source_record_id=null" in err for err in val_nq["errors"]))

        # 5. Missing source_locator
        no_locator = dict(records[0])
        no_locator["source_locator"] = ""
        val_loc = validate_records([no_locator])
        self.assertFalse(val_loc["valid"])
        self.assertTrue(any("source_locator" in err for err in val_loc["errors"]))

    def test_load_benchmark_handles_malformed_file(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write('{"valid": true}\n')
            f.write("not valid json\n")
            temp_path = Path(f.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_benchmark(temp_path)
            self.assertIn("Malformed JSON", str(ctx.exception))
        finally:
            temp_path.unlink()

    def test_run_benchmark_execution_and_reporting(self) -> None:
        # Executes run_benchmark end-to-end and checks structured return
        report = run_benchmark(self.benchmark_path, mode="planner", verbose=False)
        self.assertEqual(80, report["total_records"])
        self.assertEqual(0, report["verified_gold_records"])
        self.assertEqual(80, report["pending_records"])
        self.assertIn("candidate_metrics", report)
        self.assertIn("dataset_breakdown", report)
        self.assertEqual(4, len(report["dataset_breakdown"]))

    def test_independent_annotations_file_exists_and_exact_count(self) -> None:
        """Verify that independent_llm_annotations.jsonl exists and has exactly 80 records."""
        self.assertTrue(
            self.annotations_path.exists(),
            "independent_llm_annotations.jsonl must exist",
        )
        with open(self.annotations_path, "r", encoding="utf-8") as f:
            ann_records = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(80, len(ann_records), "Must have exactly 80 annotation records")

    def test_independent_annotations_id_bijection_with_benchmark(self) -> None:
        """Verify exact 1:1 ID match between annotations and benchmark without duplicates, omissions, or extras."""
        bench_records = load_benchmark(self.benchmark_path)
        bench_ids = [r["id"] for r in bench_records]

        with open(self.annotations_path, "r", encoding="utf-8") as f:
            ann_records = [json.loads(line) for line in f if line.strip()]
        ann_ids = [r["id"] for r in ann_records]

        # No duplicate annotation IDs
        self.assertEqual(len(ann_ids), len(set(ann_ids)), "Duplicate annotation IDs found")

        # Explicit set comparison
        bench_id_set = set(bench_ids)
        ann_id_set = set(ann_ids)

        missing = bench_id_set - ann_id_set
        extra = ann_id_set - bench_id_set

        self.assertEqual(set(), missing, f"Missing annotations for benchmark IDs: {missing}")
        self.assertEqual(set(), extra, f"Unexpected extra annotations: {extra}")
        self.assertEqual(bench_id_set, ann_id_set, "Annotation IDs must perfectly match benchmark IDs")

    def test_independent_annotations_schema_and_isolation(self) -> None:
        """Verify required fields, allowed label values, annotator type, and curator label isolation."""
        with open(self.annotations_path, "r", encoding="utf-8") as f:
            ann_records = [json.loads(line) for line in f if line.strip()]

        required_keys = {
            "id",
            "independent_annotation",
            "missing_facts",
            "annotation_rationale",
            "annotator_type",
        }
        forbidden_keys = {"candidate_label", "curator_rationale", "rationale", "gold_label"}

        for idx, rec in enumerate(ann_records, start=1):
            # Check required keys
            for key in required_keys:
                self.assertIn(key, rec, f"Annotation #{idx} missing required key '{key}'")

            # Check annotator type
            self.assertEqual(
                "LLM_independent",
                rec["annotator_type"],
                f"Annotation #{idx} has invalid annotator_type",
            )

            # Check label validity
            self.assertIn(
                rec["independent_annotation"],
                {"SUFFICIENT", "INSUFFICIENT"},
                f"Annotation #{idx} has invalid label '{rec['independent_annotation']}'",
            )

            # Check non-empty strings where expected
            self.assertTrue(rec["id"].strip())
            self.assertTrue(rec["annotation_rationale"].strip())

            # Check isolation: must NOT contain candidate or gold fields
            for forbidden in forbidden_keys:
                self.assertNotIn(
                    forbidden,
                    rec,
                    f"Annotation #{idx} violates isolation by including '{forbidden}'",
                )

    def test_independent_annotations_label_distribution(self) -> None:
        """Verify that the independent annotation artifact contains 40 SUFFICIENT and 40 INSUFFICIENT records.

        NOTE: This is an annotation distribution sanity check on the curated dataset, NOT model performance.
        """
        with open(self.annotations_path, "r", encoding="utf-8") as f:
            ann_records = [json.loads(line) for line in f if line.strip()]

        labels = [r["independent_annotation"] for r in ann_records]
        suff_count = labels.count("SUFFICIENT")
        insuff_count = labels.count("INSUFFICIENT")

        self.assertEqual(40, suff_count, "Expected exactly 40 SUFFICIENT annotations")
        self.assertEqual(40, insuff_count, "Expected exactly 40 INSUFFICIENT annotations")

    def test_benchmark_invariants_preserved_during_annotation(self) -> None:
        """Verify that benchmark.jsonl remains unchanged: all gold_label=PENDING and candidate_label balanced."""
        bench_records = load_benchmark(self.benchmark_path)
        self.assertEqual(80, len(bench_records))

        for idx, r in enumerate(bench_records, start=1):
            self.assertEqual(
                "PENDING",
                r["gold_label"],
                f"Benchmark record #{idx} ({r['id']}) gold_label must remain PENDING",
            )

        cand_labels = [r["candidate_label"] for r in bench_records]
        self.assertEqual(40, cand_labels.count("SUFFICIENT"))
        self.assertEqual(40, cand_labels.count("INSUFFICIENT"))

    def test_human_verification_subset_file_exists_and_exact_count(self) -> None:
        """Verify that human_verification_subset.jsonl and HUMAN_ANNOTATION_GUIDE.md exist and have 40 records."""
        self.assertTrue(
            self.human_subset_path.exists(),
            "human_verification_subset.jsonl must exist",
        )
        self.assertTrue(
            self.human_guide_path.exists(),
            "HUMAN_ANNOTATION_GUIDE.md must exist",
        )
        with open(self.human_subset_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(40, len(records), "Human verification subset must contain exactly 40 records")

    def test_human_verification_subset_dataset_distribution_and_uniqueness(self) -> None:
        """Verify 10 records per dataset, 5 questions per dataset with both variants, and no duplicate IDs."""
        with open(self.human_subset_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        ids = [r["id"] for r in records]
        # No duplicate IDs
        self.assertEqual(len(ids), len(set(ids)), "Duplicate IDs found in human verification subset")

        # Every ID exists in benchmark.jsonl
        bench_records = load_benchmark(self.benchmark_path)
        bench_ids = {r["id"] for r in bench_records}
        self.assertTrue(set(ids).issubset(bench_ids), "All subset IDs must exist in benchmark.jsonl")

        # 10 records per dataset
        expected_datasets = ["PopQA", "FreshQA", "Natural Questions", "EntityQuestions"]
        for ds in expected_datasets:
            ds_records = [r for r in records if r["source_dataset"] == ds]
            self.assertEqual(10, len(ds_records), f"Dataset '{ds}' must have exactly 10 records")

            # Exactly 5 unique questions, each appearing with exactly 2 variants
            q_to_count: dict[str, int] = {}
            for r in ds_records:
                q = r["original_question"]
                q_to_count[q] = q_to_count.get(q, 0) + 1

            self.assertEqual(5, len(q_to_count), f"Dataset '{ds}' must have exactly 5 distinct questions")
            for q, count in q_to_count.items():
                self.assertEqual(2, count, f"Question '{q}' in '{ds}' must have exactly 2 variants")

    def test_human_verification_subset_schema_and_blinding(self) -> None:
        """Verify strict schema, initial blank decision/rationale, and complete absence of forbidden fields."""
        with open(self.human_subset_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        required_exact_keys = {
            "id",
            "source_dataset",
            "original_question",
            "context",
            "human_decision",
            "human_rationale",
        }
        forbidden_keys = {
            "candidate_label",
            "independent_annotation",
            "curator_rationale",
            "rationale",
            "gold_label",
            "required_facts",
            "reasoning_type",
            "source_qa_answer",
            "source_record_id",
            "source_locator",
            "context_source_type",
            "annotator_type",
        }

        for idx, rec in enumerate(records, start=1):
            # Exact keys check
            self.assertEqual(
                required_exact_keys,
                set(rec.keys()),
                f"Record #{idx} ({rec.get('id')}) keys do not match exact required blinded schema",
            )

            # human_decision must be initially blank
            self.assertEqual(
                "",
                rec["human_decision"],
                f"Record #{idx} ({rec['id']}) human_decision must be initially blank",
            )

            # human_rationale must be initially blank
            self.assertEqual(
                "",
                rec["human_rationale"],
                f"Record #{idx} ({rec['id']}) human_rationale must be initially blank",
            )

            # Required string fields must be non-empty
            self.assertTrue(rec["id"].strip())
            self.assertTrue(rec["source_dataset"].strip())
            self.assertTrue(rec["original_question"].strip())
            self.assertTrue(rec["context"].strip())

            # Forbidden fields must be strictly absent
            for forbidden in forbidden_keys:
                self.assertNotIn(
                    forbidden,
                    rec,
                    f"Record #{idx} ({rec['id']}) violates blinding by including '{forbidden}'",
                )


if __name__ == "__main__":
    unittest.main()
