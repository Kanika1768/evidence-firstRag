import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.adapters.llm import (  # noqa: E402
    SchemaValidationError,
    StructuredOutputRepairRequest,
    get_schema,
    parse_structured_output,
    parse_structured_output_with_repair,
    to_dataclass,
    to_mapping,
)
from agentic_rag.contracts import (  # noqa: E402
    AnswerabilityLabel,
    ConflictEvidence,
    ConflictingEvidenceGroup,
    ContextAssessment,
    GroundedAnswer,
    QueryRewriteResult,
    RetrievalPlan,
)


class LLMAdapterSchemaTests(unittest.TestCase):
    def test_registry_contains_structured_output_schemas(self):
        self.assertIs(get_schema("RetrievalPlan").target_type, RetrievalPlan)
        self.assertIs(get_schema("QueryRewriteResult").target_type, QueryRewriteResult)
        self.assertIs(get_schema("ContextAssessment").target_type, ContextAssessment)
        self.assertIs(get_schema("GroundedAnswer").target_type, GroundedAnswer)

    def test_converts_retrieval_plan_mapping_to_dataclass_and_back(self):
        raw_plan = {
            "question": "Who owns Alice's project?",
            "required_facts": [
                {"id": "person_project", "description": "Alice project", "priority": "must"},
                {"id": "project_owner", "description": "Project owner", "priority": "must"},
            ],
            "routes": [
                {
                    "fact_id": "person_project",
                    "candidate_corpus_ids": ["directory"],
                    "reason": "Assignments are in directory.",
                }
            ],
            "stop_conditions": ["Stop when owner is known."],
        }

        plan = to_dataclass("RetrievalPlan", raw_plan)

        self.assertIsInstance(plan, RetrievalPlan)
        self.assertEqual("person_project", plan.required_facts[0].id)
        self.assertEqual(("directory",), tuple(plan.routes[0].candidate_corpus_ids))
        self.assertEqual(raw_plan, to_mapping("RetrievalPlan", plan))

    def test_parses_structured_json_string_through_registry(self):
        result = parse_structured_output(
            "QueryRewriteResult",
            """
            {
              "subqueries": [
                {
                  "id": "q0-0",
                  "fact_id": "project_owner",
                  "query": "Project Zen owner",
                  "target_corpus_ids": ["projects"],
                  "reason": "Project records contain owners."
                }
              ]
            }
            """,
        )

        self.assertIsInstance(result, QueryRewriteResult)
        self.assertEqual("projects", result.subqueries[0].target_corpus_ids[0])

    def test_converts_query_rewrite_result_mapping(self):
        result = to_dataclass(
            "QueryRewriteResult",
            {
                "subqueries": [
                    {
                        "id": "q0-0",
                        "fact_id": "project_owner",
                        "query": "Project Zen owner",
                        "target_corpus_ids": ["projects"],
                        "reason": "Project records contain owners.",
                    }
                ]
            },
        )

        self.assertIsInstance(result, QueryRewriteResult)
        self.assertEqual("q0-0", result.subqueries[0].id)

    def test_converts_context_assessment_and_grounded_answer(self):
        assessment = to_dataclass(
            "ContextAssessment",
            {
                "status": "insufficient",
                "sufficiency_score": 0.5,
                "covered_facts": [{"fact_id": "person_project", "snippet_ids": ["s1"]}],
                "missing_facts": ["Project owner"],
                "unsupported_claims": [],
                "conflicts": [
                    {
                        "fact_id": "owner",
                        "groups": [
                            {"label": "nina", "snippet_ids": ["s1"], "value": "Nina"},
                            {"label": "omar", "snippet_ids": ["s2"], "value": "Omar"},
                        ],
                        "reason": "Two owner values were found.",
                    }
                ],
                "feedback_queries": [
                    {
                        "query": "Project owner",
                        "target_corpus_ids": ["projects"],
                        "reason": "Missing required fact.",
                    }
                ],
                "reason": "One fact is missing.",
                "answerability": "useful_but_incomplete",
            },
        )
        answer = to_dataclass(
            "GroundedAnswer",
            {
                "answer": "Partial evidence found.",
                "citations": [{"claim": "person_project", "snippet_ids": ["s1"]}],
                "status": "partial",
                "missing_facts": ["Project owner"],
                "sufficiency_score": 0.5,
                "conflicts": [
                    {
                        "fact_id": "owner",
                        "groups": [
                            {"label": "nina", "snippet_ids": ["s1"], "value": "Nina"},
                            {"label": "omar", "snippet_ids": ["s2"], "value": "Omar"},
                        ],
                        "reason": "Two owner values were found.",
                    }
                ],
            },
        )

        self.assertIsInstance(assessment, ContextAssessment)
        self.assertEqual(("Project owner",), tuple(assessment.missing_facts))
        self.assertEqual(AnswerabilityLabel.USEFUL_BUT_INCOMPLETE, assessment.answerability_label)
        self.assertEqual("useful_but_incomplete", to_mapping("ContextAssessment", assessment)["answerability"])
        self.assertEqual("s1", assessment.covered_facts[0].snippet_ids[0])
        self.assertIsInstance(assessment.conflicts[0], ConflictEvidence)
        self.assertIsInstance(assessment.conflicts[0].groups[0], ConflictingEvidenceGroup)
        self.assertEqual("omar", assessment.conflicts[0].groups[1].label)
        self.assertEqual("Nina", to_mapping("ContextAssessment", assessment)["conflicts"][0]["groups"][0]["value"])
        self.assertIsInstance(answer, GroundedAnswer)
        self.assertEqual("person_project", answer.citations[0].claim)
        self.assertEqual("s2", answer.conflicts[0].groups[1].snippet_ids[0])

    def test_missing_required_field_raises_schema_error(self):
        with self.assertRaises(SchemaValidationError) as error:
            to_dataclass(
                "GroundedAnswer",
                {
                    "answer": "No citations field.",
                    "status": "partial",
                    "missing_facts": [],
                    "sufficiency_score": 0.0,
                },
            )

        self.assertIn("citations", str(error.exception))

    def test_malformed_json_raises_schema_error(self):
        with self.assertRaises(SchemaValidationError) as error:
            parse_structured_output("GroundedAnswer", '{"answer": "missing brace"')

        self.assertIn("malformed JSON", str(error.exception))

    def test_wrong_enum_value_raises_schema_error(self):
        with self.assertRaises(SchemaValidationError) as error:
            to_dataclass(
                "GroundedAnswer",
                {
                    "answer": "Unsupported status.",
                    "citations": [],
                    "status": "complete",
                    "missing_facts": [],
                    "sufficiency_score": 1.0,
                },
            )

        self.assertIn("status", str(error.exception))
        self.assertIn("answered", str(error.exception))

    def test_wrong_field_type_raises_schema_error(self):
        with self.assertRaises(SchemaValidationError) as error:
            to_dataclass(
                "QueryRewriteResult",
                {
                    "subqueries": {
                        "id": "q0-0",
                        "fact_id": "project_owner",
                        "query": "Project Zen owner",
                        "target_corpus_ids": ["projects"],
                        "reason": "Project records contain owners.",
                    }
                },
            )

        self.assertIn("subqueries", str(error.exception))
        self.assertIn("array", str(error.exception))

    def test_one_shot_repair_succeeds_with_injected_callable(self):
        calls: list[StructuredOutputRepairRequest] = []

        def repair(request: StructuredOutputRepairRequest) -> str:
            calls.append(request)
            self.assertEqual("GroundedAnswer", request.schema_name)
            self.assertIn("missing required field(s): citations", request.errors[0])
            self.assertIn('"answer": "Needs repair."', request.malformed_output)
            self.assertIn("Schema name: GroundedAnswer", request.prompt)
            self.assertIn("Validation errors:", request.prompt)
            self.assertIn("Malformed output:", request.prompt)
            return """
            {
              "answer": "Repaired answer.",
              "citations": [],
              "status": "partial",
              "missing_facts": [],
              "sufficiency_score": 0.0
            }
            """

        answer = parse_structured_output_with_repair(
            "GroundedAnswer",
            """
            {
              "answer": "Needs repair.",
              "status": "partial",
              "missing_facts": [],
              "sufficiency_score": 0.0
            }
            """,
            repair=repair,
        )

        self.assertEqual("Repaired answer.", answer.answer)
        self.assertEqual(1, len(calls))

    def test_repair_is_not_called_for_valid_output(self):
        def repair(request: StructuredOutputRepairRequest) -> str:
            self.fail("repair callable should not be called for valid output")

        answer = parse_structured_output_with_repair(
            "GroundedAnswer",
            """
            {
              "answer": "Already valid.",
              "citations": [],
              "status": "partial",
              "missing_facts": [],
              "sufficiency_score": 0.0
            }
            """,
            repair=repair,
        )

        self.assertEqual("Already valid.", answer.answer)

    def test_one_shot_repair_failure_raises_second_error(self):
        calls: list[StructuredOutputRepairRequest] = []

        def repair(request: StructuredOutputRepairRequest) -> str:
            calls.append(request)
            return """
            {
              "answer": "Still invalid.",
              "citations": [],
              "status": "complete",
              "missing_facts": [],
              "sufficiency_score": 1.0
            }
            """

        with self.assertRaises(SchemaValidationError) as error:
            parse_structured_output_with_repair(
                "GroundedAnswer",
                '{"answer": "broken"',
                repair=repair,
            )

        self.assertEqual(1, len(calls))
        self.assertIn("status", str(error.exception))
        self.assertIn("answered", str(error.exception))


if __name__ == "__main__":
    unittest.main()
