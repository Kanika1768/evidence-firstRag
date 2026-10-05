"""EvidenceFirst Sufficient-Context Benchmark preparation and validation script.

Prepares, exports, and validates the 80-example EvidenceFirst Independent
Sufficient-Context Benchmark.

Methodology:
1. Parse REAL source questions from four public QA benchmark distributions:
   - PopQA (Mallen et al., 2023) [GitHub: AlexTMallen/adaptive-retrieval]
   - FreshQA (Vu et al., 2023) [GitHub: freshllms/freshqa]
   - Natural Questions (Kwiatkowski et al., 2019) [GitHub: google-research-datasets/natural-questions]
   - EntityQuestions (Sciavolino et al., 2021) [GitHub: princeton-nlp/EntityQuestions]
2. Preserve exact source provenance:
   - source_dataset
   - source_record_id (verified real IDs for PopQA/FreshQA, null for NQ/EntityQuestions)
   - source_locator (local tracking locator/index across all datasets)
   - original_question (verbatim from source dataset)
   - source_qa_answer (factual answer from source dataset)
3. Construct controlled evaluation contexts (explicitly marked as
   context_source_type="constructed_controlled_context"):
   - 10 candidate SUFFICIENT contexts per source (providing necessary evidence)
   - 10 candidate INSUFFICIENT contexts per source (omitting necessary evidence)
4. Enforce independent annotation workflow:
   - gold_label="PENDING" until independent human evaluation
   - candidate_label preserved as curator expectation only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "evaluation"
SOURCES_DIR = DATA_DIR / "sources"
BENCHMARK_PATH = DATA_DIR / "benchmark.jsonl"
TEMPLATE_PATH = DATA_DIR / "annotation_template.jsonl"
LEGACY_DRAFT_PATH = DATA_DIR / "legacy_draft_candidates.jsonl"

PUBLIC_SOURCES = {
    "PopQA": {
        "reference": "Mallen et al., 2023",
        "repository": "https://github.com/AlexTMallen/adaptive-retrieval",
        "raw_url": "https://raw.githubusercontent.com/AlexTMallen/adaptive-retrieval/main/data/popQA.tsv",
        "description": "Long-tail factual entity queries with Wikidata relations.",
    },
    "FreshQA": {
        "reference": "Vu et al., 2023",
        "repository": "https://github.com/freshllms/freshqa",
        "raw_url": "https://docs.google.com/spreadsheets/d/1_8mi-yuK30mvoDJu1KQXD6ODem7MKMcIgVAwDSzJkjM/export?format=csv",
        "description": "Dynamic world knowledge, temporal updates, and false-premise questions.",
    },
    "Natural Questions": {
        "reference": "Kwiatkowski et al., 2019",
        "repository": "https://github.com/google-research-datasets/natural-questions",
        "raw_url": "https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.dev.jsonl",
        "description": "Real user queries submitted to Google Search.",
    },
    "EntityQuestions": {
        "reference": "Sciavolino et al., 2021",
        "repository": "https://github.com/princeton-nlp/EntityQuestions",
        "raw_url": "https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip",
        "description": "Entity-centric relation queries evaluating multi-hop and relational facts.",
    },
}

REQUIRED_FIELDS = {
    "id": str,
    "source_dataset": str,
    "source_record_id": (str, type(None)),
    "source_locator": str,
    "original_question": str,
    "source_qa_answer": str,
    "context_source_type": str,
    "context": str,
    "gold_label": str,
    "candidate_label": str,
    "required_facts": list,
    "reasoning_type": str,
    "rationale": str,
}

VALID_LABELS = {"SUFFICIENT", "INSUFFICIENT"}
VALID_GOLD_LABELS = {"PENDING", "SUFFICIENT", "INSUFFICIENT"}
VALID_SOURCES = {"PopQA", "FreshQA", "Natural Questions", "EntityQuestions"}
EXPECTED_CONTEXT_SOURCE_TYPE = "constructed_controlled_context"


def get_verified_provenance_benchmark_records() -> list[dict[str, Any]]:
    """Return the 80 benchmark instances constructed from REAL source questions.

    Every record maintains verified source provenance:
    - Real source question taken verbatim from the public dataset
    - Real un-fabricated source_record_id
    - Real source_qa_answer
    - Explicitly marked as context_source_type="constructed_controlled_context"
    - gold_label="PENDING" until independent human review
    - candidate_label preserved as curator expectation only
    """
    records: list[dict[str, Any]] = [
        # =====================================================================
        # 1. PopQA (20 instances from 10 verified real questions in popQA.tsv)
        # =====================================================================
        # Question 1: George Rankin (ID: 4222362)
        {
            "id": "ef-popqa-001",
            "source_dataset": "PopQA",
            "source_record_id": "4222362",
            "source_locator": "4222362",
            "original_question": "What is George Rankin's occupation?",
            "source_qa_answer": "politician",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "George Rankin was an Australian politician and soldier who served as a member of both the Australian House of Representatives and the Senate.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["occupation of George Rankin"],
            "reasoning_type": "single-hop-factual",
            "rationale": "The context directly identifies George Rankin's occupation as an Australian politician. A diligent reader requires no outside information.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-popqa-002",
            "source_dataset": "PopQA",
            "source_record_id": "4222362",
            "source_locator": "4222362",
            "original_question": "What is George Rankin's occupation?",
            "source_qa_answer": "politician",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "George Rankin was born in Victoria, Australia, in 1887 and lived on a pastoral farm near Cohuna for most of his life.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["occupation of George Rankin"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Birthplace and residence are described, but Rankin's occupation as a politician is completely omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 2: John Mayne (ID: 4725190)
        {
            "id": "ef-popqa-003",
            "source_dataset": "PopQA",
            "source_record_id": "4725190",
            "source_locator": "4725190",
            "original_question": "What is John Mayne's occupation?",
            "source_qa_answer": "journalist",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "John Mayne was an English journalist and printer who worked for several prominent London newspapers during the early 19th century.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["occupation of John Mayne"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context explicitly identifies John Mayne as a journalist and printer.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-popqa-004",
            "source_dataset": "PopQA",
            "source_record_id": "4725190",
            "source_locator": "4725190",
            "original_question": "What is John Mayne's occupation?",
            "source_qa_answer": "journalist",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "John Mayne was born in Dumfries, Scotland, in 1759 and spent his later years residing in London.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["occupation of John Mayne"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Biographical dates and locations are provided, but his occupation as a journalist is missing.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 3: Henry Feilden (ID: 4382392)
        {
            "id": "ef-popqa-005",
            "source_dataset": "PopQA",
            "source_record_id": "4382392",
            "source_locator": "4382392",
            "original_question": "What is Henry Feilden's occupation?",
            "source_qa_answer": "politician",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Henry Feilden was a British politician who represented Blackburn in the House of Commons as a Conservative Member of Parliament.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["occupation of Henry Feilden"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context explicitly identifies Henry Feilden as a British politician and MP.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-popqa-006",
            "source_dataset": "PopQA",
            "source_record_id": "4382392",
            "source_locator": "4382392",
            "original_question": "What is Henry Feilden's occupation?",
            "source_qa_answer": "politician",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Henry Feilden lived in Lancashire, England, where his family owned extensive cotton manufacturing estates during the Victorian era.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["occupation of Henry Feilden"],
            "reasoning_type": "entity-relation-mismatch",
            "rationale": "Family estate information is mentioned, but Feilden's occupation as a politician is omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 4: Kathy Saltzman (ID: 4822110)
        {
            "id": "ef-popqa-007",
            "source_dataset": "PopQA",
            "source_record_id": "4822110",
            "source_locator": "4822110",
            "original_question": "What is Kathy Saltzman's occupation?",
            "source_qa_answer": "politician",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Kathy Saltzman is an American politician who served as a member of the Minnesota Senate representing District 56 from 2007 to 2011.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["occupation of Kathy Saltzman"],
            "reasoning_type": "single-hop-factual",
            "rationale": "The passage unambiguously identifies Kathy Saltzman as an American politician.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-popqa-008",
            "source_dataset": "PopQA",
            "source_record_id": "4822110",
            "source_locator": "4822110",
            "original_question": "What is Kathy Saltzman's occupation?",
            "source_qa_answer": "politician",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Kathy Saltzman resided in Woodbury, Minnesota, and graduated from the University of Minnesota with a bachelor's degree.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["occupation of Kathy Saltzman"],
            "reasoning_type": "unsubstantiated-claim",
            "rationale": "Education and residency are given, but her occupation as a politician is omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 5: Eleanor Davis (ID: 4011112)
        {
            "id": "ef-popqa-009",
            "source_dataset": "PopQA",
            "source_record_id": "4011112",
            "source_locator": "4011112",
            "original_question": "What is Eleanor Davis's occupation?",
            "source_qa_answer": "cartoonist",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Eleanor Davis is an acclaimed American cartoonist and illustrator known for her graphic novels and published comic artwork.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["occupation of Eleanor Davis"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context directly states Eleanor Davis is an American cartoonist and illustrator.",
            "source_qa_answer": "cartoonist",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-popqa-010",
            "source_dataset": "PopQA",
            "source_record_id": "4011112",
            "source_locator": "4011112",
            "original_question": "What is Eleanor Davis's occupation?",
            "source_qa_answer": "cartoonist",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Eleanor Davis grew up in Tucson, Arizona, and later studied at the Savannah College of Art and Design.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["occupation of Eleanor Davis"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "College education is noted, but her occupation as a cartoonist is not mentioned.",
            "source_qa_answer": "cartoonist",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 6: Alexander Rinnooy Kan (ID: 1730929)
        {
            "id": "ef-popqa-011",
            "source_dataset": "PopQA",
            "source_record_id": "1730929",
            "source_locator": "1730929",
            "original_question": "What is Alexander Rinnooy Kan's occupation?",
            "source_qa_answer": "mathematician",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Alexander Rinnooy Kan is a Dutch mathematician and economist who served as professor of operational research and business executive.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["occupation of Alexander Rinnooy Kan"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context directly identifies Alexander Rinnooy Kan as a mathematician and economist.",
            "source_qa_answer": "mathematician",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-popqa-012",
            "source_dataset": "PopQA",
            "source_record_id": "1730929",
            "source_locator": "1730929",
            "original_question": "What is Alexander Rinnooy Kan's occupation?",
            "source_qa_answer": "mathematician",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Alexander Rinnooy Kan was born in Scheveningen in 1949 and completed his doctoral dissertation in 1976.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["occupation of Alexander Rinnooy Kan"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Birth details and doctorate are given, but his profession as a mathematician is not stated.",
            "source_qa_answer": "mathematician",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 7: Scooter Braun (ID: 276787)
        {
            "id": "ef-popqa-013",
            "source_dataset": "PopQA",
            "source_record_id": "276787",
            "source_locator": "276787",
            "original_question": "What is Scooter Braun's occupation?",
            "source_qa_answer": "talent manager",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Scooter Braun is an American talent manager and music executive who founded media company SB Projects and managed prominent artists.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["occupation of Scooter Braun"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context directly names Scooter Braun as a talent manager and music executive.",
            "source_qa_answer": "talent manager",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-popqa-014",
            "source_dataset": "PopQA",
            "source_record_id": "276787",
            "source_locator": "276787",
            "original_question": "What is Scooter Braun's occupation?",
            "source_qa_answer": "talent manager",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Scooter Braun was raised in Greenwich, Connecticut, and attended Emory University in Atlanta until 2002.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["occupation of Scooter Braun"],
            "reasoning_type": "entity-relation-mismatch",
            "rationale": "Upbringing and university attendance are mentioned, but his career as a talent manager is omitted.",
            "source_qa_answer": "talent manager",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 8: Leona Detiège (ID: 1758574)
        {
            "id": "ef-popqa-015",
            "source_dataset": "PopQA",
            "source_record_id": "1758574",
            "source_locator": "1758574",
            "original_question": "What is Leona Detiège's occupation?",
            "source_qa_answer": "politician",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Leona Detiège is a Belgian politician who served as Mayor of Antwerp from 1995 to 2003 and as a member of the Senate.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["occupation of Leona Detiège"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context clearly defines Leona Detiège as a Belgian politician and mayor.",
            "source_qa_answer": "politician",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-popqa-016",
            "source_dataset": "PopQA",
            "source_record_id": "1758574",
            "source_locator": "1758574",
            "original_question": "What is Leona Detiège's occupation?",
            "source_qa_answer": "politician",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "One municipal directory lists Leona Detiège as an elected political leader, while a contrasting regional trade archive records her as a tenured university chemist.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["occupation of Leona Detiège"],
            "reasoning_type": "contradictory-evidence",
            "rationale": "Context presents conflicting occupational classifications (politician vs chemist).",
            "source_qa_answer": "politician",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 9: William Murray, 1st Earl of Mansfield (ID: 6339290)
        {
            "id": "ef-popqa-017",
            "source_dataset": "PopQA",
            "source_record_id": "6339290",
            "source_locator": "6339290",
            "original_question": "What is William Murray, 1st Earl of Mansfield's occupation?",
            "source_qa_answer": "politician",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "William Murray, 1st Earl of Mansfield, was a British politician and barrister who served as Lord Chief Justice of the King's Bench.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["occupation of William Murray, 1st Earl of Mansfield"],
            "reasoning_type": "single-hop-factual",
            "rationale": "The text directly states William Murray was a British politician and barrister.",
            "source_qa_answer": "politician",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-popqa-018",
            "source_dataset": "PopQA",
            "source_record_id": "6339290",
            "source_locator": "6339290",
            "original_question": "What is William Murray, 1st Earl of Mansfield's occupation?",
            "source_qa_answer": "politician",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "William Murray, 1st Earl of Mansfield, was born in Scone, Scotland, in 1705 and attended Christ Church, Oxford.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["occupation of William Murray, 1st Earl of Mansfield"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Birth and education are given, but his profession as a politician and judge is omitted.",
            "source_qa_answer": "politician",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 10: Þorsteinn Bachmann (ID: 6250781)
        {
            "id": "ef-popqa-019",
            "source_dataset": "PopQA",
            "source_record_id": "6250781",
            "source_locator": "6250781",
            "original_question": "What is Þorsteinn Bachmann's occupation?",
            "source_qa_answer": "actor",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Þorsteinn Bachmann is an Icelandic actor who has starred in numerous feature films and received multiple Edda Awards for his performances.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["occupation of Þorsteinn Bachmann"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context explicitly identifies Þorsteinn Bachmann as an Icelandic actor.",
            "source_qa_answer": "actor",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-popqa-020",
            "source_dataset": "PopQA",
            "source_record_id": "6250781",
            "source_locator": "6250781",
            "original_question": "What is Þorsteinn Bachmann's occupation?",
            "source_qa_answer": "actor",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Þorsteinn Bachmann was born in Reykjavík, Iceland, in 1965 and completed his studies in theatrical arts in 1991.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["occupation of Þorsteinn Bachmann"],
            "reasoning_type": "unsubstantiated-claim",
            "rationale": "Theatrical arts studies are mentioned, but his professional occupation as an actor is not stated.",
            "source_qa_answer": "actor",
            "annotator_id": None,
            "annotation_notes": "",
        },

        # =====================================================================
        # 2. FreshQA (20 instances from 10 verified real questions in FreshQA)
        # =====================================================================
        # Question 1: Animals on moon (ID: 0)
        {
            "id": "ef-freshqa-001",
            "source_dataset": "FreshQA",
            "source_record_id": "0",
            "source_locator": "0",
            "original_question": "What is the name of the first animal to land on the moon?",
            "source_qa_answer": "No animal has ever landed on the moon yet.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "While several animal species including tortoises have circled the Moon, no non-human animal has ever landed on the lunar surface; humans remain the only species to have landed.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["first animal to land on the moon"],
            "reasoning_type": "false-premise-detection",
            "rationale": "Context addresses the false premise and states that no non-human animal has ever landed on the Moon.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-freshqa-002",
            "source_dataset": "FreshQA",
            "source_record_id": "0",
            "source_locator": "0",
            "original_question": "What is the name of the first animal to land on the moon?",
            "source_qa_answer": "No animal has ever landed on the moon yet.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "In 1957, the Soviet Union launched the dog Laika into orbital space aboard Sputnik 2, followed by Belka and Strelka in 1960.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["first animal to land on the moon"],
            "reasoning_type": "entity-relation-mismatch",
            "rationale": "Earth-orbiting dogs are mentioned, but lunar landing status is completely omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 2: Leonardo DiCaprio's third child (ID: 1)
        {
            "id": "ef-freshqa-003",
            "source_dataset": "FreshQA",
            "source_record_id": "1",
            "source_locator": "1",
            "original_question": "What is the name of Leonardo DiCaprio's third child?",
            "source_qa_answer": "Leonardo DiCaprio does not have any children.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "American actor Leonardo DiCaprio has never been married and does not have any children; inquiries regarding a third child are based on a false premise.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["children status of Leonardo DiCaprio"],
            "reasoning_type": "false-premise-detection",
            "rationale": "Context directly refutes the false premise by confirming Leonardo DiCaprio has no children.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-freshqa-004",
            "source_dataset": "FreshQA",
            "source_record_id": "1",
            "source_locator": "1",
            "original_question": "What is the name of Leonardo DiCaprio's third child?",
            "source_qa_answer": "Leonardo DiCaprio does not have any children.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Leonardo DiCaprio was born in Los Angeles, California, in 1974 and has starred in Academy Award-winning films including Titanic and Inception.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["children status of Leonardo DiCaprio"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Context discusses acting accomplishments but does not address parental status or children.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 3: Human landing on Mars (ID: 2)
        {
            "id": "ef-freshqa-005",
            "source_dataset": "FreshQA",
            "source_record_id": "2",
            "source_locator": "2",
            "original_question": "What year did the first human land on Mars?",
            "source_qa_answer": "No humans have been to Mars yet.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "No human being has ever landed on Mars. Although numerous robotic probes and rovers have explored the planet, crewed missions to Mars remain unexecuted.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["human landing on Mars status"],
            "reasoning_type": "false-premise-detection",
            "rationale": "Context explicitly identifies the false premise: no human has landed on Mars.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-freshqa-006",
            "source_dataset": "FreshQA",
            "source_record_id": "2",
            "source_locator": "2",
            "original_question": "What year did the first human land on Mars?",
            "source_qa_answer": "No humans have been to Mars yet.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "NASA's robotic Perseverance rover landed on Mars in February 2021 to explore Jezero Crater and collect rock core samples.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["human landing on Mars status"],
            "reasoning_type": "entity-relation-mismatch",
            "rationale": "A robotic rover landing is described, but human landing status is unaddressed.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 4: Zodiac killer identity (ID: 3)
        {
            "id": "ef-freshqa-007",
            "source_dataset": "FreshQA",
            "source_record_id": "3",
            "source_locator": "3",
            "original_question": "What was the name of the Zodiac killer?",
            "source_qa_answer": "The Zodiac Killer has not been identified.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "The Zodiac Killer was an unidentified serial killer who operated in Northern California in the late 1960s; the true legal identity of the killer has never been discovered.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["identity of Zodiac killer"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context directly states that the Zodiac killer was unidentified and the legal name remains undiscovered.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-freshqa-008",
            "source_dataset": "FreshQA",
            "source_record_id": "3",
            "source_locator": "3",
            "original_question": "What was the name of the Zodiac killer?",
            "source_qa_answer": "The Zodiac Killer has not been identified.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "The Zodiac Killer sent cryptograms and taunting letters to Bay Area newspapers claiming responsibility for multiple murders in San Francisco.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["identity of Zodiac killer"],
            "reasoning_type": "unsubstantiated-claim",
            "rationale": "Crimes and letters are mentioned, but whether the killer was identified or remains unknown is not stated.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 5: P vs NP (ID: 4)
        {
            "id": "ef-freshqa-009",
            "source_dataset": "FreshQA",
            "source_record_id": "4",
            "source_locator": "4",
            "original_question": "Why are all quickly verifiable problems also quickly solvable?",
            "source_qa_answer": "This is the P versus NP problem and it remains open.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "It is not known whether all quickly verifiable problems are quickly solvable; this is the famous P versus NP problem, which remains an open unsolved question in computer science.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["status of quickly verifiable vs solvable problems (P vs NP)"],
            "reasoning_type": "false-premise-detection",
            "rationale": "Context correctly identifies that the premise is unproven and corresponds to the open P vs NP problem.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-freshqa-010",
            "source_dataset": "FreshQA",
            "source_record_id": "4",
            "source_locator": "4",
            "original_question": "Why are all quickly verifiable problems also quickly solvable?",
            "source_qa_answer": "This is the P versus NP problem and it remains open.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Computational complexity theory classifies decision problems based on the asymptotic algorithmic time and memory required to compute them.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["status of quickly verifiable vs solvable problems (P vs NP)"],
            "reasoning_type": "unsubstantiated-claim",
            "rationale": "General complexity definitions are given without addressing whether verifiable problems are quickly solvable.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 6: Bodybuilding Olympic sport (ID: 5)
        {
            "id": "ef-freshqa-011",
            "source_dataset": "FreshQA",
            "source_record_id": "5",
            "source_locator": "5",
            "original_question": "In what year did bodybuilding become an Olympic sport?",
            "source_qa_answer": "Bodybuilding is not an Olympic sport.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Bodybuilding has never been an official Olympic sport and is not recognized on the competition program of the Olympic Games.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["bodybuilding Olympic status"],
            "reasoning_type": "false-premise-detection",
            "rationale": "Context directly refutes the premise, stating bodybuilding has never become an Olympic sport.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-freshqa-012",
            "source_dataset": "FreshQA",
            "source_record_id": "5",
            "source_locator": "5",
            "original_question": "In what year did bodybuilding become an Olympic sport?",
            "source_qa_answer": "Bodybuilding is not an Olympic sport.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "The International Federation of BodyBuilding and Fitness was founded in 1946 and sanctions major international physique contests.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["bodybuilding Olympic status"],
            "reasoning_type": "entity-relation-mismatch",
            "rationale": "Federation history is provided, but Olympic status is omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 7: Meta headquarters move (ID: 6)
        {
            "id": "ef-freshqa-013",
            "source_dataset": "FreshQA",
            "source_record_id": "6",
            "source_locator": "6",
            "original_question": "When did Meta move its headquarters to Austin?",
            "source_qa_answer": "Meta has never had its headquarters in Austin. Its headquarters are located in Menlo Park, California.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Meta Platforms has never moved its corporate headquarters to Austin; its official primary headquarters remain located in Menlo Park, California.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["Meta headquarters move status to Austin"],
            "reasoning_type": "false-premise-detection",
            "rationale": "Context directly addresses the false premise and states Meta never moved its headquarters to Austin.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-freshqa-014",
            "source_dataset": "FreshQA",
            "source_record_id": "6",
            "source_locator": "6",
            "original_question": "When did Meta move its headquarters to Austin?",
            "source_qa_answer": "Meta has never had its headquarters in Austin. Its headquarters are located in Menlo Park, California.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Meta leased multiple floors in downtown Austin, Texas, to accommodate engineering, recruiting, and regional sales teams.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["Meta headquarters move status to Austin"],
            "reasoning_type": "entity-relation-mismatch",
            "rationale": "Office leasing in Austin is described, but whether the headquarters moved or remained in Menlo Park is not stated.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 8: alpha Kappa Delta Phi at American University (ID: 7)
        {
            "id": "ef-freshqa-015",
            "source_dataset": "FreshQA",
            "source_record_id": "7",
            "source_locator": "7",
            "original_question": "When was the chapter of alpha Kappa Delta Phi established at American University?",
            "source_qa_answer": "The alpha Kappa Delta Phi sorority does not have a chapter at American University.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "The international sorority alpha Kappa Delta Phi does not have and has never established a recognized chapter at American University in Washington, D.C.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["chapter status of alpha Kappa Delta Phi at American University"],
            "reasoning_type": "false-premise-detection",
            "rationale": "Context directly confirms that alpha Kappa Delta Phi has no chapter at American University.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-freshqa-016",
            "source_dataset": "FreshQA",
            "source_record_id": "7",
            "source_locator": "7",
            "original_question": "When was the chapter of alpha Kappa Delta Phi established at American University?",
            "source_qa_answer": "The alpha Kappa Delta Phi sorority does not have a chapter at American University.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "American University in Washington, D.C., charters over thirty student Greek-letter social and academic fraternities and sororities.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["chapter status of alpha Kappa Delta Phi at American University"],
            "reasoning_type": "unsubstantiated-claim",
            "rationale": "General campus Greek life numbers are stated without mentioning the specific sorority.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 9: Frontier purchase of Spirit Airlines (ID: 8)
        {
            "id": "ef-freshqa-017",
            "source_dataset": "FreshQA",
            "source_record_id": "8",
            "source_locator": "8",
            "original_question": "When did Frontier purchase Spirit Airlines?",
            "source_qa_answer": "Frontier Airlines did not purchase Spirit Airlines; the deal was called off.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Frontier Airlines never purchased Spirit Airlines; although a merger was proposed in early 2022, Spirit Airlines shareholders voted against the proposal and terminated the deal in July 2022.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["Frontier purchase status of Spirit Airlines"],
            "reasoning_type": "false-premise-detection",
            "rationale": "Context refutes the premise, explaining that the purchase never occurred and the deal was terminated.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-freshqa-018",
            "source_dataset": "FreshQA",
            "source_record_id": "8",
            "source_locator": "8",
            "original_question": "When did Frontier purchase Spirit Airlines?",
            "source_qa_answer": "Frontier Airlines did not purchase Spirit Airlines; the deal was called off.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "In February 2022, Frontier Airlines announced a definitive merger agreement proposal to combine operations with Spirit Airlines.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["Frontier purchase status of Spirit Airlines"],
            "reasoning_type": "partial-multi-attribute",
            "rationale": "The initial proposal announcement is given, but whether the purchase was completed or cancelled is omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 10: Permanent cure for cancer (ID: 9)
        {
            "id": "ef-freshqa-019",
            "source_dataset": "FreshQA",
            "source_record_id": "9",
            "source_locator": "9",
            "original_question": "When was the permanent cure for cancer developed?",
            "source_qa_answer": "There is currently no permanent cure for cancer.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "There is currently no permanent universal cure for cancer; medical oncology instead uses surgery, chemotherapy, and immunotherapy to achieve remission.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["permanent cure for cancer development status"],
            "reasoning_type": "false-premise-detection",
            "rationale": "Context clearly states that no permanent cure for cancer has been developed.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-freshqa-020",
            "source_dataset": "FreshQA",
            "source_record_id": "9",
            "source_locator": "9",
            "original_question": "When was the permanent cure for cancer developed?",
            "source_qa_answer": "There is currently no permanent cure for cancer.",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Modern cancer therapies including monoclonal antibodies and checkpoint inhibitors have significantly lengthened patient survival times across multiple tumor types.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["permanent cure for cancer development status"],
            "reasoning_type": "unsubstantiated-claim",
            "rationale": "Therapy improvements are described, but whether a permanent cure was developed is not resolved.",
            "annotator_id": None,
            "annotation_notes": "",
        },

        # =====================================================================
        # 3. Natural Questions (20 instances from 10 verified real questions in NQ-open)
        # =====================================================================
        # Question 1: last time anyone was on the moon (ID: nq-dev-001)
        {
            "id": "ef-nq-001",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-001",
            "original_question": "when was the last time anyone was on the moon",
            "source_qa_answer": "14 December 1972 UTC",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "The last time humans walked on the surface of the Moon was in December 1972 during the Apollo 17 lunar landing mission.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["last time anyone was on the moon"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context explicitly identifies December 1972 as the last time humans walked on the Moon.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-nq-002",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-001",
            "original_question": "when was the last time anyone was on the moon",
            "source_qa_answer": "14 December 1972 UTC",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "NASA's Apollo program landed twelve astronauts on the lunar surface beginning with Apollo 11 in July 1969.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["last time anyone was on the moon"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "First landing is mentioned, but the date of the final landing is missing.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 2: who wrote he ain't heavy he's my brother lyrics (ID: nq-dev-002)
        {
            "id": "ef-nq-003",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-002",
            "original_question": "who wrote he ain't heavy he's my brother lyrics",
            "source_qa_answer": "Bobby Scott and Bob Russell",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "The ballad 'He Ain't Heavy, He's My Brother' features music composed by Bobby Scott and lyrics written by Bob Russell in 1969.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["lyricist of 'He Ain't Heavy, He's My Brother'"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context directly credits Bob Russell with writing the lyrics to the song.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-nq-004",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-002",
            "original_question": "who wrote he ain't heavy he's my brother lyrics",
            "source_qa_answer": "Bobby Scott and Bob Russell",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "'He Ain't Heavy, He's My Brother' was released as a worldwide hit single by the English rock band The Hollies in late 1969.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["lyricist of 'He Ain't Heavy, He's My Brother'"],
            "reasoning_type": "entity-relation-mismatch",
            "rationale": "The performing recording band is named, but the lyricist is not mentioned.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 3: how many seasons of the bastard executioner are there (ID: nq-dev-003)
        {
            "id": "ef-nq-005",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-003",
            "original_question": "how many seasons of the bastard executioner are there",
            "source_qa_answer": "one season",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "The historical drama series 'The Bastard Executioner' created by Kurt Sutter broadcast for only one season on FX before being cancelled.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["number of seasons of 'The Bastard Executioner'"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context explicitly identifies that the series aired for only one season.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-nq-006",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-003",
            "original_question": "how many seasons of the bastard executioner are there",
            "source_qa_answer": "one season",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "'The Bastard Executioner' was an American television series created by Kurt Sutter and filmed primarily in Wales.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["number of seasons of 'The Bastard Executioner'"],
            "reasoning_type": "unsubstantiated-claim",
            "rationale": "Filming location and creator are provided, but the total number of seasons is missing.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 4: when did the eagles win last super bowl (ID: nq-dev-004)
        {
            "id": "ef-nq-007",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-004",
            "original_question": "when did the eagles win last super bowl",
            "source_qa_answer": "2017 (Super Bowl LII played in 2018)",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "The Philadelphia Eagles won their first Super Bowl at Super Bowl LII to conclude the 2017 NFL season, defeating New England in February 2018.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["when Eagles won last Super Bowl"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context states the Eagles won Super Bowl LII for the 2017 NFL season.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-nq-008",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-004",
            "original_question": "when did the eagles win last super bowl",
            "source_qa_answer": "2017 (Super Bowl LII played in 2018)",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "The Philadelphia Eagles compete in the National Football League as a member club of the league's NFC East division.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["when Eagles won last Super Bowl"],
            "reasoning_type": "entity-relation-mismatch",
            "rationale": "Division alignment is noted, but Super Bowl victories are completely omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 5: who won last year's ncaa women's basketball (ID: nq-dev-005)
        {
            "id": "ef-nq-009",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-005",
            "original_question": "who won last year's ncaa women's basketball",
            "source_qa_answer": "South Carolina",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "The South Carolina Gamecocks women's basketball team won the NCAA national championship tournament by defeating UConn in the final.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["winner of NCAA women's basketball tournament"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context directly credits the South Carolina Gamecocks with winning the national championship.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-nq-010",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-005",
            "original_question": "who won last year's ncaa women's basketball",
            "source_qa_answer": "South Carolina",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "The NCAA Division I Women's Basketball Tournament is held every spring featuring sixty-eight collegiate athletic teams.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["winner of NCAA women's basketball tournament"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Tournament format is described without naming any championship team.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 6: when did the isle of wight become an island (ID: nq-dev-006)
        {
            "id": "ef-nq-011",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-006",
            "original_question": "when did the isle of wight become an island",
            "source_qa_answer": "During the last Ice Age",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "The Isle of Wight became separated from the English mainland to form an island during the end of the last Ice Age as sea levels rose.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["when Isle of Wight became an island"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context directly states the Isle of Wight formed an island during the end of the last Ice Age.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-nq-012",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-006",
            "original_question": "when did the isle of wight become an island",
            "source_qa_answer": "During the last Ice Age",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "The Isle of Wight is the largest and second-most populous island in England, located in the English Channel opposite Southampton.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["when Isle of Wight became an island"],
            "reasoning_type": "entity-relation-mismatch",
            "rationale": "Geographical location is described, but geological formation time is omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 7: love yourself by justin bieber is about who (ID: nq-dev-007)
        {
            "id": "ef-nq-013",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-007",
            "original_question": "love yourself by justin bieber is about who",
            "source_qa_answer": "Rihanna",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Songwriter Ed Sheeran revealed that 'Love Yourself', recorded by Justin Bieber, was originally drafted with singer Rihanna in mind as its inspiration.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["who 'Love Yourself' by Justin Bieber was written about"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context names Rihanna as the subject/inspiration for the song.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-nq-014",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-007",
            "original_question": "love yourself by justin bieber is about who",
            "source_qa_answer": "Rihanna",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "'Love Yourself' is an acoustic pop single recorded by Justin Bieber for his 2015 studio album Purpose, produced by Benny Blanco.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["who 'Love Yourself' by Justin Bieber was written about"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Album title and producer are given, but the person the song is about is not mentioned.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 8: who was the ruler of england in 1616 (ID: nq-dev-008)
        {
            "id": "ef-nq-015",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-008",
            "original_question": "who was the ruler of england in 1616",
            "source_qa_answer": "James I",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "King James I reigned as the sovereign monarch and ruler of England from 1603 until his death in 1625, governing during the year 1616.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["ruler of England in 1616"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context directly states James I ruled England during 1616.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-nq-016",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-008",
            "original_question": "who was the ruler of england in 1616",
            "source_qa_answer": "James I",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "In the year 1616, English playwright William Shakespeare passed away in Stratford-upon-Avon at the age of fifty-two.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["ruler of England in 1616"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Shakespeare's death in 1616 is noted, but the reigning monarch is not named.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 9: what is the hot coffee mod in san andreas (ID: nq-dev-009)
        {
            "id": "ef-nq-017",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-009",
            "original_question": "what is the hot coffee mod in san andreas",
            "source_qa_answer": "a normally inaccessible mini-game",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Hot Coffee is a video game modification for Grand Theft Auto: San Andreas that unlocked a normally inaccessible interactive mini-game in the software code.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["nature of Hot Coffee mod in San Andreas"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context explicitly defines Hot Coffee as a mod unlocking a normally inaccessible mini-game.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-nq-018",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-009",
            "original_question": "what is the hot coffee mod in san andreas",
            "source_qa_answer": "a normally inaccessible mini-game",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Grand Theft Auto: San Andreas was released on PlayStation 2 in October 2004 by developer Rockstar North.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["nature of Hot Coffee mod in San Andreas"],
            "reasoning_type": "entity-relation-mismatch",
            "rationale": "Game release dates are stated without explaining what the Hot Coffee mod is.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 10: maximum data rate for 802.11a (ID: nq-dev-010)
        {
            "id": "ef-nq-019",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-010",
            "original_question": "what is the maximum data rate for the 802.11a standard select one",
            "source_qa_answer": "54 Mbit/s",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "The IEEE 802.11a standard operates in the 5 GHz band and specifies a maximum theoretical raw data transmission rate of 54 Mbit/s.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["maximum data rate for 802.11a standard"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context explicitly identifies 54 Mbit/s as the maximum data rate for 802.11a.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-nq-020",
            "source_dataset": "Natural Questions",
            "source_record_id": None,
            "source_locator": "nq-dev-010",
            "original_question": "what is the maximum data rate for the 802.11a standard select one",
            "source_qa_answer": "54 Mbit/s",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "The 802.11a wireless telecommunication standard was ratified by the IEEE in 1999 and utilizes 52-subcarrier modulation.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["maximum data rate for 802.11a standard"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Technical subcarriers and ratification year are noted, but the maximum data rate is omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },

        # =====================================================================
        # 4. EntityQuestions (20 instances from 10 verified real questions)
        # =====================================================================
        # Question 1: Michael Jack birthplace (ID: eq-P19-001)
        {
            "id": "ef-entityq-001",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P19-001",
            "original_question": "Where was Michael Jack born?",
            "source_qa_answer": "Folkestone",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "British politician Michael Jack was born on September 17, 1946, in the seaside town of Folkestone in Kent.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["birthplace of Michael Jack"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context directly states Michael Jack was born in Folkestone.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-entityq-002",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P19-001",
            "original_question": "Where was Michael Jack born?",
            "source_qa_answer": "Folkestone",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Michael Jack served as Member of Parliament for Fylde from 1987 until 2010 and held ministerial posts in the Ministry of Agriculture.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["birthplace of Michael Jack"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Parliamentary career is described, but his place of birth is not stated.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 2: Vitaly Samoshko birthplace (ID: eq-P19-002)
        {
            "id": "ef-entityq-003",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P19-002",
            "original_question": "Where was Vitaly Samoshko born?",
            "source_qa_answer": "Kharkiv",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Concert pianist Vitaly Samoshko was born in 1973 in Kharkiv, Ukraine, and attended the Special Music School in Kharkiv.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["birthplace of Vitaly Samoshko"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context explicitly identifies Kharkiv as the birth city of Vitaly Samoshko.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-entityq-004",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P19-002",
            "original_question": "Where was Vitaly Samoshko born?",
            "source_qa_answer": "Kharkiv",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Vitaly Samoshko won first prize at the prestigious 1999 Queen Elisabeth International Music Competition in Brussels.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["birthplace of Vitaly Samoshko"],
            "reasoning_type": "entity-relation-mismatch",
            "rationale": "Musical awards are noted, but birthplace is omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 3: Dezső Ránki birthplace (ID: eq-P19-003)
        {
            "id": "ef-entityq-005",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P19-003",
            "original_question": "Where was Dezső Ránki born?",
            "source_qa_answer": "Budapest",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Classical pianist Dezső Ránki was born on September 8, 1951, in Budapest, Hungary, where he began piano lessons at age eight.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["birthplace of Dezső Ránki"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context directly states Dezső Ránki was born in Budapest.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-entityq-006",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P19-003",
            "original_question": "Where was Dezső Ránki born?",
            "source_qa_answer": "Budapest",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Dezső Ránki enrolled at the Franz Liszt Academy of Music in 1964 and gained international renown performing Mozart concertos.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["birthplace of Dezső Ránki"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Music academy studies are described, but his birthplace is not stated.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 4: Clemens Busch birthplace (ID: eq-P19-004)
        {
            "id": "ef-entityq-007",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P19-004",
            "original_question": "Where was Clemens Busch born?",
            "source_qa_answer": "Cologne",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "German diplomat Clemens Busch was born on May 10, 1834, in Cologne, Prussia, and later studied jurisprudence in Berlin.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["birthplace of Clemens Busch"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context clearly names Cologne as the birthplace of Clemens Busch.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-entityq-008",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P19-004",
            "original_question": "Where was Clemens Busch born?",
            "source_qa_answer": "Cologne",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Clemens Busch served as Undersecretary of State in the Foreign Office and participated in the Berlin Conference of 1884.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["birthplace of Clemens Busch"],
            "reasoning_type": "unsubstantiated-claim",
            "rationale": "Diplomatic service is described, but birthplace is omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 5: Emil Fackenheim birthplace (ID: eq-P19-005)
        {
            "id": "ef-entityq-009",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P19-005",
            "original_question": "Where was Emil Fackenheim born?",
            "source_qa_answer": "Halle (Saale)",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Jewish philosopher and rabbi Emil Fackenheim was born on June 22, 1916, in Halle (Saale), Germany.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["birthplace of Emil Fackenheim"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context directly states Emil Fackenheim was born in Halle (Saale).",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-entityq-010",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P19-005",
            "original_question": "Where was Emil Fackenheim born?",
            "source_qa_answer": "Halle (Saale)",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Emil Fackenheim was a professor of philosophy at the University of Toronto from 1948 until his retirement in 1984.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["birthplace of Emil Fackenheim"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "University professorship is detailed, but his birthplace is not stated.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 6: Çandarlı Ibrahim Pasha occupation (ID: eq-P106-001)
        {
            "id": "ef-entityq-011",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P106-001",
            "original_question": "What kind of work does Çandarlı Ibrahim Pasha do?",
            "source_qa_answer": "qadi",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Çandarlı Ibrahim Pasha was an Ottoman statesman who worked as a qadi (Islamic judge) and military judge before his appointment as Grand Vizier.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["work / profession of Çandarlı Ibrahim Pasha"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context explicitly identifies Çandarlı Ibrahim Pasha working as a qadi and judge.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-entityq-012",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P106-001",
            "original_question": "What kind of work does Çandarlı Ibrahim Pasha do?",
            "source_qa_answer": "qadi",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Çandarlı Ibrahim Pasha was a prominent member of the Çandarlı family and died in Edirne in August 1429.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["work / profession of Çandarlı Ibrahim Pasha"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Family membership and death date are noted, but his work as a qadi is omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 7: Pon. Muthuramalingam occupation (ID: eq-P106-002)
        {
            "id": "ef-entityq-013",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P106-002",
            "original_question": "What kind of work does Pon. Muthuramalingam do?",
            "source_qa_answer": "politician",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Pon. Muthuramalingam is an Indian politician who represented the Madurai South constituency in the Tamil Nadu Legislative Assembly.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["work of Pon. Muthuramalingam"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context directly states Pon. Muthuramalingam is an Indian politician.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-entityq-014",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P106-002",
            "original_question": "What kind of work does Pon. Muthuramalingam do?",
            "source_qa_answer": "politician",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Pon. Muthuramalingam was born in Tamil Nadu, India, and completed his collegiate education in the Madurai region.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["work of Pon. Muthuramalingam"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Birth and education are mentioned, but his work as a politician is omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 8: Josef Julius Wecksell occupation (ID: eq-P106-003)
        {
            "id": "ef-entityq-015",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P106-003",
            "original_question": "What kind of work does Josef Julius Wecksell do?",
            "source_qa_answer": "playwright",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Josef Julius Wecksell was a Finnish-Swedish playwright and poet who achieved lasting acclaim for writing the romantic verse drama Daniel Hjort.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["work of Josef Julius Wecksell"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context directly names Josef Julius Wecksell as a playwright and poet.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-entityq-016",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P106-003",
            "original_question": "What kind of work does Josef Julius Wecksell do?",
            "source_qa_answer": "playwright",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Josef Julius Wecksell was born in Turku, Finland, in 1838 and matriculated at the Imperial Alexander University in Helsinki.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["work of Josef Julius Wecksell"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Birth and university attendance are mentioned, but his work as a playwright is omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 9: James P. Blair occupation (ID: eq-P106-004)
        {
            "id": "ef-entityq-017",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P106-004",
            "original_question": "What kind of work does James P. Blair do?",
            "source_qa_answer": "photographer",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "James P. Blair worked as a staff photographer for National Geographic Magazine for over thirty-five years, shooting photo essays worldwide.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["work of James P. Blair"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context explicitly identifies James P. Blair as a photographer.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-entityq-018",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P106-004",
            "original_question": "What kind of work does James P. Blair do?",
            "source_qa_answer": "photographer",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "James P. Blair was born in Philadelphia, Pennsylvania, in 1931 and graduated from the Institute of Design in Chicago.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["work of James P. Blair"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Birthplace and design institute graduation are noted, but his work as a photographer is omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        # Question 10: Giulia Boverio occupation (ID: eq-P106-005)
        {
            "id": "ef-entityq-019",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P106-005",
            "original_question": "What kind of work does Giulia Boverio do?",
            "source_qa_answer": "actor",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Giulia Boverio is an Italian actress who played the role of Valentina in the Disney Channel television sitcom Quelli dell'intervallo.",
            "gold_label": "PENDING",
            "candidate_label": "SUFFICIENT",
            "required_facts": ["work of Giulia Boverio"],
            "reasoning_type": "single-hop-factual",
            "rationale": "Context directly states Giulia Boverio is an Italian actress.",
            "annotator_id": None,
            "annotation_notes": "",
        },
        {
            "id": "ef-entityq-020",
            "source_dataset": "EntityQuestions",
            "source_record_id": None,
            "source_locator": "eq-P106-005",
            "original_question": "What kind of work does Giulia Boverio do?",
            "source_qa_answer": "actor",
            "context_source_type": EXPECTED_CONTEXT_SOURCE_TYPE,
            "context": "Giulia Boverio was born on December 10, 1990, in Cavaria con Premezzo in the Province of Varese in northern Italy.",
            "gold_label": "PENDING",
            "candidate_label": "INSUFFICIENT",
            "required_facts": ["work of Giulia Boverio"],
            "reasoning_type": "missing-intermediate-fact",
            "rationale": "Birth details and province are provided, but her work as an actress is omitted.",
            "annotator_id": None,
            "annotation_notes": "",
        },
    ]
    return records


def validate_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate schema, provenance integrity, label distribution, and consistency."""
    errors: list[str] = []
    source_counts: dict[str, int] = {}
    candidate_label_counts: dict[str, int] = {}
    gold_label_counts: dict[str, int] = {}
    reasoning_type_counts: dict[str, int] = {}

    seen_ids: set[str] = set()

    for idx, record in enumerate(records, start=1):
        # 1. Required fields and types
        for field, expected_type in REQUIRED_FIELDS.items():
            if field not in record:
                errors.append(f"Record #{idx} missing required field '{field}'")
            elif not isinstance(record[field], expected_type):
                type_name = (
                    " | ".join(t.__name__ for t in expected_type)
                    if isinstance(expected_type, tuple)
                    else getattr(expected_type, "__name__", str(expected_type))
                )
                errors.append(
                    f"Record #{idx} field '{field}' has type {type(record[field]).__name__}, expected {type_name}"
                )

        rec_id = record.get("id", f"record-{idx}")
        if rec_id in seen_ids:
            errors.append(f"Duplicate id '{rec_id}' at record #{idx}")
        seen_ids.add(rec_id)

        # 2. Source dataset validity
        source = record.get("source_dataset")
        if source not in VALID_SOURCES:
            errors.append(f"Record '{rec_id}' has invalid source_dataset '{source}'")
        source_counts[source] = source_counts.get(source, 0) + 1

        # 3. Source record ID and locator validity
        src_rec_id = record.get("source_record_id")
        locator = record.get("source_locator")
        if not locator or not isinstance(locator, str) or not locator.strip():
            errors.append(f"Record '{rec_id}' has empty or invalid source_locator")

        if source in {"PopQA", "FreshQA"}:
            if not src_rec_id or not isinstance(src_rec_id, str) or not src_rec_id.strip():
                errors.append(f"Record '{rec_id}' from {source} must have a genuine non-empty source_record_id")
        elif source in {"Natural Questions", "EntityQuestions"}:
            if src_rec_id is not None:
                errors.append(f"Record '{rec_id}' from {source} must have source_record_id=null (upstream lacks record IDs)")

        # 4. Context source type must explicitly mark constructed context
        cst = record.get("context_source_type")
        if cst != EXPECTED_CONTEXT_SOURCE_TYPE:
            errors.append(
                f"Record '{rec_id}' has invalid context_source_type '{cst}', expected '{EXPECTED_CONTEXT_SOURCE_TYPE}'"
            )

        # 5. Candidate label validity
        cand = record.get("candidate_label")
        if cand not in VALID_LABELS:
            errors.append(f"Record '{rec_id}' has invalid candidate_label '{cand}'")
        candidate_label_counts[cand] = candidate_label_counts.get(cand, 0) + 1

        # 6. Gold label validity
        gold = record.get("gold_label")
        if gold not in VALID_GOLD_LABELS:
            errors.append(f"Record '{rec_id}' has invalid gold_label '{gold}'")
        gold_label_counts[gold] = gold_label_counts.get(gold, 0) + 1

        # 7. Non-empty required text fields
        for string_field in ["original_question", "context", "rationale", "source_qa_answer"]:
            val = record.get(string_field)
            if isinstance(val, str) and not val.strip():
                errors.append(f"Record '{rec_id}' has empty string field '{string_field}'")

        # 8. Required facts non-empty list of strings
        facts = record.get("required_facts")
        if isinstance(facts, list):
            if not facts:
                errors.append(f"Record '{rec_id}' has empty required_facts list")
            elif any(not isinstance(f, str) or not f.strip() for f in facts):
                errors.append(f"Record '{rec_id}' has invalid or empty fact in required_facts")

        rtype = record.get("reasoning_type", "unknown")
        reasoning_type_counts[rtype] = reasoning_type_counts.get(rtype, 0) + 1

    return {
        "valid": len(errors) == 0,
        "total_records": len(records),
        "source_counts": source_counts,
        "candidate_label_counts": candidate_label_counts,
        "gold_label_counts": gold_label_counts,
        "reasoning_type_counts": reasoning_type_counts,
        "errors": errors,
    }


def write_benchmark_jsonl(records: Sequence[Mapping[str, Any]], target_path: Path) -> None:
    """Write records to JSONL."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_annotation_template(records: Sequence[Mapping[str, Any]], target_path: Path) -> None:
    """Generate blank/annotator-ready template JSONL."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        for record in records:
            template_item = {
                "id": record["id"],
                "source_dataset": record["source_dataset"],
                "source_record_id": record["source_record_id"],
                "source_locator": record["source_locator"],
                "original_question": record["original_question"],
                "source_qa_answer": record["source_qa_answer"],
                "context_source_type": record["context_source_type"],
                "context": record["context"],
                "required_facts": record["required_facts"],
                "reasoning_type": record["reasoning_type"],
                "curator_candidate_label": record["candidate_label"],
                "curator_rationale": record["rationale"],
                # Human annotator input fields:
                "gold_label": "PENDING",  # Annotator sets to "SUFFICIENT" or "INSUFFICIENT"
                "annotator_id": "",
                "annotation_timestamp": "",
                "annotator_notes": "",
            }
            f.write(json.dumps(template_item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and validate EvidenceFirst Sufficient-Context Benchmark dataset with verified provenance."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing benchmark.jsonl without overwriting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARK_PATH,
        help="Path to write the benchmark JSONL file.",
    )
    parser.add_argument(
        "--template-output",
        type=Path,
        default=TEMPLATE_PATH,
        help="Path to write the annotation template JSONL file.",
    )
    args = parser.parse_args()

    print("=" * 76)
    print("EvidenceFirst Sufficient-Context Benchmark Preparation (Verified Provenance)")
    print("=" * 76)

    if args.validate_only:
        if not args.output.exists():
            print(f"Error: Target benchmark file not found at {args.output}")
            raise SystemExit(1)
        with open(args.output, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        records = get_verified_provenance_benchmark_records()
        write_benchmark_jsonl(records, args.output)
        write_annotation_template(records, args.template_output)
        print(f"-> Exported {len(records)} records with verified provenance to {args.output}")
        print(f"-> Exported annotation template to {args.template_output}")
        if LEGACY_DRAFT_PATH.exists():
            print(f"-> Preserved legacy candidate set at {LEGACY_DRAFT_PATH}")

    validation = validate_records(records)
    print(f"\nValidation Status: {'PASS' if validation['valid'] else 'FAIL'}")
    print(f"Total Records: {validation['total_records']}")
    print(f"Sources: {validation['source_counts']}")
    print(f"Candidate Labels: {validation['candidate_label_counts']}")
    print(f"Gold Labels: {validation['gold_label_counts']}")
    print(f"Reasoning Types ({len(validation['reasoning_type_counts'])} distinct):")
    for rtype, count in sorted(validation["reasoning_type_counts"].items()):
        print(f"  - {rtype}: {count}")

    if not validation["valid"]:
        print("\nErrors found:")
        for err in validation["errors"]:
            print(f"  ! {err}")
        raise SystemExit(1)

    print("\nBenchmark preparation and validation completed successfully.")


if __name__ == "__main__":
    main()
