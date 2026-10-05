"""Generates the blinded human-verification subset for the EvidenceFirst sufficient-context benchmark."""

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = ROOT / "data" / "evaluation" / "benchmark.jsonl"
SUBSET_PATH = ROOT / "data" / "evaluation" / "human_verification_subset.jsonl"
DEFAULT_SEED = 42


def generate_blinded_human_verification_subset(
    benchmark_path: Path = BENCHMARK_PATH,
    output_path: Path = SUBSET_PATH,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    with open(benchmark_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    target_datasets = ["PopQA", "FreshQA", "Natural Questions", "EntityQuestions"]
    blinded_records = []

    for ds in target_datasets:
        ds_records = [r for r in records if r["source_dataset"] == ds]
        # Group records by question while preserving benchmark encounter order
        q_order = []
        q_to_records = defaultdict(list)
        for r in ds_records:
            q = r["original_question"]
            if q not in q_to_records:
                q_order.append(q)
            q_to_records[q].append(r)

        if len(q_order) != 10:
            raise ValueError(f"Expected 10 distinct questions for {ds}, found {len(q_order)}")

        # Deterministic sampling of 5 questions per dataset using fixed seed
        rng = random.Random(seed)
        sampled_indices = sorted(rng.sample(range(len(q_order)), 5))

        for idx in sampled_indices:
            q = q_order[idx]
            variants = q_to_records[q]
            if len(variants) != 2:
                raise ValueError(f"Expected 2 variants for question '{q}', found {len(variants)}")
            for v in variants:
                blinded_records.append({
                    "id": v["id"],
                    "source_dataset": v["source_dataset"],
                    "original_question": v["original_question"],
                    "context": v["context"],
                    "human_decision": "",
                    "human_rationale": "",
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in blinded_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return blinded_records


if __name__ == "__main__":
    records = generate_blinded_human_verification_subset()
    print(f"Generated {len(records)} blinded records in {SUBSET_PATH}")
