import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_demo(script_name):
    script = ROOT / "examples" / script_name

    print("\n" + "=" * 72)
    print(f"RUNNING: {script_name}")
    print("=" * 72)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
    )

    if result.returncode != 0:
        print(f"\nDemo failed: {script_name}")
        sys.exit(result.returncode)


def main():
    print("=" * 72)
    print("EVIDENCEFIRST RAG — FINAL DEMONSTRATION")
    print("=" * 72)

    print(
        "\nThis demonstration shows:\n"
        "  1. Context sufficiency and selective answering\n"
        "  2. Missing-evidence recovery\n"
        "  3. Grounded citations and execution trace\n"
        "  4. Baseline vs iterative EvidenceFirst evaluation\n"
    )

    run_demo("in_memory_pipeline.py")
    run_demo("evaluation_demo.py")

    print("\n" + "=" * 72)
    print("EVIDENCEFIRST RAG — DEMONSTRATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
