#!/usr/bin/env python3
"""Download the demo corpus listed in data/corpus/manifest.csv and verify SHA-256 checksums.

The PDFs are U.S. Government works (public domain) published by NIST. They are
not committed to git; this script restores the exact files the index was built from.

Usage:
  python scripts/download_corpus.py [--force]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "corpus" / "manifest.csv"
PDF_DIR = ROOT / "data" / "corpus" / "pdfs"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the EvidenceFirst demo corpus.")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist.")
    args = parser.parse_args()

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    failures = 0
    for row in rows:
        dest = PDF_DIR / row["filename"]
        if dest.exists() and not args.force and sha256(dest) == row["sha256"]:
            print(f"ok      {row['filename']}")
            continue
        request = urllib.request.Request(row["url"], headers={"User-Agent": "evidencefirst-rag-corpus/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            dest.write_bytes(response.read())
        if sha256(dest) != row["sha256"]:
            print(f"MISMATCH {row['filename']}: checksum differs from manifest (publisher may have revised it)")
            failures += 1
        else:
            print(f"fetched {row['filename']}")

    print(f"{len(rows) - failures}/{len(rows)} documents verified in {PDF_DIR}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
