"""One-time, EXPLICIT download of the local FinBERT weights (Phase 18).

    uv run python scripts/download_finbert.py [--yes]

The scoring path (``options_system.sentiment.score_backfill`` / ``FinbertScorer``)
loads weights with ``local_files_only=True`` and will NEVER download anything — by
design (see docs/SENTIMENT.md). This script is the single deliberate exception: it
prints exactly what it will fetch, where it will go, and the approximate size, then
downloads via ``huggingface_hub`` and prints the resolved snapshot revision hash.

FinBERT (``ProsusAI/finbert``) is a free, open model; no account, key, or card.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODEL = "ProsusAI/finbert"
# Skip TF/Flax duplicates of the weights — only the PyTorch/tokenizer files are used.
IGNORE_PATTERNS = ["*.h5", "*.msgpack", "*.ot", "*.tflite"]
APPROX_SIZE = "~440 MB (PyTorch weights ~418 MB + tokenizer/config)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args(argv)

    from huggingface_hub import constants, snapshot_download

    dest = Path(constants.HF_HUB_CACHE)
    print("This will download, ONE TIME, the local FinBERT scoring weights:")
    print(f"  model:        {MODEL} (free/open; no account, no key, no card)")
    print(f"  destination:  {dest} (the local Hugging Face cache)")
    print(f"  approx. size: {APPROX_SIZE}")
    print(f"  skipped:      {IGNORE_PATTERNS} (TF/Flax duplicates)")
    print("The scoring path itself stays local_files_only=True and never downloads.")
    if not args.yes:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted — nothing downloaded.")
            return 1

    path = snapshot_download(MODEL, ignore_patterns=IGNORE_PATTERNS)
    revision = Path(path).name
    print(f"Done. snapshot: {path}")
    print(f"Resolved revision hash: {revision}")
    print("Scoring will stamp this revision as model_version_or_hash on every score.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
