"""One-time data prep for the Tinker autoresearch target.

Downloads roneneldan/TinyStories from HuggingFace and writes plain UTF-8 text to
``data/train.txt`` and ``data/val.txt``. Tokenization is deferred to ``train.py``
so the agent can swap base models (and therefore tokenizers) without re-running
this step.

Usage:
    python prepare.py                      # default: 50k train / 1k val stories
    python prepare.py --train 200000 --val 2000

Requirements:
    pip install datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset


DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=int, default=50_000)
    parser.add_argument("--val", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("roneneldan/TinyStories")
    train = ds["train"].shuffle(seed=args.seed).select(range(args.train))
    val = ds["validation"].shuffle(seed=args.seed).select(range(args.val))

    train_path = DATA_DIR / "train.txt"
    val_path = DATA_DIR / "val.txt"

    _write_stories(train_path, train)
    _write_stories(val_path, val)

    print(f"wrote {train_path}  ({train_path.stat().st_size:,} bytes, {len(train):,} stories)")
    print(f"wrote {val_path}    ({val_path.stat().st_size:,} bytes, {len(val):,} stories)")


def _write_stories(path: Path, dataset) -> None:
    sep = "\n\n<|endoftext|>\n\n"
    with path.open("w", encoding="utf-8") as f:
        for row in dataset:
            text = (row.get("text") or "").strip()
            if text:
                f.write(text)
                f.write(sep)


if __name__ == "__main__":
    main()
