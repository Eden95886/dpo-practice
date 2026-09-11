"""Stream English Wikipedia from Hugging Face without downloading the full 11.6GB dump.

Requires a logged-in Hugging Face account (hf auth login). Anonymous downloads
often return HTTP 403 on the large English subset.

Usage:
    python wikipedia/stream_wikipedia.py
    python wikipedia/stream_wikipedia.py --n 20
"""

from __future__ import annotations

import argparse

from datasets import load_dataset

DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.en"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="number of titles to print")
    parser.add_argument("--preview-chars", type=int, default=500)
    args = parser.parse_args()

    print(f"loading stream {DATASET} / {CONFIG} ...", flush=True)
    ds = load_dataset(DATASET, CONFIG, split="train", streaming=True)

    for i, ex in enumerate(ds):
        if i == 0:
            print("id:", ex.get("id"))
            print("url:", ex.get("url"))
            print("title:", ex.get("title"))
            text = ex.get("text") or ""
            print("text_chars:", len(text))
            print("--- text preview ---")
            print(text[: args.preview_chars])
            print("--- titles ---")
        print(f"{i:02d}  {ex.get('title')}")
        if i + 1 >= args.n:
            break


if __name__ == "__main__":
    main()
