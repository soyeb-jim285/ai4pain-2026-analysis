"""Load both splits and write numpy+parquet caches so all other scripts are fast."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import CLASSES, SIGNALS, load_split  # noqa: E402


def main() -> None:
    for split in ("train", "validation"):
        t0 = time.time()
        tensor, meta = load_split(split)
        dt = time.time() - t0
        print(
            f"{split:>10}: tensor={tensor.shape} dtype={tensor.dtype} "
            f"meta={len(meta)} subjects={meta['subject'].nunique()} "
            f"(loaded in {dt:.1f}s)"
        )
        cls_counts = meta["class"].value_counts().reindex(CLASSES, fill_value=0)
        print(f"             class counts: {dict(cls_counts)}")
        print(f"             signals: {SIGNALS}")
    print("\nCache primed.")


if __name__ == "__main__":
    main()
