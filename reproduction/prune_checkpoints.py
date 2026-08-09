#!/usr/bin/env python3
"""Keep one checkpoint from a timm training run and drop the redundant copies.

`timm.utils.CheckpointSaver` writes the same tensor state under three names
every epoch: `last.pth.tar`, `checkpoint-<epoch>.pth.tar`, and — whenever the
eval metric improves — `model_best.pth.tar`. On any filesystem that supports it
these are `os.link()` hardlinks to a single inode, so the extra names are
byte-identical to the file they duplicate and carry no additional information.

For a short, truncated run only the best checkpoint is worth keeping or
publishing. This script removes the duplicates at the end of the training step
so the run leaves exactly one file per distinct set of weights, which keeps the
recorded output set a set of genuinely distinct artifacts.

    python reproduction/prune_checkpoints.py \\
        --dir ./output/resnet18-fashion-mnist --keep model_best.pth.tar
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="timm --output/--experiment dir")
    ap.add_argument(
        "--keep",
        default="model_best.pth.tar",
        help="the one checkpoint filename to keep (default: model_best.pth.tar)",
    )
    args = ap.parse_args()

    out = Path(args.dir)
    keep = out / args.keep
    if not keep.is_file():
        raise SystemExit(f"error: checkpoint to keep does not exist: {keep}")

    keep_stat = keep.stat()
    removed = []
    for path in sorted(out.glob("*.pth.tar")):
        if path.resolve() == keep.resolve():
            continue
        st = path.stat()
        same = st.st_ino == keep_stat.st_ino and st.st_dev == keep_stat.st_dev
        os.remove(path)
        removed.append(f"{path.name}{' (hardlink of ' + args.keep + ')' if same else ''}")

    print(f"kept  {keep}  ({keep_stat.st_size} bytes)")
    print("removed " + (", ".join(removed) if removed else "nothing"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
