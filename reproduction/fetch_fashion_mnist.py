#!/usr/bin/env python3
"""Download the Fashion-MNIST dataset used by this reproduction row.

Kept as its own pipeline stage (rather than a side effect of `train.py
--dataset-download`) so the dataset fetch is a recorded step with its own
inputs and outputs in the lineage graph, and so training and evaluation both
read the *same* on-disk copy.

Downloads from the public torchvision mirror; no credentials are required.
"""

from __future__ import annotations

import argparse

from torchvision.datasets import FashionMNIST


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="./data", help="download root")
    args = ap.parse_args()

    for split, is_train in (("train", True), ("validation", False)):
        ds = FashionMNIST(root=args.data_dir, train=is_train, download=True)
        print(f"fashion-mnist {split}: {len(ds)} examples -> {args.data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
