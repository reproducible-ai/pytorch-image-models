# Reproduction row — `timm` ResNet-18 / Fashion-MNIST (truncated)

This directory holds the extra material for one row of the **Reproducible AI**
campaign, run against this fork of
[`huggingface/pytorch-image-models`](https://github.com/huggingface/pytorch-image-models)
("timm", Apache-2.0).

## What the row trains

A **ResNet-18 (11.2M params) trained from scratch on Fashion-MNIST at 28x28,
single channel, for 5 epochs**, batch size 256, SGD (lr 0.05, momentum 0.9,
weight decay 5e-4), cosine schedule, no warmup, single-process data loading
(`-j 0`), using the repository's own `train.py`.

**This is a deliberately TRUNCATED run and the result is not converged.** timm's
published recipes train for hundreds of epochs on ImageNet-scale data; five
epochs on Fashion-MNIST exists here to produce a *real* checkpoint and a *real*
held-out number in a few GPU-minutes, so that the reproduction record has
something honest to rebuild. Do not read the reported top-1 as a timm result.

## Pipeline

| step | command | produces |
|---|---|---|
| `fetch_dataset` | `python reproduction/fetch_fashion_mnist.py --data-dir ./data` | `data/FashionMNIST/raw/` |
| `train` | `python train.py --data-dir ./data --dataset torch/fashion_mnist ... --epochs 5` then `python reproduction/prune_checkpoints.py` | `output/resnet18-fashion-mnist/model_best.pth.tar`, `summary.csv`, `args.yaml` |
| `evaluate` | `python validate.py ... --checkpoint output/resnet18-fashion-mnist/model_best.pth.tar --results-file ./metrics/eval.json` | `metrics/eval.json` (held-out top-1 / top-5) |

The exact, recorded commands live in
[`.treqs/workflows/timm-resnet18-fashion-mnist.yaml`](../.treqs/workflows/timm-resnet18-fashion-mnist.yaml).
The evaluation is a **separate step** so the metric is computed and recorded in
its own right rather than being read out of the training log.

Fashion-MNIST is downloaded from the public torchvision mirror; **no credentials
are required** to run any step, and the whole download is about 30 MB.

## Notes on the fork, and two things worth knowing about upstream

* **`timm/data/loader.py` carries a one-line fix** (the only upstream source
  change in this fork): `persistent_workers` is now gated on `num_workers > 0`.
  Without it the documented `-j 0` / `--workers 0` (single-process data loading)
  raises `ValueError: persistent_workers option needs num_workers > 0` before
  the first batch. This row uses `-j 0` so the recorded environment is the
  environment the workload actually ran in, with no DataLoader worker
  subprocesses in the way.
* **`--in-chans 1` needs an explicit `--mean`/`--std`.**
  `timm.data.config.resolve_data_config` sets `input_size` from `in_chans` but
  leaves `mean`/`std` at the 3-channel ImageNet defaults unless they are given
  explicitly, so a single-channel run dies on the first batch with
  `RuntimeError: output with shape [1, 28, 28] doesn't match the broadcast
  shape [3, 28, 28]`. The commands here pass `--mean 0.2860 --std 0.3530`
  (Fashion-MNIST statistics) rather than patching upstream.
* **The training step prunes its own duplicate checkpoints.**
  `timm.utils.CheckpointSaver` writes each epoch's state under three names —
  `last.pth.tar`, `checkpoint-<epoch>.pth.tar` and `model_best.pth.tar` — as
  `os.link()` hardlinks to one inode, so two of the three carry no information
  the third does not. `reproduction/prune_checkpoints.py` runs at the end of the
  training step and keeps only `model_best.pth.tar`, so the run leaves one file
  per distinct set of weights.
* Nothing is installed with `pip install -e .`; `train.py` and `validate.py` are
  run from the repository root and import the checked-out `timm/` package
  directly.
* Upstream's optional Weights & Biases logging (`--log-wandb`) is **not**
  enabled in this row, so no experiment-tracker link is recorded.
