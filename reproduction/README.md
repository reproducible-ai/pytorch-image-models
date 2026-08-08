# Reproduction row — `timm` ResNet-18 / CIFAR-10 (truncated)

This directory holds the extra material for one row of the **Reproducible AI**
campaign, run against this fork of
[`huggingface/pytorch-image-models`](https://github.com/huggingface/pytorch-image-models)
("timm", Apache-2.0).

## What the row trains

A **ResNet-18 (11.2M params) trained from scratch on CIFAR-10 at 32x32 for 5
epochs**, batch size 256, SGD (lr 0.05, momentum 0.9, weight decay 5e-4), cosine
schedule, no warmup, single-process data loading (`-j 0`), using the
repository's own `train.py`.

**This is a deliberately TRUNCATED run and the result is not converged.** timm's
published recipes train for hundreds of epochs on ImageNet-scale data; five
epochs on CIFAR-10 exists here to produce a *real* checkpoint and a *real*
held-out number in a few GPU-minutes, so that the reproduction record has
something honest to rebuild. Do not read the reported top-1 as a timm result.

## Pipeline

| step | command | produces |
|---|---|---|
| `fetch_dataset` | `python reproduction/fetch_cifar10.py --data-dir ./data` | `data/cifar-10-batches-py/` |
| `train` | `python train.py --data-dir ./data --dataset torch/cifar10 ... --epochs 5` | `output/resnet18-cifar10/model_best.pth.tar`, `summary.csv`, `args.yaml` |
| `evaluate` | `python validate.py ... --checkpoint output/resnet18-cifar10/model_best.pth.tar --results-file ./metrics/eval.json` | `metrics/eval.json` (held-out top-1 / top-5) |

The exact, recorded commands live in
[`.treqs/workflows/timm-resnet18-cifar10.yaml`](../.treqs/workflows/timm-resnet18-cifar10.yaml).
The evaluation is a **separate step** so the metric is computed and recorded in
its own right rather than being read out of the training log.

CIFAR-10 is downloaded from the public torchvision mirror; **no credentials are
required** to run any step.

## Notes on the fork

* `timm/data/loader.py` carries a one-line fix: `persistent_workers` is now
  gated on `num_workers > 0`. Without it the documented `-j 0` (single-process
  data loading) raises
  `ValueError: persistent_workers option needs num_workers > 0`. This row uses
  `-j 0` so that the recorded environment is the environment the workload
  actually ran in, with no DataLoader worker subprocesses in the way.
* Nothing is installed with `pip install -e .`; `train.py` and `validate.py` are
  run from the repository root and import the checked-out `timm/` package
  directly.
* Upstream's optional Weights & Biases logging (`--log-wandb`) is **not**
  enabled in this row, so no experiment-tracker link is recorded.
