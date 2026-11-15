# -*- coding: utf-8 -*-
"""
cargo_cnn.model - the network
=============================================================================
A deliberately SMALL convolutional network.

    input 4 x 64 x 64
      block 1   2 x (3x3 conv, batch-norm, ReLU)    16 maps   -> max-pool 32x32
      block 2                                        32 maps   -> max-pool 16x16
      block 3                                        64 maps   -> max-pool  8x8
      block 4                                       128 maps
      global average pool + global max pool  -> 256 numbers
      dropout 0.3, one linear unit           -> logit
      sigmoid(logit) = P(released)

WHY SMALL
  There are a few hundred labelled cells, not tens of thousands. A large
  network (ResNet-50: 25 M parameters) would memorise every training cell
  and generalise badly. This one has ~0.3 M parameters, and is trained with
  heavy augmentation, dropout and weight decay - all ways of stopping it from
  memorising.

WHY AVERAGE AND MAX POOL TOGETHER
  Average pooling answers "how much signal overall" (what brightness and fill
  ratio measure). Max pooling answers "is there any very strong local feature
  anywhere" - a single bright punctum, a sharp rim. Released vs trapped cargo
  differs in both, so both are kept.
=============================================================================
"""

import os

import torch
from torch import nn


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CellNet(nn.Module):
    def __init__(self, in_ch=4, width=16, dropout=0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(in_ch, w), nn.MaxPool2d(2),
            _block(w, 2 * w), nn.MaxPool2d(2),
            _block(2 * w, 4 * w), nn.MaxPool2d(2),
            _block(4 * w, 8 * w),
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(16 * w, 1)

    def forward(self, x):
        f = self.features(x)
        f = torch.cat([f.mean(dim=(2, 3)), f.amax(dim=(2, 3))], dim=1)
        return self.head(self.drop(f)).squeeze(1)


def n_parameters(model):
    return sum(p.numel() for p in model.parameters())


def pick_device(requested="auto"):
    """
    cuda covers BOTH NVIDIA and AMD-ROCm (PyTorch-for-ROCm reports itself as
    cuda), so the RX 7800 XT in the rocm312 environment is found here.
    """
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_working_device(device, log=print, train=True, force=False):
    """
    Prove the GPU can actually run this network before relying on it, and
    work around the failure seen with PyTorch-for-ROCm on Windows.

    THE PROBLEM
      On AMD + Windows, PyTorch hands batch-normalisation to AMD's MIOpen
      library. During TRAINING, MIOpen compiles a small GPU program at run
      time (MIOpenBatchNormFwdTrainSpatialHIP.cpp). On some ROCm-for-Windows
      builds that compile fails with "'type_traits' file not found" and the
      run dies with miopenStatusUnknownError. Cellpose never hits this
      because it only predicts.

    THE FIX, in order
      1. try one forward+backward pass on the GPU;
      2. if that fails, switch MIOpen off (torch.backends.cudnn.enabled =
         False - on ROCm this flag controls MIOpen). PyTorch then uses its own
         pre-built GPU kernels, which need no run-time compilation. Slower
         than MIOpen in principle; irrelevant for a network this small;
      3. if that fails too, use the CPU and say so.

    Returns the device to use. force=True runs the test even on the CPU
    (used by the tests).
    """
    import torch

    if device.type == "cpu" and not force:
        return device

    # Known case: PyTorch-for-ROCm on Windows. MIOpen's run-time compiled
    # batch-norm kernel does not build there, and every attempt prints a long
    # compiler error before the fallback below takes over. Skip MIOpen up
    # front instead - same result, no alarming output. (The self-test still
    # runs, so a different failure is still caught.)
    if (os.name == "nt" and getattr(torch.version, "hip", None)
            and torch.backends.cudnn.enabled):
        torch.backends.cudnn.enabled = False
        log("AMD GPU on Windows: using PyTorch's own GPU kernels (MIOpen's "
            "batch-norm kernel does not build on ROCm for Windows).")

    def attempt():
        m = CellNet().to(device)
        x = torch.randn(4, 4, 64, 64, device=device)
        if train:
            m.train()
            m(x).sum().backward()
        else:
            m.eval()
            with torch.no_grad():
                m(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    try:
        attempt()
        return device
    except Exception as exc:
        first = (str(exc).strip().splitlines() or ["?"])[0][:120]
    log(f"GPU self-test failed ({first}).")

    if torch.backends.cudnn.enabled:
        log("  This is a known problem of AMD's MIOpen library with ROCm on "
            "Windows\n  (a kernel it compiles at run time does not build). "
            "Switching MIOpen off and\n  using PyTorch's own GPU kernels "
            "instead...")
        torch.backends.cudnn.enabled = False
        try:
            attempt()
            log("  -> works. Training continues on the GPU.")
            return device
        except Exception as exc:
            first = (str(exc).strip().splitlines() or ["?"])[0][:120]
            log(f"  -> still failing ({first}).")

    log("  Falling back to the CPU. Training still works, it just takes "
        "several minutes.")
    return torch.device("cpu")
