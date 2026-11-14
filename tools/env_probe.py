# -*- coding: utf-8 -*-
"""
env_probe.py - asked by "Cargo Analysis.bat" about each conda environment.

Prints ONE line the launcher can parse:
    PROBE|OK|GPU <name>|<env folder>
    PROBE|OK|CPU|<env folder>
    PROBE|MISSING|<module> <module> ...

Checking for packages with find_spec does not import them, so this is fast;
only torch is imported, to ask whether its GPU works.
"""
import importlib.util
import sys

NEED = ("cellpose", "torch", "tifffile", "PIL", "scipy", "numpy")

missing = [n for n in NEED if importlib.util.find_spec(n) is None]
if missing:
    print("PROBE|MISSING|" + " ".join(missing))
    sys.exit(0)

try:
    import torch
    if torch.cuda.is_available():      # also true for AMD ROCm builds
        dev = "GPU " + " ".join(torch.cuda.get_device_name(0).split())
    else:
        dev = "CPU"
except Exception:
    dev = "CPU"
print(f"PROBE|OK|{dev}|{sys.prefix}")
