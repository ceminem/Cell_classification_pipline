# -*- coding: utf-8 -*-
"""
env_probe.py - asked by "Cargo Analysis.bat" and "Setup on this PC.bat"
about one Python environment.

Prints ONE line the launcher can parse:
    PROBE|OK|GPU <name>|<env folder>
    PROBE|OK|CPU|<env folder>
    PROBE|MISSING|<module> <module> ...
    PROBE|BROKEN|<module>: <error message>|<module>

MISSING = not installed. BROKEN = installed but does not load. On a fresh
Windows PC that is almost always "DLL load failed" because the Microsoft
Visual C++ runtime is missing (numpy, torch and opencv need it).

Every package is really imported (not only looked up), so an environment that
would crash at start is never chosen.
"""
import importlib
import importlib.util
import sys

NEED = ("numpy", "scipy", "PIL", "tifffile", "torch", "cellpose")


def clean(text):
    """One line, no characters that break the batch parser."""
    text = " ".join(str(text).split())
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("Ä", "Ae"),
                 ("Ö", "Oe"), ("Ü", "Ue"), ("ß", "ss")):
        text = text.replace(a, b)
    for ch in '|!^&<>%"':
        text = text.replace(ch, " ")
    text = text.encode("ascii", "replace").decode("ascii")
    return text[:400]


missing = [n for n in NEED if importlib.util.find_spec(n) is None]
if missing:
    print("PROBE|MISSING|" + " ".join(missing))
    sys.exit(0)

def essence(exc):
    """The informative part of an import error.

    numpy wraps the real error in a long help text that ends with
    "Original error was: <the real error>"; keep only that part.
    Otherwise use the innermost cause (the DLL error behind an ImportError).
    """
    text = str(exc)
    if "Original error was:" in text:
        return text.split("Original error was:", 1)[1].strip()
    inner = exc
    while inner.__cause__ is not None or inner.__context__ is not None:
        inner = inner.__cause__ or inner.__context__
    return str(inner) if inner is not exc else text


DETAIL = "--detail" in sys.argv

for name in NEED:
    try:
        importlib.import_module(name)
    except Exception as exc:          # ImportError, OSError (DLL) ...
        if DETAIL:
            # full message for a human, used by "Setup on this PC.bat"
            import traceback
            print(f"Loading '{name}' failed. Full message:")
            print("-" * 60)
            traceback.print_exc(file=sys.stdout)
            print("-" * 60)
            print("ESSENCE|" + clean(essence(exc)))
        else:
            print(f"PROBE|BROKEN|{name}: {clean(essence(exc))}|{name}")
        sys.exit(0)

try:
    import torch
    if torch.cuda.is_available():      # also true for AMD ROCm builds
        dev = "GPU " + clean(torch.cuda.get_device_name(0))
    else:
        dev = "CPU"
except Exception:
    dev = "CPU"
print(f"PROBE|OK|{dev}|{sys.prefix}")
