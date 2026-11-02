"""Focused external operator checks in an existing compatible environment only."""

import argparse
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source", default="/nfs/hpc/share/bellaak/research/alpha-sweep/alpha-sweep-mobile-eps0316")
    args = parser.parse_args()
    blockers = []
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        blockers.append("Python >=3.10,<3.13 required")
    for package, expected in (("torch", "2.5.1"), ("torchvision", "0.20.1"), ("numpy", "1.26.4")):
        try:
            actual = version(package).split("+")[0]
        except PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            blockers.append(f"{package}: {actual}, expected {expected}")
    try:
        if int(version("PyYAML").split(".")[0]) < 6:
            blockers.append("PyYAML >=6 required")
    except PackageNotFoundError:
        blockers.append("PyYAML missing")
    root = Path(args.native_source).resolve()
    if not (root / "src" / "alpha_sweep_training" / "mobile_candidate.py").is_file():
        blockers.append("Existing native candidate source required")
    if blockers:
        print("CPU CHECKS UNRUN: " + "; ".join(blockers), file=sys.stderr)
        return 2
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "tests"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ALPHA_SWEEP_DIAGNOSTIC_TESTS"] = "1"
    os.environ["ALPHA_SWEEP_CUDA_TESTS"] = "0"
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromName("test_operator.OperatorChecks")
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
