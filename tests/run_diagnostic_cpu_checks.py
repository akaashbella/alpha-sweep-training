"""Explicit CPU-only suite, gated before project/Torch imports. No installations."""

import os
import sys
from importlib.metadata import PackageNotFoundError, version


def main():
    blockers = []
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        blockers.append(f"Python {sys.version.split()[0]}; requires >=3.10,<3.13")
    for name, expected in (("torch", "2.5.1"), ("torchvision", "0.20.1"), ("numpy", "1.26.4")):
        try:
            actual = version(name)
        except PackageNotFoundError:
            actual = "missing"
        if actual.split("+")[0] != expected:
            blockers.append(f"{name} {actual}; requires {expected}")
    try:
        if int(version("PyYAML").split(".")[0]) < 6:
            blockers.append("PyYAML requires >=6")
    except PackageNotFoundError:
        blockers.append("PyYAML missing; requires >=6")
    if blockers:
        print("CPU CHECKS BLOCKED (no tests/project imports executed): " + "; ".join(blockers), file=sys.stderr)
        return 2
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ALPHA_SWEEP_DIAGNOSTIC_TESTS"] = "1"
    os.environ["ALPHA_SWEEP_CUDA_TESTS"] = "0"
    import unittest
    # Explicit classes, never discovery of CudaContracts or any research command.
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        "test_computation_contracts.CpuContracts", "test_diagnostic_contracts.DiagnosticContracts",
        "test_diagnostic_integration.DiagnosticRunnerWiring", "test_mobile_candidate.CandidateContracts"])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
