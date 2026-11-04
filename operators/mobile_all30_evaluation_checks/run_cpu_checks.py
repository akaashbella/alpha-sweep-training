"""Run isolated operator wiring checks; no native compute, checkpoints or Slurm."""

import argparse
import ast
import os
from pathlib import Path
import sys
import unittest


def main():
    sys.dont_write_bytecode = True
    checks = Path(__file__).resolve().parent
    bundle = checks.parent / "mobile_all30_evaluation_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source-root", type=Path, default=checks.parents[1],
                        help="Read only StopRequest/signal helper source; no installed research environment needed")
    args = parser.parse_args()
    native = args.native_source_root.resolve()
    for name in ("interruption.py", "allocation_signals.py"):
        if not (native / "src/alpha_sweep_training" / name).is_file():
            parser.error(f"Missing helper source: {native}; supply --native-source-root with the existing checkout")
    os.environ["ALPHA_SWEEP_CHECK_NATIVE_SOURCE_ROOT"] = str(native)
    sources = sorted(checks.glob("*.py")) + sorted(bundle.glob("*.py"))
    for path in sources:
        compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
        ast.parse(path.read_bytes(), filename=str(path), feature_version=(3, 11))
    print(f"Python {sys.version.split()[0]}; syntax and Python 3.11 grammar checks passed for {len(sources)} files", flush=True)
    sys.path.insert(0, str(bundle))
    sys.path.insert(0, str(checks))
    suite = unittest.defaultTestLoader.discover(str(checks), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
