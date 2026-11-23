"""Allow python -m analysis.loss_geometry_stage1 from an uninstalled checkout."""

from pathlib import Path
import sys

# Reuse this checkout's contracts, without installing or importing execution code.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from .cli import main

raise SystemExit(main())
