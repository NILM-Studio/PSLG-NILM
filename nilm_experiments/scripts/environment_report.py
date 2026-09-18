"""Record reproducibility metadata only; never load any dataset or fit a model."""
import importlib.metadata
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nilm_lab.common import code_snapshot, write_json

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/environment_report.py OUTPUT.json")
    versions = {}
    for name in ["torch", "numpy", "pandas", "scikit-learn", "scipy", "pytest", "PyYAML"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    write_json(sys.argv[1], {"platform": platform.platform(), "python": sys.version,
                             "packages": versions, "code_sha256": code_snapshot(), "real_data_experiments": False})
