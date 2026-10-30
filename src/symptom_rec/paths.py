"""Where the project's files are, relative to the repository root."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KB_DIR = REPO_ROOT / "kb"
VIGNETTES_FILE = REPO_ROOT / "tests" / "cases" / "vignettes.yaml"
DATA_DIR = REPO_ROOT / "data" / "raw"   # the Agnos CSV goes here; never committed
BUILD_DIR = REPO_ROOT / "build"         # everything learned from the data; never committed
