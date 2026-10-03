"""SignBridge core package.

Project paths are resolved from this file rather than the working directory, so
the apps behave the same however they are launched, on macOS and Windows.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

ASL_DATA_DIR = DATA_DIR / "asl"
ASL_MODEL_PATH = MODELS_DIR / "asl_classifier.pkl"
