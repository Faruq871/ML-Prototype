from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MAX_UPLOAD_MB = 100
RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_CV_FOLDS = 5
SUPPORTED_EXTENSIONS = {"csv", "xlsx"}
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
