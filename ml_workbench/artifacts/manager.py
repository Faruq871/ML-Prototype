from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json
import joblib
from ml_workbench.config import ARTIFACT_DIR

class ArtifactManager:
    def __init__(self, directory: Path | str = ARTIFACT_DIR):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save_model(self, estimator, algorithm_name: str, task: str, metadata: dict | None = None) -> Path:
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in algorithm_name).strip("-")
        artifact_id = uuid4().hex[:10]
        path = self.directory / f"{slug}-{artifact_id}.joblib"
        joblib.dump({"estimator": estimator, "task": task, "algorithm": algorithm_name,
                     "metadata": metadata or {}, "created_at": datetime.now(timezone.utc).isoformat()}, path)
        return path

    def save_report(self, report: dict, algorithm_name: str) -> Path:
        artifact_id = uuid4().hex[:10]
        path = self.directory / f"report-{artifact_id}.json"
        path.write_text(json.dumps({"algorithm": algorithm_name, "report": report}, indent=2, default=str), encoding="utf-8")
        return path
