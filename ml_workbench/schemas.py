from dataclasses import dataclass, field
from typing import Any

@dataclass
class DatasetProfile:
    rows: int
    columns: int
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    missing_cells: int = 0
    missing_ratio: float = 0.0
    duplicate_rows: int = 0
    memory_mb: float = 0.0
    high_cardinality_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    class_count: int | None = None
    minority_class_ratio: float | None = None

@dataclass
class Recommendation:
    algorithm_id: int
    name: str
    score: float
    rationale: list[str]
    caveats: list[str] = field(default_factory=list)

@dataclass
class TrainingResult:
    task: str
    algorithm_id: int
    algorithm_name: str
    estimator: Any
    preprocessor: Any
    metrics: dict[str, Any]
    predictions: Any
    output_frame: Any
    metadata: dict[str, Any] = field(default_factory=dict)
