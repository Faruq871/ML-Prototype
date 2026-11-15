class WorkbenchError(Exception):
    """Base exception for expected user-facing workbench errors."""


class DatasetError(WorkbenchError):
    """Raised when an uploaded dataset cannot be used."""


class RecommendationError(WorkbenchError):
    """Raised for unsupported recommendation tasks."""


class TrainingError(WorkbenchError):
    """Raised when model training or evaluation cannot proceed."""
