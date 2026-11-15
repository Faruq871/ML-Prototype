from __future__ import annotations
from io import BytesIO
from pathlib import Path
import pandas as pd
from ml_workbench.config import MAX_UPLOAD_MB
from ml_workbench.exceptions import DatasetError


def load_dataframe(file_bytes: bytes, filename: str, max_size_mb: int = MAX_UPLOAD_MB) -> pd.DataFrame:
    """Load CSV/XLSX bytes into a DataFrame with basic upload validation."""
    if not file_bytes:
        raise DatasetError("File kosong. Silakan pilih dataset yang berisi data.")
    if len(file_bytes) > max_size_mb * 1024 * 1024:
        raise DatasetError(f"Ukuran file melewati batas {max_size_mb} MB.")

    suffix = Path(filename).suffix.lower().lstrip(".")
    try:
        if suffix == "csv":
            for encoding in ("utf-8-sig", "cp1252", "latin-1"):
                try:
                    return pd.read_csv(BytesIO(file_bytes), encoding=encoding, sep=None, engine="python")
                except Exception:  # try a second common encoding
                    continue
            raise DatasetError("CSV tidak dapat dibaca. Periksa delimiter, encoding, dan format file.")
        if suffix == "xlsx":
            return pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
        raise DatasetError("Format tidak didukung. Gunakan file CSV atau XLSX.")
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"Gagal membaca file: {exc}") from exc


def validate_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise DatasetError("Dataset harus berupa pandas DataFrame.")
    if data.empty:
        raise DatasetError("Dataset tidak memiliki baris data.")
    if len(data.columns) == 0:
        raise DatasetError("Dataset tidak memiliki kolom.")
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].astype(str).tolist()
        raise DatasetError(f"Nama kolom duplikat ditemukan: {duplicates}")
    clean = data.copy()
    clean.columns = [str(col).strip() for col in clean.columns]
    if clean.columns.duplicated().any():
        duplicates = clean.columns[clean.columns.duplicated()].tolist()
        raise DatasetError(f"Nama kolom menjadi duplikat setelah spasi dibersihkan: {duplicates}")
    if any(not col for col in clean.columns):
        raise DatasetError("Terdapat nama kolom kosong. Perbaiki header dataset.")
    return clean
