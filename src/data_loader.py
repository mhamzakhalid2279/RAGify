"""
Universal Data Loader
─────────────────────
Handles CSV, Excel (.xlsx/.xls), JSON, JSONL with:
  • Automatic encoding detection (UTF-8, Latin-1, CP1252, ISO-8859-1)
  • Separator sniffing (, ; | \\t)
  • Multi-sheet Excel support
  • JSON / JSONL arrays, dict-of-lists, nested data/records keys
  • Full cleaning pipeline
  • Schema inference with stats
"""

from __future__ import annotations

import io
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chardet
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

ENCODINGS_TO_TRY = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-8-sig"]
SEPARATORS_TO_TRY = [",", ";", "|", "\t", " "]


def _detect_encoding(raw: bytes) -> str:
    """Detect file encoding with chardet fallback."""
    result = chardet.detect(raw[:65536])  # sample first 64 KB
    detected = (result.get("encoding") or "utf-8").lower()
    # Normalise chardet aliases
    alias_map = {
        "ascii": "utf-8",
        "utf-8-sig": "utf-8-sig",
        "windows-1252": "cp1252",
    }
    return alias_map.get(detected, detected)


def _read_text(raw: bytes) -> Tuple[str, str]:
    """Return (text, encoding) trying multiple encodings."""
    # First try chardet suggestion
    enc = _detect_encoding(raw)
    order = [enc] + [e for e in ENCODINGS_TO_TRY if e != enc]
    for e in order:
        try:
            return raw.decode(e), e
        except (UnicodeDecodeError, LookupError):
            continue
    # Last resort – ignore errors
    return raw.decode("utf-8", errors="replace"), "utf-8 (lossy)"


def _sniff_separator(text: str) -> str:
    """Return the most likely CSV separator."""
    sample = "\n".join(text.splitlines()[:20])
    counts = {sep: sample.count(sep) for sep in SEPARATORS_TO_TRY}
    # Prefer separators that produce consistent column counts
    best_sep, best_score = ",", -1
    for sep, cnt in counts.items():
        if cnt == 0:
            continue
        rows = [line.split(sep) for line in sample.splitlines() if line.strip()]
        if not rows:
            continue
        col_counts = [len(r) for r in rows]
        mode_count = max(set(col_counts), key=col_counts.count)
        consistency = col_counts.count(mode_count) / len(col_counts)
        score = consistency * math.log1p(cnt)
        if score > best_score and mode_count > 1:
            best_score = score
            best_sep = sep
    return best_sep


def _normalise_column(name: str) -> str:
    """Lowercase, strip, replace spaces/special chars with underscore."""
    name = str(name).strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^\w]+", "_", name).strip("_").lower()
    return name or "col"


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full cleaning pipeline to a DataFrame."""
    # 1. Drop fully-unnamed columns (Unnamed: 0, Unnamed: 1 …)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    # 2. Normalise column names
    seen: Dict[str, int] = {}
    new_cols = []
    for c in df.columns:
        nc = _normalise_column(c)
        if nc in seen:
            seen[nc] += 1
            nc = f"{nc}_{seen[nc]}"
        else:
            seen[nc] = 0
        new_cols.append(nc)
    df.columns = new_cols  # type: ignore[assignment]

    # 3. Drop full-duplicate rows
    df = df.drop_duplicates()

    # 4. Fill NaN with sensible defaults per dtype
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("")
        else:
            df[col] = df[col].fillna(0)

    # 5. Numeric coercion — try to convert object columns that look numeric
    for col in df.select_dtypes(include="object").columns:
        coerced = pd.to_numeric(df[col].str.replace(",", "").str.strip(), errors="coerce")
        if coerced.notna().mean() > 0.8:  # >80% convertible → cast
            df[col] = coerced.fillna(0)

    # 6. Reset index
    df = df.reset_index(drop=True)
    return df


def _infer_schema(df: pd.DataFrame) -> List[Dict]:
    """Build a schema list: one entry per column with type/stats/samples."""
    schema = []
    for col in df.columns:
        series = df[col]
        n_unique = series.nunique()
        null_count = int(series.isna().sum()) + int((series == "").sum())

        entry: Dict = {
            "column": col,
            "dtype": str(series.dtype),
            "unique_count": n_unique,
            "null_count": null_count,
            "sample_values": series.dropna().iloc[:5].tolist(),
        }

        if pd.api.types.is_numeric_dtype(series):
            entry.update(
                {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": round(float(series.mean()), 4),
                }
            )

        schema.append(entry)
    return schema


# ──────────────────────────────────────────────
# JSON / JSONL helpers
# ──────────────────────────────────────────────

def _parse_json_to_df(raw_text: str) -> Optional[pd.DataFrame]:
    """Try to parse JSON/JSONL into a DataFrame."""
    text = raw_text.strip()

    # JSONL — one JSON object per line
    if text.startswith("{") and "\n" in text:
        try:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            return pd.json_normalize(records)
        except Exception:
            pass

    # Regular JSON
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(obj, list):
        return pd.json_normalize(obj)
    if isinstance(obj, dict):
        # Try common keys: data, records, results, items, rows
        for key in ("data", "records", "results", "items", "rows"):
            if key in obj and isinstance(obj[key], list):
                return pd.json_normalize(obj[key])
        # dict-of-lists format
        try:
            return pd.DataFrame(obj)
        except Exception:
            pass

    return None


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

class LoadedData:
    """Container returned by load_file()."""

    def __init__(self):
        self.tables: Dict[str, pd.DataFrame] = {}      # name → cleaned DataFrame
        self.schemas: Dict[str, List[Dict]] = {}       # name → schema list
        self.file_type: str = ""
        self.encoding: str = ""
        self.separator: Optional[str] = None
        self.errors: List[str] = []

    @property
    def primary(self) -> pd.DataFrame:
        """Return the first / largest table."""
        if not self.tables:
            raise ValueError("No tables loaded")
        return next(iter(self.tables.values()))

    @property
    def summary(self) -> str:
        lines = [f"**File type:** {self.file_type}  |  **Encoding:** {self.encoding}"]
        for name, df in self.tables.items():
            lines.append(
                f"• **{name}** — {len(df):,} rows × {len(df.columns)} cols"
            )
        return "\n".join(lines)


def load_file(filepath: str) -> LoadedData:
    """
    Load any supported file into a LoadedData object.
    Supported: .csv, .tsv, .txt, .xlsx, .xls, .json, .jsonl
    """
    path = Path(filepath)
    result = LoadedData()
    suffix = path.suffix.lower()

    # ── Excel ──────────────────────────────────
    if suffix in (".xlsx", ".xls", ".xlsm"):
        result.file_type = "Excel"
        result.encoding = "binary"
        try:
            xl = pd.ExcelFile(filepath)
            for sheet in xl.sheet_names:
                try:
                    df = xl.parse(sheet, dtype=str)
                    df = _clean_dataframe(df)
                    if df.empty:
                        continue
                    name = _normalise_column(str(sheet)) or f"sheet_{len(result.tables)}"
                    result.tables[name] = df
                    result.schemas[name] = _infer_schema(df)
                except Exception as e:
                    result.errors.append(f"Sheet '{sheet}': {e}")
        except Exception as e:
            result.errors.append(f"Excel load error: {e}")

    # ── CSV / TSV / TXT ────────────────────────
    elif suffix in (".csv", ".tsv", ".txt"):
        result.file_type = "CSV"
        raw = path.read_bytes()
        text, enc = _read_text(raw)
        result.encoding = enc
        sep = "\t" if suffix == ".tsv" else _sniff_separator(text)
        result.separator = sep
        try:
            df = pd.read_csv(
                io.StringIO(text),
                sep=sep,
                dtype=str,
                on_bad_lines="skip",
                low_memory=False,
            )
            df = _clean_dataframe(df)
            name = path.stem.lower().replace(" ", "_")[:40]
            result.tables[name] = df
            result.schemas[name] = _infer_schema(df)
        except Exception as e:
            result.errors.append(f"CSV parse error: {e}")

    # ── JSON / JSONL ────────────────────────────
    elif suffix in (".json", ".jsonl"):
        result.file_type = "JSON"
        raw = path.read_bytes()
        text, enc = _read_text(raw)
        result.encoding = enc
        df = _parse_json_to_df(text)
        if df is not None:
            df = _clean_dataframe(df)
            name = path.stem.lower().replace(" ", "_")[:40]
            result.tables[name] = df
            result.schemas[name] = _infer_schema(df)
        else:
            result.errors.append("Could not parse JSON into tabular format.")

    else:
        result.errors.append(f"Unsupported file type: {suffix}")

    return result