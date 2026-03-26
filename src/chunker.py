"""
Intelligent Chunker
────────────────────
Three strategies:
  • row_window   — sliding row windows (best for tall tables)
  • column_group — groups related columns (best for wide 30+ col tables)
  • hybrid       — row_window × column_group combined

Auto-mode selects based on shape.
Always prepends:
  1. Global overview chunk (all tables)
  2. Per-table schema chunk
  3. Per-table statistical summary chunk
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import LoadedData


# ──────────────────────────────────────────────
# Chunk data class
# ──────────────────────────────────────────────

class Chunk:
    __slots__ = ("text", "metadata")

    def __init__(self, text: str, metadata: dict):
        self.text = text
        self.metadata = metadata  # table, chunk_type, row_start, row_end, cols, etc.

    def __repr__(self):
        return f"Chunk(type={self.metadata.get('chunk_type')!r}, len={len(self.text)})"


# ──────────────────────────────────────────────
# Strategy helpers
# ──────────────────────────────────────────────

def _df_to_text_rows(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    max_val_len: int = 200,
) -> List[str]:
    """Convert DataFrame rows to 'col: val | col: val …' strings."""
    sub = df[cols] if cols else df
    rows = []
    for _, row in sub.iterrows():
        parts = []
        for c in sub.columns:
            val = str(row[c])
            if len(val) > max_val_len:
                val = val[:max_val_len] + "…"
            parts.append(f"{c}: {val}")
        rows.append(" | ".join(parts))
    return rows


def _row_window_chunks(
    df: pd.DataFrame,
    table_name: str,
    window: int = 10,
    overlap: int = 2,
) -> List[Chunk]:
    """Sliding window over rows."""
    chunks = []
    n = len(df)
    step = max(1, window - overlap)
    for start in range(0, n, step):
        end = min(start + window, n)
        sub = df.iloc[start:end]
        row_texts = _df_to_text_rows(sub)
        text = (
            f"[Table: {table_name} | Rows {start+1}–{end} of {n}]\n"
            + "\n".join(row_texts)
        )
        chunks.append(
            Chunk(
                text,
                {
                    "table": table_name,
                    "chunk_type": "row_window",
                    "row_start": start,
                    "row_end": end,
                    "cols": list(df.columns),
                },
            )
        )
    return chunks


def _group_columns(cols: List[str], group_size: int = 8) -> List[List[str]]:
    """Split column list into groups of group_size."""
    return [cols[i : i + group_size] for i in range(0, len(cols), group_size)]


def _column_group_chunks(
    df: pd.DataFrame,
    table_name: str,
    row_sample: int = 20,
    group_size: int = 8,
) -> List[Chunk]:
    """For wide tables: group columns, sample rows per group."""
    chunks = []
    col_groups = _group_columns(list(df.columns), group_size)
    # Sample rows evenly
    indices = np.linspace(0, len(df) - 1, min(row_sample, len(df)), dtype=int).tolist()
    sample_df = df.iloc[indices]

    for gi, group in enumerate(col_groups):
        row_texts = _df_to_text_rows(sample_df, cols=group)
        text = (
            f"[Table: {table_name} | Column Group {gi+1}/{len(col_groups)}: "
            f"{', '.join(group)}]\n" + "\n".join(row_texts)
        )
        chunks.append(
            Chunk(
                text,
                {
                    "table": table_name,
                    "chunk_type": "column_group",
                    "col_group": gi,
                    "cols": group,
                },
            )
        )
    return chunks


def _hybrid_chunks(
    df: pd.DataFrame,
    table_name: str,
    window: int = 8,
    overlap: int = 1,
    group_size: int = 6,
) -> List[Chunk]:
    """Hybrid: column groups × row windows."""
    chunks = []
    col_groups = _group_columns(list(df.columns), group_size)
    n = len(df)
    step = max(1, window - overlap)

    for gi, group in enumerate(col_groups):
        for start in range(0, n, step):
            end = min(start + window, n)
            sub = df.iloc[start:end]
            row_texts = _df_to_text_rows(sub, cols=group)
            text = (
                f"[Table: {table_name} | ColGroup {gi+1} × Rows {start+1}–{end}]\n"
                + "\n".join(row_texts)
            )
            chunks.append(
                Chunk(
                    text,
                    {
                        "table": table_name,
                        "chunk_type": "hybrid",
                        "col_group": gi,
                        "row_start": start,
                        "row_end": end,
                        "cols": group,
                    },
                )
            )
    return chunks


# ──────────────────────────────────────────────
# Header / overview chunks
# ──────────────────────────────────────────────

def _schema_chunk(table_name: str, schema: List[dict]) -> Chunk:
    lines = [f"[Schema: {table_name}]"]
    for col_info in schema:
        col = col_info["column"]
        dtype = col_info["dtype"]
        n_uniq = col_info["unique_count"]
        nulls = col_info["null_count"]
        samples = ", ".join(str(v) for v in col_info["sample_values"][:3])
        line = f"  {col} ({dtype}) | unique={n_uniq} | nulls={nulls} | e.g. {samples}"
        if "mean" in col_info:
            line += f" | mean={col_info['mean']}, min={col_info['min']}, max={col_info['max']}"
        lines.append(line)
    return Chunk("\n".join(lines), {"table": table_name, "chunk_type": "schema"})


def _stats_chunk(df: pd.DataFrame, table_name: str) -> Chunk:
    lines = [f"[Statistical Summary: {table_name} — {len(df)} rows × {len(df.columns)} columns]"]
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        desc = df[num_cols].describe().round(3)
        lines.append(desc.to_string())
    else:
        lines.append("No numeric columns detected.")
    cat_cols = df.select_dtypes(include="object").columns.tolist()[:5]
    if cat_cols:
        lines.append("\nTop categorical values:")
        for c in cat_cols:
            top = df[c].value_counts().head(5).to_dict()
            top_str = ", ".join(f"{k!r}:{v}" for k, v in top.items())
            lines.append(f"  {c}: {top_str}")
    return Chunk("\n".join(lines), {"table": table_name, "chunk_type": "stats"})


def _global_overview_chunk(data: LoadedData, description: str) -> Chunk:
    lines = [
        "=== GLOBAL OVERVIEW ===",
        f"Purpose: {description}",
        f"Number of tables: {len(data.tables)}",
    ]
    for name, df in data.tables.items():
        lines.append(f"  • {name}: {len(df):,} rows × {len(df.columns)} columns")
        lines.append(f"    Columns: {', '.join(df.columns.tolist()[:20])}")
    return Chunk("\n".join(lines), {"table": "_global", "chunk_type": "overview"})


# ──────────────────────────────────────────────
# Auto strategy selector
# ──────────────────────────────────────────────

def _pick_strategy(df: pd.DataFrame) -> Tuple[str, dict]:
    n_rows, n_cols = df.shape
    if n_cols >= 30:
        if n_rows > 200:
            return "hybrid", {"window": 8, "overlap": 1, "group_size": 6}
        return "column_group", {"row_sample": 25, "group_size": 8}
    if n_rows > 500:
        return "row_window", {"window": 12, "overlap": 2}
    if n_rows > 100:
        return "row_window", {"window": 10, "overlap": 2}
    return "row_window", {"window": 8, "overlap": 1}


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────

def build_chunks(
    data: LoadedData,
    description: str,
    strategy: str = "auto",
) -> List[Chunk]:
    """
    Build the full chunk list for a LoadedData object.

    Parameters
    ----------
    data        : LoadedData from data_loader.load_file()
    description : user-provided chatbot purpose string
    strategy    : 'auto' | 'row_window' | 'column_group' | 'hybrid'

    Returns
    -------
    List[Chunk] ordered as: overview, per-table schema+stats, content chunks
    """
    all_chunks: List[Chunk] = []

    # 1. Global overview
    all_chunks.append(_global_overview_chunk(data, description))

    for table_name, df in data.tables.items():
        schema = data.schemas[table_name]

        # 2. Schema chunk
        all_chunks.append(_schema_chunk(table_name, schema))

        # 3. Stats chunk
        all_chunks.append(_stats_chunk(df, table_name))

        # 4. Content chunks
        if strategy == "auto":
            strat, kwargs = _pick_strategy(df)
        else:
            strat = strategy
            kwargs = {}

        if strat == "row_window":
            content_chunks = _row_window_chunks(df, table_name, **kwargs)
        elif strat == "column_group":
            content_chunks = _column_group_chunks(df, table_name, **kwargs)
        else:
            content_chunks = _hybrid_chunks(df, table_name, **kwargs)

        all_chunks.extend(content_chunks)

    return all_chunks
