"""
RAGlitheon — Instant RAG Chatbot Builder  (single-file HuggingFace Spaces deployment)
==================================================================================
Full parity with:
  data_loader.py  ·  chunker.py  ·  embedder.py
  retriever.py    ·  llm_client.py  ·  rag_pipeline.py  ·  app.py

Compatibility:
  ✓ Python 3.13+          ✓ Gradio 5.x (type="messages" chatbot)
  ✓ No submodule imports  ✓ Docker SDK on HuggingFace Spaces
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# CRITICAL MONKEY-PATCH — must run before `import gradio`
# Fixes: TypeError: argument of type 'bool' is not iterable
# Root cause: gradio_client.utils._json_schema_to_python_type passes boolean
#             values (valid JSON Schema) to get_type() which does `"const" in schema`
#             — this crashes on booleans in Python 3.13.
# Fix: wrap get_type and _json_schema_to_python_type to guard against non-dict schemas.
# ══════════════════════════════════════════════════════════════════════════════
import gradio_client.utils as _gcu

_original_get_type               = _gcu.get_type
_original_json_schema_to_python  = _gcu._json_schema_to_python_type

def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _original_get_type(schema)

def _safe_json_schema_to_python_type(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _original_json_schema_to_python(schema, defs)

_gcu.get_type                   = _safe_get_type
_gcu._json_schema_to_python_type = _safe_json_schema_to_python_type
# ══════════════════════════════════════════════════════════════════════════════



# ── stdlib ────────────────────────────────────────────────────────────────────
import io, json, math, os, re, time, traceback, unicodedata
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────────────────────────
import chardet
import faiss
import gradio as gr
import numpy as np
import pandas as pd
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer

# ══════════════════════════════════════════════════════════════════════════════
# ① DATA LOADER  (full port of data_loader.py)
# ══════════════════════════════════════════════════════════════════════════════

ENCODINGS_TO_TRY  = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-8-sig"]
SEPARATORS_TO_TRY = [",", ";", "|", "\t", " "]


def _detect_encoding(raw: bytes) -> str:
    result  = chardet.detect(raw[:65536])
    detected = (result.get("encoding") or "utf-8").lower()
    alias = {"ascii": "utf-8", "utf-8-sig": "utf-8-sig", "windows-1252": "cp1252"}
    return alias.get(detected, detected)


def _read_text(raw: bytes) -> Tuple[str, str]:
    enc   = _detect_encoding(raw)
    order = [enc] + [e for e in ENCODINGS_TO_TRY if e != enc]
    for e in order:
        try:
            return raw.decode(e), e
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8 (lossy)"


def _sniff_separator(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    best_sep, best_score = ",", -1.0
    for sep in SEPARATORS_TO_TRY:
        cnt = sample.count(sep)
        if cnt == 0:
            continue
        rows = [line.split(sep) for line in sample.splitlines() if line.strip()]
        if not rows:
            continue
        col_counts  = [len(r) for r in rows]
        mode_count  = max(set(col_counts), key=col_counts.count)
        consistency = col_counts.count(mode_count) / len(col_counts)
        score       = consistency * math.log1p(cnt)
        if score > best_score and mode_count > 1:
            best_score = score
            best_sep   = sep
    return best_sep


def _normalise_column(name: str) -> str:
    name = str(name).strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r"[^\w]+", "_", name).strip("_").lower()
    return name or "col"


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    # Normalise column names (unique)
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
    # Drop full duplicates
    df = df.drop_duplicates()
    # Fill NaN
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("")
        else:
            df[col] = df[col].fillna(0)
    # Numeric coercion
    for col in df.select_dtypes(include="object").columns:
        coerced = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce")
        if coerced.notna().mean() > 0.8:
            df[col] = coerced.fillna(0)
    df = df.reset_index(drop=True)
    return df


def _infer_schema(df: pd.DataFrame) -> List[Dict]:
    schema = []
    for col in df.columns:
        series    = df[col]
        null_cnt  = int(series.isna().sum()) + int((series == "").sum())
        entry: Dict = {
            "column":        col,
            "dtype":         str(series.dtype),
            "unique_count":  series.nunique(),
            "null_count":    null_cnt,
            "sample_values": series.dropna().iloc[:5].tolist(),
        }
        if pd.api.types.is_numeric_dtype(series):
            entry.update({"min": float(series.min()), "max": float(series.max()),
                          "mean": round(float(series.mean()), 4)})
        schema.append(entry)
    return schema


def _parse_json_to_df(raw_text: str) -> Optional[pd.DataFrame]:
    text = raw_text.strip()
    if text.startswith("{") and "\n" in text:
        try:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            return pd.json_normalize(records)
        except Exception:
            pass
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, list):
        return pd.json_normalize(obj)
    if isinstance(obj, dict):
        for key in ("data", "records", "results", "items", "rows"):
            if key in obj and isinstance(obj[key], list):
                return pd.json_normalize(obj[key])
        try:
            return pd.DataFrame(obj)
        except Exception:
            pass
    return None


class LoadedData:
    def __init__(self):
        self.tables:    Dict[str, pd.DataFrame] = {}
        self.schemas:   Dict[str, List[Dict]]   = {}
        self.file_type: str                     = ""
        self.encoding:  str                     = ""
        self.separator: Optional[str]           = None
        self.errors:    List[str]               = []

    @property
    def primary(self) -> pd.DataFrame:
        if not self.tables:
            raise ValueError("No tables loaded")
        return next(iter(self.tables.values()))

    @property
    def summary(self) -> str:
        lines = [f"**File type:** {self.file_type}  |  **Encoding:** {self.encoding}"]
        for name, df in self.tables.items():
            lines.append(f"• **{name}** — {len(df):,} rows × {len(df.columns)} cols")
        return "\n".join(lines)


def load_file(filepath: str) -> LoadedData:
    path   = Path(filepath)
    result = LoadedData()
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls", ".xlsm"):
        result.file_type = "Excel"
        result.encoding  = "binary"
        try:
            xl = pd.ExcelFile(filepath)
            for sheet in xl.sheet_names:
                try:
                    df = xl.parse(sheet, dtype=str)
                    df = _clean_dataframe(df)
                    if df.empty:
                        continue
                    name = _normalise_column(str(sheet)) or f"sheet_{len(result.tables)}"
                    result.tables[name]  = df
                    result.schemas[name] = _infer_schema(df)
                except Exception as e:
                    result.errors.append(f"Sheet '{sheet}': {e}")
        except Exception as e:
            result.errors.append(f"Excel load error: {e}")

    elif suffix in (".csv", ".tsv", ".txt"):
        result.file_type = "CSV"
        raw              = path.read_bytes()
        text, enc        = _read_text(raw)
        result.encoding  = enc
        sep              = "\t" if suffix == ".tsv" else _sniff_separator(text)
        result.separator = sep
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str,
                             on_bad_lines="skip", low_memory=False)
            df   = _clean_dataframe(df)
            name = path.stem.lower().replace(" ", "_")[:40]
            result.tables[name]  = df
            result.schemas[name] = _infer_schema(df)
        except Exception as e:
            result.errors.append(f"CSV parse error: {e}")

    elif suffix in (".json", ".jsonl"):
        result.file_type = "JSON"
        raw              = path.read_bytes()
        text, enc        = _read_text(raw)
        result.encoding  = enc
        df               = _parse_json_to_df(text)
        if df is not None:
            df   = _clean_dataframe(df)
            name = path.stem.lower().replace(" ", "_")[:40]
            result.tables[name]  = df
            result.schemas[name] = _infer_schema(df)
        else:
            result.errors.append("Could not parse JSON into tabular format.")
    else:
        result.errors.append(f"Unsupported file type: {suffix}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# ② CHUNKER  (full port of chunker.py)
# ══════════════════════════════════════════════════════════════════════════════

class Chunk:
    __slots__ = ("text", "metadata")
    def __init__(self, text: str, metadata: dict):
        self.text     = text
        self.metadata = metadata
    def __repr__(self):
        return f"Chunk(type={self.metadata.get('chunk_type')!r}, len={len(self.text)})"


def _df_to_text_rows(df: pd.DataFrame, cols: Optional[List[str]] = None,
                     max_val_len: int = 200) -> List[str]:
    sub  = df[cols] if cols else df
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


def _row_window_chunks(df: pd.DataFrame, table_name: str,
                       window: int = 10, overlap: int = 2) -> List[Chunk]:
    chunks, n, step = [], len(df), max(1, window - overlap)
    for start in range(0, n, step):
        end       = min(start + window, n)
        sub       = df.iloc[start:end]
        row_texts = _df_to_text_rows(sub)
        text      = (f"[Table: {table_name} | Rows {start+1}–{end} of {n}]\n"
                     + "\n".join(row_texts))
        chunks.append(Chunk(text, {"table": table_name, "chunk_type": "row_window",
                                   "row_start": start, "row_end": end,
                                   "cols": list(df.columns)}))
    return chunks


def _group_columns(cols: List[str], group_size: int = 8) -> List[List[str]]:
    return [cols[i:i+group_size] for i in range(0, len(cols), group_size)]


def _column_group_chunks(df: pd.DataFrame, table_name: str,
                         row_sample: int = 20, group_size: int = 8) -> List[Chunk]:
    chunks     = []
    col_groups = _group_columns(list(df.columns), group_size)
    indices    = np.linspace(0, len(df)-1, min(row_sample, len(df)), dtype=int).tolist()
    sample_df  = df.iloc[indices]
    for gi, group in enumerate(col_groups):
        row_texts = _df_to_text_rows(sample_df, cols=group)
        text = (f"[Table: {table_name} | Column Group {gi+1}/{len(col_groups)}: "
                f"{', '.join(group)}]\n" + "\n".join(row_texts))
        chunks.append(Chunk(text, {"table": table_name, "chunk_type": "column_group",
                                   "col_group": gi, "cols": group}))
    return chunks


def _hybrid_chunks(df: pd.DataFrame, table_name: str,
                   window: int = 8, overlap: int = 1, group_size: int = 6) -> List[Chunk]:
    chunks, col_groups, n, step = [], _group_columns(list(df.columns), group_size), len(df), max(1, window-overlap)
    for gi, group in enumerate(col_groups):
        for start in range(0, n, step):
            end       = min(start + window, n)
            sub       = df.iloc[start:end]
            row_texts = _df_to_text_rows(sub, cols=group)
            text      = (f"[Table: {table_name} | ColGroup {gi+1} × Rows {start+1}–{end}]\n"
                         + "\n".join(row_texts))
            chunks.append(Chunk(text, {"table": table_name, "chunk_type": "hybrid",
                                       "col_group": gi, "row_start": start,
                                       "row_end": end, "cols": group}))
    return chunks


def _schema_chunk(table_name: str, schema: List[dict]) -> Chunk:
    lines = [f"[Schema: {table_name}]"]
    for ci in schema:
        col = ci["column"]; dtype = ci["dtype"]
        line = (f"  {col} ({dtype}) | unique={ci['unique_count']} | nulls={ci['null_count']} "
                f"| e.g. {', '.join(str(v) for v in ci['sample_values'][:3])}")
        if "mean" in ci:
            line += f" | mean={ci['mean']}, min={ci['min']}, max={ci['max']}"
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
            lines.append(f"  {c}: {', '.join(f'{k!r}:{v}' for k,v in top.items())}")
    return Chunk("\n".join(lines), {"table": table_name, "chunk_type": "stats"})


def _global_overview_chunk(data: LoadedData, description: str) -> Chunk:
    lines = ["=== GLOBAL OVERVIEW ===", f"Purpose: {description}",
             f"Number of tables: {len(data.tables)}"]
    for name, df in data.tables.items():
        lines.append(f"  • {name}: {len(df):,} rows × {len(df.columns)} columns")
        lines.append(f"    Columns: {', '.join(df.columns.tolist()[:20])}")
    return Chunk("\n".join(lines), {"table": "_global", "chunk_type": "overview"})


def _pick_strategy(df: pd.DataFrame) -> Tuple[str, dict]:
    nr, nc = df.shape
    if nc >= 30:
        if nr > 200: return "hybrid",       {"window": 8, "overlap": 1, "group_size": 6}
        return           "column_group",    {"row_sample": 25, "group_size": 8}
    if nr > 500:     return "row_window",   {"window": 12, "overlap": 2}
    if nr > 100:     return "row_window",   {"window": 10, "overlap": 2}
    return               "row_window",      {"window": 8,  "overlap": 1}


def build_chunks(data: LoadedData, description: str, strategy: str = "auto") -> List[Chunk]:
    all_chunks: List[Chunk] = [_global_overview_chunk(data, description)]
    for table_name, df in data.tables.items():
        schema = data.schemas[table_name]
        all_chunks.append(_schema_chunk(table_name, schema))
        all_chunks.append(_stats_chunk(df, table_name))
        if strategy == "auto":
            strat, kwargs = _pick_strategy(df)
        else:
            strat, kwargs = strategy, {}
        if strat == "row_window":
            all_chunks.extend(_row_window_chunks(df, table_name, **kwargs))
        elif strat == "column_group":
            all_chunks.extend(_column_group_chunks(df, table_name, **kwargs))
        else:
            all_chunks.extend(_hybrid_chunks(df, table_name, **kwargs))
    return all_chunks


# ══════════════════════════════════════════════════════════════════════════════
# ③ EMBEDDER  (full port of embedder.py)
# ══════════════════════════════════════════════════════════════════════════════

EMBED_MODEL_NAME                    = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _MODEL


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = _get_model().encode(texts, batch_size=batch_size, show_progress_bar=False,
                                convert_to_numpy=True, normalize_embeddings=True)
    return vecs.astype(np.float32)


class FAISSIndex:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        texts       = [c.text for c in chunks]
        self.vecs   = embed_texts(texts)
        dim         = self.vecs.shape[1]
        self.index  = faiss.IndexFlatIP(dim)
        self.index.add(self.vecs)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        q_vec            = embed_texts([query])
        scores, indices  = self.index.search(q_vec, min(top_k, len(self.chunks)))
        return [(int(idx), float(sc)) for idx, sc in zip(indices[0], scores[0]) if idx >= 0]


# ══════════════════════════════════════════════════════════════════════════════
# ④ RETRIEVER  (full port of retriever.py)
# ══════════════════════════════════════════════════════════════════════════════

def _tokenise(text: str) -> List[str]:
    return re.findall(r"\b[a-z0-9_]+\b", text.lower())


class TinyBM25:
    k1 = 1.5
    b  = 0.75

    def __init__(self, corpus: List[str]):
        self.n          = len(corpus)
        self.tokenised  = [_tokenise(doc) for doc in corpus]
        self.dl         = [len(t) for t in self.tokenised]
        self.avgdl      = sum(self.dl) / max(self.n, 1)
        df_count: Dict[str, int] = defaultdict(int)
        self.tf: List[Dict[str, int]] = []
        for toks in self.tokenised:
            freq: Dict[str, int] = defaultdict(int)
            for t in toks: freq[t] += 1
            self.tf.append(dict(freq))
            for t in set(toks): df_count[t] += 1
        self.idf = {term: math.log((self.n - df + 0.5) / (df + 0.5) + 1)
                    for term, df in df_count.items()}

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        sc, dl, tf_map = 0.0, self.dl[doc_idx], self.tf[doc_idx]
        for t in query_tokens:
            if t not in tf_map: continue
            idf = self.idf.get(t, 0.0); tf = tf_map[t]
            sc += idf * tf * (self.k1+1) / (tf + self.k1*(1-self.b+self.b*dl/self.avgdl))
        return sc

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        tokens = _tokenise(query)
        if not tokens: return []
        scores = [(i, self.score(tokens, i)) for i in range(self.n) if self.score(tokens, i) > 0]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def _rrf_fuse(ranked_lists: List[List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
    scores: Dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (idx, _) in enumerate(ranked):
            scores[idx] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _mmr(query_vec: np.ndarray, candidate_indices: List[int],
         chunk_vecs: np.ndarray, top_k: int = 6, lambda_: float = 0.6) -> List[int]:
    if not candidate_indices: return []
    selected: List[int] = []
    remaining = list(candidate_indices)
    q = query_vec[0]
    cand_vecs = chunk_vecs[remaining]
    query_sims = cand_vecs @ q
    while remaining and len(selected) < top_k:
        if not selected:
            best_local = int(np.argmax(query_sims))
        else:
            sel_vecs      = chunk_vecs[selected]
            cand_vecs2    = chunk_vecs[remaining]
            redundancy    = (cand_vecs2 @ sel_vecs.T).max(axis=1)
            q_sims_rem    = cand_vecs2 @ q
            mmr_scores    = lambda_ * q_sims_rem - (1 - lambda_) * redundancy
            best_local    = int(np.argmax(mmr_scores))
        chosen = remaining.pop(best_local)
        selected.append(chosen)
        query_sims = np.delete(query_sims, best_local)
    return selected


class HybridRetriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks      = chunks
        corpus           = [c.text for c in chunks]
        self.bm25        = TinyBM25(corpus)
        self.faiss_index = FAISSIndex(chunks)
        self.chunk_vecs  = self.faiss_index.vecs

    def retrieve(self, query: str, top_k: int = 6, bm25_k: int = 20,
                 dense_k: int = 20, mmr_lambda: float = 0.65) -> List[Chunk]:
        bm25_results  = self.bm25.search(query, top_k=bm25_k)
        dense_results = self.faiss_index.search(query, top_k=dense_k)
        fused         = _rrf_fuse([bm25_results, dense_results])
        cand_indices  = [idx for idx, _ in fused[:top_k*3]]
        if not cand_indices: return []
        q_vec         = embed_texts([query])
        diverse       = _mmr(q_vec, cand_indices, self.chunk_vecs, top_k=top_k, lambda_=mmr_lambda)
        return [self.chunks[i] for i in diverse]


# ══════════════════════════════════════════════════════════════════════════════
# ⑤ LLM CLIENT  (full port of llm_client.py)
# ══════════════════════════════════════════════════════════════════════════════

# PRIMARY_MODEL   = "HuggingFaceH4/zephyr-7b-beta"
# FALLBACK_MODELS = [
#     "mistralai/Mistral-7B-Instruct-v0.3",
#     "microsoft/Phi-3-mini-4k-instruct",
#     "tiiuae/falcon-7b-instruct",
# ]
# DEFAULT_LLM_PARAMS: Dict[str, Any] = {"max_tokens": 1024, "temperature": 0.3, "top_p": 0.92}
# FALLBACK_TRIGGERS = {"rate limit", "503", "500", "429", "overloaded", "unavailable", "quota"}
PRIMARY_MODEL   = "mistralai/Mistral-7B-Instruct-v0.3"
FALLBACK_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]
DEFAULT_LLM_PARAMS: Dict[str, Any] = {"max_tokens": 1024, "temperature": 0.3, "top_p": 0.92}
FALLBACK_TRIGGERS = {
    "rate limit", "503", "500", "429", "overloaded", 
    "unavailable", "quota", "model_not_supported", "not supported"
}
VALID_LLM_KEYS    = {"max_tokens", "temperature", "top_p", "stop", "seed"}


def _should_fallback(msg: str) -> bool:
    return any(t in msg.lower() for t in FALLBACK_TRIGGERS)


def _make_client(model: str) -> InferenceClient:
    token = os.environ.get("HF_TOKEN")
    return InferenceClient(model=model, token=token, timeout=90)


def stream_response(messages: List[Dict[str, str]], system_prompt: str,
                    params: Optional[Dict[str, Any]] = None,
                    max_retries: int = 2) -> Generator[str, None, None]:
    gen_params   = {k: v for k, v in {**DEFAULT_LLM_PARAMS, **(params or {})}.items()
                    if k in VALID_LLM_KEYS}
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    last_error    = ""
    for model in [PRIMARY_MODEL] + FALLBACK_MODELS:
        client = _make_client(model)
        for attempt in range(max_retries):
            try:
                stream = client.chat_completion(messages=full_messages, stream=True, **gen_params)
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta: yield delta
                return
            except Exception as e:
                last_error = str(e)
                if _should_fallback(last_error):
                    time.sleep(1.5 * (attempt + 1)); break
                else:
                    yield f"\n\n[⚠ Error: {last_error}]"; return
    yield f"\n\n[⚠ All models unavailable. Last error: {last_error}]"


def generate_sync(prompt: str, system_prompt: str = "You are a helpful assistant.",
                  max_tokens: int = 512) -> str:
    return "".join(stream_response(
        messages=[{"role": "user", "content": prompt}],
        system_prompt=system_prompt,
        params={"max_tokens": max_tokens, "temperature": 0.5},
    ))


# ══════════════════════════════════════════════════════════════════════════════
# ⑥ RAG PIPELINE  (full port of rag_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

ANALYTICAL_PATTERNS = [
    r"\b(average|mean|median|sum|total|count|max|min|maximum|minimum|how many|percentage|ratio|trend)\b",
    r"\b(compare|comparison|difference|versus|vs\.?|ranking|top \d+|bottom \d+)\b",
    r"\b(distribution|correlation|group by|breakdown|analysis|statistics)\b",
]
LOOKUP_PATTERNS = [
    r"\b(what is|what are|who is|find|show|list|display|get|fetch|look up|search for)\b",
    r"\b(value|record|entry|row|column|field)\b",
]


def classify_query(query: str) -> str:
    q = query.lower()
    for pat in ANALYTICAL_PATTERNS:
        if re.search(pat, q): return "analytical"
    for pat in LOOKUP_PATTERNS:
        if re.search(pat, q): return "lookup"
    return "conversational"


def _schema_summary_text(data: LoadedData) -> str:
    lines = []
    for name, schema in data.schemas.items():
        cols = [f"{c['column']} ({c['dtype']})" for c in schema]
        lines.append(f"Table '{name}' [{len(data.tables[name])} rows]: " + ", ".join(cols))
    return "\n".join(lines)


def build_system_prompt(description: str, data: LoadedData) -> str:
    return f"""You are an expert data analyst and assistant for the following database application.

PURPOSE OF THIS CHATBOT:
{description}

DATABASE SCHEMA:
{_schema_summary_text(data)}

INSTRUCTIONS:
1. Answer questions using ONLY the information provided in the context chunks below.
2. For analytical questions: perform calculations, identify trends, compare values.
3. For lookup questions: extract specific records or fields requested.
4. For conversational questions: be friendly and helpful using the database context.
5. If the answer is not in the provided context, say so clearly — do NOT fabricate data.
6. Always cite which table/column your answer comes from when relevant.
7. Format numbers clearly; use markdown tables for structured data comparisons.
8. Be concise but thorough. Think step by step for complex questions.
"""


def generate_example_questions(description: str, data: LoadedData, n: int = 5) -> List[str]:
    schema_text = _schema_summary_text(data)
    prompt = (f'Given this database chatbot purpose: "{description}"\n'
              f"And these tables: {schema_text}\n\n"
              f"Generate exactly {n} diverse, specific, useful example questions a user might ask.\n"
              f'Return ONLY a JSON array of strings. Example: ["Q1?", "Q2?", "Q3?"]')
    raw = generate_sync(prompt, max_tokens=300)
    try:
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            questions = json.loads(match.group())
            if isinstance(questions, list):
                return [str(q) for q in questions[:n]]
    except Exception:
        pass
    lines = [l.strip().lstrip("0123456789.-) ") for l in raw.splitlines() if "?" in l]
    return lines[:n] if lines else ["What are the main trends in this data?",
                                    "Show me the top 5 entries.",
                                    "What is the average value?",
                                    "How many unique categories are there?",
                                    "What is the most common value?"]


class RAGPipeline:
    def __init__(self, data: LoadedData, description: str):
        self.data          = data
        self.description   = description
        self.system_prompt = build_system_prompt(description, data)
        chunks             = build_chunks(data, description, strategy="auto")
        self.retriever     = HybridRetriever(chunks)
        self.memory: deque = deque(maxlen=16)   # 8-turn sliding window
        self._example_questions: Optional[List[str]] = None

    @property
    def example_questions(self) -> List[str]:
        if self._example_questions is None:
            self._example_questions = generate_example_questions(self.description, self.data, n=5)
        return self._example_questions

    def _build_context(self, query: str, qtype: str) -> str:
        top_k  = 8 if qtype == "analytical" else 6
        chunks = self.retriever.retrieve(query, top_k=top_k)
        if not chunks: return "No relevant data found in the database."
        sections = []
        for i, chunk in enumerate(chunks, 1):
            meta  = chunk.metadata
            label = f"[Context {i} | type={meta.get('chunk_type')} table={meta.get('table')}]"
            sections.append(f"{label}\n{chunk.text}")
        return "\n\n---\n\n".join(sections)

    def chat(self, user_message: str) -> Generator[str, None, None]:
        qtype   = classify_query(user_message)
        context = self._build_context(user_message, qtype)
        augmented = (f"[Query type: {qtype}]\n\nRETRIEVED CONTEXT:\n{context}"
                     f"\n\nUSER QUESTION: {user_message}")
        self.memory.append({"role": "user", "content": augmented})
        response_parts: List[str] = []
        for token in stream_response(messages=list(self.memory),
                                      system_prompt=self.system_prompt):
            response_parts.append(token)
            yield token
        self.memory.append({"role": "assistant", "content": "".join(response_parts)})

    def session_info(self) -> Dict:
        return {"tables":       list(self.data.tables.keys()),
                "total_rows":   sum(len(df) for df in self.data.tables.values()),
                "total_chunks": len(self.retriever.chunks),
                "memory_turns": len(self.memory) // 2}


# ══════════════════════════════════════════════════════════════════════════════
# ⑦ GRADIO UI  (full port of app.py — Gradio 5 compatible)
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
:root {
  --brand-primary: #7C3AED;
  --brand-secondary: #4F46E5;
  --brand-accent: #06B6D4;
  --brand-success: #10B981;
  --bg-dark: #0F0F14;
  --bg-card: #1A1A2E;
  --bg-input: #16213E;
  --text-primary: #F1F5F9;
  --text-muted: #94A3B8;
  --border-color: #334155;
}
.gradio-container {
  background: var(--bg-dark) !important;
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
  max-width: 1200px !important;
  margin: 0 auto !important;
}
footer { display: none !important; }
.header-banner {
  background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary), var(--brand-accent));
  border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
  text-align: center; box-shadow: 0 20px 60px rgba(0,0,0,0.6);
}
.header-banner h1 { color:#fff; font-size:2rem; font-weight:800; margin:0 0 0.4rem; }
.header-banner p  { color:rgba(255,255,255,0.85); font-size:1rem; margin:0; }
textarea, input[type=text] {
  background: var(--bg-input) !important;
  border: 1px solid var(--border-color) !important;
  border-radius: 10px !important;
  color: var(--text-primary) !important;
  font-size: 0.9rem !important;
  transition: border-color 0.2s !important;
}
textarea:focus, input[type=text]:focus {
  border-color: var(--brand-primary) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(124,58,237,0.2) !important;
}
.message.user {
  background: linear-gradient(135deg, var(--brand-primary), var(--brand-secondary)) !important;
  color: white !important; border-radius: 14px 14px 4px 14px !important;
}
.message.bot {
  background: var(--bg-card) !important; color: var(--text-primary) !important;
  border: 1px solid var(--border-color) !important; border-radius: 14px 14px 14px 4px !important;
}
.sidebar-stat {
  display:flex; justify-content:space-between; align-items:center;
  padding:0.5rem 0; border-bottom:1px solid rgba(51,65,85,0.5);
}
.sidebar-stat .label { color:#94A3B8; font-size:0.82rem; }
.sidebar-stat .value { color:#06B6D4; font-weight:700; font-size:0.88rem; }
.sidebar-title {
  color:#94A3B8; font-size:0.8rem; font-weight:700;
  text-transform:uppercase; letter-spacing:1px;
  padding-bottom:0.4rem; border-bottom:1px solid #334155;
}
.badge {
  display:inline-block; background:rgba(124,58,237,0.2); color:#c4b5fd;
  border:1px solid rgba(124,58,237,0.4); border-radius:999px;
  padding:0.2rem 0.7rem; font-size:0.75rem; font-weight:600;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--brand-primary); }
.gr-markdown { color: var(--text-primary) !important; }
.gr-markdown h2, .gr-markdown h3 { color: #c4b5fd; }
.gr-markdown code {
  background: var(--bg-input); padding: 0.2em 0.5em;
  border-radius: 4px; font-size: 0.85em; color: var(--brand-accent);
}
.gr-markdown table { border-collapse:collapse; width:100%; }
.gr-markdown table th {
  background:rgba(124,58,237,0.2); color:#c4b5fd;
  padding:0.5rem 1rem; text-align:left;
}
.gr-markdown table td {
  border-bottom:1px solid var(--border-color);
  padding:0.5rem 1rem; color:var(--text-primary);
}
"""

HEADER_HTML = """
<div class="header-banner">
  <h1>🤖 RAGlitheon — Instant RAG Chatbot Builder</h1>
  <p>Upload any database · Describe your chatbot · Start chatting — 100% free, open-source LLMs</p>
</div>
"""

STEPS_HTML = """
<div style="background:linear-gradient(135deg,rgba(124,58,237,0.12),rgba(6,182,212,0.08));
            border:1px solid rgba(124,58,237,0.3);border-radius:12px;
            padding:1.2rem 1.5rem;margin-bottom:0.8rem;">
  <div style="display:flex;gap:2rem;flex-wrap:wrap;justify-content:center;">
    <div style="text-align:center;">
      <div style="font-size:1.8rem">📁</div>
      <strong style="color:#c4b5fd">Step 1</strong>
      <p style="color:#94a3b8;margin:0.2rem 0 0;font-size:0.82rem">Upload CSV / Excel / JSON</p>
    </div>
    <div style="text-align:center;">
      <div style="font-size:1.8rem">✏️</div>
      <strong style="color:#c4b5fd">Step 2</strong>
      <p style="color:#94a3b8;margin:0.2rem 0 0;font-size:0.82rem">Describe chatbot purpose</p>
    </div>
    <div style="text-align:center;">
      <div style="font-size:1.8rem">🚀</div>
      <strong style="color:#c4b5fd">Step 3</strong>
      <p style="color:#94a3b8;margin:0.2rem 0 0;font-size:0.82rem">Build &amp; chat instantly</p>
    </div>
    <div style="text-align:center;">
      <div style="font-size:1.8rem">🆓</div>
      <strong style="color:#c4b5fd">Free</strong>
      <p style="color:#94a3b8;margin:0.2rem 0 0;font-size:0.82rem">Open-source LLMs only</p>
    </div>
  </div>
</div>
"""

FORMATS_HTML = """
<p style="color:#94a3b8;font-size:0.78rem;margin:0.3rem 0;">
  <strong style="color:#c4b5fd">Supported:</strong>
  CSV (auto-delimiter) &nbsp;·&nbsp; Excel .xlsx/.xls (all sheets) &nbsp;·&nbsp;
  JSON / JSONL &nbsp;·&nbsp; TSV
</p>
"""


def _make_sidebar_html(data: LoadedData, info: Dict) -> str:
    table_items = "".join(
        f'<div class="sidebar-stat"><span class="label">📋 {name}</span>'
        f'<span class="value">{len(df):,} rows</span></div>'
        for name, df in data.tables.items()
    )
    return f"""<div>
      <div class="sidebar-title">📊 Database Info</div>
      {table_items}
      <div class="sidebar-stat">
        <span class="label">Total Rows</span><span class="value">{info['total_rows']:,}</span>
      </div>
      <div class="sidebar-stat">
        <span class="label">Index Chunks</span><span class="value">{info['total_chunks']:,}</span>
      </div>
      <div class="sidebar-stat">
        <span class="label">File Type</span><span class="value">{data.file_type}</span>
      </div>
      <div class="sidebar-stat">
        <span class="label">Encoding</span><span class="value">{data.encoding}</span>
      </div>
    </div>"""


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_pipeline(file_obj, description: str, progress=gr.Progress()):
    if file_obj is None:
        return (None, "❌ Please upload a file first.",
                gr.update(visible=True), gr.update(visible=False),
                "", *[gr.update(value="", visible=False)] * 5)

    if not description or len(description.strip()) < 10:
        return (None, "❌ Please provide a description (at least 10 characters).",
                gr.update(visible=True), gr.update(visible=False),
                "", *[gr.update(value="", visible=False)] * 5)
    try:
        progress(0.10, desc="📂 Loading and parsing file…")
        loaded: LoadedData = load_file(file_obj.name)

        if not loaded.tables:
            errs = "; ".join(loaded.errors) or "Unknown parse error"
            return (None, f"❌ Could not parse file: {errs}",
                    gr.update(visible=True), gr.update(visible=False),
                    "", *[gr.update(value="", visible=False)] * 5)

        progress(0.35, desc="✂️ Chunking data intelligently…")
        progress(0.55, desc="🔢 Building embeddings (~30 s)…")
        pipeline = RAGPipeline(loaded, description.strip())

        progress(0.85, desc="💡 Generating example questions…")
        questions = pipeline.example_questions
        sidebar   = _make_sidebar_html(loaded, pipeline.session_info())

        q_updates = [gr.update(value=questions[i], visible=True) if i < len(questions)
                     else gr.update(value="", visible=False) for i in range(5)]

        progress(1.0, desc="✅ Done!")
        return (pipeline,
                f"✅ Chatbot ready! Built from **{Path(file_obj.name).name}**",
                gr.update(visible=False), gr.update(visible=True),
                sidebar, *q_updates)

    except Exception as e:
        print(f"[ERROR] build_pipeline:\n{traceback.format_exc()}")
        return (None, f"❌ Build failed: {e}",
                gr.update(visible=True), gr.update(visible=False),
                "", *[gr.update(value="", visible=False)] * 5)


# ── Chat handler ──────────────────────────────────────────────────────────────

def chat_respond(user_message: str, history: List[Dict[str, str]],
                 pipeline_state) -> Generator:
    if not user_message or not user_message.strip():
        yield history, ""; return

    if pipeline_state is None:
        yield (history + [{"role": "user",      "content": user_message},
                           {"role": "assistant", "content": "⚠️ No pipeline loaded. Please build the chatbot first."}],
               ""); return

    history = history + [{"role": "user", "content": user_message},
                          {"role": "assistant", "content": ""}]
    yield history, ""

    full_response = ""
    for token in pipeline_state.chat(user_message.strip()):
        full_response += token
        history[-1] = {"role": "assistant", "content": full_response}
        yield history, ""

    history[-1] = {"role": "assistant", "content": full_response}
    yield history, ""


def reset_to_builder(pipeline_state):
    del pipeline_state
    return (None, [], gr.update(visible=True), gr.update(visible=False),
            gr.update(value="", visible=False))


# ── Gradio Blocks ─────────────────────────────────────────────────────────────

def create_app():
    with gr.Blocks(
        css=CSS,
        title="RAGlitheon — RAG Chatbot Builder",
        theme=gr.themes.Base(primary_hue="violet", secondary_hue="indigo",
                             neutral_hue="slate"),
    ) as app:

        pipeline_state = gr.State(None)
        gr.HTML(HEADER_HTML)

        # ── STAGE 1: BUILDER ──────────────────────────────────────────────────
        with gr.Column(visible=True) as builder_panel:
            gr.HTML(STEPS_HTML)
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Markdown("### 📁 Upload Your Database")
                    gr.HTML(FORMATS_HTML)
                    file_input = gr.File(
                        label="Drop your file here or click to browse",
                        file_types=[".csv", ".xlsx", ".xls", ".json", ".jsonl", ".tsv", ".txt"],
                    )
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        gr.Markdown("Chunking strategy is set automatically. Override here if needed.")
                        chunk_strategy = gr.Radio(
                            choices=["auto", "row_window", "column_group", "hybrid"],
                            value="auto", label="Chunking Strategy",
                        )
                with gr.Column(scale=1):
                    gr.Markdown("### ✏️ Describe Your Chatbot")
                    description_input = gr.Textbox(
                        placeholder=(
                            "Example: This is a sales analytics chatbot. "
                            "Help users explore revenue trends, find top customers, "
                            "compare product performance, and answer questions about "
                            "sales targets and regional breakdowns."
                        ),
                        label="What should this chatbot help users do?",
                        lines=7, max_lines=12,
                    )
                    gr.Markdown("_💡 The more specific your description, the smarter the chatbot._")
                    build_btn = gr.Button("🚀  Build My Chatbot", variant="primary", size="lg")

            build_status = gr.Markdown(value="", visible=True)

        # ── STAGE 2: CHAT ─────────────────────────────────────────────────────
        with gr.Column(visible=False) as chat_panel:
            with gr.Row():
                with gr.Column(scale=1, min_width=240):
                    sidebar_html = gr.HTML(value="")
                    gr.Markdown("---")
                    gr.Markdown("### 💡 Example Questions")
                    example_btns: List[gr.Button] = []
                    for _ in range(5):
                        b = gr.Button(value="", visible=False, size="sm")
                        example_btns.append(b)
                    gr.Markdown("---")
                    reset_btn = gr.Button("🔄 Build New Chatbot", variant="secondary", size="sm")

                with gr.Column(scale=3):
                    gr.Markdown("### 💬 Chat with Your Data")
                    chatbot = gr.Chatbot(
                        value=[], height=480, show_label=False,
                        bubble_full_width=False, render_markdown=True,
                        type="messages",
                        avatar_images=(None, "https://api.dicebear.com/9.x/bottts-neutral/svg?seed=RAGlitheon"),
                    )
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Ask anything about your data…",
                            show_label=False, lines=1, max_lines=5, scale=5,
                        )
                        send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=80)
                    clear_btn = gr.Button("🗑 Clear Chat", variant="secondary", size="sm")

        # ── EVENT WIRING ──────────────────────────────────────────────────────

        build_btn.click(
            fn=lambda: gr.update(visible=True, value="⏳ Building your RAG chatbot… (1–3 min)"),
            outputs=[build_status],
        ).then(
            fn=build_pipeline,
            inputs=[file_input, description_input],
            outputs=[pipeline_state, build_status, builder_panel,
                     chat_panel, sidebar_html, *example_btns],
            show_progress="full",
        )

        chat_input.submit(
            fn=chat_respond,
            inputs=[chat_input, chatbot, pipeline_state],
            outputs=[chatbot, chat_input],
            show_progress="hidden",
        )
        send_btn.click(
            fn=chat_respond,
            inputs=[chat_input, chatbot, pipeline_state],
            outputs=[chatbot, chat_input],
            show_progress="hidden",
        )

        for btn in example_btns:
            btn.click(fn=lambda q: q, inputs=[btn], outputs=[chat_input]).then(
                fn=chat_respond,
                inputs=[chat_input, chatbot, pipeline_state],
                outputs=[chatbot, chat_input],
            )

        clear_btn.click(fn=lambda: [], outputs=[chatbot])
        reset_btn.click(
            fn=reset_to_builder,
            inputs=[pipeline_state],
            outputs=[pipeline_state, chatbot, builder_panel, chat_panel, build_status],
        )

    return app


if __name__ == "__main__":
    create_app().launch(
        server_name="0.0.0.0",   # bind all interfaces (required for HF Spaces)
        show_api=False,           # CRITICAL: prevents get_api_info() crash in gradio_client
        ssr_mode=False,           # CRITICAL: disables SSR which triggers the crash
        show_error=True,
    )
