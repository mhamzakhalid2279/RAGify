"""
RAG Pipeline — Full Orchestration
──────────────────────────────────
Fixed: generate_sync now uses max_tokens (not max_new_tokens)
"""

from __future__ import annotations

import json
import re
from collections import deque
from typing import Deque, Dict, Generator, List

from .chunker import build_chunks
from .data_loader import LoadedData
from .llm_client import generate_sync, stream_response
from .retriever import HybridRetriever

# ──────────────────────────────────────────────
# Query classifier
# ──────────────────────────────────────────────

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
    """Return 'analytical' | 'lookup' | 'conversational'."""
    q = query.lower()
    for pat in ANALYTICAL_PATTERNS:
        if re.search(pat, q):
            return "analytical"
    for pat in LOOKUP_PATTERNS:
        if re.search(pat, q):
            return "lookup"
    return "conversational"


# ──────────────────────────────────────────────
# Dynamic system prompt
# ──────────────────────────────────────────────

def _schema_summary_text(data: LoadedData) -> str:
    lines = []
    for name, schema in data.schemas.items():
        cols = [f"{c['column']} ({c['dtype']})" for c in schema]
        lines.append(f"Table '{name}' [{len(data.tables[name])} rows]: " + ", ".join(cols))
    return "\n".join(lines)


def build_system_prompt(description: str, data: LoadedData) -> str:
    schema_text = _schema_summary_text(data)
    return f"""You are an expert data analyst and assistant for the following database application.

PURPOSE OF THIS CHATBOT:
{description}

DATABASE SCHEMA:
{schema_text}

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


# ──────────────────────────────────────────────
# Example question generator
# ──────────────────────────────────────────────

def generate_example_questions(description: str, data: LoadedData, n: int = 5) -> List[str]:
    schema_text = _schema_summary_text(data)
    prompt = f"""Given this database chatbot purpose: "{description}"
And these tables: {schema_text}

Generate exactly {n} diverse, specific, useful example questions a user might ask about this database.
Return ONLY a JSON array of strings, nothing else. Example:
["Question 1?", "Question 2?", "Question 3?"]"""

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
    return lines[:n] if lines else [
        "What are the main trends in this data?",
        "Show me the top 5 entries.",
        "What is the average value?",
        "How many unique categories are there?",
        "What is the most common value?",
    ]


# ──────────────────────────────────────────────
# Main RAG Pipeline
# ──────────────────────────────────────────────

class RAGPipeline:
    """Fully orchestrated RAG chatbot instance. One per user session."""

    def __init__(self, data: LoadedData, description: str):
        self.data = data
        self.description = description
        self.system_prompt = build_system_prompt(description, data)

        chunks = build_chunks(data, description, strategy="auto")
        self.retriever = HybridRetriever(chunks)

        # 8-turn sliding memory window (16 messages = 8 pairs)
        self.memory: Deque[Dict[str, str]] = deque(maxlen=16)
        self._example_questions: List[str] | None = None

    @property
    def example_questions(self) -> List[str]:
        if self._example_questions is None:
            self._example_questions = generate_example_questions(
                self.description, self.data, n=5
            )
        return self._example_questions

    def _build_context(self, query: str, qtype: str) -> str:
        top_k = 8 if qtype == "analytical" else 6
        chunks = self.retriever.retrieve(query, top_k=top_k)
        if not chunks:
            return "No relevant data found in the database."
        sections = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.metadata
            label = f"[Context {i} | type={meta.get('chunk_type')} table={meta.get('table')}]"
            sections.append(f"{label}\n{chunk.text}")
        return "\n\n---\n\n".join(sections)

    def chat(self, user_message: str) -> Generator[str, None, None]:
        """Stream a response. Maintains sliding memory internally."""
        qtype = classify_query(user_message)
        context = self._build_context(user_message, qtype)

        augmented = (
            f"[Query type: {qtype}]\n\n"
            f"RETRIEVED CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {user_message}"
        )

        self.memory.append({"role": "user", "content": augmented})

        response_parts = []
        for token in stream_response(
            messages=list(self.memory),
            system_prompt=self.system_prompt,
        ):
            response_parts.append(token)
            yield token

        full_response = "".join(response_parts)
        self.memory.append({"role": "assistant", "content": full_response})

    def session_info(self) -> Dict:
        return {
            "tables": list(self.data.tables.keys()),
            "total_rows": sum(len(df) for df in self.data.tables.values()),
            "total_chunks": len(self.retriever.chunks),
            "memory_turns": len(self.memory) // 2,
        }