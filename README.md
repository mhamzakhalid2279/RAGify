---
title: RAGify — Instant RAG Chatbot Builder
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.14.0"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Upload any DB file → get an instant AI chatbot. 100% free.
tags:
  - rag
  - chatbot
  - nlp
  - data-analysis
  - gradio
  - huggingface
---

# 🤖 RAGify — Instant RAG Chatbot Builder

**Upload any database file → describe your chatbot's purpose → start chatting with your data instantly.**

No API keys. No paid services. 100% open-source, free-tier LLMs.

## ✨ Features

| Feature | Details |
|---|---|
| **Universal Data Loader** | CSV, TSV, Excel (all sheets), JSON, JSONL — chardet encoding + separator detection |
| **Intelligent Chunking** | Auto-selects `row_window`, `column_group`, or `hybrid` based on table shape |
| **Hybrid Search** | FAISS dense + TinyBM25 keyword → RRF fusion → MMR diversification |
| **Free LLMs** | Zephyr-7B → Mistral-7B → Phi-3 → Falcon-7B (auto-fallback chain) |
| **Streaming** | Word-by-word token streaming |
| **Memory** | 8-turn sliding context window |
| **Example Questions** | AI-generated starter questions tailored to your data |

## 🚀 Deployment

This Space uses a single `app.py` — no subfolders needed.

Upload only:
```
app.py
requirements.txt
README.md
```

### Optional: HF Token for higher rate limits
Space Settings → Repository Secrets → `HF_TOKEN`

## 🧠 Architecture

```
File + Description
      ↓
Data Loader (chardet + separator sniff + multi-sheet Excel + JSON)
      ↓
Chunker (row_window / column_group / hybrid — auto selected)
      ↓
Embedder (MiniLM-L6-v2) + FAISS IndexFlatIP
      ↓
HybridRetriever: TinyBM25 + FAISS → RRF → MMR
      ↓
RAGPipeline: query classifier + dynamic prompt + 8-turn memory
      ↓
LLM: Zephyr-7B → Mistral-7B → Phi-3 → Falcon-7B (fallback chain)
      ↓
Gradio 5 Chat UI (type="messages", streaming)
```