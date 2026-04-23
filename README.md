# 🧠 DocMind – RAG Chatbot

A beautiful, production-grade **Retrieval-Augmented Generation (RAG)** chatbot built with:

- **Streamlit** – sleek dark UI
- **FAISS** – lightning-fast local vector similarity search
- **sentence-transformers** – free local embeddings (no API key needed)
- **LangChain** – orchestration layer
- **PyMuPDF** – PDF text extraction
- **OpenAI GPT** (optional) or **flan-t5-base** (free local LLM)

---

## 🚀 Quick Start

### 1. Clone / copy the project files

```
docmind/
├── app.py              ← Streamlit UI
├── rag_pipeline.py     ← RAG logic (ingest + query)
├── requirements.txt    ← Python dependencies
├── .env.example        ← Copy this to .env and fill in your keys
└── README.md
```

### 2. Create & activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run downloads the embedding model (~90 MB). Subsequent runs use the cache.

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

| Variable | Description |
|---|---|
| `USE_OPENAI` | `true` → use OpenAI GPT; `false` → use free flan-t5-base |
| `OPENAI_API_KEY` | Your OpenAI key (only needed when `USE_OPENAI=true`) |
| `OPENAI_MODEL` | e.g. `gpt-3.5-turbo` or `gpt-4o` |
| `EMBED_MODEL` | HuggingFace embedding model ID |
| `FAISS_INDEX_PATH` | Where the FAISS index is persisted |

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧩 How It Works

```
PDF Upload
    │
    ▼
PyMuPDF text extraction (page-by-page)
    │
    ▼
RecursiveCharacterTextSplitter → overlapping chunks
    │
    ▼
sentence-transformers → dense vector embeddings
    │
    ▼
FAISS index (saved to disk)
    │
    ▼
User question → embed → cosine similarity search → top-K chunks
    │
    ▼
Prompt = system instructions + retrieved context + question
    │
    ▼
LLM (OpenAI GPT or flan-t5) → Answer + source page numbers
```

---

## ⚙️ Tuning Parameters

In the sidebar you can adjust:

| Parameter | Effect |
|---|---|
| **Chunk size** | Larger = more context per chunk, but less precise retrieval |
| **Chunk overlap** | Higher overlap reduces boundary cut-off issues |
| **Top-K** | More chunks = richer context, but larger prompts |

---

## 🆓 Running 100% Free (No API Key)

Set `USE_OPENAI=false` in `.env`. The app will use:
- **Embeddings:** `all-MiniLM-L6-v2` (local, ~90 MB)
- **LLM:** `google/flan-t5-base` (local, ~250 MB)

Answers will be shorter and less fluent than GPT, but entirely free and private.

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `langchain` | RAG orchestration |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Local embeddings |
| `PyMuPDF` | PDF parsing |
| `openai` | GPT API (optional) |
| `python-dotenv` | `.env` loading |

---

## 🛠️ Troubleshooting

**PDF shows 0 chunks** → The PDF is likely scanned/image-based. Use a PDF with selectable text, or add OCR preprocessing with `pytesseract`.

**Out of memory** → Lower chunk size, or switch to a smaller embedding model.

**Slow on first run** → The embedding model is being downloaded. Subsequent runs are fast.

**OpenAI errors** → Double-check your `OPENAI_API_KEY` in `.env`.