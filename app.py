import streamlit as st
import os
import time
from pathlib import Path
from rag_pipeline import RAGPipeline

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind – RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:ital,wght@0,300;0,400;1,300&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --panel:     #16161f;
    --border:    #2a2a3a;
    --accent:    #7c6af7;
    --accent2:   #e46af7;
    --accent3:   #6af7d4;
    --text:      #e8e8f0;
    --muted:     #6b6b85;
    --radius:    14px;
    --font-head: 'Syne', sans-serif;
    --font-mono: 'DM Mono', monospace;
}

html, body, [class*="css"]  { font-family: var(--font-mono); }
.stApp                       { background: var(--bg); color: var(--text); }
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
header[data-testid="stHeader"] { background: transparent !important; }

::-webkit-scrollbar            { width: 4px; }
::-webkit-scrollbar-track      { background: transparent; }
::-webkit-scrollbar-thumb      { background: var(--border); border-radius: 2px; }

.hero-title {
    font-family: var(--font-head);
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff 10%, var(--accent) 55%, var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 999px;
    font-size: 0.7rem; font-weight: 600;
    letter-spacing: 0.05em; text-transform: uppercase;
}
.badge-idle  { background: #1a1a2e; color: var(--muted);  border: 1px solid var(--border); }
.badge-ready { background: #0d2b1f; color: #4ade80;        border: 1px solid #4ade8040; }

.msg-wrap { display: flex; gap: 14px; margin-bottom: 18px; align-items: flex-start; }
.msg-wrap.user { flex-direction: row-reverse; }
.avatar {
    width: 34px; height: 34px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
}
.avatar-user { background: linear-gradient(135deg, var(--accent), var(--accent2)); }
.avatar-ai   { background: #1a1a2e; border: 1px solid var(--border); }
.bubble {
    padding: 14px 18px; border-radius: var(--radius);
    max-width: 80%; font-size: 0.875rem; line-height: 1.7;
}
.bubble-user {
    background: linear-gradient(135deg, #7c6af722, #e46af711);
    border: 1px solid #7c6af744; color: var(--text);
}
.bubble-ai {
    background: var(--panel); border: 1px solid var(--border); color: var(--text);
}

.src-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.src-pill {
    font-size: 0.65rem; padding: 3px 10px;
    background: #1a1a2e; border: 1px solid #7c6af744;
    border-radius: 999px; color: var(--accent);
}

[data-testid="stFileUploadDropzone"] {
    background: var(--panel) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--accent) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity .2s, transform .1s !important;
}
.stButton > button:hover  { opacity: .85 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

.stChatInput textarea {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--font-mono) !important;
}
.stChatInput textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px #7c6af722 !important;
}

.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 14px; }
.stat-card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 12px 14px;
}
.stat-label { font-size: 0.62rem; text-transform: uppercase; letter-spacing: .12em; color: var(--muted); margin-bottom: 4px; }
.stat-val   { font-family: var(--font-head); font-size: 1.3rem; font-weight: 800; color: var(--text); }

.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
}

hr { border-color: var(--border) !important; }
.stMarkdown p { color: var(--text); }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────────
for key, val in [
    ("rag", None), ("messages", []),
    ("doc_stats", {"chunks": 0, "pages": 0, "filename": ""}),
    ("ready", False),
]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">DocMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">RAG · FAISS · Embeddings</div>', unsafe_allow_html=True)
    st.markdown("<hr style='margin:1rem 0'>", unsafe_allow_html=True)

    if st.session_state.ready:
        st.markdown('<span class="badge badge-ready">● Ready to chat</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-idle">○ No document loaded</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    with st.expander("⚙️ Chunking settings"):
        chunk_size    = st.slider("Chunk size (chars)", 300, 2000, 800, 50)
        chunk_overlap = st.slider("Chunk overlap (chars)", 0, 400, 100, 20)
        top_k         = st.slider("Top-K chunks to retrieve", 1, 10, 4)

    process_btn = st.button("⚡  Build Knowledge Base", use_container_width=True)

    if process_btn and uploaded_file:
        tmp_path = Path("/tmp") / uploaded_file.name
        tmp_path.write_bytes(uploaded_file.read())

        prog = st.progress(0, text="Extracting text…")
        try:
            rag = RAGPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k)
            prog.progress(30, text="Chunking document…")
            stats = rag.ingest(str(tmp_path))
            prog.progress(80, text="Building FAISS index…")
            time.sleep(0.3)
            prog.progress(100, text="Done!")

            st.session_state.update(
                rag=rag, ready=True, messages=[], doc_stats=stats
            )
            st.success(f"✅ Indexed {stats['chunks']} chunks from {stats['pages']} pages!")
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.ready:
        s = st.session_state.doc_stats
        st.markdown("<hr style='margin:1rem 0'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Chunks</div>
                <div class="stat-val">{s['chunks']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Pages</div>
                <div class="stat-val">{s['pages']}</div>
            </div>
        </div>
        <div style="font-size:0.68rem; color:var(--muted); word-break:break-all;">📄 {s['filename']}</div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='margin:1rem 0'>", unsafe_allow_html=True)
        if st.button("🗑️  Clear & Reset", use_container_width=True):
            for k in ["rag", "messages", "ready"]:
                st.session_state[k] = None if k == "rag" else ([] if k == "messages" else False)
            st.session_state.doc_stats = {"chunks": 0, "pages": 0, "filename": ""}
            st.rerun()

    st.markdown("<hr style='margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.68rem; color:var(--muted); line-height:1.8">
    <b style="color:#e8e8f0">Stack</b><br>
    FAISS · sentence-transformers<br>
    LangChain · PyMuPDF · OpenAI<br><br>
    <b style="color:#e8e8f0">Flow</b><br>
    PDF → chunks → embeddings<br>
    → FAISS index → retrieval<br>
    → LLM answer
    </div>
    """, unsafe_allow_html=True)

# ── Main panel ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.4rem 0 0.4rem">
    <span style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#e8e8f0;">
        Chat with your document
    </span>
    <span style="font-size:0.72rem; color:#6b6b85; margin-left:12px;">
        Semantic retrieval + LLM generation
    </span>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr style='margin:0.5rem 0 1.2rem'>", unsafe_allow_html=True)

# Message history
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding:4rem 0; color:#6b6b85;">
        <div style="font-size:3rem; margin-bottom:1rem;">🧠</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.15rem; font-weight:700;
                    color:#e8e8f030; margin-bottom:0.6rem;">
            Upload a PDF to begin
        </div>
        <div style="font-size:0.75rem; max-width:380px; margin:0 auto; line-height:1.8;">
            Load a PDF in the sidebar → Build the knowledge base → Ask anything about your document.
            All retrieval is done locally with FAISS.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        role, content = msg["role"], msg["content"]
        sources = msg.get("sources", [])

        if role == "user":
            st.markdown(f"""
            <div class="msg-wrap user">
                <div class="avatar avatar-user">👤</div>
                <div class="bubble bubble-user">{content}</div>
            </div>""", unsafe_allow_html=True)
        else:
            pills = "".join(f'<span class="src-pill">p.{p}</span>' for p in sources)
            src_html = f'<div class="src-row">{pills}</div>' if pills else ""
            st.markdown(f"""
            <div class="msg-wrap">
                <div class="avatar avatar-ai">🧠</div>
                <div class="bubble bubble-ai">
                    {content}
                    {src_html}
                </div>
            </div>""", unsafe_allow_html=True)

# Chat input
if st.session_state.ready:
    query = st.chat_input("Ask anything about your document…")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.spinner("Retrieving relevant chunks & generating answer…"):
            try:
                result = st.session_state.rag.query(query)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": result["answer"],
                    "sources": result.get("source_pages", []),
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error generating answer: {e}",
                })
        st.rerun()
else:
    st.chat_input("Upload and process a PDF first…", disabled=True)