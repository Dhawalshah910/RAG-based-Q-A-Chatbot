"""
rag_pipeline.py
Core RAG pipeline: PDF ingestion → chunking → embeddings → FAISS → retrieval → LLM answer.
Supports both OpenAI and a free HuggingFace-only mode (set USE_OPENAI=false in .env).
"""

import os
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_OPENAI     = os.getenv("USE_OPENAI", "true").lower() == "true" and bool(OPENAI_API_KEY)
EMBED_MODEL    = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_PATH     = os.getenv("FAISS_INDEX_PATH", "/tmp/docmind_faiss")


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Parameters
    ----------
    chunk_size    : characters per chunk
    chunk_overlap : overlap between consecutive chunks
    top_k         : number of chunks to retrieve per query
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100, top_k: int = 4):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k         = top_k
        self.vectorstore: FAISS | None = None

        # Embedding model (local, no API key required)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # LLM backend
        if USE_OPENAI:
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model_name=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                temperature=0.2,
                openai_api_key=OPENAI_API_KEY,
            )
        else:
            # Free fallback: HuggingFace pipeline (flan-t5-base)
            from langchain_community.llms import HuggingFacePipeline
            from transformers import pipeline as hf_pipeline
            pipe = hf_pipeline(
                "text2text-generation",
                model="google/flan-t5-base",
                max_new_tokens=512,
            )
            self.llm = HuggingFacePipeline(pipeline=pipe)

    # ── Ingestion ─────────────────────────────────────────────────────────────────

    def _extract_text(self, pdf_path: str) -> tuple[list[Document], int]:
        """Extract text from PDF page by page, return Documents + page count."""
        docs: list[Document] = []
        with fitz.open(pdf_path) as pdf:
            n_pages = len(pdf)
            for page_num, page in enumerate(pdf, start=1):
                text = page.get_text("text")
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
                if text:
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": pdf_path, "page": page_num},
                    ))
        return docs, n_pages

    def _chunk_documents(self, docs: list[Document]) -> list[Document]:
        """Split documents into overlapping chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        return splitter.split_documents(docs)

    def ingest(self, pdf_path: str) -> dict[str, Any]:
        """
        Full ingestion pipeline.
        Returns stats dict: {chunks, pages, filename}.
        """
        docs, n_pages = self._extract_text(pdf_path)
        if not docs:
            raise ValueError("Could not extract any text from the PDF. Is it scanned?")

        chunks = self._chunk_documents(docs)
        if not chunks:
            raise ValueError("Chunking produced zero chunks.")

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(FAISS_PATH)

        return {
            "chunks":   len(chunks),
            "pages":    n_pages,
            "filename": Path(pdf_path).name,
        }

    # ── Query ─────────────────────────────────────────────────────────────────────

    def _build_prompt(self, context: str, question: str) -> str:
        return f"""You are an expert document assistant. Use ONLY the context below to answer the question.
If the answer is not contained in the context, say "I couldn't find relevant information in the document."
Be concise, accurate, and structured.

--- CONTEXT ---
{context}

--- QUESTION ---
{question}

--- ANSWER ---"""

    def query(self, question: str) -> dict[str, Any]:
        """
        Retrieve relevant chunks and generate an answer.
        Returns dict: {answer, source_pages, chunks_used}.
        """
        if self.vectorstore is None:
            raise RuntimeError("No vectorstore loaded. Call ingest() first.")

        # Semantic retrieval
        retrieved = self.vectorstore.similarity_search(question, k=self.top_k)

        # Build context
        context_parts = []
        source_pages: list[int] = []
        for doc in retrieved:
            context_parts.append(doc.page_content)
            pg = doc.metadata.get("page")
            if pg and pg not in source_pages:
                source_pages.append(pg)

        context = "\n\n---\n\n".join(context_parts)
        prompt  = self._build_prompt(context, question)

        # Generate
        if USE_OPENAI:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content
        else:
            answer = self.llm.invoke(prompt)

        return {
            "answer":       answer.strip(),
            "source_pages": sorted(source_pages),
            "chunks_used":  len(retrieved),
        }

    # ── Persistence ───────────────────────────────────────────────────────────────

    def load_index(self, path: str = FAISS_PATH) -> None:
        """Load a previously saved FAISS index from disk."""
        self.vectorstore = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )