"""
finley/memory.py — Cross-session semantic memory for advice Q&A pairs.

Stores past advice exchanges as embeddings. On new questions, retrieves
semantically similar past exchanges to give Finley continuity across sessions.

Only advice-type Q&As are stored (not data lookups — those answers go stale).
"""

import json
import os
import time
from typing import Optional

import numpy as np

_MEMORY_DIR  = ".finley_memory"
_QA_FILE     = os.path.join(_MEMORY_DIR, "conversations.json")
_EMB_FILE    = os.path.join(_MEMORY_DIR, "embeddings.npy")
_MODEL_NAME  = "all-MiniLM-L6-v2"
_MAX_STORED  = 200   # cap to prevent unbounded growth
_THRESHOLD   = 0.55  # cosine similarity cutoff
_TOP_K       = 2     # max past exchanges to inject


def _load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(_MODEL_NAME)


class MemoryStore:
    def __init__(self):
        self._model  = None   # lazy-load on first use
        self._qa: list[dict]    = []
        self._emb: Optional[np.ndarray] = None
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def store(self, question: str, answer: str) -> None:
        """Embed the question and persist the Q&A pair."""
        vec = self._embed(question)
        entry = {
            "question":  question,
            "answer":    answer[:600],   # trim very long answers
            "timestamp": int(time.time()),
        }
        self._qa.append(entry)
        if self._emb is None:
            self._emb = vec.reshape(1, -1)
        else:
            self._emb = np.vstack([self._emb, vec])

        # Keep only the most recent _MAX_STORED entries
        if len(self._qa) > _MAX_STORED:
            self._qa  = self._qa[-_MAX_STORED:]
            self._emb = self._emb[-_MAX_STORED:]

        self._save()

    def search(self, question: str) -> list[dict]:
        """Return top-k past Q&As most similar to question (above threshold)."""
        if self._emb is None or len(self._qa) == 0:
            return []

        vec    = self._embed(question)
        scores = self._cosine(vec, self._emb)
        top    = np.argsort(scores)[::-1][:_TOP_K]

        results = []
        for idx in top:
            if scores[idx] >= _THRESHOLD:
                results.append({**self._qa[idx], "score": float(scores[idx])})
        return results

    def format_context(self, matches: list[dict]) -> str:
        """Format retrieved matches as a short context block for the prompt."""
        if not matches:
            return ""
        lines = ["Relevant context from previous conversations:"]
        for m in matches:
            lines.append(f'  Q: {m["question"]}')
            lines.append(f'  A: {m["answer"]}')
        return "\n".join(lines)

    def count(self) -> int:
        return len(self._qa)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        if self._model is None:
            self._model = _load_model()
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.astype(np.float32)

    @staticmethod
    def _cosine(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        # Both vec and matrix rows are already L2-normalised by sentence-transformers
        return matrix @ vec

    def _load(self) -> None:
        if os.path.exists(_QA_FILE):
            with open(_QA_FILE, encoding="utf-8") as f:
                self._qa = json.load(f)
        if os.path.exists(_EMB_FILE) and self._qa:
            self._emb = np.load(_EMB_FILE).astype(np.float32)
            # Guard against file/json mismatch
            if len(self._emb) != len(self._qa):
                self._emb = None
                self._qa  = []

    def _save(self) -> None:
        os.makedirs(_MEMORY_DIR, exist_ok=True)
        with open(_QA_FILE, "w", encoding="utf-8") as f:
            json.dump(self._qa, f, indent=2)
        np.save(_EMB_FILE, self._emb)
