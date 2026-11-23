import threading
import time
from typing import Dict, Optional

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from backend.db import DATABASE_URL
from backend.ingest import get_embedding_model

_encode_lock = threading.Lock()


def semantic_search(query: str, top_k: int = 5, lecture_id: Optional[int] = None) -> Dict:
    t0 = time.perf_counter()
    model = get_embedding_model()
    with _encode_lock:
        q_emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)

    t_db0 = time.perf_counter()
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        if lecture_id is not None:
            rows = conn.execute(
                """
                SELECT c.id, l.id, l.title, l.filename, l.hls_path,
                       c.start_sec, c.end_sec, c.timestamp_label, c.content,
                       1 - (c.embedding <=> %s) AS similarity
                FROM transcript_chunks c
                JOIN lectures l ON l.id = c.lecture_id
                WHERE c.lecture_id = %s
                ORDER BY c.embedding <=> %s
                LIMIT %s;
                """,
                (q_emb, lecture_id, q_emb, top_k),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT c.id, l.id, l.title, l.filename, l.hls_path,
                       c.start_sec, c.end_sec, c.timestamp_label, c.content,
                       1 - (c.embedding <=> %s) AS similarity
                FROM transcript_chunks c
                JOIN lectures l ON l.id = c.lecture_id
                ORDER BY c.embedding <=> %s
                LIMIT %s;
                """,
                (q_emb, q_emb, top_k),
            ).fetchall()
    t_db = (time.perf_counter() - t_db0) * 1000.0
    t_total = (time.perf_counter() - t0) * 1000.0

    results = [
        {
            "chunk_id": r[0],
            "lecture_id": r[1],
            "lecture_title": r[2],
            "filename": r[3],
            "hls_path": r[4] or "",
            "start_sec": r[5],
            "end_sec": r[6],
            "timestamp": r[7],
            "content": r[8],
            "score": round(float(r[9]), 4),
        }
        for r in rows
    ]
    return {
        "results": results,
        "db_latency_ms": round(t_db, 2),
        "total_latency_ms": round(t_total, 2),
    }


def lexical_search(query: str, top_k: int = 5, lecture_id: Optional[int] = None) -> Dict:
    t0 = time.perf_counter()
    with psycopg.connect(DATABASE_URL) as conn:
        if lecture_id is not None:
            rows = conn.execute(
                """
                SELECT c.id, l.id, l.title, l.filename, l.hls_path,
                       c.start_sec, c.end_sec, c.timestamp_label, c.content,
                       ts_rank(to_tsvector('spanish', c.content), plainto_tsquery('spanish', %s)) AS rank
                FROM transcript_chunks c
                JOIN lectures l ON l.id = c.lecture_id
                WHERE c.lecture_id = %s
                  AND to_tsvector('spanish', c.content) @@ plainto_tsquery('spanish', %s)
                ORDER BY rank DESC
                LIMIT %s;
                """,
                (query, lecture_id, query, top_k),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT c.id, l.id, l.title, l.filename, l.hls_path,
                       c.start_sec, c.end_sec, c.timestamp_label, c.content,
                       ts_rank(to_tsvector('spanish', c.content), plainto_tsquery('spanish', %s)) AS rank
                FROM transcript_chunks c
                JOIN lectures l ON l.id = c.lecture_id
                WHERE to_tsvector('spanish', c.content) @@ plainto_tsquery('spanish', %s)
                ORDER BY rank DESC
                LIMIT %s;
                """,
                (query, query, top_k),
            ).fetchall()
    t_total = (time.perf_counter() - t0) * 1000.0

    results = [
        {
            "chunk_id": r[0],
            "lecture_id": r[1],
            "lecture_title": r[2],
            "filename": r[3],
            "hls_path": r[4] or "",
            "start_sec": r[5],
            "end_sec": r[6],
            "timestamp": r[7],
            "content": r[8],
            "score": round(float(r[9]), 4),
        }
        for r in rows
    ]
    return {
        "results": results,
        "total_latency_ms": round(t_total, 2),
    }
