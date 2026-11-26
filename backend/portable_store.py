import os
import sqlite3
import time
from typing import Dict, List, Optional

import numpy as np

SQLITE_DB_PATH = os.getenv(
    "PORTABLE_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "demo_laptop_db.sqlite"),
)


def init_sqlite(db_path: str = SQLITE_DB_PATH) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lectures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                course TEXT NOT NULL DEFAULT 'Servicios Cloud',
                duration_sec INTEGER DEFAULT 0,
                hls_path TEXT DEFAULT ''
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transcript_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lecture_id INTEGER NOT NULL REFERENCES lectures(id) ON DELETE CASCADE,
                start_sec INTEGER NOT NULL,
                end_sec INTEGER NOT NULL,
                timestamp_label TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL
            );
            """
        )
        conn.commit()


def sqlite_list_lectures(db_path: str = SQLITE_DB_PATH) -> List[Dict]:
    init_sqlite(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT l.id, l.filename, l.title, l.course, l.duration_sec, l.hls_path,
                   COUNT(c.id) AS chunk_count
            FROM lectures l
            LEFT JOIN transcript_chunks c ON c.lecture_id = l.id
            GROUP BY l.id
            ORDER BY l.course, (COALESCE(l.hls_path, '') <> '') DESC, l.title;
            """
        ).fetchall()
    return [
        {
            "id": r[0],
            "filename": r[1],
            "title": r[2],
            "course": r[3],
            "duration_sec": r[4],
            "hls_path": r[5] or "",
            "chunk_count": r[6],
        }
        for r in rows
    ]


def sqlite_save_lecture_chunks(
    filename: str,
    title: str,
    course: str,
    duration_sec: int,
    hls_path: str,
    chunks: List[Dict],
    embeddings: np.ndarray,
    db_path: str = SQLITE_DB_PATH,
) -> int:
    init_sqlite(db_path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO lectures (filename, title, course, duration_sec, hls_path)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                title = excluded.title,
                course = excluded.course,
                duration_sec = excluded.duration_sec,
                hls_path = excluded.hls_path;
            """,
            (filename, title, course, duration_sec, hls_path),
        )
        lecture_id = cur.execute(
            "SELECT id FROM lectures WHERE filename = ?;", (filename,)
        ).fetchone()[0]
        cur.execute("DELETE FROM transcript_chunks WHERE lecture_id = ?;", (lecture_id,))
        for c, emb in zip(chunks, embeddings):
            vec_bytes = np.asarray(emb, dtype=np.float32).tobytes()
            cur.execute(
                """
                INSERT INTO transcript_chunks (lecture_id, start_sec, end_sec, timestamp_label, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (lecture_id, c["start_sec"], c["end_sec"], c["timestamp_label"], c["content"], vec_bytes),
            )
        conn.commit()
    return lecture_id


def sqlite_semantic_search(
    q_emb: np.ndarray,
    top_k: int = 5,
    lecture_id: Optional[int] = None,
    course: Optional[str] = None,
    db_path: str = SQLITE_DB_PATH,
) -> Dict:
    t_db0 = time.perf_counter()
    init_sqlite(db_path)
    with sqlite3.connect(db_path) as conn:
        if lecture_id is not None:
            rows = conn.execute(
                """
                SELECT c.id, l.id, l.title, l.course, l.filename, l.hls_path,
                       c.start_sec, c.end_sec, c.timestamp_label, c.content, c.embedding
                FROM transcript_chunks c
                JOIN lectures l ON l.id = c.lecture_id
                WHERE c.lecture_id = ?;
                """,
                (lecture_id,),
            ).fetchall()
        elif course:
            rows = conn.execute(
                """
                SELECT c.id, l.id, l.title, l.course, l.filename, l.hls_path,
                       c.start_sec, c.end_sec, c.timestamp_label, c.content, c.embedding
                FROM transcript_chunks c
                JOIN lectures l ON l.id = c.lecture_id
                WHERE l.course = ?;
                """,
                (course,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT c.id, l.id, l.title, l.course, l.filename, l.hls_path,
                       c.start_sec, c.end_sec, c.timestamp_label, c.content, c.embedding
                FROM transcript_chunks c
                JOIN lectures l ON l.id = c.lecture_id;
                """
            ).fetchall()

    results = []
    if rows:
        mat = np.vstack([np.frombuffer(r[10], dtype=np.float32) for r in rows])
        scores = mat @ q_emb.astype(np.float32)
        top_idx = np.argsort(-scores)[:top_k]
        for idx in top_idx:
            r = rows[int(idx)]
            results.append(
                {
                    "chunk_id": r[0],
                    "lecture_id": r[1],
                    "lecture_title": r[2],
                    "course": r[3],
                    "filename": r[4],
                    "hls_path": r[5] or "",
                    "start_sec": r[6],
                    "end_sec": r[7],
                    "timestamp": r[8],
                    "content": r[9],
                    "score": round(float(scores[int(idx)]), 4),
                }
            )
    t_db = (time.perf_counter() - t_db0) * 1000.0
    return {"results": results, "db_latency_ms": round(t_db, 2)}
