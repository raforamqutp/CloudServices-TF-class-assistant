import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from backend.db import DATABASE_URL, init_db

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def parse_timestamp_to_sec(ts: str) -> int:
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return 0


def format_sec_label(sec: int) -> str:
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_transcript_file(filepath: str, max_words_per_chunk: int = 95) -> List[Dict]:
    pattern = re.compile(r"^\[(\d{2}:\d{2}:\d{2}|\d{2}:\d{2})\]\s*(.+)$")
    entries: List[Tuple[int, str]] = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = pattern.match(line)
            if not m:
                continue
            sec = parse_timestamp_to_sec(m.group(1))
            text = m.group(2).strip()
            if len(text) < 4 or text.lower().count("no,") > 5:
                continue
            entries.append((sec, text))

    chunks: List[Dict] = []
    if not entries:
        return chunks

    curr_texts: List[str] = []
    curr_words = 0
    start_sec = entries[0][0]
    last_sec = start_sec

    for sec, text in entries:
        words = len(text.split())
        if curr_words > 0 and (curr_words + words > max_words_per_chunk or (sec - start_sec) > 90):
            chunks.append(
                {
                    "start_sec": start_sec,
                    "end_sec": last_sec,
                    "timestamp_label": format_sec_label(start_sec),
                    "content": " ".join(curr_texts),
                }
            )
            overlap_text = curr_texts[-1] if curr_texts else ""
            curr_texts = [overlap_text, text] if overlap_text else [text]
            curr_words = sum(len(t.split()) for t in curr_texts)
            start_sec = last_sec
        else:
            curr_texts.append(text)
            curr_words += words
        last_sec = sec

    if curr_texts:
        chunks.append(
            {
                "start_sec": start_sec,
                "end_sec": last_sec,
                "timestamp_label": format_sec_label(start_sec),
                "content": " ".join(curr_texts),
            }
        )

    return chunks


def ingest_transcript(filepath: str, title: str, hls_path: str = "") -> Dict:
    t0 = time.perf_counter()
    filename = os.path.basename(filepath)
    chunks = parse_transcript_file(filepath)
    if not chunks:
        return {"filename": filename, "chunks": 0, "status": "empty"}

    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    t_emb_start = time.perf_counter()
    embeddings = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    t_emb = time.perf_counter() - t_emb_start

    duration_sec = chunks[-1]["end_sec"]

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        row = conn.execute(
            """
            INSERT INTO lectures (filename, title, duration_sec, hls_path)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (filename) DO UPDATE
            SET title = EXCLUDED.title,
                duration_sec = EXCLUDED.duration_sec,
                hls_path = EXCLUDED.hls_path
            RETURNING id;
            """,
            (filename, title, duration_sec, hls_path),
        ).fetchone()
        lecture_id = row[0]

        conn.execute("DELETE FROM transcript_chunks WHERE lecture_id = %s;", (lecture_id,))

        with conn.cursor() as cur:
            for c, emb in zip(chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO transcript_chunks (lecture_id, start_sec, end_sec, timestamp_label, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                    (
                        lecture_id,
                        c["start_sec"],
                        c["end_sec"],
                        c["timestamp_label"],
                        c["content"],
                        emb.astype(np.float32),
                    ),
                )
        conn.commit()

    total_time = time.perf_counter() - t0
    return {
        "lecture_id": lecture_id,
        "filename": filename,
        "title": title,
        "chunks": len(chunks),
        "duration_sec": duration_sec,
        "embed_time_s": round(t_emb, 3),
        "total_time_s": round(total_time, 3),
        "status": "ok",
    }


def ingest_all_transcriptions(trans_dir: str = r"D:\Projects\Cloud\transcriptions") -> List[Dict]:
    init_db()
    results = []
    files = sorted(f for f in os.listdir(trans_dir) if f.startswith("transcripcion_") and f.endswith(".txt"))
    for f in files:
        full_path = os.path.join(trans_dir, f)
        if "demo_gestion_conocimiento" in f:
            title = "Gestión del Conocimiento (Clase Demo)"
            hls_rel = "/hls/demo_gestion_conocimiento/index.m3u8"
        else:
            title = f.replace("transcripcion_", "Servicios Cloud - ").replace(".txt", "").replace("_", " ").title()
            hls_rel = "/hls/test_hls_gpu/index.m3u8" if "20261019" in f else ""
        res = ingest_transcript(full_path, title, hls_rel)
        results.append(res)
        print(
            f"[INGEST] {res['filename']}: {res['chunks']} chunks | "
            f"embed={res.get('embed_time_s', 0)}s | total={res.get('total_time_s', 0)}s"
        )
    return results


if __name__ == "__main__":
    ingest_all_transcriptions()
