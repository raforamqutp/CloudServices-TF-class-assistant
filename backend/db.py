import os
import psycopg
from pgvector.psycopg import register_vector

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://cloud_user:cloud_pass@localhost:5433/lecture_db",
)


def get_connection():
    return psycopg.connect(DATABASE_URL)


def init_db():
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lectures (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                duration_sec INTEGER DEFAULT 0,
                hls_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcript_chunks (
                id SERIAL PRIMARY KEY,
                lecture_id INTEGER REFERENCES lectures(id) ON DELETE CASCADE,
                start_sec INTEGER NOT NULL,
                end_sec INTEGER NOT NULL,
                timestamp_label TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding vector(384) NOT NULL
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
            ON transcript_chunks
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_fts_es
            ON transcript_chunks
            USING gin (to_tsvector('spanish', content));
        """)
        conn.commit()


if __name__ == "__main__":
    init_db()
