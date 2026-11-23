import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import psycopg
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.db import DATABASE_URL
from backend.ingest import get_embedding_model
from backend.retriever import semantic_search


def _load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), "..", "mi_agente", ".env")
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


_load_env_file()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_embedding_model()
    yield


app = FastAPI(title="CloudLecture Assistant", version="1.0.0", lifespan=lifespan)

HLS_DIR = os.getenv("HLS_DIR", r"D:\Projects\Cloud\frames")
if os.path.isdir(HLS_DIR):
    app.mount("/hls", StaticFiles(directory=HLS_DIR), name="hls")


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    lecture_id: Optional[int] = None


@app.get("/api/lectures")
def list_lectures():
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            "SELECT id, filename, title, duration_sec, hls_path FROM lectures ORDER BY (hls_path <> '') DESC, id;"
        ).fetchall()
    return [
        {
            "id": r[0],
            "filename": r[1],
            "title": r[2],
            "duration_sec": r[3],
            "hls_path": r[4] or "",
        }
        for r in rows
    ]


@app.get("/api/search")
def search(q: str = Query(..., min_length=2), top_k: int = 5, lecture_id: Optional[int] = None):
    return semantic_search(q, top_k=top_k, lecture_id=lecture_id)


@app.post("/api/chat")
def chat(req: ChatRequest):
    retrieval = semantic_search(req.question, top_k=req.top_k, lecture_id=req.lecture_id)
    hits = retrieval["results"]
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()

    answer = ""
    used_llm = False
    model_used = ""
    llm_latency_ms = 0.0
    if api_key and hits:
        from google import genai
        from google.genai import types

        context = "\n".join(
            f"- [{h['timestamp']}] ({h['lecture_title']}): {h['content']}"
            for h in hits
        )
        prompt = (
            "Responde en español de forma clara y concisa usando únicamente los siguientes "
            "fragmentos de clase. Cita siempre la marca de tiempo exacta [MM:SS] o [HH:MM:SS]:\n\n"
            f"{context}\n\nPregunta: {req.question}"
        )
        client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=15000))
        for candidate in ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite"):
            try:
                t_llm0 = time.perf_counter()
                resp = client.models.generate_content(model=candidate, contents=prompt)
                llm_latency_ms = round((time.perf_counter() - t_llm0) * 1000.0, 2)
                answer = resp.text
                used_llm = True
                model_used = candidate
                break
            except Exception:
                continue

    if not used_llm:
        if not hits:
            answer = "No se encontraron coincidencias para esta consulta."
        else:
            lines = ["Fragmentos más relevantes encontrados:"]
            for h in hits[:3]:
                lines.append(f"• [{h['timestamp']}] {h['lecture_title']}: {h['content'][:220]}...")
            answer = "\n".join(lines)

    return {
        "answer": answer,
        "used_llm": used_llm,
        "model_used": model_used,
        "retrieval_latency_ms": retrieval["total_latency_ms"],
        "llm_latency_ms": llm_latency_ms,
        "hits": hits,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    return r"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Asistente de Clases Grabadas</title>
  <link href="https://vjs.zencdn.net/8.10.0/video-js.css" rel="stylesheet" />
  <style>
    * { box-sizing: border-box; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background:#0f172a; color:#f8fafc; margin:0; padding:24px; }
    .container { max-width:1340px; margin:auto; }
    .topbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:18px; }
    .topbar h1 { margin:0; font-size:21px; font-weight:700; color:#f8fafc; }
    .topbar select { padding:8px 12px; border-radius:6px; border:1px solid #475569; background:#1e293b; color:#f8fafc; font-size:13.5px; min-width:300px; }
    .grid { display:grid; grid-template-columns: 1.25fr 1fr; gap:20px; }
    .card { background:#1e293b; border-radius:10px; padding:18px; border:1px solid #334155; }
    h2 { margin:0 0 12px 0; font-size:16px; color:#cbd5e1; font-weight:600; }
    .search-row { display:flex; gap:8px; margin-bottom:14px; }
    input { flex:1; padding:10px 12px; border-radius:6px; border:1px solid #475569; background:#0f172a; color:#fff; font-size:14px; }
    button { padding:10px 18px; border-radius:6px; border:none; background:#2563eb; color:#fff; font-weight:600; cursor:pointer; font-size:14px; }
    button:hover { background:#1d4ed8; }
    button:disabled { opacity:0.6; cursor:wait; }
    .hit { background:#0f172a; border:1px solid #334155; border-radius:8px; padding:10px 12px; margin-bottom:10px; }
    .ts-btn { background:#0ea5e9; color:#0f172a; font-weight:700; padding:3px 9px; border-radius:4px; font-size:12px; cursor:pointer; border:none; margin-right:8px; }
    .ts-btn:hover { background:#38bdf8; }
    .meta { font-size:12px; color:#94a3b8; margin-bottom:6px; }
    .answer { white-space:pre-wrap; background:#0f172a; padding:14px; border-radius:8px; border-left:3px solid #38bdf8; margin-bottom:14px; font-size:13.5px; line-height:1.5; }
    #status { font-size:12px; color:#94a3b8; margin-bottom:10px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="topbar">
      <h1>Asistente de Clases Grabadas</h1>
      <select id="lectureSelect" onchange="onLectureChange()"></select>
    </div>
    <div class="grid">
      <div class="card">
        <h2 id="videoTitle">Reproductor de Video</h2>
        <video id="player" class="video-js vjs-default-skin vjs-big-play-centered" controls preload="auto" style="width:100%; height:410px;"></video>
        <p class="meta" style="margin-top:10px;">Selecciona cualquier marca de tiempo en los resultados para ir al segundo exacto de la grabación.</p>
      </div>
      <div class="card">
        <h2>Consulta en Lenguaje Natural</h2>
        <div class="search-row">
          <input id="q" placeholder="Escribe tu pregunta sobre la clase..." onkeydown="if(event.key==='Enter') ask()"/>
          <button id="askBtn" onclick="ask()">Consultar</button>
        </div>
        <div id="status"></div>
        <div id="answer" class="answer" style="display:none;"></div>
        <div id="hits"></div>
      </div>
    </div>
  </div>
  <script src="https://vjs.zencdn.net/8.10.0/video.min.js"></script>
  <script>
    const player = videojs('player');
    let lectures = [];
    let currentHls = '';

    async function loadLectures() {
      const res = await fetch('/api/lectures');
      lectures = await res.json();
      const sel = document.getElementById('lectureSelect');
      sel.innerHTML = '<option value="">Todas las sesiones indexadas</option>' +
        lectures.map(l => `<option value="${l.id}">${l.title}</option>`).join('');
      const demo = lectures.find(l => l.hls_path && l.hls_path.includes('demo_gestion_conocimiento')) ||
                   lectures.find(l => l.hls_path);
      if (demo) {
        sel.value = String(demo.id);
        setPlayerSource(demo.hls_path, demo.title);
      }
    }

    function setPlayerSource(hlsPath, title) {
      if (!hlsPath || hlsPath === currentHls) return;
      currentHls = hlsPath;
      document.getElementById('videoTitle').textContent = title || 'Reproductor de Video';
      player.src({ src: hlsPath, type: 'application/x-mpegURL' });
    }

    function onLectureChange() {
      const val = document.getElementById('lectureSelect').value;
      const lec = lectures.find(l => String(l.id) === val);
      if (lec && lec.hls_path) {
        setPlayerSource(lec.hls_path, lec.title);
      }
    }

    function jumpTo(sec, hlsPath, title) {
      if (hlsPath && hlsPath !== currentHls) {
        setPlayerSource(hlsPath, title);
        player.one('loadedmetadata', () => {
          player.currentTime(sec);
          player.play();
        });
      } else {
        player.currentTime(sec);
        player.play();
      }
    }

    async function ask() {
      const q = document.getElementById('q').value.trim();
      if (!q) return;
      const btn = document.getElementById('askBtn');
      const status = document.getElementById('status');
      btn.disabled = true;
      status.textContent = 'Buscando fragmentos relevantes...';
      const val = document.getElementById('lectureSelect').value;
      const payload = { question: q, top_k: 5 };
      if (val) payload.lecture_id = parseInt(val, 10);

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        status.textContent = `Recuperación: ${data.retrieval_latency_ms} ms` +
          (data.used_llm ? ` | Síntesis: ${data.llm_latency_ms} ms` : '');
        const ansDiv = document.getElementById('answer');
        ansDiv.style.display = 'block';
        const safeHtml = data.answer
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        ansDiv.innerHTML = safeHtml;
        const hitsDiv = document.getElementById('hits');
        hitsDiv.innerHTML = data.hits.map(h => `
          <div class="hit">
            <div class="meta">
              <button class="ts-btn" onclick="jumpTo(${h.start_sec}, '${h.hls_path}', '${h.lecture_title.replace(/'/g, '')}')">${h.timestamp}</button>
              <b>${h.lecture_title}</b> &middot; Similitud: ${h.score}
            </div>
            <div style="font-size:13px; color:#e2e8f0;">${h.content}</div>
          </div>
        `).join('');
      } finally {
        btn.disabled = false;
      }
    }

    loadLectures();
  </script>
</body>
</html>"""
