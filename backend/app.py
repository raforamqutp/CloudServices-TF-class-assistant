import os
import re
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import psycopg
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.db import DATABASE_URL
from backend.ingest import format_sec_label, get_embedding_model, ingest_transcript, parse_transcript_file
from backend.portable_store import sqlite_list_lectures, sqlite_save_lecture_chunks
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

TICS_VIDEO_DIR = os.getenv("TICS_VIDEO_DIR", r"D:\UTP\2026 - Agosto\TICS\AG-files\grabaciones_clases")
if os.path.isdir(TICS_VIDEO_DIR):
    app.mount("/media/tics", StaticFiles(directory=TICS_VIDEO_DIR), name="tics_media")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "..", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/media/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploaded_media")


class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    lecture_id: Optional[int] = None
    course: Optional[str] = None


@app.get("/api/lectures")
def list_lectures():
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=2) as conn:
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
    except Exception:
        return sqlite_list_lectures()


@app.post("/api/ingest-video")
async def ingest_video_endpoint(
    file: UploadFile = File(...),
    course: str = Form("TICS"),
    title: str = Form(""),
    max_seconds: int = Form(0),
):
    t0 = time.perf_counter()
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(file.filename or "clase.mp4"))
    video_path = os.path.join(UPLOAD_DIR, safe_name)
    content = await file.read()
    with open(video_path, "wb") as out:
        out.write(content)

    final_title = title.strip() or os.path.splitext(safe_name)[0].replace("_", " ")
    media_url = f"/media/uploads/{safe_name}"

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        txt_path = os.path.join(tmpdir, f"transcripcion_{os.path.splitext(safe_name)[0]}.txt")

        from faster_whisper import WhisperModel
        import ctranslate2

        has_cuda = ctranslate2.get_cuda_device_count() > 0
        device = "cuda" if has_cuda else "cpu"
        compute_type = "float16" if has_cuda else "int8"
        whisper_size = "medium" if has_cuda else "tiny"

        t_asr0 = time.perf_counter()
        wmodel = WhisperModel(whisper_size, device=device, compute_type=compute_type)

        windows = [0, max_seconds, max_seconds * 2, max_seconds * 3] if max_seconds > 0 else [0]
        lines = []
        for offset_sec in windows:
            ff_cmd = ["ffmpeg", "-y"]
            if offset_sec > 0:
                ff_cmd.extend(["-ss", str(offset_sec)])
            ff_cmd.extend(["-i", video_path])
            if max_seconds > 0:
                ff_cmd.extend(["-t", str(max_seconds)])
            ff_cmd.extend(["-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", audio_path])
            subprocess.run(ff_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            segments, _ = wmodel.transcribe(audio_path, language="es", vad_filter=True)
            lines = []
            for seg in segments:
                ts = format_sec_label(offset_sec + int(seg.start))
                txt = seg.text.strip()
                if txt:
                    lines.append(f"[{ts}] {txt}")
            if len(lines) >= 2:
                break

        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write("\n".join(lines) + "\n")
        asr_time = round(time.perf_counter() - t_asr0, 2)

        chunks = parse_transcript_file(txt_path)
        if not chunks:
            return {"status": "error", "detail": "No se detectó voz en la grabación."}

        try:
            res = ingest_transcript(txt_path, final_title, hls_path=media_url, course=course.strip() or "General")
            lecture_id = res["lecture_id"]
        except Exception:
            emb_model = get_embedding_model()
            embs = emb_model.encode([c["content"] for c in chunks], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
            lecture_id = sqlite_save_lecture_chunks(
                os.path.basename(txt_path),
                final_title,
                course.strip() or "General",
                chunks[-1]["end_sec"],
                media_url,
                chunks,
                np.asarray(embs, dtype=np.float32),
            )

    return {
        "status": "ok",
        "lecture_id": lecture_id,
        "course": course.strip() or "General",
        "title": final_title,
        "chunks": len(chunks),
        "lines": len(lines),
        "device": device,
        "whisper_model": whisper_size,
        "asr_time_s": asr_time,
        "total_time_s": round(time.perf_counter() - t0, 2),
        "hls_path": media_url,
    }


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=2),
    top_k: int = 5,
    lecture_id: Optional[int] = None,
    course: Optional[str] = None,
):
    return semantic_search(q, top_k=top_k, lecture_id=lecture_id, course=course)


@app.post("/api/chat")
def chat(req: ChatRequest):
    retrieval = semantic_search(req.question, top_k=req.top_k, lecture_id=req.lecture_id, course=req.course)
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
            "Eres un asistente académico. Los siguientes fragmentos provienen de una transcripción "
            "automática de voz (ASR) de una clase universitaria, por lo que pueden contener errores "
            "fonéticos menores o palabras cortadas. Redacta una respuesta clara, coherente y natural en español "
            "corrigiendo esos errores evidentes por contexto académico sin inventar hechos ajenos a los fragmentos. "
            "Cita siempre la marca de tiempo exacta [MM:SS] o [HH:MM:SS]:\n\n"
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
    body { font-family: 'Segoe UI', system-ui, sans-serif; background:#f8fafc; color:#0f172a; margin:0; padding:20px; }
    .container { max-width:1320px; margin:auto; }
    .topbar { display:flex; flex-wrap:wrap; align-items:center; justify-content:space-between; gap:12px; margin-bottom:14px; padding-bottom:12px; border-bottom:1px solid #cbd5e1; }
    .topbar h1 { margin:0; font-size:22px; font-weight:700; color:#0f172a; }
    .course-bar { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:14px; align-items:center; }
    .course-pill { padding:7px 14px; border-radius:999px; border:1px solid #cbd5e1; background:#ffffff; color:#334155; font-size:13.5px; font-weight:600; cursor:pointer; }
    .course-pill:hover { border-color:#1d4ed8; color:#1d4ed8; }
    .course-pill.active { background:#1d4ed8; color:#ffffff; border-color:#1d4ed8; }
    .controls-row { display:flex; gap:10px; align-items:center; }
    .controls-row select { padding:8px 12px; border-radius:6px; border:1px solid #94a3b8; background:#ffffff; color:#0f172a; font-size:14px; font-weight:500; min-width:290px; }
    .btn-outline { padding:8px 14px; border-radius:6px; border:1.5px solid #1d4ed8; background:#ffffff; color:#1d4ed8; font-weight:600; cursor:pointer; font-size:14px; }
    .btn-outline:hover { background:#eff6ff; }
    .ingest-panel { display:none; background:#ffffff; border:1.5px solid #93c5fd; border-radius:10px; padding:14px 16px; margin-bottom:14px; box-shadow:0 1px 3px rgba(15,23,42,0.06); }
    .ingest-row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
    .ingest-row select, .ingest-row input[type="text"] { padding:9px 11px; border-radius:6px; border:1px solid #94a3b8; background:#ffffff; color:#0f172a; font-size:14px; }
    .ingest-row input[type="text"] { min-width:180px; flex:1; }
    #ingestStatus { font-size:13px; color:#1e40af; margin-top:8px; font-weight:600; }
    .grid { display:grid; grid-template-columns: 1.15fr 1fr; gap:18px; }
    .card { background:#ffffff; border-radius:10px; padding:18px; border:1px solid #cbd5e1; box-shadow:0 1px 3px rgba(15,23,42,0.06); }
    h2 { margin:0 0 12px 0; font-size:17px; color:#1e293b; font-weight:700; }
    .session-list { margin-top:12px; max-height:190px; overflow-y:auto; border:1px solid #e2e8f0; border-radius:8px; padding:6px; background:#f8fafc; }
    .session-item { display:flex; justify-content:space-between; align-items:center; gap:8px; width:100%; text-align:left; padding:8px 10px; margin-bottom:4px; border-radius:6px; border:1px solid transparent; background:#ffffff; color:#0f172a; font-size:13.5px; cursor:pointer; }
    .session-item:hover { border-color:#93c5fd; background:#eff6ff; }
    .session-item.active { border-color:#1d4ed8; background:#dbeafe; font-weight:600; }
    .badge { font-size:11.5px; padding:2px 7px; border-radius:999px; font-weight:600; white-space:nowrap; }
    .badge-course { background:#e2e8f0; color:#334155; }
    .badge-video { background:#dcfce7; color:#166534; }
    .badge-text { background:#fef3c7; color:#92400e; }
    .search-row { display:flex; gap:8px; margin-bottom:10px; }
    input { flex:1; padding:10px 12px; border-radius:6px; border:1px solid #94a3b8; background:#ffffff; color:#0f172a; font-size:14.5px; }
    .btn-primary { padding:10px 18px; border-radius:6px; border:none; background:#1d4ed8; color:#ffffff; font-weight:600; cursor:pointer; font-size:14.5px; }
    .btn-primary:hover { background:#1e40af; }
    .btn-primary:disabled { opacity:0.6; cursor:wait; }
    .scope-note { font-size:12.5px; color:#475569; margin-bottom:10px; }
    .hit { background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:12px 14px; margin-bottom:10px; }
    .ts-btn { background:#1d4ed8; color:#ffffff; font-weight:700; padding:4px 10px; border-radius:4px; font-size:13px; cursor:pointer; border:none; margin-right:8px; }
    .ts-btn:hover { background:#1e40af; }
    .meta { font-size:13px; color:#475569; margin-bottom:6px; display:flex; flex-wrap:wrap; align-items:center; gap:6px; }
    .answer { white-space:pre-wrap; background:#eff6ff; color:#0f172a; padding:14px; border-radius:8px; border:1px solid #bfdbfe; border-left:4px solid #1d4ed8; margin-bottom:12px; font-size:14.5px; line-height:1.5; }
    #status { font-size:13px; color:#475569; margin-bottom:10px; font-weight:500; }
  </style>
</head>
<body>
  <div class="container">
    <div class="topbar">
      <h1>Asistente de Clases Grabadas</h1>
      <div class="controls-row">
        <select id="lectureSelect" onchange="onLectureChange()"></select>
        <button class="btn-outline" onclick="toggleIngest()">+ Agregar clase</button>
      </div>
    </div>
    <div id="ingestPanel" class="ingest-panel">
      <div class="ingest-row">
        <select id="ingCourseSelect" onchange="onIngestCourseChange()"></select>
        <input id="ingNewCourse" type="text" placeholder="Nombre del nuevo curso..." style="display:none;" />
        <input id="ingTitle" type="text" placeholder="Título de la sesión (opcional)" />
        <select id="ingScope">
          <option value="240">Demo rápida: 4 min con voz (ideal sin GPU)</option>
          <option value="0">Clase completa</option>
        </select>
        <input id="ingFile" type="file" accept="video/mp4,video/*" style="flex:1; padding:7px;" />
        <button id="ingBtn" class="btn-primary" onclick="uploadAndIngest()">Procesar grabación</button>
      </div>
      <div id="ingestStatus"></div>
    </div>
    <div id="courseBar" class="course-bar"></div>
    <div class="grid">
      <div class="card">
        <h2 id="videoTitle">Reproductor de Video</h2>
        <video id="player" class="video-js vjs-default-skin vjs-big-play-centered" controls preload="auto" style="width:100%; height:390px;"></video>
        <p class="meta" style="margin-top:10px;">Selecciona un curso y una grabación para reproducir o haz clic en cualquier marca de tiempo.</p>
        <div id="sessionList" class="session-list"></div>
      </div>
      <div class="card">
        <h2>Consulta en Lenguaje Natural</h2>
        <div class="search-row">
          <input id="q" placeholder="Escribe tu pregunta sobre el curso o sesión..." onkeydown="if(event.key==='Enter') ask()"/>
          <button id="askBtn" class="btn-primary" onclick="ask()">Consultar</button>
        </div>
        <div id="scopeNote" class="scope-note"></div>
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
    let selectedCourse = '';
    let selectedLectureId = '';
    let currentMedia = '';

    function getFilteredLectures() {
      return selectedCourse ? lectures.filter(l => l.course === selectedCourse) : lectures;
    }

    function renderCourses() {
      const counts = {};
      lectures.forEach(l => { counts[l.course] = (counts[l.course] || 0) + 1; });
      const courses = Object.keys(counts).sort();
      const bar = document.getElementById('courseBar');
      const pills = [
        `<button class="course-pill ${selectedCourse === '' ? 'active' : ''}" onclick="selectCourse('')">Todos los cursos (${lectures.length})</button>`,
        ...courses.map(c =>
          `<button class="course-pill ${selectedCourse === c ? 'active' : ''}" onclick="selectCourse('${c.replace(/'/g, "\\'")}')">${c} (${counts[c]})</button>`
        )
      ];
      bar.innerHTML = pills.join('');

      const ingSel = document.getElementById('ingCourseSelect');
      if (ingSel) {
        const prev = ingSel.value;
        ingSel.innerHTML = courses.map(c => `<option value="${c}">${c}</option>`).join('') +
          '<option value="__new__">+ Nuevo curso...</option>';
        if (prev && (courses.includes(prev) || prev === '__new__')) {
          ingSel.value = prev;
        } else if (selectedCourse && courses.includes(selectedCourse)) {
          ingSel.value = selectedCourse;
        }
        onIngestCourseChange();
      }
    }

    function onIngestCourseChange() {
      const sel = document.getElementById('ingCourseSelect');
      const inp = document.getElementById('ingNewCourse');
      if (sel && inp) {
        inp.style.display = sel.value === '__new__' ? 'block' : 'none';
      }
    }

    function renderSessionControls() {
      const list = getFilteredLectures();
      const sel = document.getElementById('lectureSelect');
      const allLabel = selectedCourse ? `Todas las sesiones de ${selectedCourse}` : 'Todas las sesiones registradas';
      sel.innerHTML = `<option value="">${allLabel}</option>` +
        list.map(l => `<option value="${l.id}">${l.title}</option>`).join('');
      sel.value = selectedLectureId;

      const listDiv = document.getElementById('sessionList');
      listDiv.innerHTML = list.map(l => {
        const isActive = String(l.id) === String(selectedLectureId);
        const tag = l.hls_path && l.chunk_count > 0
          ? '<span class="badge badge-video">Video + Texto</span>'
          : l.hls_path
            ? '<span class="badge badge-video">Video</span>'
            : '<span class="badge badge-text">Solo texto</span>';
        return `<button class="session-item ${isActive ? 'active' : ''}" onclick="selectLecture(${l.id})">
          <span>${l.title}</span>
          <span>${tag}</span>
        </button>`;
      }).join('');

      updateScopeNote();
    }

    function updateScopeNote() {
      const note = document.getElementById('scopeNote');
      if (selectedLectureId) {
        const lec = lectures.find(l => String(l.id) === String(selectedLectureId));
        note.textContent = lec ? `Ámbito de búsqueda: ${lec.course} → ${lec.title}` : '';
      } else if (selectedCourse) {
        note.textContent = `Ámbito de búsqueda: todas las sesiones transcritas de ${selectedCourse}`;
      } else {
        note.textContent = 'Ámbito de búsqueda: todos los cursos registrados';
      }
    }

    function selectCourse(course) {
      selectedCourse = course;
      selectedLectureId = '';
      renderCourses();
      renderSessionControls();
      const firstPlayable = getFilteredLectures().find(l => l.hls_path);
      if (firstPlayable) {
        setPlayerSource(firstPlayable.hls_path, firstPlayable.title);
      }
    }

    function selectLecture(id) {
      selectedLectureId = String(id);
      const lec = lectures.find(l => String(l.id) === selectedLectureId);
      renderSessionControls();
      if (lec && lec.hls_path) {
        setPlayerSource(lec.hls_path, lec.title);
      }
    }

    async function loadLectures() {
      const res = await fetch('/api/lectures');
      lectures = await res.json();
      const demo = lectures.find(l => l.hls_path && l.hls_path.includes('demo_gestion_conocimiento')) ||
                   lectures.find(l => l.hls_path);
      if (demo) {
        selectedCourse = demo.course;
        selectedLectureId = String(demo.id);
        setPlayerSource(demo.hls_path, demo.title);
      }
      renderCourses();
      renderSessionControls();
    }

    function setPlayerSource(mediaPath, title) {
      if (!mediaPath || mediaPath === currentMedia) return;
      currentMedia = mediaPath;
      document.getElementById('videoTitle').textContent = title || 'Reproductor de Video';
      const mime = mediaPath.toLowerCase().endsWith('.m3u8') ? 'application/x-mpegURL' : 'video/mp4';
      player.src({ src: mediaPath, type: mime });
    }

    function onLectureChange() {
      selectedLectureId = document.getElementById('lectureSelect').value;
      renderSessionControls();
      const lec = lectures.find(l => String(l.id) === selectedLectureId);
      if (lec && lec.hls_path) {
        setPlayerSource(lec.hls_path, lec.title);
      }
    }

    function jumpTo(sec, mediaPath, title, lectureId) {
      if (lectureId) {
        selectedLectureId = String(lectureId);
        renderSessionControls();
      }
      if (!mediaPath) return;
      if (mediaPath !== currentMedia) {
        setPlayerSource(mediaPath, title);
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
      const payload = { question: q, top_k: 5 };
      if (selectedLectureId) {
        payload.lecture_id = parseInt(selectedLectureId, 10);
      } else if (selectedCourse) {
        payload.course = selectedCourse;
      }

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
              <button class="ts-btn" onclick="jumpTo(${h.start_sec}, '${h.hls_path}', '${h.lecture_title.replace(/'/g, '')}', ${h.lecture_id})">${h.timestamp}</button>
              <span class="badge badge-course">${h.course || ''}</span>
              <b>${h.lecture_title}</b> &middot; Similitud: ${h.score}
            </div>
            <div style="font-size:13.5px; color:#1e293b; line-height:1.45;">${h.content}</div>
          </div>
        `).join('');
      } finally {
        btn.disabled = false;
      }
    }

    function toggleIngest() {
      const p = document.getElementById('ingestPanel');
      p.style.display = p.style.display === 'block' ? 'none' : 'block';
    }

    async function uploadAndIngest() {
      const fileInput = document.getElementById('ingFile');
      const status = document.getElementById('ingestStatus');
      const btn = document.getElementById('ingBtn');
      if (!fileInput.files || !fileInput.files.length) {
        status.textContent = 'Selecciona un archivo MP4 para procesar.';
        return;
      }
      const selCourse = document.getElementById('ingCourseSelect').value;
      const chosenCourse = selCourse === '__new__'
        ? (document.getElementById('ingNewCourse').value.trim() || 'Nuevo Curso')
        : selCourse;
      const fd = new FormData();
      fd.append('file', fileInput.files[0]);
      fd.append('course', chosenCourse);
      fd.append('title', document.getElementById('ingTitle').value.trim());
      fd.append('max_seconds', document.getElementById('ingScope').value);

      btn.disabled = true;
      status.textContent = 'Extrayendo audio, transcribiendo y generando vectores...';
      try {
        const res = await fetch('/api/ingest-video', { method: 'POST', body: fd });
        const data = await res.json();
        if (data.status === 'ok') {
          status.textContent = `Listo en ${data.total_time_s} s (${data.device.toUpperCase()} / Whisper ${data.whisper_model}): ${data.lines} líneas, ${data.chunks} fragmentos listos para consulta.`;
          await loadLectures();
          selectedCourse = data.course;
          selectedLectureId = String(data.lecture_id);
          renderCourses();
          renderSessionControls();
          setPlayerSource(data.hls_path, data.title);
        } else {
          status.textContent = data.detail || 'Error al procesar el video.';
        }
      } catch (e) {
        status.textContent = 'Error de conexión durante la ingesta.';
      } finally {
        btn.disabled = false;
      }
    }

    loadLectures();
  </script>
</body>
</html>"""
