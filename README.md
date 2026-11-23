# Asistente de Clases Grabadas (Local-First)

Herramienta local para transcodificar, transcribir, indexar vectorialmente y consultar grabaciones de clases universitarias en lenguaje natural con salto directo al segundo exacto del video.

## Arquitectura y Componentes

- **Procesamiento multimedia (`backend/media_pipeline.py`):** Extracción de audio mono a 64 kbps y transcodificación adaptativa HLS (`.m3u8` / `.ts` de 6 s) con FFmpeg 7.0 acelerado por hardware (NVIDIA NVENC), más transcripción local con `faster-whisper-medium` en CUDA.
- **Almacenamiento vectorial (`backend/db.py`, `backend/ingest.py`):** PostgreSQL 18 con la extensión `pgvector` (índice HNSW de distancia coseno, 384 dimensiones con `paraphrase-multilingual-MiniLM-L12-v2`) e índice GIN en español.
- **Recuperación y agente (`backend/retriever.py`, `mi_agente/agent.py`):** Búsqueda k-NN con marcas temporales `[HH:MM:SS]` integrada como herramienta en Google ADK y síntesis con Gemini Flash (con respaldo automático entre modelos Flash).
- **Interfaz web (`backend/app.py`):** Backend en FastAPI + reproductor adaptativo Video.js para reproducir HLS y saltar al segundo citado en las respuestas.

## Requisitos Previos

- Python 3.13 (64 bits)
- FFmpeg 7.0 (con soporte NVIDIA NVENC si se usa GPU)
- PostgreSQL con extensión `pgvector` (vía `docker-compose.yml` o WSL2 Ubuntu mediante `scripts/setup_pgvector_wsl.sh`)
- Clave de API de Google Gemini configurada en `mi_agente/.env` (ver `.env.example`)

## Instalación Rápida

```bash
pip install -r requirements.txt
copy .env.example mi_agente\.env
```

Edita `mi_agente/.env` colocando tu `GOOGLE_API_KEY` y la cadena `DATABASE_URL`.

## Uso en Windows

- **Panel interactivo:** Ejecuta `control_plataforma.bat` para iniciar, detener o verificar el estado de los servicios.
- **Inicio directo:** Ejecuta `iniciar_plataforma.bat` (levanta PostgreSQL + servidor FastAPI en `http://127.0.0.1:8080` y abre el navegador).
- **Apagado limpio:** Ejecuta `detener_plataforma.bat`.

Para levantar únicamente el agente en la consola de desarrollo de Google ADK:

```bash
iniciar_agente_adk.bat
```

## Estructura del Repositorio

```text
├── backend/
│   ├── app.py                # Servidor FastAPI + interfaz web Video.js
│   ├── db.py                 # Inicialización de tablas e índices HNSW / GIN
│   ├── ingest.py             # Segmentación temporal, embeddings e ingesta
│   ├── media_pipeline.py     # Pipeline FFmpeg (NVENC/HLS/MP3) + faster-whisper
│   └── retriever.py          # Búsqueda vectorial HNSW y léxica FTS
├── mi_agente/
│   └── agent.py              # Definición del agente Google ADK
├── scripts/
│   ├── benchmark_rag.py      # Evaluación Hit@3, Hit@5, MRR y latencia
│   └── setup_pgvector_wsl.sh # Configuración de PostgreSQL + pgvector en WSL
├── proyecto_final/           # Diagramas C4/UML, resultados y documento IEEE
├── docker-compose.yml        # Alternativa de despliegue PostgreSQL + pgvector
├── control_plataforma.bat    # Menú de control (iniciar / detener / estado)
├── iniciar_plataforma.bat    # Script de arranque rápido
└── detener_plataforma.bat    # Script de apagado limpio
```
