from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.client import User
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.compute import Server
from diagrams.programming.framework import FastAPI
from diagrams.programming.language import Python
from diagrams.gcp.compute import Run
from diagrams.gcp.storage import GCS
from diagrams.gcp.ml import AIPlatform

graph_attr = {
    "fontsize": "14",
    "bgcolor": "white",
    "pad": "0.4",
    "splines": "ortho",
    "nodesep": "0.6",
    "ranksep": "0.8"
}

with Diagram(
    "Arquitectura Híbrida CloudLecture Assistant (Local-First + Extensión Cloud)",
    filename="/mnt/d/Projects/Cloud/proyecto_final/fig_mingrammer_arquitectura",
    show=False,
    direction="LR",
    graph_attr=graph_attr,
):
    student = User("Estudiante (UTP)")

    with Cluster("Estación Local (GPU NVIDIA RTX 4070 SUPER)"):
        ui = FastAPI("Interfaz Web\n(FastAPI + Video.js)")
        orch = Python("Orquestador + Agent\n(Google ADK + Whisper)")
        gpu = Server("FFmpeg 7.0\n(NVENC HLS + MP3 64k)")
        db = PostgreSQL("PostgreSQL 18\n(pgvector HNSW)")

        ui >> Edge(label="Consulta / Chat") >> orch
        orch >> Edge(label="Transcodifica") >> gpu
        orch >> Edge(label="k-NN (384-d)") >> db

    with Cluster("Servicios Cloud (Google Cloud Platform)"):
        gemini = AIPlatform("Google Gemini API\n(gemini-3.5-flash)")
        gcs = GCS("Cloud Storage\n(Respaldo MP4/HLS)")
        crun = Run("Cloud Run Jobs\n(Serverless Opcional)")

    student >> Edge(label="HTTP / HLS") >> ui
    orch >> Edge(label="Prompt + Top-k") >> gemini
    orch >> Edge(label="Sync opcional", style="dashed") >> gcs
    orch >> Edge(label="Fallback sin GPU", style="dashed") >> crun
