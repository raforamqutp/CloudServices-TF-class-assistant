import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from google.adk.agents.llm_agent import Agent
from backend.retriever import semantic_search


def buscar_momentos_clase(pregunta: str, top_k: int = 5) -> dict:
    """Busca en las transcripciones indexadas los fragmentos relevantes y su marca de tiempo."""
    data = semantic_search(pregunta, top_k=top_k)
    return {
        "resultados": data["results"],
        "latencia_ms": data["total_latency_ms"],
    }


root_agent = Agent(
    model="gemini-3.5-flash",
    name="academic_lecture_agent",
    description="Asistente para búsqueda semántica y consulta de clases grabadas.",
    instruction=(
        "Eres un asistente académico para consultar grabaciones de clase. "
        "Antes de responder sobre cualquier tema visto en clase, llama a `buscar_momentos_clase`. "
        "Los fragmentos provienen de transcripción automática (ASR) y pueden tener errores fonéticos menores; "
        "redacta la respuesta en español claro corrigiendo por contexto académico sin inventar hechos externos. "
        "Incluye siempre la marca de tiempo exacta [MM:SS] o [HH:MM:SS] y la sesión correspondiente."
    ),
    tools=[buscar_momentos_clase],
)
