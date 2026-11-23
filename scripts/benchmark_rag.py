import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import psycopg
from backend.db import DATABASE_URL
from backend.retriever import lexical_search, semantic_search

EVAL_QUERIES = [
    ("¿Cómo funciona la virtualización y qué es un hipervisor o máquina virtual?", ["virtual", "máquina", "hipervisor", "vmware", "virtualbox"]),
    ("¿Qué es un contenedor Docker y en qué se diferencia de una máquina virtual?", ["contenedor", "docker", "imagen", "virtual"]),
    ("¿Para qué sirve Kubernetes en la orquestación de contenedores?", ["kubernetes", "pod", "nodo", "clúster", "contenedor"]),
    ("¿Cómo se define infraestructura como código usando Terraform?", ["terraform", "infraestructura", "código", "recurso"]),
    ("¿Qué ventajas tiene la computación en la nube frente a servidores locales?", ["nube", "cloud", "servidor", "costo", "escala"]),
    ("¿Cómo funciona el almacenamiento de objetos en Amazon S3?", ["s3", "bucket", "objeto", "almacenamiento"]),
    ("¿Qué es Amazon EC2 y cómo se lanzan instancias?", ["ec2", "instancia", "amazon", "aws"]),
    ("¿Cómo se configuran permisos, usuarios y roles en AWS IAM?", ["iam", "usuario", "rol", "permiso", "política"]),
    ("¿Qué es una red privada virtual VPC y para qué sirven las subredes?", ["vpc", "subred", "red", "pública", "privada"]),
    ("¿Cómo ayuda un balanceador de carga a la alta disponibilidad?", ["balanceador", "carga", "tráfico", "disponibilidad"]),
    ("¿Qué diferencia hay entre IaaS, PaaS y SaaS?", ["iaas", "paas", "saas", "servicio", "plataforma"]),
    ("¿Qué es la arquitectura de microservicios frente a un monolito?", ["microservicio", "monolito", "servicio", "api"]),
    ("¿Cómo se monitorean métricas y registros en la nube?", ["monitoreo", "métrica", "log", "cloudwatch", "alerta"]),
    ("¿Qué es una base de datos relacional frente a una NoSQL?", ["base de datos", "sql", "nosql", "tabla", "rds"]),
    ("¿Cómo funciona el escalado automático cuando sube la demanda?", ["escalado", "auto", "demanda", "instancia"]),
    ("¿Qué es una API REST y cómo se comunican los servicios web?", ["api", "rest", "http", "json", "endpoint"]),
    ("¿Cómo se protege la seguridad y el cifrado en la nube?", ["seguridad", "cifrado", "firewall", "acceso"]),
    ("¿Qué es el proyecto final del curso y qué entregables pide el docente?", ["proyecto", "artículo", "paper", "github", "entrega"]),
    ("¿Cómo se despliega una aplicación usando Git y repositorios?", ["git", "github", "repositorio", "código"]),
    ("¿Qué costos y modelos de facturación existen en AWS o Google Cloud?", ["costo", "precio", "facturación", "pago", "gratis"]),
    ("¿Qué son las comunidades de práctica y para qué sirven?", ["comunidades de práctica", "comunidad", "aprender", "intereses"]),
    ("¿Qué riesgo existe cuando un trabajador experimentado deja la organización?", ["perder conocimiento", "trabajador", "organización", "experiencia"]),
    ("¿Qué papel cumple el liderazgo en la gestión del conocimiento?", ["liderazgo", "supervisores", "ejemplo", "motiva"]),
    ("¿Por qué es importante aprender de los errores en una organización?", ["errores", "experiencias", "aprender"]),
    ("¿Cómo influye la cultura organizacional en compartir conocimiento?", ["cultura organizacional", "confianza", "compartir", "conocimiento"]),
]


def is_relevant(content: str, expected_keywords: list) -> bool:
    low = content.lower()
    return any(kw in low for kw in expected_keywords)


def main():
    with psycopg.connect(DATABASE_URL) as conn:
        lectures_count = conn.execute("SELECT count(*) FROM lectures;").fetchone()[0]
        chunks_count = conn.execute("SELECT count(*) FROM transcript_chunks;").fetchone()[0]

    semantic_search("calentamiento", top_k=1)

    sem_hit3, sem_hit5, sem_mrr, sem_lats, sem_db_lats = 0, 0, [], [], []
    lex_hit3, lex_hit5, lex_mrr, lex_lats = 0, 0, [], []

    for q, kws in EVAL_QUERIES:
        s_res = semantic_search(q, top_k=5)
        sem_lats.append(s_res["total_latency_ms"])
        sem_db_lats.append(s_res["db_latency_ms"])
        s_ranks = [i + 1 for i, h in enumerate(s_res["results"]) if is_relevant(h["content"], kws)]
        if any(r <= 3 for r in s_ranks):
            sem_hit3 += 1
        if s_ranks:
            sem_hit5 += 1
            sem_mrr.append(1.0 / s_ranks[0])
        else:
            sem_mrr.append(0.0)

        l_res = lexical_search(q, top_k=5)
        lex_lats.append(l_res["total_latency_ms"])
        l_ranks = [i + 1 for i, h in enumerate(l_res["results"]) if is_relevant(h["content"], kws)]
        if any(r <= 3 for r in l_ranks):
            lex_hit3 += 1
        if l_ranks:
            lex_hit5 += 1
            lex_mrr.append(1.0 / l_ranks[0])
        else:
            lex_mrr.append(0.0)

    n = len(EVAL_QUERIES)
    summary = {
        "lectures_indexed": lectures_count,
        "total_chunks": chunks_count,
        "num_eval_queries": n,
        "semantic_hnsw": {
            "hit_at_3_pct": round(100.0 * sem_hit3 / n, 1),
            "hit_at_5_pct": round(100.0 * sem_hit5 / n, 1),
            "mrr": round(sum(sem_mrr) / n, 3),
            "avg_total_latency_ms": round(sum(sem_lats) / n, 2),
            "avg_db_latency_ms": round(sum(sem_db_lats) / n, 2),
        },
        "lexical_fts": {
            "hit_at_3_pct": round(100.0 * lex_hit3 / n, 1),
            "hit_at_5_pct": round(100.0 * lex_hit5 / n, 1),
            "mrr": round(sum(lex_mrr) / n, 3),
            "avg_total_latency_ms": round(sum(lex_lats) / n, 2),
        },
    }

    out_path = os.path.join(ROOT_DIR, "scripts", "benchmark_results_v_final.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
