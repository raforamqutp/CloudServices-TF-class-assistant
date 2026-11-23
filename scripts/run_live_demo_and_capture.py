import json
import time
import urllib.request

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

BASE_URL = "http://127.0.0.1:8080"


def http_json(method: str, path: str, payload=None):
    url = BASE_URL + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    lectures = http_json("GET", "/api/lectures")
    demo = next(l for l in lectures if "demo_gestion_conocimiento" in l["filename"])
    demo_id = demo["id"]
    print(f"Clase Demo encontrada: id={demo_id} | {demo['title']} | hls={demo['hls_path']}")

    demo_questions = [
        "¿Qué son las comunidades de práctica y para qué sirven según la clase?",
        "¿Qué riesgo existe cuando un trabajador experimentado deja la organización?",
        "¿Qué papel cumple el liderazgo en la gestión del conocimiento?",
    ]

    demo_log = []
    for q in demo_questions:
        res = http_json("POST", "/api/chat", {"question": q, "top_k": 5, "lecture_id": demo_id})
        demo_log.append(
            {
                "question": q,
                "used_llm": res["used_llm"],
                "model_used": res.get("model_used", ""),
                "retrieval_latency_ms": res["retrieval_latency_ms"],
                "llm_latency_ms": res["llm_latency_ms"],
                "answer": res["answer"],
                "top_hit": res["hits"][0] if res["hits"] else None,
            }
        )
        print("\n--- DEMO Q&A ---")
        print("Q:", q)
        print(
            f"used_llm={res['used_llm']} ({res.get('model_used','')}) | retrieval={res['retrieval_latency_ms']} ms | llm={res['llm_latency_ms']} ms"
        )
        print("A:", res["answer"][:350])

    with open(r"D:\Projects\Cloud\scripts\demo_interaction_log.json", "w", encoding="utf-8") as f:
        json.dump(demo_log, f, indent=2, ensure_ascii=False)

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1440,920")
    opts.add_argument("--autoplay-policy=no-user-gesture-required")
    opts.add_argument("--mute-audio")
    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(BASE_URL)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#lectureSelect option"))
        )
        time.sleep(2)
        inp = driver.find_element(By.ID, "q")
        inp.clear()
        inp.send_keys(demo_questions[0])
        driver.find_element(By.ID, "askBtn").click()

        WebDriverWait(driver, 30).until(
            lambda d: d.find_element(By.ID, "answer").is_displayed()
            and len(d.find_elements(By.CSS_SELECTOR, ".ts-btn")) > 0
        )
        driver.find_elements(By.CSS_SELECTOR, ".ts-btn")[0].click()
        time.sleep(3)
        shot_path = r"D:\Projects\Cloud\proyecto_final\fig_demo_ui_real.png"
        driver.save_screenshot(shot_path)
        print("\nCaptura real de la interfaz guardada en:", shot_path)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
