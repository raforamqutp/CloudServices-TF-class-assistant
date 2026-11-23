import json
import os
import subprocess
import threading
import time
from typing import Dict

import psutil
import torch
from faster_whisper import WhisperModel


def _probe_duration(video_path: str) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ],
        text=True,
    ).strip()
    return float(out)


def _format_hms(sec: float) -> str:
    total = int(sec)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_audio_mp3(video_path: str, audio_out: str) -> Dict:
    os.makedirs(os.path.dirname(audio_out), exist_ok=True)
    duration = _probe_duration(video_path)
    t0 = time.perf_counter()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ab",
            "64k",
            "-c:a",
            "libmp3lame",
            audio_out,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    elapsed = time.perf_counter() - t0
    in_mb = os.path.getsize(video_path) / (1024 * 1024)
    out_mb = os.path.getsize(audio_out) / (1024 * 1024)
    return {
        "duration_s": round(duration, 2),
        "elapsed_s": round(elapsed, 2),
        "speed_x": round(duration / elapsed, 1),
        "input_mb": round(in_mb, 2),
        "output_mb": round(out_mb, 2),
        "reduction_pct": round((1.0 - out_mb / in_mb) * 100.0, 1),
        "audio_path": audio_out,
    }


def transcode_hls(video_path: str, hls_dir: str, use_gpu: bool = True, limit_seconds: int = 0) -> Dict:
    os.makedirs(hls_dir, exist_ok=True)
    playlist = os.path.join(hls_dir, "index.m3u8")
    duration = float(limit_seconds) if limit_seconds > 0 else _probe_duration(video_path)

    cmd = ["ffmpeg", "-y"]
    if use_gpu:
        cmd += ["-hwaccel", "cuda"]
    cmd += ["-i", video_path]
    if limit_seconds > 0:
        cmd += ["-t", str(limit_seconds)]

    if use_gpu:
        cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "26"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "26"]

    cmd += [
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-hls_time",
        "6",
        "-hls_list_size",
        "0",
        "-hls_playlist_type",
        "vod",
        "-hls_segment_filename",
        os.path.join(hls_dir, "seg_%04d.ts"),
        playlist,
    ]

    cpu_samples = []
    stop_flag = threading.Event()

    def _sample_cpu():
        psutil.cpu_percent(interval=None)
        while not stop_flag.is_set():
            cpu_samples.append(psutil.cpu_percent(interval=0.5))

    monitor = threading.Thread(target=_sample_cpu, daemon=True)
    monitor.start()
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elapsed = time.perf_counter() - t0
    stop_flag.set()
    monitor.join(timeout=2)

    avg_cpu = round(sum(cpu_samples) / len(cpu_samples), 1) if cpu_samples else 0.0
    ts_files = [f for f in os.listdir(hls_dir) if f.endswith(".ts")]
    return {
        "encoder": "h264_nvenc" if use_gpu else "libx264",
        "duration_s": round(duration, 2),
        "elapsed_s": round(elapsed, 2),
        "speed_x": round(duration / elapsed, 1),
        "avg_cpu_pct": avg_cpu,
        "segments": len(ts_files),
        "playlist_path": playlist,
    }


def transcribe_audio_whisper(audio_path: str, transcript_out: str, model_size: str = "medium") -> Dict:
    os.makedirs(os.path.dirname(transcript_out), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    t_load0 = time.perf_counter()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    load_s = time.perf_counter() - t_load0

    t0 = time.perf_counter()
    segments, info = model.transcribe(
        audio_path,
        language="es",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
    )

    lines = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            lines.append(f"[{_format_hms(seg.start)}] {text}")

    elapsed = time.perf_counter() - t0
    with open(transcript_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {
        "model": f"faster-whisper-{model_size}",
        "device": device,
        "compute_type": compute_type,
        "model_load_s": round(load_s, 2),
        "audio_duration_s": round(info.duration, 2),
        "elapsed_s": round(elapsed, 2),
        "speed_x": round(info.duration / elapsed, 1),
        "lines": len(lines),
        "transcript_path": transcript_out,
    }


def run_demo_pipeline():
    video_path = r"D:\Projects\Cloud\video-demo\GestionDelConocimiento - GMT20260919-193007_Recording_1920x852.mp4"
    audio_out = r"D:\Projects\Cloud\audio_snippets\demo_gestion_conocimiento.mp3"
    hls_gpu_dir = r"D:\Projects\Cloud\frames\demo_gestion_conocimiento"
    hls_cpu_bench_dir = r"D:\Projects\Cloud\frames\demo_bench_cpu_300s"
    hls_gpu_bench_dir = r"D:\Projects\Cloud\frames\demo_bench_gpu_300s"
    transcript_out = r"D:\Projects\Cloud\transcriptions\transcripcion_demo_gestion_conocimiento.txt"

    print("[1/4] Extrayendo audio 64 kbps mono del video demo...")
    audio_metrics = extract_audio_mp3(video_path, audio_out)
    print(json.dumps(audio_metrics, indent=2))

    print("[2/4] Benchmark HLS 300 s (CPU libx264 vs GPU h264_nvenc)...")
    bench_cpu = transcode_hls(video_path, hls_cpu_bench_dir, use_gpu=False, limit_seconds=300)
    bench_gpu = transcode_hls(video_path, hls_gpu_bench_dir, use_gpu=True, limit_seconds=300)
    print("CPU 300s:", bench_cpu)
    print("GPU 300s:", bench_gpu)

    print("[3/4] Transcodificando video demo COMPLETO (2 h 16 min) a HLS con NVENC...")
    full_hls = transcode_hls(video_path, hls_gpu_dir, use_gpu=True, limit_seconds=0)
    print(json.dumps(full_hls, indent=2))

    print("[4/4] Transcribiendo audio COMPLETO con faster-whisper-medium en GPU CUDA...")
    asr_metrics = transcribe_audio_whisper(audio_out, transcript_out, model_size="medium")
    print(json.dumps(asr_metrics, indent=2))

    report = {
        "video_file": os.path.basename(video_path),
        "audio_extraction": audio_metrics,
        "hls_benchmark_300s_cpu": bench_cpu,
        "hls_benchmark_300s_gpu": bench_gpu,
        "hls_full_video_gpu": full_hls,
        "whisper_transcription": asr_metrics,
    }
    out_json = r"D:\Projects\Cloud\scripts\demo_pipeline_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("Pipeline demo completado. Metricas guardadas en:", out_json)


if __name__ == "__main__":
    run_demo_pipeline()
