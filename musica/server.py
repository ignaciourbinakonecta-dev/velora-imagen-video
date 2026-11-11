"""Servidor de música de Velora (GPU Música en SaladCloud): ACE-Step v1 3.5B (Apache-2.0, uso comercial).

GET  /ready   -> 200 cuando el modelo está cargado (sonda de Salad), 503 mientras carga.
POST /musica  -> JSON {prompt: estilo (géneros, ánimo, instrumentos), segundos: 10-240, seed, pasos: 27-60}
              <- {"mp3_b64": ..., "segundos": ...}   (instrumental: sin letra)
"""
import base64
import os
import random
import subprocess
import threading
import time
import traceback

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

CKPT = "/app/checkpoints"
TMP = "/tmp/velora_musica"
os.makedirs(TMP, exist_ok=True)
app = FastAPI()
_pipe = None
_error = None
_lock = threading.Lock()


def _cargar():
    global _pipe, _error
    try:
        import torch
        from acestep.pipeline_ace_step import ACEStepPipeline
        t0 = time.time()
        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        p = ACEStepPipeline(checkpoint_dir=CKPT, dtype="bfloat16" if bf16 else "float32")
        # calentamiento: carga los pesos y compila lo necesario con una pista corta
        p(audio_duration=8, prompt="calm piano", lyrics="[instrumental]", infer_step=10,
          save_path=os.path.join(TMP, "calentamiento.wav"), manual_seeds=[1])
        _pipe = p
        print(f">> ACE-Step listo en {time.time() - t0:.0f} s", flush=True)
    except Exception as e:  # noqa: BLE001
        _error = f"{e!r}"
        traceback.print_exc()


threading.Thread(target=_cargar, daemon=True).start()


class Pedido(BaseModel):
    prompt: str
    segundos: float = 90
    seed: int | None = None
    pasos: int = 60


@app.get("/ready")
def ready():
    if _pipe is not None:
        return {"status": "ready"}
    return JSONResponse({"status": "error" if _error else "loading", "error": _error}, status_code=503)


@app.get("/health")
def health():
    return {"ok": True, "cargado": _pipe is not None, "error": _error}


@app.post("/musica")
def musica(p: Pedido):
    if _pipe is None:
        raise HTTPException(503, _error or "cargando modelo")
    seg = max(10.0, min(240.0, float(p.segundos)))
    seed = p.seed if p.seed is not None else random.randint(1, 2**31 - 1)
    wav = os.path.join(TMP, f"m_{time.time_ns()}.wav")
    t0 = time.time()
    with _lock:
        try:
            salida = _pipe(audio_duration=seg, prompt=p.prompt, lyrics="[instrumental]",
                           infer_step=max(20, min(60, int(p.pasos))), manual_seeds=[seed], save_path=wav)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            raise HTTPException(500, f"{e!r}"[:500])
    archivo = next((x for x in (salida or []) if str(x).endswith(".wav")), wav)
    mp3 = archivo.rsplit(".", 1)[0] + ".mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", archivo, "-b:a", "192k", mp3], check=True)
    with open(mp3, "rb") as f:
        datos = f.read()
    for x in (archivo, mp3):
        try:
            os.remove(x)
        except OSError:
            pass
    print(f">> {seg:.0f} s de música en {time.time() - t0:.1f} s (seed {seed})", flush=True)
    return {"mp3_b64": base64.b64encode(datos).decode(), "segundos": seg, "seed": seed}


if __name__ == "__main__":
    uvicorn.run(app, host="::", port=int(os.getenv("PORT", "3000")))
