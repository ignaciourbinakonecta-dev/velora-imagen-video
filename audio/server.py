"""Servidor de voz de Velora (GPU Audio en SaladCloud): IndexTTS-2.5 (bilibili, voz clonada + 8 emociones).

GET  /ready  -> 200 cuando el modelo está cargado (sonda de Salad), 503 mientras carga.
POST /tts    -> JSON {text, lang: "es"|"ja"|"en", voz_b64: wav de referencia (5-15 s) en base64,
                      emo_vector: [feliz, enojo, triste, miedo, asco, melancolía, sorpresa, calma] (0-1, opcional),
                      emo_b64: audio de referencia de EMOCIÓN en base64 (opcional), emo_alpha (0-1),
                      emo_text: descripción de la emoción (opcional), duration_factor (0.5-2, >1 = más lento)}
             <- {"wav_b64": ..., "segundos": ...}
Las referencias se guardan por huella: IndexTTS reutiliza su análisis si la misma voz vuelve a pedirse.
"""
import base64
import hashlib
import os
import threading
import time
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # /app/index-tts: el modelo usa rutas relativas

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

app = FastAPI()
_tts = None
_error = None
_lock = threading.Lock()
TMP = "/tmp/velora"
os.makedirs(TMP, exist_ok=True)
LANGS = {"es": "ES", "ja": "JA", "en": "EN", "zh": "ZH"}


def _cargar():
    global _tts, _error
    try:
        from indextts.infer_v2_5 import IndexTTS2
        t0 = time.time()
        _tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints",
                         use_bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
                         use_qwen_emo=True)
        print(f">> IndexTTS-2.5 listo en {time.time() - t0:.0f} s", flush=True)
    except Exception as e:  # noqa: BLE001
        _error = f"{e!r}"
        traceback.print_exc()


threading.Thread(target=_cargar, daemon=True).start()


def _archivo(b64: str, sufijo: str) -> str:
    datos = base64.b64decode(b64)
    ruta = os.path.join(TMP, hashlib.sha1(datos).hexdigest()[:16] + sufijo)
    if not os.path.exists(ruta):
        with open(ruta, "wb") as f:
            f.write(datos)
    return ruta


class Pedido(BaseModel):
    text: str
    lang: str = "es"
    voz_b64: str
    emo_vector: list[float] | None = None
    emo_b64: str | None = None
    emo_alpha: float = 1.0
    emo_text: str | None = None
    duration_factor: float = 1.0


@app.get("/ready")
def ready():
    if _tts is not None:
        return {"status": "ready"}
    return JSONResponse({"status": "error" if _error else "loading", "error": _error}, status_code=503)


@app.get("/health")
def health():
    return {"ok": True, "cargado": _tts is not None, "error": _error}


@app.post("/tts")
def tts(p: Pedido):
    if _tts is None:
        raise HTTPException(503, _error or "cargando modelo")
    voz = _archivo(p.voz_b64, ".wav")
    kw = {"emo_alpha": max(0.0, min(1.0, p.emo_alpha)),
          "duration_factor": max(0.5, min(2.0, p.duration_factor))}
    if p.emo_b64:
        kw["emo_audio_prompt"] = _archivo(p.emo_b64, "_emo.wav")
    elif p.emo_text:
        kw.update(use_emo_text=True, emo_text=p.emo_text)
    elif p.emo_vector and any(v > 0 for v in p.emo_vector):
        kw["emo_vector"] = [max(0.0, min(1.0, float(v))) for v in (list(p.emo_vector) + [0.0] * 8)[:8]]
    salida = os.path.join(TMP, f"out_{time.time_ns()}.wav")
    t0 = time.time()
    with _lock:  # una sola GPU: de a un pedido
        try:
            _tts.infer(spk_audio_prompt=voz, text=p.text, output_path=salida,
                       lang=LANGS.get(p.lang.lower(), p.lang.upper()), use_random=False, **kw)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            raise HTTPException(500, f"{e!r}"[:500])
    with open(salida, "rb") as f:
        wav = f.read()
    os.remove(salida)
    import soundfile as sf
    import io
    seg = sf.info(io.BytesIO(wav)).duration
    print(f">> {seg:.1f} s de voz en {time.time() - t0:.1f} s ({p.lang}, {list(kw)})", flush=True)
    return {"wav_b64": base64.b64encode(wav).decode(), "segundos": seg}


if __name__ == "__main__":
    uvicorn.run(app, host="::", port=int(os.getenv("PORT", "3000")))
