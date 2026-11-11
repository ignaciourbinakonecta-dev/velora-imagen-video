# Velora Audio: IndexTTS-2.5 (bilibili Model Use License: uso comercial gratis bajo 100M MAU / RMB 1.000M al año)
# con los modelos ya incluidos, para arrancar en SaladCloud sin descargar nada.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_CACHE_DIR=/tmp/uvcache GIT_LFS_SKIP_SMUDGE=1 HF_HUB_DISABLE_PROGRESS_BARS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 curl ca-certificates build-essential python3-pip \
    && rm -rf /var/lib/apt/lists/* && pip3 install --no-cache-dir uv

# código fijado a un commit concreto (reproducible)
ARG INDEXTTS_SHA=d9e41aac89fd00b3d71497fddb287b7f24613712
WORKDIR /app
RUN git clone https://github.com/index-tts/index-tts.git && cd index-tts && git checkout ${INDEXTTS_SHA}
WORKDIR /app/index-tts
RUN uv sync --frozen --python 3.11 && rm -rf /tmp/uvcache \
    && uv pip install --python .venv/bin/python "fastapi==0.115.*" "uvicorn==0.30.*" soundfile && rm -rf /tmp/uvcache

# pesos de IndexTTS-2.5 + modelos auxiliares (w2v-bert-2.0, CAMPPlus, BigVGAN)
RUN .venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('IndexTeam/IndexTTS-2.5', local_dir='checkpoints')" \
    && .venv/bin/python -c "from indextts.utils.model_download import ensure_models_available; print(ensure_models_available('checkpoints'))" \
    && rm -rf /root/.cache/huggingface
# preparar el frontend de texto (normalizador, japonés) para que no se arme en cada arranque; si falla, no es fatal
RUN .venv/bin/python -c "from indextts.utils.front import TextNormalizer; t=TextNormalizer(enable_glossary=True); t.load(); \
from indextts.utils.ja_g2p import JapaneseG2PProcessor; JapaneseG2PProcessor(g2p_ratio=0); \
from indextts.utils.nemo_tn import normalize_text as n; print(n('Tengo 25 años', 'es'))" || echo "aviso: frontend se preparará al arrancar"

COPY server.py /app/index-tts/server.py
EXPOSE 3000
CMD ["/app/index-tts/.venv/bin/python", "/app/index-tts/server.py"]
