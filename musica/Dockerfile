# Velora Música: ACE-Step v1 3.5B (Apache-2.0 en código y pesos: uso comercial) con el modelo ya incluido.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_CACHE_DIR=/tmp/uvcache HF_HUB_DISABLE_PROGRESS_BARS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 ca-certificates build-essential python3-pip \
    && rm -rf /var/lib/apt/lists/* && pip3 install --no-cache-dir uv

ARG ACESTEP_SHA=1bee4c9f5b43e30995f8d4d33b3919197ce1bd68
WORKDIR /app
RUN git clone https://github.com/ace-step/ACE-Step.git && cd ACE-Step && git checkout ${ACESTEP_SHA}
RUN uv venv /app/venv --python 3.10 \
    && uv pip install --python /app/venv/bin/python "torch==2.6.0" "torchaudio==2.6.0" "torchvision==0.21.0" \
         --index-url https://download.pytorch.org/whl/cu124 \
    && uv pip install --python /app/venv/bin/python -e /app/ACE-Step \
    && uv pip install --python /app/venv/bin/python "fastapi==0.115.*" "uvicorn==0.30.*" \
    && rm -rf /tmp/uvcache

# pesos del modelo dentro de la imagen (music_dcae_f8c8, music_vocoder, ace_step_transformer, umt5-base)
RUN /app/venv/bin/python -c "from huggingface_hub import snapshot_download; \
snapshot_download('ACE-Step/ACE-Step-v1-3.5B', local_dir='/app/checkpoints')" && rm -rf /root/.cache/huggingface

COPY server.py /app/server.py
EXPOSE 3000
CMD ["/app/venv/bin/python", "/app/server.py"]
