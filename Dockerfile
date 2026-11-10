# Velora Video v2 = v1 (ComfyUI + Wan 2.2 fp8 horneados, ver Dockerfile.base) + post-proceso:
#   - RIFE 4.7 (ComfyUI-Frame-Interpolation, commit fijo) -> 16 fps a 32 fps
#   - RealESRGAN x4plus anime 6B -> 480p a 1080p
# Parte de la imagen v1, así las máquinas de Salad que ya la tienen solo bajan las capas nuevas (~0,3 GB).
FROM ghcr.io/ignaciourbinakonecta-dev/velora-video:wan22-v1
LABEL org.opencontainers.image.source="https://github.com/ignaciourbinakonecta-dev/velora-imagen-video"
LABEL org.opencontainers.image.description="Velora Video v2 - ComfyUI + Wan 2.2 + RIFE + RealESRGAN anime"

ARG FI_SHA=26545cc2dd95bc3d27f056016300673bdeee78f5
COPY bajar.py /tmp/bajar.py
RUN python3 /tmp/bajar.py \
      https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/archive/${FI_SHA}.zip /tmp/fi/fi.zip && \
    python3 -c "import zipfile; zipfile.ZipFile('/tmp/fi/fi.zip').extractall('/tmp/fi')" && \
    mv /tmp/fi/ComfyUI-Frame-Interpolation-${FI_SHA} /opt/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation && \
    rm -rf /tmp/fi && \
    python3 -m pip install --no-cache-dir einops kornia scipy opencv-contrib-python-headless && \
    python3 /tmp/bajar.py https://huggingface.co/marduk191/rife/resolve/main/rife47.pth \
      /opt/ComfyUI/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife47.pth && \
    python3 /tmp/bajar.py \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth \
      /opt/ComfyUI/models/upscale_models/RealESRGAN_x4plus_anime_6B.pth && \
    python3 -c "import einops, kornia, scipy, cv2; print('deps OK', cv2.__version__)"

COPY velora_nodo.py /opt/ComfyUI/custom_nodes/velora_comfy_nodes/__init__.py
