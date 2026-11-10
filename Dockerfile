# Velora Video: ComfyUI API + Wan 2.2 I2V 14B (fp8) horneados en la imagen.
# Salad no cobra mientras baja la imagen, así que los modelos grandes ya no se descargan con la GPU cobrando.
# Al encender solo se bajan: codificador de texto umt5 (6,3 GB) y LightX2V alto (1,1 GB, modo rápido).
FROM ghcr.io/saladtechnologies/comfyui-api:comfy0.35.0-api1.19.2-torch2.13.0-cuda13.0-runtime
LABEL org.opencontainers.image.source="https://github.com/ignaciourbinakonecta-dev/velora-imagen-video"
LABEL org.opencontainers.image.description="Velora Video - ComfyUI + Wan 2.2 I2V 14B fp8 (Apache 2.0)"

COPY bajar.py /tmp/bajar.py
# una capa por modelo grande
RUN python3 /tmp/bajar.py \
    https://huggingface.co/Comfy-Org/Wan-2.2/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors \
    /opt/ComfyUI/models/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
RUN python3 /tmp/bajar.py \
    https://huggingface.co/Comfy-Org/Wan-2.2/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors \
    /opt/ComfyUI/models/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
RUN python3 /tmp/bajar.py \
    https://huggingface.co/Comfy-Org/Wan-2.1/resolve/main/split_files/vae/wan_2.1_vae.safetensors \
    /opt/ComfyUI/models/vae/wan_2.1_vae.safetensors && \
    python3 /tmp/bajar.py \
    https://huggingface.co/Comfy-Org/Wan-2.2/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors \
    /opt/ComfyUI/models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors

# nodo propio: sube cada clip terminado a Hugging Face (evita el límite de 100 s del gateway de Salad)
COPY velora_nodo.py /opt/ComfyUI/custom_nodes/velora_comfy_nodes/__init__.py
