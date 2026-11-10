"""Nodos de ComfyUI de Velora Studio.

VeloraSubirHF: guarda el video terminado en MP4 y lo sube a un repo (dataset) de Hugging Face.
Usa la librería de Python huggingface_hub (ComfyUI ya la trae) y el HF_TOKEN del entorno del contenedor.
Así la entrega no depende de la CLI `hf` ni del gateway de Salad (que corta a los 100 s).
"""
import io
import os
import time
import uuid

import folder_paths


class VeloraSubirHF:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
            "filename_prefix": ("STRING", {"default": "velora"}),
            "repo": ("STRING", {"default": ""}),
            "carpeta": ("STRING", {"default": "clips"}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "subir"
    OUTPUT_NODE = True
    CATEGORY = "velora"

    def subir(self, video, filename_prefix, repo, carpeta):
        from huggingface_hub import HfApi

        nombre = f"{os.path.basename(filename_prefix) or 'velora'}_{uuid.uuid4().hex[:8]}.mp4"
        ruta = os.path.join(folder_paths.get_output_directory(), nombre)
        video.save_to(ruta)  # formato/codec automáticos -> mp4 h264
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        ultimo = None
        for intento in range(5):
            try:
                api.upload_file(path_or_fileobj=ruta, path_in_repo=f"{carpeta}/{nombre}", repo_id=repo,
                                repo_type="dataset", commit_message=f"clip {nombre}")
                ultimo = None
                break
            except Exception as e:  # noqa: BLE001
                ultimo = e
                time.sleep(5 * (intento + 1))
        if ultimo is not None:
            try:  # avisar al estudio en vez de dejarlo esperando
                api.upload_file(path_or_fileobj=io.BytesIO(f"error al subir: {ultimo}".encode()),
                                path_in_repo=f"{carpeta}/_error.txt", repo_id=repo, repo_type="dataset")
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"No se pudo subir el clip a {repo}: {ultimo}")
        return {"ui": {"text": [f"{carpeta}/{nombre}"]}}


NODE_CLASS_MAPPINGS = {"VeloraSubirHF": VeloraSubirHF}
NODE_DISPLAY_NAME_MAPPINGS = {"VeloraSubirHF": "Velora: subir a Hugging Face"}
