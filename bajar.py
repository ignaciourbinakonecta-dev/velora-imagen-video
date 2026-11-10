"""Descarga un archivo con reintentos (se usa al construir la imagen, en GitHub Actions)."""
import os
import sys
import time
import urllib.request

url, destino = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(destino), exist_ok=True)
for intento in range(1, 6):
    try:
        urllib.request.urlretrieve(url, destino)
        print(f"OK {destino} {os.path.getsize(destino) / 1e9:.2f} GB")
        break
    except Exception as e:  # noqa: BLE001
        print(f"intento {intento} falló: {e}")
        time.sleep(15 * intento)
else:
    sys.exit(1)
