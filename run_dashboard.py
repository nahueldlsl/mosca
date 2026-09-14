"""
Lanzador del Panel Visual Interactivo de AlphaFly Truco Uruguayo.
Inicia el servidor local FastAPI / Uvicorn y abre automáticamente el navegador web.
"""

import sys
import time
import webbrowser
import threading
import uvicorn

PORT = 8000
HOST = "127.0.0.1"
URL = f"http://{HOST}:{PORT}"

def open_browser():
    time.sleep(1.2)
    print(f"\n[AlphaFly] Abriendo navegador en: {URL}")
    try:
        webbrowser.open(URL)
    except Exception as e:
        print(f"[AlphaFly] Abre manualmente en tu navegador: {URL} ({e})")

if __name__ == "__main__":
    print("=" * 65)
    print("  🧠 INICIANDO PANEL INTERACTIVO DE ALPHAFLY TRUCO URUGUAYO 🇺🇾")
    print(f"  Conectoma FlyEM Male CNS v1.0 & Telemetría en Tiempo Real")
    print(f"  URL Local: {URL}")
    print("=" * 65)

    threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("app_truco_fly:app", host=HOST, port=PORT, log_level="info")
