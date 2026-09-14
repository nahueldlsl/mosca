"""
Script para descargar los archivos del conectoma de FlyEM Male CNS v1.0.
Permite descargar con reanudación (HTTP Range) y barras de progreso en tiempo real.
"""

import os
import sys
import argparse
import requests
from tqdm import tqdm

BASE_GCS_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"

# Lista ordenada por prioridad (los más importantes primero para poder trabajar de inmediato)
FILES_INFO = [
    # 1. Anotaciones de neuronas, tipos y clases (~14 MB)
    ("body-annotations-male-cns-v1.0-minconf-0.5.feather", "Anotaciones celulares curadas (tipos, clases, soma)"),
    # 2. Neurotransmisores predichos (~41 MB)
    ("body-neurotransmitters-male-cns-v1.0.feather", "Predicciones de neurotransmisores por neurona"),
    # 3. Pesos de conectoma trazados (~484 MB)
    ("connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather", "Grafo de conectividad (solo neuronas trazadas)"),
    # 4. Pesos de conectoma significativos (~479 MB)
    ("connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather", "Grafo de conectividad (conexiones significativas)"),
    # 5. Estadísticas de segmentos (~742 MB)
    ("body-stats-male-cns-v1.0-minconf-0.5.feather", "Estadísticas de sinapsis por cuerpo"),
    # 6. Grafo completo de pesos (~1002 MB)
    ("connectome-weights-male-cns-v1.0-minconf-0.5.feather", "Grafo completo de conectividad"),
    # 7. Neurotransmisores a nivel T-bar (~2.5 GB)
    ("tbar-neurotransmitters-male-cns-v1.0.feather", "Predicciones de neurotransmisores por T-bar"),
    # 8. Pares de sinapsis trazados (~2.8 GB)
    ("syn-partners-male-cns-v1.0-minconf-0.5-traced-only.feather", "Pares sinápticos trazados (coordenadas pre y post)"),
    # 9. Pares de sinapsis significativos (~2.8 GB)
    ("syn-partners-male-cns-v1.0-minconf-0.5-significant-only.feather", "Pares sinápticos significativos"),
    # 10. Pares de sinapsis completos (~6.5 GB)
    ("syn-partners-male-cns-v1.0-minconf-0.5.feather", "Todos los pares sinápticos"),
    # 11. Puntos sinápticos completos (~12.5 GB)
    ("syn-points-male-cns-v1.0-minconf-0.5.feather", "Todas las coordenadas 3D de puntos sinápticos"),
]

def download_file(filename: str, dest_dir: str) -> bool:
    url = BASE_GCS_URL + filename
    os.makedirs(dest_dir, exist_ok=True)
    local_path = os.path.join(dest_dir, filename)

    # Verificar si el archivo ya existe y qué tamaño tiene
    head_resp = requests.head(url, timeout=30)
    if head_resp.status_code != 200:
        print(f"Error al verificar {filename}: HTTP {head_resp.status_code}")
        return False

    remote_size = int(head_resp.headers.get("content-length", 0))
    initial_bytes = 0
    headers = {}

    if os.path.exists(local_path):
        existing_size = os.path.getsize(local_path)
        if existing_size == remote_size and remote_size > 0:
            print(f"[YA DESCARGADO] {filename} ({remote_size / (1024*1024):.2f} MB)")
            return True
        elif existing_size < remote_size:
            print(f"[REANUDANDO] {filename} desde {existing_size / (1024*1024):.2f} MB de {remote_size / (1024*1024):.2f} MB")
            headers["Range"] = f"bytes={existing_size}-"
            initial_bytes = existing_size
        else:
            # Si es mayor (archivo corrupto), reiniciar
            print(f"[REINICIANDO] {filename}")
            initial_bytes = 0

    mode = "ab" if initial_bytes > 0 else "wb"
    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        if resp.status_code not in (200, 206):
            print(f"Error al descargar {filename}: HTTP {resp.status_code}")
            return False

        with open(local_path, mode) as f, tqdm(
            desc=filename[:35],
            total=remote_size,
            initial=initial_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    return True

def main():
    parser = argparse.ArgumentParser(description="Descargador de datos del cerebro de la mosca (FlyEM Male CNS v1.0)")
    parser.add_argument("--output-dir", default=os.path.join("data", "flat-connectome"), help="Directorio destino")
    parser.add_argument("--priority-only", action="store_true", help="Descargar solo tablas principales (~2.7 GB en vez de 29 GB)")
    parser.add_argument("--core-only", action="store_true", help="Descargar solo anotaciones + neurotransmisores + red trazada (~540 MB)")
    parser.add_argument("--files", nargs="*", help="Descargar archivos específicos por nombre")
    args = parser.parse_args()

    print("=" * 70)
    print(" FlyEM Male CNS v1.0 - Descarga de Conectoma")
    print(" Destino:", os.path.abspath(args.output_dir))
    print("=" * 70)

    if args.files:
        targets = [f for f in FILES_INFO if f[0] in args.files]
    elif args.core_only:
        targets = FILES_INFO[:3]
    elif args.priority_only:
        targets = FILES_INFO[:6]
    else:
        targets = FILES_INFO

    total_est = sum(
        int(requests.head(BASE_GCS_URL + f[0], timeout=15).headers.get("content-length", 0))
        for f in targets
    )
    print(f"Archivos a descargar: {len(targets)} (~{total_est / (1024*1024*1024):.2f} GB total)\n")

    success_count = 0
    for idx, (fname, desc) in enumerate(targets, 1):
        print(f"[{idx}/{len(targets)}] {desc}")
        ok = download_file(fname, args.output_dir)
        if ok:
            success_count += 1
        print()

    print("=" * 70)
    print(f"Completado: {success_count}/{len(targets)} archivos listos en {args.output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
