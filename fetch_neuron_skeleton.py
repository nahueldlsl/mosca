"""
Script para descargar y visualizar esqueletos neuronales en formato SWC
desde el dataset de FlyEM Male CNS v1.0.
"""

import os
import argparse
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SWC_BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/"

def download_skeleton(body_id: int, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{body_id}.swc"
    local_path = os.path.join(out_dir, filename)
    url = SWC_BASE_URL + filename

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"Esqueleto ya en caché: {local_path}")
        return local_path

    print(f"Descargando esqueleto para la neurona bodyId={body_id} desde {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(local_path, "wb") as f:
            f.write(resp.read())
        print(f"Descargado con éxito: {local_path} ({os.path.getsize(local_path) / 1024:.1f} KB)")
        return local_path
    except Exception as e:
        print(f"Error descargando esqueleto para {body_id}: {e}")
        return None

def parse_swc(swc_path: str):
    """
    Parsea un archivo SWC estándar.
    Retorna un DataFrame con columnas: [id, type, x, y, z, radius, parent]
    """
    rows = []
    with open(swc_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 7:
                rows.append({
                    "id": int(parts[0]),
                    "type": int(parts[1]),
                    "x": float(parts[2]),
                    "y": float(parts[3]),
                    "z": float(parts[4]),
                    "radius": float(parts[5]),
                    "parent": int(parts[6])
                })
    return pd.DataFrame(rows)

def plot_skeleton_3d(df: pd.DataFrame, body_id: int, out_dir: str):
    """Grafica el esqueleto en 3D usando Matplotlib."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Mapear IDs a posiciones
    node_coords = {row["id"]: (row["x"], row["y"], row["z"]) for _, row in df.iterrows()}

    # Dibujar segmentos
    for _, row in df.iterrows():
        parent_id = row["parent"]
        if parent_id in node_coords:
            p_coord = node_coords[parent_id]
            ax.plot(
                [row["x"], p_coord[0]],
                [row["y"], p_coord[1]],
                [row["z"], p_coord[2]],
                color="#1f77b4",
                alpha=0.6,
                linewidth=1.2
            )

    # Resaltar el soma (tipo 1 o nodo raíz con parent=-1)
    soma_nodes = df[(df["type"] == 1) | (df["parent"] == -1)]
    if not soma_nodes.empty:
        ax.scatter(
            soma_nodes["x"], soma_nodes["y"], soma_nodes["z"],
            color="red", s=40, label="Soma / Raíz"
        )

    ax.set_title(f"Morfología Neuronal 3D - Body ID: {body_id}\n({len(df)} nodos)", fontsize=12)
    ax.set_xlabel("X (8nm)")
    ax.set_ylabel("Y (8nm)")
    ax.set_zlabel("Z (8nm)")
    ax.legend(loc="upper right")

    out_png = os.path.join(out_dir, f"{body_id}_skeleton_3d.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"Visualización 3D guardada en: {out_png}")
    return out_png

def main():
    parser = argparse.ArgumentParser(description="Descarga y visualiza esqueletos neuronales SWC.")
    parser.add_argument("--body", type=int, default=100001, help="Body ID de la neurona (ej: 100001)")
    parser.add_argument("--out-dir", default=os.path.join("data", "skeletons"), help="Directorio destino")
    args = parser.parse_args()

    swc_path = download_skeleton(args.body, args.out_dir)
    if swc_path:
        df = parse_swc(swc_path)
        print(f"Esqueleto parseado con {len(df)} nodos.")
        plot_skeleton_3d(df, args.body, args.out_dir)

if __name__ == "__main__":
    main()
