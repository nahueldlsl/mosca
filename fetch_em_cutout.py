"""
Script para extraer recortes (cutouts) 2D o subvolúmenes 3D de microscopía electrónica (EM)
directamente desde Google Cloud Storage (gs://flyem-male-cns/em/em-clahe-jpeg).
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cloudvolume import CloudVolume

EM_CLOUD_PATH = "precomputed://gs://flyem-male-cns/em/em-clahe-jpeg"

def get_volume():
    """Inicializa la conexión CloudVolume al volumen EM con CLAHE y JPEG."""
    print(f"Conectando a {EM_CLOUD_PATH} (HTTPS)...")
    vol = CloudVolume(EM_CLOUD_PATH, use_https=True, progress=True)
    return vol

def extract_cutout(vol, x: int, y: int, z: int, dx: int = 500, dy: int = 500, dz: int = 1):
    """
    Extrae un bloque o corte en las coordenadas dadas.
    Coordenadas en vóxeles (resolución de 8nm isotropic).
    """
    x_end = x + dx
    y_end = y + dy
    z_end = z + dz

    print(f"Descargando cutout: X[{x}:{x_end}], Y[{y}:{y_end}], Z[{z}:{z_end}]...")
    # CloudVolume soporta indexación por rangos [x, y, z, channel]
    if dz == 1:
        cutout = vol[x:x_end, y:y_end, z, 0]
    else:
        cutout = vol[x:x_end, y:y_end, z:z_end, 0]

    return cutout

def save_cutout(cutout, out_dir: str, prefix: str = "em_cutout"):
    os.makedirs(out_dir, exist_ok=True)
    
    # Si es un corte 2D
    if cutout.ndim == 2 or (cutout.ndim == 3 and cutout.shape[2] == 1):
        slice_2d = cutout.squeeze()
        
        # Guardar en NumPy
        npy_path = os.path.join(out_dir, f"{prefix}.npy")
        np.save(npy_path, slice_2d)
        
        # Guardar en PNG
        img = Image.fromarray(slice_2d)
        png_path = os.path.join(out_dir, f"{prefix}.png")
        img.save(png_path)
        
        print(f"Corte 2D guardado en:")
        print(f" - {png_path} ({slice_2d.shape[1]}x{slice_2d.shape[0]} px)")
        print(f" - {npy_path}")
        return png_path
    else:
        # Subvolumen 3D
        npy_path = os.path.join(out_dir, f"{prefix}_3d.npy")
        np.save(npy_path, cutout)
        
        # Guardar corte central en PNG
        mid_z = cutout.shape[2] // 2
        mid_slice = cutout[:, :, mid_z].squeeze()
        img = Image.fromarray(mid_slice)
        png_path = os.path.join(out_dir, f"{prefix}_mid_slice.png")
        img.save(png_path)
        
        print(f"Volumen 3D guardado en {npy_path} (forma: {cutout.shape})")
        print(f"Corte central guardado en {png_path}")
        return png_path

def main():
    parser = argparse.ArgumentParser(description="Extrae cortes de microscopía EM del cerebro de la mosca.")
    parser.add_argument("--x", type=int, default=40000, help="Coordenada X central/inicial (default: 40000)")
    parser.add_argument("--y", type=int, default=40000, help="Coordenada Y central/inicial (default: 40000)")
    parser.add_argument("--z", type=int, default=20000, help="Coordenada Z (plano de corte) (default: 20000)")
    parser.add_argument("--size", type=int, default=500, help="Tamaño en píxeles (ancho y alto) (default: 500)")
    parser.add_argument("--depth", type=int, default=1, help="Profundidad en capas Z (default: 1 para imagen 2D)")
    parser.add_argument("--out-dir", default=os.path.join("data", "em_cutouts"), help="Carpeta de salida")
    parser.add_argument("--no-plot", action="store_true", help="No mostrar ventana gráfica de matplotlib")
    args = parser.parse_args()

    vol = get_volume()
    print(f"Dimensiones del volumen total (voxels): {vol.shape}")
    print(f"Resolución de vóxel (nm): {vol.resolution}")

    cutout = extract_cutout(vol, args.x, args.y, args.z, dx=args.size, dy=args.size, dz=args.depth)
    prefix = f"em_x{args.x}_y{args.y}_z{args.z}_s{args.size}"
    png_path = save_cutout(cutout, args.out_dir, prefix=prefix)

    if not args.no_plot:
        plt.figure(figsize=(8, 8))
        if cutout.ndim == 2 or (cutout.ndim == 3 and cutout.shape[2] == 1):
            plt.imshow(cutout.squeeze().T, cmap="gray")
        else:
            plt.imshow(cutout[:, :, cutout.shape[2] // 2].squeeze().T, cmap="gray")
        plt.title(f"FlyEM Male CNS - Microscopía EM (X:{args.x}, Y:{args.y}, Z:{args.z})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{prefix}_plot.png"), dpi=150)
        print(f"Gráfico con anotaciones guardado en: {os.path.join(args.out_dir, f'{prefix}_plot.png')}")

if __name__ == "__main__":
    main()
