#!/usr/bin/env python3
"""
Author: Antoni Martorell
Affiliation: Systems, Robotics and Vision Group (SRV),
             University of the Balearic Islands (UIB)
Contact: antoni.martorell@uib.es
License: This code is provided for research and academic purposes.
"""

import open3d as o3d
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
MESH_FILE = "/home/uib/derelictes_ws/src/acoustic_postprocessing/results/mb_mesh.ply"
SSS_TIF   = "/home/uib/derelictes_ws/src/acoustic_postprocessing/results/sss_mosaic.tif"
OUTPUT_MESH = "/home/uib/derelictes_ws/src/acoustic_postprocessing/results/fused_data.ply"

COLORMAP = cm.viridis     # gray, viridis, inferno, etc.
NODATA_VALUE = 0

# -----------------------------º
# 1. CARGAR MALLA MB
# -----------------------------
print("Cargando malla MB...")
mesh = o3d.io.read_triangle_mesh(MESH_FILE)
mesh.compute_vertex_normals()

vertices = np.asarray(mesh.vertices)
print(f"Vértices: {len(vertices)}")

# -----------------------------
# 2. CARGAR MOSAICO SSS
# -----------------------------
print("Cargando mosaico SSS...")
with rasterio.open(SSS_TIF) as src:
    sss = src.read(1)
    transform = src.transform
    nodata = src.nodata

# -----------------------------
# 3. PROYECCIÓN SSS → VÉRTICES
# -----------------------------
print("Proyectando intensidad SSS sobre la malla...")

intensity = np.zeros(len(vertices), dtype=np.float32)

for i, (x, y, z) in enumerate(vertices):
    try:
        row, col = src.index(x, y)
        if 0 <= row < sss.shape[0] and 0 <= col < sss.shape[1]:
            val = sss[row, col]
            if nodata is not None and val == nodata:
                intensity[i] = NODATA_VALUE
            else:
                intensity[i] = val
        else:
            intensity[i] = NODATA_VALUE
    except Exception:
        intensity[i] = NODATA_VALUE

# -----------------------------
# 4. NORMALIZAR + COLOR MAP
# -----------------------------
print("Normalizando intensidad y aplicando colormap...")

valid = intensity > NODATA_VALUE
print(f"Vértices con intensidad válida: {np.sum(valid)} / {len(valid)}")

if not np.any(valid):
    raise RuntimeError(
        "ERROR: Ningún vértice del mesh intersecta con el mosaico SSS. "
        "Revisa CRS, sistema de coordenadas y solapamiento espacial."
    )
imin, imax = intensity[valid].min(), intensity[valid].max()
int_norm = (intensity - imin) / (imax - imin + 1e-6)

colors = COLORMAP(int_norm)[:, :3]  # RGBA → RGB

mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

# -----------------------------
# 5. GUARDAR RESULTADO
# -----------------------------
print(f"Guardando malla texturizada: {OUTPUT_MESH}")
o3d.io.write_triangle_mesh(OUTPUT_MESH, mesh)

print("✅ Proyección completada correctamente")
