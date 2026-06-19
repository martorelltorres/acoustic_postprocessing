#!/usr/bin/env python3
"""
Standalone (non-ROS) version of sss_mb_fusion.py with hardcoded paths.
Textures the MB mesh with SSS intensity. Output: fused_data.ply.

Author: Antoni Martorell (SRV, UIB)
"""

import open3d as o3d
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Paths and config
MESH_FILE = "/home/uib/derelictes_ws/src/acoustic_postprocessing/results/mb_mesh.ply"
SSS_TIF   = "/home/uib/derelictes_ws/src/acoustic_postprocessing/results/sss_mosaic.tif"
OUTPUT_MESH = "/home/uib/derelictes_ws/src/acoustic_postprocessing/results/fused_data.ply"

COLORMAP = cm.viridis     # gray, viridis, inferno, etc.
NODATA_VALUE = 0

# Load MB mesh
print("Loading MB mesh...")
mesh = o3d.io.read_triangle_mesh(MESH_FILE)
mesh.compute_vertex_normals()

vertices = np.asarray(mesh.vertices)
print(f"Vertices: {len(vertices)}")

# Load SSS mosaic
print("Loading SSS mosaic...")
with rasterio.open(SSS_TIF) as src:
    sss = src.read(1)
    transform = src.transform
    nodata = src.nodata

# Sample SSS intensity at each vertex (x, y) UTM
print("Projecting SSS intensity onto the mesh...")

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

# Normalize and apply colormap to vertices
print("Normalizing intensity and applying colormap...")

valid = intensity > NODATA_VALUE
print(f"Vertices with valid intensity: {np.sum(valid)} / {len(valid)}")

if not np.any(valid):
    raise RuntimeError(
        "ERROR: No mesh vertex intersects the SSS mosaic. "
        "Check CRS, coordinate system and spatial overlap."
    )
imin, imax = intensity[valid].min(), intensity[valid].max()
int_norm = (intensity - imin) / (imax - imin + 1e-6)

colors = COLORMAP(int_norm)[:, :3]  # RGBA -> RGB

mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

# Save result
print(f"Saving textured mesh: {OUTPUT_MESH}")
o3d.io.write_triangle_mesh(OUTPUT_MESH, mesh)

print("Projection completed.")
