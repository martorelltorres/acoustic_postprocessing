#!/usr/bin/env python3
"""
Genera las entradas de make_pipeline_animation.py a partir de results/.

Salidas:
  results/anim_data/stages.npz      V, T, inten, cloud
  results/anim_data/footprints.npz  mb_traj, sss_traj

Uso:  python3 make_anim_data.py [results_dir]

DOS RESTRICCIONES QUE IMPONE EL CONSUMIDOR (make_pipeline_animation.py):

1. DECIMAR LA MALLA. Las etapas 3 y 4 construyen un `Poly3DCollection` con todos los
   triángulos en CADA uno de sus 100 fotogramas, y matplotlib 3D los ordena por
   profundidad a mano en cada dibujado. Con los 335 000 triángulos de la superficie real
   no termina. Se muestrea el DEM a ~0.9 m, que deja unos pocos miles.

2. ORDENAR LA NUBE POR LA MISIÓN. La etapa 2 dibuja `cloud[:n]` creciendo, así que el
   ORDEN de las filas es lo que se ve. `mb_pointcloud.xyz` sale de un voxel_down_sample
   de Open3D y NO está ordenado por tiempo (medido: la distancia entre puntos
   consecutivos tiene mediana 0.40 m pero p90 6.8 m y máximo 51 m; son saltos de voxel).
   Sin reordenar, la nube aparece como moteado disperso en vez de construirse a lo largo
   de la trayectoria. Se ordena por el índice del fix de navegación más cercano.

Author: Antoni Martorell (SRV, UIB)
"""

import os
import sys

import numpy as np
import rasterio
from scipy.ndimage import (binary_erosion, binary_fill_holes, binary_opening,
                           distance_transform_edt, gaussian_filter, median_filter)
from scipy.spatial import cKDTree

from make_media import load_raster, crop_to_data

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CELL_M = 0.9        # m/celda de la malla decimada
N_CLOUD = 25000     # puntos de la nube en la etapa 2
N_TRAJ = 1500       # fixes por trayectoria en la etapa 1


def coarse_surface(dem_tif, cell_m=CELL_M):
    """DEM raster -> (V, T) decimado a `cell_m`, sin el fleco dentado del borde."""
    dem, ext = crop_to_data(*load_raster(dem_tif))
    finite = np.isfinite(dem)

    core = binary_fill_holes(binary_opening(finite, iterations=3))
    valid = binary_erosion(core, iterations=6)

    idx = distance_transform_edt(~finite, return_distances=False, return_indices=True)
    z = gaussian_filter(median_filter(dem[tuple(idx)], size=5), 1.0)

    ny, nx = dem.shape
    res = (ext[1] - ext[0]) / nx
    step = max(1, int(round(cell_m / res)))

    z = z[::step, ::step]
    valid = valid[::step, ::step]

    ny, nx = z.shape
    X, Y = np.meshgrid(np.linspace(ext[0], ext[1], nx), np.linspace(ext[3], ext[2], ny))

    ids = -np.ones((ny, nx), np.int64)
    ids[valid] = np.arange(valid.sum())
    V = np.column_stack([X[valid], Y[valid], z[valid]])

    q = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    a, b = ids[:-1, :-1][q], ids[:-1, 1:][q]
    c, d = ids[1:, :-1][q], ids[1:, 1:][q]
    T = np.vstack([np.column_stack([a, c, b]), np.column_stack([b, c, d])])

    return V, T


def sample_sss(pts_xy, sss_tif):
    """Intensidad del sidescan en cada (x, y). 0 = fuera de la franja (el consumidor
    usa `face_i > 0` como máscara de cobertura)."""
    with rasterio.open(sss_tif) as src:
        a = src.read(1)
        nodata = src.nodata
        rows, cols = src.index(pts_xy[:, 0], pts_xy[:, 1])

    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < a.shape[0]) & (cols >= 0) & (cols < a.shape[1])

    val = np.zeros(len(pts_xy), np.float32)
    val[inside] = a[rows[inside], cols[inside]]

    if nodata is not None:
        val[val == nodata] = 0.0

    return val


def ordered_cloud(xyz_file, traj_xy, n=N_CLOUD, seed=0):
    """Submuestrea la nube y la ORDENA por el fix de navegación más cercano."""
    pts = np.loadtxt(xyz_file)

    rng = np.random.default_rng(seed)
    sel = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
    pts = pts[sel]

    _, near = cKDTree(traj_xy).query(pts[:, :2])
    return pts[np.argsort(near, kind="stable")]


def load_nav(res):
    cache = os.path.join(res, "media", ".nav_cache.npz")
    if not os.path.isfile(cache):
        raise SystemExit(
            f"Falta {cache}. Lo genera la extracción de navegación de make_media.py; "
            "sin él no hay trayectorias para la etapa 1.")

    d = np.load(cache)
    return d


def decimate(a, n):
    step = max(1, len(a) // n)
    return a[::step]


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PKG_ROOT, "results")
    out = os.path.join(res, "anim_data")
    os.makedirs(out, exist_ok=True)

    dem_tif = os.path.join(res, "tif", "mb_pointcloud.tif")
    sss_tif = os.path.join(res, "tif", "sss_mosaic.tif")
    xyz = os.path.join(res, "pointcloud", "mb_pointcloud.xyz")

    V, T = coarse_surface(dem_tif)
    print(f"[anim-data] malla decimada a {CELL_M} m: {len(V):,} vértices, {len(T):,} triángulos")

    inten = sample_sss(V[:, :2], sss_tif)
    print(f"[anim-data] intensidad SSS: {(inten > 0).sum():,} vértices con dato "
          f"({(inten > 0).mean() * 100:.1f}%)")

    nav = load_nav(res)
    mb_traj = np.column_stack([nav["xm"], nav["ym"]])
    sss_traj = np.column_stack([nav["xs"], nav["ys"]])

    cloud = ordered_cloud(xyz, mb_traj)
    print(f"[anim-data] nube: {len(cloud):,} puntos, ordenados a lo largo de la misión")

    mb_traj = decimate(mb_traj, N_TRAJ)
    sss_traj = decimate(sss_traj, N_TRAJ)
    print(f"[anim-data] trayectorias: MB {len(mb_traj):,}, SSS {len(sss_traj):,}")

    np.savez_compressed(os.path.join(out, "stages.npz"),
                        V=V, T=T, inten=inten, cloud=cloud)
    np.savez_compressed(os.path.join(out, "footprints.npz"),
                        mb_traj=mb_traj, sss_traj=sss_traj)

    print(f"[anim-data] listo -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
