#!/usr/bin/env python3
"""
Builds the inputs of make_pipeline_animation.py from results/.

Outputs:
  results/media/stages.npz      V, T, inten, cloud
  results/media/footprints.npz  mb_traj, sss_traj

Usage:  python3 make_nav_cache.py        # once: writes results/media/.nav_cache.npz
        python3 make_anim_data.py [results_dir]

TWO CONSTRAINTS IMPOSED BY THE CONSUMER (make_pipeline_animation.py):

1. DECIMATE THE MESH. Stages 3 and 4 build a `Poly3DCollection` with every triangle on
   EACH of their 100 frames, and matplotlib 3D depth-sorts them by hand on every draw.
   With the 335,000 triangles of the real surface it never finishes. The DEM is sampled
   at ~0.9 m, which leaves a few thousand.

2. ORDER THE CLOUD ALONG THE MISSION. Stage 2 draws a growing `cloud[:n]`, so the ROW
   ORDER is what the viewer sees. `mb_pointcloud.xyz` comes out of an Open3D
   voxel_down_sample and is NOT time-ordered (consecutive points: median gap 0.40 m but
   p90 6.8 m, max 51 m — those are voxel jumps). Unsorted, the cloud shows up as scattered
   speckle instead of building along the track, so rows are sorted by the index of the
   nearest navigation fix.

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

CELL_M = 0.9        # m/cell of the decimated mesh
N_CLOUD = 25000     # cloud points in stage 2
N_TRAJ = 1500       # fixes per trajectory in stage 1


def coarse_surface(dem_tif, cell_m=CELL_M):
    """DEM raster -> (V, T) decimated to `cell_m`, without the jagged border fringe."""
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
    """Sidescan intensity at each (x, y). 0 = outside the swath (the consumer uses
    `face_i > 0` as its coverage mask)."""
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
    """Subsamples the cloud and ORDERS it by the nearest navigation fix."""
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
            f"{cache} is missing. Generate it with `python3 make_nav_cache.py`; "
            "without it there are no trajectories for stage 1.")

    d = np.load(cache)
    return d


def decimate(a, n):
    step = max(1, len(a) // n)
    return a[::step]


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PKG_ROOT, "results")
    # results/media/, like every other media product (see results/README.md). The old
    # anim_data/ directory matched neither the README nor the files on disk.
    out = os.path.join(res, "media")
    os.makedirs(out, exist_ok=True)

    dem_tif = os.path.join(res, "tif", "mb_pointcloud.tif")
    sss_tif = os.path.join(res, "tif", "sss_mosaic.tif")
    xyz = os.path.join(res, "pointcloud", "mb_pointcloud.xyz")

    V, T = coarse_surface(dem_tif)
    print(f"[anim-data] mesh decimated to {CELL_M} m: {len(V):,} vertices, {len(T):,} triangles")

    inten = sample_sss(V[:, :2], sss_tif)
    print(f"[anim-data] SSS intensity: {(inten > 0).sum():,} vertices with data "
          f"({(inten > 0).mean() * 100:.1f}%)")

    nav = load_nav(res)
    mb_traj = np.column_stack([nav["xm"], nav["ym"]])
    sss_traj = np.column_stack([nav["xs"], nav["ys"]])

    cloud = ordered_cloud(xyz, mb_traj)
    print(f"[anim-data] cloud: {len(cloud):,} points, ordered along the mission")

    mb_traj = decimate(mb_traj, N_TRAJ)
    sss_traj = decimate(sss_traj, N_TRAJ)
    print(f"[anim-data] trajectories: MB {len(mb_traj):,}, SSS {len(sss_traj):,}")

    np.savez_compressed(os.path.join(out, "stages.npz"),
                        V=V, T=T, inten=inten, cloud=cloud)
    np.savez_compressed(os.path.join(out, "footprints.npz"),
                        mb_traj=mb_traj, sss_traj=sss_traj)

    print(f"[anim-data] done -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
