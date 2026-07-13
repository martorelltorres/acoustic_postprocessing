#!/usr/bin/env python3
"""
MB + SSS fusion GIF: the multibeam 3D relief is tinted with sidescan intensity while the
camera orbits.

Output: results/media/07_mb_sss_fusion.gif

Three acts: bare bathymetry, a front sweeping across the relief projecting the sidescan
intensity, and the complete fusion.

WHY OPEN3D AND NOT MATPLOTLIB: matplotlib 3D has no z-buffer (Poly3DCollection sorts faces
with the painter's algorithm) and draws back faces over front ones. Open3D rasterizes with
GL. The legacy Visualizer is used with `visible=False`: filament's OffscreenRenderer
renders fine but leaves the process hanging on exit. Only ONE window can be created per
process — after destroy_window(), GLFW does not restart.

WHY THE DEM AND NOT mb_mesh.ply: the Poisson mesh interpolates the raw cloud and is noisy
at 20 cm — a bristly seafloor with holes where the mesh is not manifold. The DEM is the
MEDIAN of ~100 points per 10 cm cell: same multibeam geometry, far better estimator, and a
regular grid triangulates without holes. SSS intensity is sampled exactly as in
sss_mb_fusion.py, so the fusion is the same.

Usage:  python3 make_fusion_gif.py [results_dir]

Author: Antoni Martorell (SRV, UIB)
"""

import os
import sys

import numpy as np
import open3d as o3d
import rasterio
from scipy.ndimage import binary_erosion, binary_fill_holes, binary_opening, \
    distance_transform_edt, gaussian_filter, median_filter
from PIL import Image, ImageDraw, ImageFont

from make_media import PAGE, INK, INK2, MUTED, CREDIT, CMAP_DEPTH, CMAP_BS, \
    load_raster, crop_to_data

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

W = H = 800
# NO vertical exaggeration. The DEM has real 0.8 m steps between adjacent 0.25 m cells
# (p99 slope of 73°) in its eastern half: that is the documented attitude "fanning", and
# only the SLAM fixes it. It is not high-frequency noise — neither a 5x5 median, nor
# despiking, nor a 0.30 m cell brings it down. Exaggerating the vertical turned it into a
# forest of 2 m needles that does not exist. At x1.0 the 6.5 m of relief over 55 m still
# reads, and what you see is what is there.
VERT_EXAG = 1.0
FPS = 16

ACT1 = 22             # orbit over the bare bathymetry
ACT2 = 42             # the sidescan sweeps across and is projected
ACT3 = 30             # orbit over the complete fusion
TOTAL = ACT1 + ACT2 + ACT3

NO_SSS = np.array([0.16, 0.16, 0.15])   # neutral gray: outside the sidescan swath


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def dem_surface(dem_tif, erode_px=10, smooth=0.8, spike_m=0.35):
    """DEM raster -> (XYZ vertices in UTM, triangles). Cells with data only."""
    dem, ext = crop_to_data(*load_raster(dem_tif))
    finite = np.isfinite(dem)

    # The DEM's flank is a fringe of single-ping cells with noisy Z. Eroding is not enough
    # because the fringe comes in connected clumps: OPEN first (kills threads and specks),
    # fill the interior holes, and only then erode the shore.
    core = binary_fill_holes(binary_opening(finite, iterations=3))
    valid = binary_erosion(core, iterations=erode_px)

    # NEAREST-NEIGHBOUR fill, not the global median: otherwise the smoothing drags the
    # borders towards the median and raises a jagged lip along the shore.
    idx = distance_transform_edt(~finite, return_distances=False, return_indices=True)

    # DESPIKING against the local median: the raster equivalent of surface_relative_filter.
    # The eastern half of the DEM has single-ping cells (outer-beam curl) where the Z step
    # between adjacent 10 cm cells reaches 0.84 m at p99, i.e. 83° of slope. That is noise,
    # not relief, and neither the gaussian nor a 5x5 median removes it (the noise is
    # correlated over more than half a cell) — the value has to be replaced.
    z = dem[tuple(idx)]
    for _ in range(2):
        med = median_filter(z, size=5)
        spike = np.abs(z - med) > spike_m
        z = np.where(spike, med, z)

    z = gaussian_filter(z, smooth)

    zlo, zhi = np.percentile(z[valid], (1, 99))
    z = np.clip(z, zlo, zhi)

    ny, nx = dem.shape
    xs = np.linspace(ext[0], ext[1], nx)
    ys = np.linspace(ext[3], ext[2], ny)          # row 0 = north
    X, Y = np.meshgrid(xs, ys)

    idx = -np.ones((ny, nx), np.int64)
    idx[valid] = np.arange(valid.sum())
    verts = np.column_stack([X[valid], Y[valid], z[valid]])

    # One quad -> two triangles, only when all four corners have data.
    q = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    a, b = idx[:-1, :-1][q], idx[:-1, 1:][q]
    c, d = idx[1:, :-1][q], idx[1:, 1:][q]
    tris = np.vstack([np.column_stack([a, c, b]), np.column_stack([b, c, d])])

    return verts, tris


def sample_sss(verts, sss_tif):
    """Sidescan intensity at each vertex (x, y). Returns (intensity, valid)."""
    with rasterio.open(sss_tif) as src:
        a = src.read(1)
        nodata = src.nodata
        rows, cols = src.index(verts[:, 0], verts[:, 1])

    rows = np.asarray(rows)
    cols = np.asarray(cols)

    inside = (rows >= 0) & (rows < a.shape[0]) & (cols >= 0) & (cols < a.shape[1])

    val = np.zeros(len(verts), np.float32)
    val[inside] = a[rows[inside], cols[inside]]

    if nodata is not None:
        val[val == nodata] = 0.0

    return val, val > 0


def build_colors(verts, sss_tif):
    z = verts[:, 2]
    zlo, zhi = np.percentile(z, (2, 98))
    depth_rgb = CMAP_DEPTH((np.clip(z, zlo, zhi) - zlo) / (zhi - zlo))[:, :3]

    val, valid = sample_sss(verts, sss_tif)

    # The gray ramp reaches almost black and Open3D's lighting kills it entirely, so it is
    # compressed into [0.18, 1.0], which on screen still reads as low backscatter.
    ilo, ihi = np.percentile(val[valid], (2, 98))
    norm = np.clip((val - ilo) / max(ihi - ilo, 1e-6), 0, 1)
    sss_rgb = CMAP_BS(0.18 + 0.82 * norm)[:, :3]
    sss_rgb[~valid] = NO_SSS

    return depth_rgb, sss_rgb, valid


def orbit_front(az_deg, elev_deg=30.0):
    a, e = np.radians(az_deg), np.radians(elev_deg)
    return [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]


def caption(img, act, wipe, cover):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)

    try:
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        small = ImageFont.truetype("DejaVuSans.ttf", 16)
        tiny = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        big = small = tiny = ImageFont.load_default()

    d.text((34, 28), "Multibeam + sidescan fusion", font=big, fill=INK)

    sub = {0: "Multibeam bathymetry",
           1: f"Projecting sidescan intensity   {wipe * 100:3.0f}%",
           2: f"Fused  ·  {cover * 100:.0f}% of the seafloor has sidescan coverage"}[act]
    d.text((36, 70), sub, font=small, fill=INK2)

    if act >= 1:
        d.rectangle([36, H - 92, 52, H - 78], fill=tuple((NO_SSS * 255).astype(int)))
        d.text((60, H - 93), "no sidescan coverage", font=tiny, fill=MUTED)

    d.text((36, H - 40), CREDIT, font=tiny, fill=MUTED)
    return np.array(im)


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PKG_ROOT, "results")
    out = os.path.join(res, "media")
    os.makedirs(out, exist_ok=True)

    verts, tris = dem_surface(os.path.join(res, "tif", "mb_pointcloud.tif"))
    print(f"[fusion] surface: {len(verts):,} vertices, {len(tris):,} triangles")

    depth_rgb, sss_rgb, valid = build_colors(verts, os.path.join(res, "tif", "sss_mosaic.tif"))
    cover = float(valid.mean())
    print(f"[fusion] with SSS data: {valid.sum():,} ({cover * 100:.1f}%)")

    # Centered: UTM coordinates are ~4.4e6 and the z-buffer loses precision.
    c = verts.mean(axis=0)
    v = verts - c
    v[:, 2] *= VERT_EXAG

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.compute_vertex_normals()

    xlo, xhi = v[:, 0].min(), v[:, 0].max()

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H, visible=False)
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.background_color = hex2rgb(PAGE)
    opt.mesh_show_back_face = True

    ctr = vis.get_view_control()
    # FOV at its minimum (5°) = near-orthographic. At the default 60° with the camera
    # close, the relief warped and looked like a canyon.
    ctr.change_field_of_view(step=-90)
    frames = []

    for i in range(TOTAL):
        if i < ACT1:
            act, wipe, cols = 0, 0.0, depth_rgb
        elif i < ACT1 + ACT2:
            act = 1
            wipe = (i - ACT1 + 1) / ACT2
            front = xlo + wipe * (xhi - xlo)
            cols = np.where((v[:, 0] < front)[:, None], sss_rgb, depth_rgb)
        else:
            act, wipe, cols = 2, 1.0, sss_rgb

        mesh.vertex_colors = o3d.utility.Vector3dVector(cols)
        vis.update_geometry(mesh)

        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_front(orbit_front(-70 + 360.0 * i / TOTAL))
        ctr.set_zoom(0.48)

        vis.poll_events()
        vis.update_renderer()

        img = (np.asarray(vis.capture_screen_float_buffer(True)) * 255).astype(np.uint8)
        frames.append(caption(img, act, wipe, cover))

        if i % 20 == 0:
            print(f"[fusion] frame {i}/{TOTAL}")

    vis.destroy_window()

    gif = os.path.join(out, "07_mb_sss_fusion.gif")
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(gif, save_all=True, append_images=pil[1:],
                duration=int(1000 / FPS), loop=0, optimize=True)

    print(f"[fusion] done -> {gif}  ({len(frames)} frames, {W}x{H})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
