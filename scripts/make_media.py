#!/usr/bin/env python3
"""
Builds outreach material (social media) from results/.

Outputs in results/media/:
  01_bathymetry.png     shaded bathymetry + AUV track
  02_backscatter.png    multibeam backscatter + track
  03_sss_mosaic.png     sidescan mosaic + track
  04_survey_reveal.gif  the track draws itself over the bathymetry
  05_relief_lightsweep.gif  the sun orbits the DEM and reveals the micro-relief
  06_two_surveys.png    both missions (MB and SSS) in the same UTM frame
  08_layer_stack.png    schematic 3D stack (5 layers): track / SSS / multibeam / stereo
                        photomosaic (mosaics/lawnmower.png) / synthetic seafloor
                        (rock, sand, posidonia). Each dataset fills the width.
  09_octagon_vs_lawnmower.png  why nested octagons beat the classic lawnmower. Two 3D
                        stacks: on top the pattern the AUV flies (the lawnmower WITH
                        cross-tracks, so that it gets loop closures too), below the
                        photomosaic that came out of that same area. The argument:
                        90° vs 45° turns and INS drift. Uses the REAL Andratx track
                        (multibeam_SLAM) and the mosaics in results/media/mosaics/.

Usage:  python3 make_media.py [results_dir]

COLOR DECISIONS (measured, not aesthetic):
  - Depth and backscatter are MAGNITUDE -> single-hue sequential ramp. No `turbo` or
    `viridis` here: a rainbow invents boundaries the data does not have. Depth uses the
    blue ramp (light=shallow -> dark=deep) and backscatter neutral gray
    (light=hard/reflective), which is the sonar convention.
  - The two tracks are IDENTITY -> two categorical hues. Orange (#d95926) and magenta
    (#d55181): worst-case dE=48.8 under protanopia/deuteranopia (threshold 12), and >=3:1
    contrast against both the map's deepest blue and black. The prettier orange+yellow
    pair only reaches dE=11.4 and fails.
  - Every track carries a dark casing because it crosses light and dark areas of the map,
    and is labelled directly as well as in the legend: identity never rests on color alone.

Author: Antoni Martorell (SRV, UIB)
"""

import os
import sys
import textwrap

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, label as cc_label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, LightSource, Normalize
from matplotlib.patheffects import withStroke
from matplotlib.patches import Polygon, Rectangle, Circle, Arc, FancyBboxPatch
from matplotlib.transforms import Affine2D
import rasterio
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── tokens ─────────────────────────────────────────────────────────────────────
PAGE   = "#0d0d0d"      # page plane (dark)
INK    = "#ffffff"      # primary ink
INK2   = "#c3c2b7"      # secondary ink
MUTED  = "#898781"      # axes / labels
MB_COL  = "#d95926"     # multibeam track  (orange slot, dark theme)
SSS_COL = "#d55181"     # sidescan track   (magenta slot, dark theme)
STEREO_COL = "#4fb9a6"  # "next step" layer (teal): distinct from orange/magenta/blue
LAWN_COL  = "#828a94"   # lawnmower (neutral baseline)
OCT_COL   = "#d55181"   # octagons (magenta, the pattern being highlighted)
CROSS_COL = "#f4c94b"   # amber: self-crossings = loop closures
TURN_COL  = "#e0554e"   # red: 90° turns = where the INS injects drift

# REAL octagon track (Andratx 3 m mission), which lives in the SLAM package.
SLAM_TRAJ = os.path.join(PKG_ROOT, "..", "multibeam_SLAM", "results", "raw_trajectory.npy")

# Blue sequential ramp (steps 100..700 of the reference palette)
DEPTH_STEPS = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
               "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
               "#184f95", "#104281", "#0d366b"]
CMAP_DEPTH = LinearSegmentedColormap.from_list("depth", DEPTH_STEPS[::-1])  # deep=dark
CMAP_BS = LinearSegmentedColormap.from_list("bs", ["#0f0f0e", "#4a4a46", "#8f8e86", "#d3d2c9", "#f5f5f1"])

CREDIT = "SPARUS II AUV  ·  Andratx, Mallorca  ·  2 Jul 2026  ·  SRV — UIB"


# ── utilities ──────────────────────────────────────────────────────────────────
def load_raster(path):
    """Returns (float array with NaN at nodata, extent in UTM)."""
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        nod = src.nodata
        b = src.bounds

    if nod is not None:
        a[a == nod] = np.nan
    a[~np.isfinite(a)] = np.nan

    return a, (b.left, b.right, b.bottom, b.top)


def crop_to_data(a, extent, pad_px=12):
    """Crops to the bbox of the pixels with data. The rasters carry a lot of empty."""
    rows, cols = np.nonzero(np.isfinite(a))
    if rows.size == 0:
        return a, extent

    r0, r1 = max(rows.min() - pad_px, 0), min(rows.max() + pad_px + 1, a.shape[0])
    c0, c1 = max(cols.min() - pad_px, 0), min(cols.max() + pad_px + 1, a.shape[1])

    x0, x1, y0, y1 = extent
    px = (x1 - x0) / a.shape[1]
    py = (y1 - y0) / a.shape[0]

    return a[r0:r1, c0:c1], (x0 + c0 * px, x0 + c1 * px, y1 - r1 * py, y1 - r0 * py)


def shade(a, cmap, vert_exag=6.0, azdeg=315, altdeg=45):
    """Colors with `cmap` and multiplies by a hillshade. NaN -> transparent."""
    finite = np.isfinite(a)
    vmin, vmax = np.nanpercentile(a, (2, 98))
    norm = Normalize(vmin, vmax)

    filled = np.where(finite, a, np.nanmedian(a))
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    rgb = ls.shade(filled, cmap=cmap, norm=norm, vert_exag=vert_exag,
                   blend_mode="soft")

    rgba = np.dstack([rgb[:, :, :3], finite.astype(float)])
    return rgba, norm


def frame(figsize=(9, 9)):
    fig = plt.figure(figsize=figsize, facecolor=PAGE)
    ax = fig.add_axes([0.06, 0.08, 0.80, 0.80])
    ax.set_facecolor(PAGE)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return fig, ax


def titles(fig, title):
    """A short label only. No subtitle: nobody reads it on social media and it stole canvas."""
    fig.text(0.06, 0.945, title, color=INK, fontsize=25, fontweight="bold", va="top")
    fig.text(0.06, 0.028, CREDIT, color=MUTED, fontsize=9.5, va="center")


def scalebar(ax, extent, meters=None):
    x0, x1, y0, y1 = extent
    span = x1 - x0
    if meters is None:
        meters = max(5, int(round(span / 5 / 5)) * 5)

    xa = x0 + span * 0.06
    ya = y0 + (y1 - y0) * 0.055
    halo = [withStroke(linewidth=3.2, foreground="#000000")]

    ax.plot([xa, xa + meters], [ya, ya], color=INK, lw=3, solid_capstyle="butt",
            path_effects=halo, zorder=6)
    ax.text(xa + meters / 2, ya + (y1 - y0) * 0.018, f"{meters} m", color=INK,
            fontsize=11, ha="center", fontweight="bold", path_effects=halo, zorder=6)


def north_arrow(ax, extent):
    x0, x1, y0, y1 = extent
    xa = x1 - (x1 - x0) * 0.075
    ya = y0 + (y1 - y0) * 0.055
    dy = (y1 - y0) * 0.075
    halo = [withStroke(linewidth=3.2, foreground="#000000")]

    ax.annotate("", xy=(xa, ya + dy), xytext=(xa, ya),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.2), zorder=6)
    ax.text(xa, ya + dy * 1.15, "N", color=INK, fontsize=12, ha="center",
            fontweight="bold", path_effects=halo, zorder=6)


def track(ax, x, y, color, label=None, lw=2.0, z=5):
    """Track with a dark casing: it crosses light and dark areas of the map."""
    ax.plot(x, y, color="#0d0d0d", lw=lw + 1.8, solid_capstyle="round", zorder=z)
    line, = ax.plot(x, y, color=color, lw=lw, solid_capstyle="round", zorder=z + 1,
                    label=label)
    return line


def miles(n):
    """Thousands separator with a thin space: 194 046, not 194,046."""
    return f"{n:,}".replace(",", " ")


def colorbar(fig, ax, mappable, label, as_depth=False):
    # Inside the axes, not flush to the edge: at 0.945 the labels ran off the canvas.
    cax = fig.add_axes([0.885, 0.135, 0.016, 0.40])
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, color=INK2, fontsize=11)
    cb.ax.tick_params(colors=MUTED, labelsize=9.5)
    cb.outline.set_edgecolor("#2c2c2a")

    if as_depth:
        # The DEM's Z is a negative elevation. It is labelled as positive DEPTH, which is
        # how a bathymetric map is read, without touching the data or the shading.
        cb.ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{-v:.0f}")
        )


# Vehicle sprite. The nose is the YELLOW section and sits on the RIGHT of the PNG
# (centroid of yellow pixels at x=516 of 647), so the sprite advances towards +x.
AUV_PNG = "sparusII.png"
AUV_LEN_PX = 132          # sprite length before zoom; it gets rotated, so leave slack
AUV_ZOOM = 0.36           # ~48 px on the 750 canvas. It is an ICON, not to scale: the
                          # Sparus II is 1.6 m, which would be ~20 px and invisible.


def load_auv_sprite(path, length_px=AUV_LEN_PX):
    """Sparus II PNG -> RGBA with transparent background, cropped and scaled.

    The PNG is opaque with a white background, but the hull is ALSO white: thresholding
    "white -> transparent" punches holes in it. Only the white CONNECTED TO THE BORDER
    (components touching the frame), which is the real background, is erased. That way the
    16,834 white pixels inside the hull survive.
    """
    a = np.array(Image.open(path).convert("RGBA"))
    rgb = a[:, :, :3].astype(int)

    near_white = (rgb > 238).all(axis=2)
    lab, _ = cc_label(near_white)

    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    border.discard(0)

    a[np.isin(lab, list(border)), 3] = 0

    ys, xs = np.nonzero(a[:, :, 3] > 0)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    im = Image.fromarray(a)
    h = max(1, round(im.height * length_px / im.width))

    return im.resize((length_px, h), Image.LANCZOS)


def auv_rotated(sprite, heading_deg):
    """Rotates the sprite to the heading, never leaving it belly-up.

    Rotating naively leaves the vehicle upside down on westward headings. When the heading
    points left, the sprite is mirrored vertically BEFORE rotating, so the nose follows the
    heading while the back stays up. The standard trick for side-view sprites.
    """
    im = sprite
    if np.cos(np.radians(heading_deg)) < 0:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)

    return np.array(im.rotate(heading_deg, resample=Image.BICUBIC, expand=True))


# ── figures ────────────────────────────────────────────────────────────────────
def fig_bathymetry(res, out, nav):
    dem, ext = crop_to_data(*load_raster(os.path.join(res, "tif", "mb_pointcloud.tif")))
    rgba, norm = shade(dem, CMAP_DEPTH)

    fig, ax = frame()
    ax.imshow(rgba, extent=ext, origin="upper", interpolation="bilinear")
    track(ax, nav["xm"], nav["ym"], MB_COL, lw=1.5)

    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    scalebar(ax, ext); north_arrow(ax, ext)

    ax.text(0.025, 0.975, "AUV track", transform=ax.transAxes, color=MB_COL,
            fontsize=12, fontweight="bold", va="top",
            path_effects=[withStroke(linewidth=3.2, foreground="#000000")])

    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_DEPTH)
    colorbar(fig, ax, sm, "depth (m)", as_depth=True)

    titles(fig, "Multibeam bathymetry")
    fig.savefig(os.path.join(out, "01_bathymetry.png"), dpi=120, facecolor=PAGE)
    plt.close(fig)


def fig_backscatter(res, out, nav):
    bs, ext = crop_to_data(*load_raster(os.path.join(res, "tif", "mb_intensity.tif")))
    fig, ax = frame()

    norm = Normalize(*np.nanpercentile(bs, (2, 98)))
    ax.imshow(np.ma.masked_invalid(bs), extent=ext, origin="upper", cmap=CMAP_BS,
              norm=norm, interpolation="bilinear")
    track(ax, nav["xm"], nav["ym"], MB_COL, lw=2.2)

    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    scalebar(ax, ext); north_arrow(ax, ext)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_BS)
    colorbar(fig, ax, sm, "backscatter (bright = hard)")

    titles(fig, "Multibeam backscatter")
    fig.savefig(os.path.join(out, "02_backscatter.png"), dpi=120, facecolor=PAGE)
    plt.close(fig)


def fig_sss(res, out, nav):
    sss, ext = crop_to_data(*load_raster(os.path.join(res, "tif", "sss_mosaic.tif")))
    fig, ax = frame()

    norm = Normalize(*np.nanpercentile(sss, (2, 98)))
    ax.imshow(np.ma.masked_invalid(sss), extent=ext, origin="upper", cmap=CMAP_BS,
              norm=norm, interpolation="bilinear")
    track(ax, nav["xs"], nav["ys"], SSS_COL, lw=2.2)

    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    scalebar(ax, ext); north_arrow(ax, ext)

    ax.text(0.025, 0.975, "AUV track", transform=ax.transAxes, color=SSS_COL,
            fontsize=12, fontweight="bold", va="top",
            path_effects=[withStroke(linewidth=3.2, foreground="#000000")])

    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_BS)
    colorbar(fig, ax, sm, "backscatter (bright = hard)")

    titles(fig, "Sidescan sonar mosaic")
    fig.savefig(os.path.join(out, "03_sss_mosaic.png"), dpi=120, facecolor=PAGE)
    plt.close(fig)


def fig_two_surveys(res, out, nav):
    sss, ext = crop_to_data(*load_raster(os.path.join(res, "tif", "sss_mosaic.tif")))
    _, mb_ext = crop_to_data(*load_raster(os.path.join(res, "tif", "mb_intensity.tif")))

    fig, ax = frame()
    norm = Normalize(*np.nanpercentile(sss, (2, 98)))
    ax.imshow(np.ma.masked_invalid(sss), extent=ext, origin="upper", cmap=CMAP_BS,
              norm=norm, interpolation="bilinear", alpha=0.85)

    x0, x1, y0, y1 = mb_ext
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=MB_COL, lw=1.6,
            ls="--", zorder=4)

    l1 = track(ax, nav["xm"], nav["ym"], MB_COL, label="Multibeam  13:52", lw=1.5)
    l2 = track(ax, nav["xs"], nav["ys"], SSS_COL, label="Sidescan  10:44", lw=2.2)

    halo = [withStroke(linewidth=3.2, foreground="#000000")]
    ax.text(nav["xm"][len(nav["xm"]) // 2], nav["ym"][len(nav["ym"]) // 2] + 3,
            "multibeam", color=MB_COL, fontsize=12, fontweight="bold", ha="center",
            path_effects=halo, zorder=7)
    ax.text(nav["xs"][len(nav["xs"]) // 2], nav["ys"][len(nav["ys"]) // 2] - 4,
            "sidescan", color=SSS_COL, fontsize=12, fontweight="bold", ha="center",
            path_effects=halo, zorder=7)

    leg = ax.legend(handles=[l1, l2], loc="upper left", frameon=True, fontsize=11)
    leg.get_frame().set_facecolor("#111110")
    leg.get_frame().set_edgecolor("#2c2c2a")
    for t in leg.get_texts():
        t.set_color(INK2)

    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    scalebar(ax, ext); north_arrow(ax, ext)

    titles(fig, "Multibeam + sidescan missions")
    fig.savefig(os.path.join(out, "06_two_surveys.png"), dpi=120, facecolor=PAGE)
    plt.close(fig)


def gif_reveal(res, out, nav, frames=72, fps=18):
    dem, ext = crop_to_data(*load_raster(os.path.join(res, "tif", "mb_pointcloud.tif")))
    rgba, _ = shade(dem, CMAP_DEPTH)

    x, y = nav["xm"], nav["ym"]
    # How many fixes each frame reveals. The FULL polyline up to that point is drawn:
    # subsampling the track (x[::step]) cut the lawnmower's corners and the shape no longer
    # matched 01_bathymetry.png.
    cuts = np.linspace(2, len(x), frames).astype(int)

    fig, ax = frame(figsize=(7.5, 7.5))
    ax.imshow(rgba, extent=ext, origin="upper", interpolation="bilinear")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    scalebar(ax, ext); north_arrow(ax, ext)

    casing, = ax.plot([], [], color="#0d0d0d", lw=4.0, solid_capstyle="round", zorder=5)
    line, = ax.plot([], [], color=MB_COL, lw=2.2, solid_capstyle="round", zorder=6)

    sprite = load_auv_sprite(os.path.join(out, AUV_PNG))
    imbox = OffsetImage(np.zeros((2, 2, 4)), zoom=AUV_ZOOM, interpolation="bilinear")
    auv = AnnotationBbox(imbox, (x[0], y[0]), frameon=False, pad=0.0, zorder=8,
                         box_alignment=(0.5, 0.5))
    ax.add_artist(auv)

    titles(fig, "Multibeam mission")
    hud = ax.text(0.025, 0.975, "", transform=ax.transAxes, color=INK, fontsize=12,
                  va="top", fontweight="bold",
                  path_effects=[withStroke(linewidth=3.2, foreground="#000000")], zorder=8)

    # Heading by differences over a ~2 s window (nav at 20 Hz). With two consecutive fixes
    # the navigation noise spins the sprite like a top.
    win = 40

    def update(i):
        k = cuts[i]
        casing.set_data(x[:k], y[:k])
        line.set_data(x[:k], y[:k])

        j = max(0, k - win)
        dx, dy = x[k - 1] - x[j], y[k - 1] - y[j]
        heading = np.degrees(np.arctan2(dy, dx)) if (dx or dy) else 0.0

        imbox.set_data(auv_rotated(sprite, heading))
        # AnnotationBbox draws the image at `xybox`, not at `xy`. Setting only `xy` leaves
        # the sprite pinned at the start point, merely rotating in place.
        auv.xy = auv.xybox = (x[k - 1], y[k - 1])

        hud.set_text(f"t = {nav['tm'][k - 1] - nav['tm'][0]:.0f} s")
        return casing, line, auv, hud

    anim = FuncAnimation(fig, update, frames=len(cuts), blit=False)
    anim.save(os.path.join(out, "04_survey_reveal.gif"),
              writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": PAGE})
    plt.close(fig)


def gif_lightsweep(res, out, frames=48, fps=16):
    """The sun orbits the DEM and the relief appears and disappears.

    A replacement for a 3D orbit: matplotlib has no z-buffer (Poly3DCollection sorts faces
    with the painter's algorithm) and at grazing angles it drew the jagged swath edge as
    stalactites that do not exist. The light sweep is 2D, does not lie about the geometry,
    and shows the micro-relief better than a rotation.
    """
    dem, ext = crop_to_data(*load_raster(os.path.join(res, "tif", "mb_pointcloud.tif")))

    finite = np.isfinite(dem)
    filled = np.where(finite, dem, np.nanmedian(dem))
    vmin, vmax = np.nanpercentile(dem, (2, 98))
    norm = Normalize(vmin, vmax)

    fig, ax = frame(figsize=(7.5, 7.5))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")

    im = ax.imshow(np.zeros((*dem.shape, 4)), extent=ext, origin="upper",
                   interpolation="bilinear")
    scalebar(ax, ext); north_arrow(ax, ext)

    titles(fig, "Seafloor relief")

    hud = ax.text(0.975, 0.975, "", transform=ax.transAxes, color=INK, fontsize=12,
                  va="top", ha="right", fontweight="bold",
                  path_effects=[withStroke(linewidth=3.2, foreground="#000000")], zorder=8)

    def update(i):
        az = 360.0 * i / frames
        ls = LightSource(azdeg=az, altdeg=32)
        rgb = ls.shade(filled, cmap=CMAP_DEPTH, norm=norm, vert_exag=9.0,
                       blend_mode="soft")[:, :, :3]
        im.set_data(np.dstack([rgb, finite.astype(float)]))
        hud.set_text(f"light from {az:3.0f}\u00b0")
        return im, hud

    anim = FuncAnimation(fig, update, frames=frames, blit=True)
    anim.save(os.path.join(out, "05_relief_lightsweep.gif"),
              writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": PAGE})
    plt.close(fig)


# ── 3D figure: per-sensor layer stack ───────────────────────────────────────────
# Oblique (cabinet) projection: a point (u,v) of the plane -> canvas. This is NOT metric 3D
# (matplotlib has no z-buffer; see gif_lightsweep), it is a DIAGRAM: each product is a sheet
# and the v axis (north) recedes up and to the right. Sheets stack by level. Same skew trick
# as the "Affine transform of an image" example.
def _raster_rgba(a, cmap, pct=(2, 98)):
    """Single-band raster -> RGBA with alpha=finite (NaN transparent)."""
    vmin, vmax = np.nanpercentile(a, pct)
    rgba = cmap(Normalize(vmin, vmax)(a))
    rgba[..., 3] = np.isfinite(a).astype(float)
    return rgba


def _fbm(shape, rng, scales, weights):
    """Fractal noise: sum of octaves of smoothed white noise. Comes out ~N(0,1)."""
    out = np.zeros(shape, np.float32)
    for sc, w in zip(scales, weights):
        out += w * gaussian_filter(rng.standard_normal(shape).astype(np.float32), sc)
    return (out - out.mean()) / (out.std() + 1e-9)


def _procedural_seafloor(w, h, seed=7):
    """Synthetic, ILLUSTRATIVE seafloor: grainy sand with caustics, posidonia meadows
    (dark organic patches) and rocks with hemispheric shading."""
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.float32)

    # 1) sand: light/dark mix modulated by low-frequency grain
    grain = _fbm((h, w), rng, (1.5, 4.0, 11.0), (0.5, 0.3, 0.2))
    tg = np.clip(0.5 + 0.30 * grain, 0, 1)[..., None]
    img[:] = np.array([0.66, 0.56, 0.39]) * (1 - tg) + np.array([0.86, 0.77, 0.58]) * tg

    # caustics: faint net of sunlight filtered through the waves
    caus = _fbm((h, w), rng, (2.2, 5.5), (0.6, 0.4))
    caus = np.clip((caus - 0.35) * 1.7, 0, 1)[..., None]
    img += caus * np.array([0.10, 0.10, 0.07])

    # 2) posidonia: large contiguous patches (low frequency), DARK olive green with faint
    # internal mottling. Real posidonia from above looks almost black/brown.
    meadow = _fbm((h, w), rng, (14.0, 30.0, 60.0), (0.55, 0.30, 0.15))
    mask = np.clip((meadow - np.percentile(meadow, 58)) / 0.55 + 0.5, 0, 1)
    tex = _fbm((h, w), rng, (0.8, 2.5), (0.6, 0.4))[..., None]
    pos = np.clip(np.array([0.09, 0.16, 0.09]) + 0.07 * tex * np.array([0.35, 0.55, 0.28]), 0, 1)
    m3 = (mask ** 1.35)[..., None]
    img = img * (1 - m3) + pos * m3

    # 3) rocks: ellipses with hemispheric shading (light from the upper left)
    yy, xx = np.mgrid[0:h, 0:w]
    L = np.array([-0.55, -0.55, 0.63])
    for _ in range(30):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        rx = rng.uniform(0.020, 0.055) * w
        ry = rx * rng.uniform(0.7, 1.2)
        ang = rng.uniform(0, np.pi)
        dx = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang)
        dy = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
        d2 = (dx / rx) ** 2 + (dy / ry) ** 2
        inside = d2 < 1.0
        if not inside.any():
            continue
        zz = np.sqrt(np.clip(1.0 - d2, 0, 1))
        nx, ny_, nz = dx / rx, dy / ry, zz * 1.4
        nn = np.sqrt(nx * nx + ny_ * ny_ + nz * nz) + 1e-6
        sh = np.clip(0.32 + 0.85 * (nx * L[0] + ny_ * L[1] + nz * L[2]) / nn, 0.14, 1.2)
        base = np.array([0.47, 0.46, 0.43]) * rng.uniform(0.78, 1.12)
        rock = np.clip(base[None, None, :] * sh[..., None], 0, 1)
        img[inside] = rock[inside]

    # 4) water-column tint: a slight cool shift so it reads as submerged
    img = img * 0.94 + 0.06 * np.array([0.10, 0.22, 0.24])
    return np.clip(img, 0, 1)


def _rot2(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _raster_data_xy(a, ext, step=6):
    """World (x, y) of the pixels that carry data, subsampled. Feeds the footprint bbox.

    The extents are axis-aligned boxes around a diagonal strip of data, so they are far too
    loose to size the sheet with: the strip itself is what has to be measured.
    """
    sub = a[::step, ::step]
    r, c = np.nonzero(np.isfinite(sub))
    ny, nx = a.shape
    x = ext[0] + (c * step + 0.5) * (ext[1] - ext[0]) / nx
    y = ext[3] - (r * step + 0.5) * (ext[3] - ext[2]) / ny
    return np.column_stack([x, y])


def fig_layer_stack(res, out, nav):
    """3D stack: track / SSS / multibeam / stereo photomosaic / real seafloor.

    Every layer shares one rotated survey frame, so the products are co-registered sheet to
    sheet and the stack reads as a single area sensed four ways.
    """
    dem, dem_ext = crop_to_data(*load_raster(os.path.join(res, "tif", "mb_pointcloud.tif")))
    sss, sss_ext = crop_to_data(*load_raster(os.path.join(res, "tif", "sss_mosaic.tif")))

    # Optical layer: the lawnmower photomosaic, already de-rotated and cropped to its data.
    # It is a plain PNG with no georeferencing, so it cannot go through the survey frame
    # below; it is fitted into the footprint rectangle instead. Missing -> the layer falls
    # back to the empty "next step" frame.
    mos_png = os.path.join(out, "mosaics", "lawnmower.png")
    mos = _load_mosaic(mos_png, rot180=True) if os.path.isfile(mos_png) else None

    # ── survey frame: ONE transform for every layer ─────────────────────────────
    # All four products are drawn through the same world->sheet affine, so a given patch of
    # seabed lands on the same (u, v) of every sheet and the corner posts tie the SAME
    # footprint through the stack. That co-registration is the whole point of the figure:
    # one area, four sensors. Fitting each layer to its own bbox (what this did before) let
    # them drift apart and silently rescaled each one.
    #
    # The frame is rotated so the survey's major axis is horizontal. In raw UTM the
    # footprint runs diagonally across an axis-aligned sheet, which wastes most of the sheet
    # on nodata — the data covered barely a third of it.
    #
    # The sheet is sized to the MULTIBEAM survey (track + DEM, ~68 x 36 m): the area where
    # all four sensors have data. The sidescan swath is ~100 m across-track (+-50 m range),
    # so sizing the sheet to its full extent would shrink every other layer to a third of
    # the sheet to make room for its far-range wings. Instead the wings are clipped to the
    # sheet, which is also what the figure is claiming: one common area, four sensors.
    mb_xy = np.column_stack([nav["xm"], nav["ym"]])
    _, ev = np.linalg.eigh(np.cov(mb_xy.T))
    theta = -np.arctan2(ev[1, -1], ev[0, -1])
    c0 = mb_xy.mean(axis=0)

    foot = np.vstack([mb_xy, _raster_data_xy(dem, dem_ext)])
    foot = (foot - c0) @ _rot2(theta).T
    (fx0, fy0), (fx1, fy1) = foot.min(axis=0) - 2.0, foot.max(axis=0) + 2.0

    # Oblique projection geometry. SX: shear; SY: foreshortening (higher = more upright).
    SX, SY, THK = 0.48, 0.34, 0.05
    PW, MX, MY = 1.0, 0.03, 0.035
    LEVELS = 5                                    # 0 seafloor .. 4 track (top)

    # The sheet IS the survey footprint: its aspect comes from the rotated bbox, so the data
    # fills it edge to edge instead of floating in the middle of an arbitrary rectangle.
    SC = PW * (1 - 2 * MX) / (fx1 - fx0)          # metres -> sheet units
    PH = (fy1 - fy0) * SC / (1 - 2 * MY)
    GAP = SY * PH + 0.09                          # spacing > on-screen sheet height

    def iso(level):
        return Affine2D().from_values(1, 0, SX, SY, 0, level * GAP)

    def proj(u, v, level):
        return (u + SX * v, SY * v + level * GAP)

    # UTM -> sheet: recentre, rotate into the survey frame, scale, inset by the margins.
    W2S = (Affine2D().translate(-c0[0], -c0[1]).rotate(theta)
           .translate(-fx0, -fy0).scale(SC).translate(PW * MX, PH * MY))

    # Where the footprint lands on every sheet (used to place the non-georeferenced mosaic).
    RECT = (PW * MX, PW * (1 - MX), PH * MY, PH * (1 - MY))

    def fit_rect(w, h):                           # pixel box -> RECT, aspect preserved
        s = min((RECT[1] - RECT[0]) / w, (RECT[3] - RECT[2]) / h)
        ox = 0.5 * (RECT[0] + RECT[1] - w * s)
        oy = 0.5 * (RECT[2] + RECT[3] - h * s)
        return [ox, ox + w * s, oy, oy + h * s]

    # Canvas limits + figure size (no dead band: height = real aspect).
    top = LEVELS - 1
    content_r = PW + SX * PH
    xL, xR = -0.06, content_r + max(0.98, content_r * 0.72)
    yB, yT = -THK - 0.10, SY * PH + top * GAP + 0.14
    figW, axw, axh = 9.8, 0.96, 0.86
    figH = (yT - yB) / (xR - xL) * (axw * figW) / axh

    fig = plt.figure(figsize=(figW, figH), facecolor=PAGE)
    ax = fig.add_axes([0.02, 0.04, axw, axh])
    ax.set_facecolor(PAGE); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")

    # Assembly posts: join the sheet's 4 corners between base and top.
    for (u, v) in [(0, 0), (PW, 0), (PW, PH), (0, PH)]:
        p0, p1 = proj(u, v, 0), proj(u, v, top)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#38382f", lw=0.8,
                ls=(0, (4, 3)), zorder=1)

    def walls(level, edge, edge_dark, z):
        """Front and side walls -> give the layer its sheet thickness."""
        BL, BR, TR = proj(0, 0, level), proj(PW, 0, level), proj(PW, PH, level)
        front = [BL, BR, (BR[0], BR[1] - THK), (BL[0], BL[1] - THK)]
        right = [BR, TR, (TR[0], TR[1] - THK), (BR[0], BR[1] - THK)]
        ax.add_patch(Polygon(front, closed=True, fc=edge, ec="none", zorder=z))
        ax.add_patch(Polygon(right, closed=True, fc=edge_dark, ec="none", zorder=z))

    def corners(level):
        return [proj(0, 0, level), proj(PW, 0, level), proj(PW, PH, level), proj(0, PH, level)]

    def outline(level, edge, z, **kw):
        ax.add_patch(Polygon(corners(level), closed=True, fill=False, ec=edge, zorder=z, **kw))

    def clip(im, level):
        """Trim a layer to its sheet. The sidescan overflows it by design (see above)."""
        im.set_clip_path(Polygon(corners(level), transform=ax.transData))
        return im

    def side_label(level, text, color, sub=None):
        px, py = proj(PW, PH * 0.5, level)
        halo = [withStroke(linewidth=3.4, foreground="#000000")]
        ax.text(px + 0.04, py, text, color=color, fontsize=15, fontweight="bold",
                va="center", ha="left", path_effects=halo, zorder=level * 10 + 9)
        if sub:
            ax.text(px + 0.04, py - 0.055, sub, color=MUTED, fontsize=11,
                    va="center", ha="left", path_effects=halo, zorder=level * 10 + 9)

    tr = ax.transData

    # ── level 0: real seafloor (rock · sand · posidonia) ────────────────────────
    z = 0
    walls(0, "#241d12", "#150f07", z)
    sea = _procedural_seafloor(680, int(680 * PH / PW))
    clip(ax.imshow(sea, extent=[0, PW, 0, PH], origin="upper",
                   transform=iso(0) + tr, interpolation="bilinear", zorder=z + 2), 0)
    outline(0, "#6b5a3a", z + 4, lw=1.1)
    side_label(0, "Seafloor", "#d8c48f")

    # ── level 1: stereo cameras — optical photomosaic ───────────────────────────
    z = 10
    walls(1, "#141414", "#0c0c0c", z)
    if mos is not None:
        # No georeferencing on the PNG: it is fitted to the footprint rectangle. Its major
        # axis is already horizontal (_load_mosaic de-rotates it), which is the survey frame
        # the other layers land in, so it lines up with them.
        clip(ax.imshow(mos, extent=fit_rect(mos.shape[1], mos.shape[0]), origin="upper",
                       transform=iso(1) + tr, interpolation="bilinear", zorder=z + 2), 1)
        outline(1, STEREO_COL, z + 4, lw=1.4, alpha=0.9)
        side_label(1, "Stereo cameras", STEREO_COL)
    else:
        outline(1, STEREO_COL, z + 3, lw=1.4, ls="--", alpha=0.85)
        for t in np.linspace(0.14, 0.86, 6):             # faint "to do" grid
            a0, a1 = proj(t * PW, 0.08 * PH, 1), proj(t * PW, 0.92 * PH, 1)
            b0, b1 = proj(0.08 * PW, t * PH, 1), proj(0.92 * PW, t * PH, 1)
            ax.plot([a0[0], a1[0]], [a0[1], a1[1]], color=STEREO_COL, lw=0.5, alpha=0.16, zorder=z + 2)
            ax.plot([b0[0], b1[0]], [b0[1], b1[1]], color=STEREO_COL, lw=0.5, alpha=0.16, zorder=z + 2)
        cx, cy = proj(PW * 0.5, PH * 0.5, 1)             # stereo camera icon
        ax.add_patch(Rectangle((cx - 0.085, cy - 0.042), 0.17, 0.084, fc="#101816",
                               ec=STEREO_COL, lw=1.5, alpha=0.95, zorder=z + 3))
        for dx in (-0.042, 0.042):
            ax.add_patch(Circle((cx + dx, cy), 0.026, fc="none", ec=STEREO_COL, lw=1.6, zorder=z + 4))
            ax.add_patch(Circle((cx + dx, cy), 0.011, fc=STEREO_COL, ec="none", alpha=0.55, zorder=z + 4))
        ax.text(cx, cy - 0.085, "?", color=STEREO_COL, fontsize=18, fontweight="bold",
                ha="center", va="center", alpha=0.8, zorder=z + 4)
        side_label(1, "Stereo cameras", STEREO_COL, sub="— next step —")

    # ── level 2: multibeam bathymetry (shaded relief) ───────────────────────────
    z = 20
    walls(2, "#10233f", "#091729", z)
    rgba_dem, _ = shade(dem, CMAP_DEPTH)
    clip(ax.imshow(rgba_dem, extent=dem_ext, origin="upper",
                   transform=W2S + iso(2) + tr, interpolation="bilinear", zorder=z + 2), 2)
    outline(2, "#2c5488", z + 4, lw=1.0)
    side_label(2, "Multibeam bathymetry", "#7db3ef")

    # ── level 3: sidescan mosaic ───────────────────────────────────────────────
    z = 30
    walls(3, "#26261f", "#161610", z)
    clip(ax.imshow(_raster_rgba(sss, CMAP_BS), extent=sss_ext, origin="upper",
                   transform=W2S + iso(3) + tr, interpolation="bilinear", zorder=z + 2), 3)
    outline(3, "#4a4a44", z + 4, lw=1.0)
    side_label(3, "Side-scan sonar", INK2)

    # ── level 4: AUV track (glass panel) ───────────────────────────────────────
    z = 40
    walls(4, "#161d2e", "#0b1120", z)
    panel = [proj(0, 0, 4), proj(PW, 0, 4), proj(PW, PH, 4), proj(0, PH, 4)]
    ax.add_patch(Polygon(panel, closed=True, fc="#0f1626", ec="none", alpha=0.55, zorder=z + 1))
    outline(4, "#3a4a6b", z + 6, lw=1.0)

    # A single clean track: the two missions overlap almost entirely and together were an
    # illegible tangle. Orange = "AUV track", as in 01/02 of this same set.
    tr4 = iso(4) + tr
    uv = W2S.transform(mb_xy)                             # same frame as the sensor layers
    um, vm = uv[:, 0], uv[:, 1]
    ax.plot(um, vm, color="#0d0d0d", lw=3.4, transform=tr4, solid_capstyle="round", zorder=z + 2)
    ax.plot(um, vm, color=MB_COL, lw=1.9, transform=tr4, solid_capstyle="round", zorder=z + 3)

    try:                                                  # Sparus sprite at the end of the track
        sprite = load_auv_sprite(os.path.join(out, AUV_PNG))
        win = min(40, len(um) - 1)
        cxe, cye = proj(um[-1], vm[-1], 4)
        p0 = proj(um[-1 - win], vm[-1 - win], 4)
        heading = np.degrees(np.arctan2(cye - p0[1], cxe - p0[0]))
        imbox = OffsetImage(auv_rotated(sprite, heading), zoom=AUV_ZOOM * 0.9,
                            interpolation="bilinear")
        ax.add_artist(AnnotationBbox(imbox, (cxe, cye), frameon=False, pad=0.0,
                                     box_alignment=(0.5, 0.5), zorder=z + 5))
    except FileNotFoundError:
        pass
    side_label(4, "AUV trajectory", INK)

    # North arrow: the survey frame is rotated, so north is not up any more. Its direction
    # goes through the same shear as the sheets, or it would point somewhere the data does not.
    nu, nv = _rot2(theta) @ np.array([0.0, 1.0])
    dx, dy = 0.16 * (nu + SX * nv), 0.16 * SY * nv
    bx, by = content_r + 0.50, yT - 0.13
    ax.annotate("", xy=(bx + dx, by + dy), xytext=(bx, by), zorder=90,
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3, shrinkA=0, shrinkB=0))
    ax.text(bx + dx * 1.34, by + dy * 1.34, "N", color=MUTED, fontsize=11,
            fontweight="bold", ha="center", va="center", zorder=90)

    ax.set_xlim(xL, xR); ax.set_ylim(yB, yT)
    titles(fig, "Sensing the seabed")
    fig.savefig(os.path.join(out, "08_layer_stack.png"), dpi=130, facecolor=PAGE)
    plt.close(fig)


# ── figure: octagons versus the classic lawnmower ───────────────────────────────
def _self_crossings(xy, gap=6, grid=1.0):
    """Points where the polyline crosses itself (real revisits).

    Each self-crossing is a place the AUV comes back to: material for a loop closure that
    corrects INS drift. A single-pass lawnmower has none. Orientation (ccw) intersection
    test, vectorized over every pair of non-consecutive segments; crossings are clustered
    onto a `grid` m grid.
    """
    p, q = xy[:-1], xy[1:]
    i, j = np.triu_indices(len(p), k=gap)
    A, B, C, D = p[i], q[i], p[j], q[j]

    def ccw(a, b, c):
        return (c[:, 1] - a[:, 1]) * (b[:, 0] - a[:, 0]) > (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])

    hit = (ccw(A, C, D) != ccw(B, C, D)) & (ccw(A, B, C) != ccw(A, B, D))
    A, B, C, D = A[hit], B[hit], C[hit], D[hit]
    r, s = B - A, D - C
    den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    den[den == 0] = 1e-12
    t = ((C - A)[:, 0] * s[:, 1] - (C - A)[:, 1] * s[:, 0]) / den
    pts = A + t[:, None] * r
    if len(pts) == 0:
        return pts
    _, idx = np.unique(np.round(pts / grid).astype(int), axis=0, return_index=True)
    return pts[np.sort(idx)]


def _lawnmower_crosstrack(x_half, y_half, n_strips=11, n_cross=4, inset=0.92):
    """Classic boustrophedon PLUS perpendicular tie-lines (cross-tracks).

    Without cross-tracks a single-pass lawnmower never crosses itself and there is nothing
    to close. With them, each tie-line traverses the `n_strips` strips: n_strips*n_cross
    revisits. The strips are pulled inwards (`inset`) so the cross-tracks' link segments run
    outside them rather than resting on their ends (otherwise the self-crossing detector
    counts the shared vertices as crossings).
    """
    ys = np.linspace(-y_half * inset, y_half * inset, n_strips)
    pts = []
    for k, yy in enumerate(ys):
        a, b = (-x_half, x_half) if k % 2 == 0 else (x_half, -x_half)
        pts += [(a, yy), (b, yy)]

    bx, by = pts[-1]
    sy = 1.0 if by >= 0 else -1.0
    xs = np.linspace(-x_half * 0.72, x_half * 0.72, n_cross)
    if bx < 0:
        xs = xs[::-1]
    pts += [(bx, sy * y_half)]                            # leaves the area sideways
    for xx in xs:
        pts += [(xx, sy * y_half), (xx, -sy * y_half)]
        sy = -sy
    return np.array(pts, float)


def _corners(xy, thresh=85.0, step=None):
    """Vertices where the heading changes by >= `thresh` degrees.

    Each one is a manoeuvre the AUV comes out of with a fresh heading error. On the REAL
    track, resample by arc length (`step` metres) before differentiating the heading, or the
    INS noise spreads the turn over dozens of samples and it goes unseen.
    """
    if step:
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        u = np.arange(0.0, s[-1], step)
        xy = np.c_[np.interp(u, s, xy[:, 0]), np.interp(u, s, xy[:, 1])]
    d = np.diff(xy, axis=0)
    h = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    dh = np.abs((np.diff(h) + 180.0) % 360.0 - 180.0)
    return xy[1:-1][dh >= thresh]


def _principal_rotation(xy):
    """Rotation that brings the cloud's major axis to the horizontal."""
    _, v = np.linalg.eigh(np.cov(xy.T))
    t = -np.arctan2(v[1, -1], v[0, -1])
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _load_mosaic(path, long_side=1150, pct=0.3, rot180=False):
    """RGBA photomosaic -> float array, major axis horizontal, cropped to the data.

    The mosaics come in a UTM frame (the footprint lies diagonally) and weigh ~90 MB.
    Subsample first, de-rotate after: rotating 66 Mpx takes minutes, rotating 1 Mpx is
    instant. PIL.rotate(+ang), with `ang` the major-axis angle in image coordinates (y
    downwards), leaves that axis horizontal.

    `rot180` resolves the direction by hand. PCA yields the AXIS, not the direction: the
    sign of an eigenvector is arbitrary, so which end of the survey ends up on the left is
    a coin toss, and the PNG carries no georeferencing to settle it. The footprint is too
    symmetric to decide it automatically either (matching it against the multibeam coverage
    scores 0.536 vs 0.541 IoU for the two options).

    The crop is by percentile rather than minimum box: four stray patches at the mosaic's
    tips inflated the box and left the sheet half empty.
    """
    im = Image.open(path).convert("RGBA")
    f = max(1, int(round(max(im.size) / float(long_side))))
    if f > 1:
        im = im.resize((im.width // f, im.height // f), Image.LANCZOS)

    ys, xs = np.nonzero(np.array(im)[..., 3] > 8)
    _, v = np.linalg.eigh(np.cov(np.c_[xs - xs.mean(), ys - ys.mean()].T))
    ang = np.degrees(np.arctan2(v[1, -1], v[0, -1])) + (180.0 if rot180 else 0.0)
    im = im.rotate(ang, resample=Image.BICUBIC, expand=True)

    a = np.array(im).astype(np.float32) / 255.0
    ys, xs = np.nonzero(a[..., 3] > 0.03)
    x0, x1 = np.percentile(xs, [pct, 100 - pct]).astype(int)
    y0, y1 = np.percentile(ys, [pct, 100 - pct]).astype(int)
    a = a[y0:y1 + 1, x0:x1 + 1]
    a[..., :3] = np.clip(a[..., :3] * 1.14, 0, 1)         # the mosaic sits on a black page
    return a


def _chip(ax, x, y, text, color, z=60):
    """Data pill under the column label. The bbox hugs the text."""
    ax.text(x, y, text, color=color, fontsize=10.6, fontweight="bold", ha="center",
            va="center", zorder=z,
            bbox=dict(boxstyle="round,pad=0.44", fc="#17171a", ec=color, lw=1.0))


def _adv_card(fig, rect, accent, glyph, head, body, ar=1.5):
    """Advantage card: accent bar, drawn icon, title and body.

    `ar` = x-units per y-unit of the axes (this is not equal aspect): without it the icons
    come out vertically squashed.
    """
    ca = fig.add_axes(rect)
    ca.set_xlim(0, 1); ca.set_ylim(0, 1); ca.axis("off")
    ca.add_patch(FancyBboxPatch((0.015, 0.04), 0.97, 0.92, boxstyle="round,pad=0.012,rounding_size=0.05",
                                fc="#151513", ec="#2c2c2a", lw=1.0,
                                transform=ca.transAxes, clip_on=False))
    ca.add_patch(Rectangle((0.015, 0.04), 0.022, 0.92, fc=accent, ec="none", clip_on=False))
    gx, gy = 0.135, 0.73                                  # icon centre
    if glyph == "turn90":                                 # 90° corner = the turn that bleeds
        x0, y0 = gx - 0.075, gy - 0.055 * ar
        ca.plot([x0, x0], [y0, y0 + 0.10 * ar], color=accent, lw=2.6, solid_capstyle="round")
        ca.plot([x0, x0 + 0.145], [y0 + 0.10 * ar, y0 + 0.10 * ar], color=accent, lw=2.6,
                solid_capstyle="round")
        ca.add_patch(Arc((x0, y0 + 0.10 * ar), 0.085, 0.085 * ar, theta1=270, theta2=360,
                         color=accent, lw=1.2, alpha=0.8))
        ca.annotate("", xy=(x0 + 0.17, y0 + 0.10 * ar), xytext=(x0 + 0.145, y0 + 0.10 * ar),
                    arrowprops=dict(arrowstyle="-|>", color=accent, lw=2.6))
    elif glyph == "loop":                                 # circular arrow = loop closure
        ca.add_patch(Arc((gx, gy), 0.14, 0.14 * ar, angle=0, theta1=300, theta2=210,
                         color=accent, lw=2.4))
        aa = np.radians(300)
        ca.annotate("", xy=(gx + 0.075 * np.cos(aa) + 0.03, gy + 0.075 * ar * np.sin(aa) + 0.03),
                    xytext=(gx + 0.075 * np.cos(aa), gy + 0.075 * ar * np.sin(aa)),
                    arrowprops=dict(arrowstyle="-|>", color=accent, lw=2.4))
    elif glyph == "rose":                                 # 8 headings
        for k in range(8):
            a = np.radians(360 * k / 8)
            ca.annotate("", xy=(gx + 0.085 * np.cos(a), gy + 0.085 * ar * np.sin(a)),
                        xytext=(gx, gy), arrowprops=dict(arrowstyle="-|>", color=accent, lw=1.5))
    ca.text(0.29, 0.80, head, color=INK, fontsize=14.5, fontweight="bold", va="center")
    # matplotlib's `wrap=True` wraps to the FIGURE width, not the axes' -> the text spilled
    # into the neighbouring card. Wrapped by hand to a safe character count.
    ca.text(0.065, 0.58, textwrap.fill(body, 46), color=INK2, fontsize=10.4, va="top",
            ha="left", linespacing=1.34)


def fig_octagon_vs_lawnmower(res, out):
    """Two strategies in 3D (pattern on top, photomosaic below) over the same area.

    Each column is an oblique stack of two sheets, with the same shear as figure 08: on top
    what the AUV flies, below what the cameras return. The octagon track is REAL (Andratx);
    the lawnmower is synthetic over the same footprint, with cross-tracks so that it gets
    loop closures too — that way the comparison is not won cheaply and the real argument
    stands: the boustrophedon's 90° turns accumulate drift, the octagon's 45° ones do not.
    """
    mos_dir = os.path.join(out, "mosaics")
    need = [SLAM_TRAJ] + [os.path.join(mos_dir, f) for f in ("lawnmower.png", "octogon.png")]
    missing = [p for p in need if not os.path.isfile(p)]
    if missing:
        print(f"[media] 09 octagons: {missing} missing, skipping.")
        return

    # ── data ────────────────────────────────────────────────────────────────────
    xy = np.load(SLAM_TRAJ)[:, :2].astype(float)
    xy -= xy.mean(0)
    cross_o = _self_crossings(xy[::3])                    # real octagon revisits
    R = _principal_rotation(xy)                           # major axis -> horizontal (fills the sheet)
    oct_xy, cross_o = xy @ R.T, cross_o @ R.T

    xr, yr = np.abs(oct_xy[:, 0]).max(), np.abs(oct_xy[:, 1]).max()
    corners_o = _corners(oct_xy, thresh=25.0, step=3.0)   # ~86 vertices, none beyond 45°
    N_STRIP, N_CROSS = 9, 4
    lawn = _lawnmower_crosstrack(xr, yr, N_STRIP, N_CROSS)
    turns_l = _corners(lawn)                              # corners, all exactly 90°
    cross_l = _self_crossings(lawn, gap=2, grid=0.5)
    cross_l = cross_l[np.abs(cross_l[:, 1]) < yr * 0.99]  # only those inside the area

    mos = {"lawn": _load_mosaic(os.path.join(mos_dir, "lawnmower.png")),
           "oct": _load_mosaic(os.path.join(mos_dir, "octogon.png"))}

    # ── oblique projection geometry (same family as figure 08) ─────────────────
    SX, SY, THK = 0.38, 0.60, 0.045
    PW, MX, MY = 1.0, 0.035, 0.045
    foot = (-xr, xr, -yr, yr)
    boxes = [foot] + [(0, m.shape[1], 0, m.shape[0]) for m in mos.values()]
    min_aspect = min((b[1] - b[0]) / (b[3] - b[2]) for b in boxes)
    PH = (PW * (1 - 2 * MX) / min_aspect) / (1 - 2 * MY)
    GAP = SY * PH + 0.165                                 # spacing between sheets
    COLW = PW + SX * PH                                   # projected width of a sheet
    OX = (0.0, COLW + 0.30)                               # u origin of each column

    def iso(level, ox):
        return Affine2D().from_values(1, 0, SX, SY, ox, level * GAP)

    def proj(u, v, level, ox):
        return (ox + u + SX * v, SY * v + level * GAP)

    def fit_ext(bx):                                      # data -> extent that fills the sheet
        s = min(PW * (1 - 2 * MX) / (bx[1] - bx[0]), PH * (1 - 2 * MY) / (bx[3] - bx[2]))
        w, hh = (bx[1] - bx[0]) * s, (bx[3] - bx[2]) * s
        ox, oy = (PW - w) / 2, (PH - hh) / 2
        return [ox, ox + w, oy, oy + hh]

    def fit_xy(p, bx):
        s = min(PW * (1 - 2 * MX) / (bx[1] - bx[0]), PH * (1 - 2 * MY) / (bx[3] - bx[2]))
        ox, oy = (PW - (bx[1] - bx[0]) * s) / 2, (PH - (bx[3] - bx[2]) * s) / 2
        return np.c_[ox + (p[:, 0] - bx[0]) * s, oy + (p[:, 1] - bx[2]) * s]

    y_top = SY * PH + GAP
    xL, xR_ = -0.56, OX[1] + COLW + 0.04
    yB, yT = -THK - 0.05, y_top + 0.195

    # ── canvas: isometric scene on top, cards below ─────────────────────────────
    figW = 14.2
    scene_x, scene_w = 0.026, 0.955
    scene_h_in = (scene_w * figW) * (yT - yB) / (xR_ - xL)
    cards_in, title_in = 3.00, 1.20
    figH = title_in + scene_h_in + cards_in

    fig = plt.figure(figsize=(figW, figH), facecolor=PAGE)
    fig.text(0.030, 1 - 0.30 / figH, "Octagons vs. the lawnmower", color=INK, fontsize=26,
             fontweight="bold", va="top")
    fig.text(0.030, 1 - 0.88 / figH, "The survey pattern decides whether SLAM can undo the "
             "navigation drift — and what the photomosaic ends up looking like.",
             color=INK2, fontsize=13, va="top")

    ax = fig.add_axes([scene_x, cards_in / figH, scene_w, scene_h_in / figH])
    ax.set_facecolor(PAGE); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")
    ax.set_xlim(xL, xR_); ax.set_ylim(yB, yT)
    tr = ax.transData
    halo = [withStroke(linewidth=3.6, foreground="#0d0d0d")]

    def walls(level, ox, edge, edge_dark, z):
        BL, BR, TR = proj(0, 0, level, ox), proj(PW, 0, level, ox), proj(PW, PH, level, ox)
        ax.add_patch(Polygon([BL, BR, (BR[0], BR[1] - THK), (BL[0], BL[1] - THK)],
                             closed=True, fc=edge, ec="none", zorder=z))
        ax.add_patch(Polygon([BR, TR, (TR[0], TR[1] - THK), (BR[0], BR[1] - THK)],
                             closed=True, fc=edge_dark, ec="none", zorder=z))

    def outline(level, ox, edge, z, **kw):
        c = [proj(0, 0, level, ox), proj(PW, 0, level, ox),
             proj(PW, PH, level, ox), proj(0, PH, level, ox)]
        ax.add_patch(Polygon(c, closed=True, fill=False, ec=edge, zorder=z, **kw))

    # assembly posts: tie the photomosaic to the track that produced it
    for ox in OX:
        for (u, v) in [(0, 0), (PW, 0), (PW, PH), (0, PH)]:
            p0, p1 = proj(u, v, 0, ox), proj(u, v, 1, ox)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#38382f", lw=0.8,
                    ls=(0, (4, 3)), zorder=1)

    # ── level 0: the photomosaics (what the cameras return) ─────────────────────
    for ox, key in zip(OX, ("lawn", "oct")):
        img = mos[key]
        walls(0, ox, "#1b2422", "#101715", 2)
        ax.imshow(img, extent=fit_ext((0, img.shape[1], 0, img.shape[0])), origin="upper",
                  transform=iso(0, ox) + tr, interpolation="bilinear", zorder=4)
        outline(0, ox, "#3d4a47", 6, lw=1.0)

    # ── level 1: the two strategies (what the AUV flies) ────────────────────────
    def pattern(ox, path, color, cross, turns, header, sub, chips, turn_label, turn_col):
        walls(1, ox, "#161d2e", "#0b1120", 20)
        glass = Polygon([proj(0, 0, 1, ox), proj(PW, 0, 1, ox),
                         proj(PW, PH, 1, ox), proj(0, PH, 1, ox)],
                        closed=True, fc="#0e1522", ec="none", alpha=0.75, zorder=21)
        ax.add_patch(glass)
        t1 = iso(1, ox) + tr
        p = fit_xy(path, foot)
        drawn = [ax.plot(p[:, 0], p[:, 1], color="#0d0d0d", lw=3.0, transform=t1,
                         solid_capstyle="round", zorder=22)[0],
                 ax.plot(p[:, 0], p[:, 1], color=color, lw=1.6, transform=t1,
                         solid_capstyle="round", zorder=23)[0]]

        # the turns: large red square = 90° (injects drift), small dot = 45°
        hard = turn_label.startswith("90")
        q = fit_xy(turns, foot)
        drawn.append(ax.scatter(q[:, 0], q[:, 1], s=18 if hard else 10,
                                marker="s" if hard else "o", facecolor=turn_col,
                                edgecolor="#0d0d0d", linewidth=0.45 if hard else 0.3,
                                transform=t1, zorder=24))
        c = fit_xy(cross, foot)                           # self-crossings = loop closures
        drawn.append(ax.scatter(c[:, 0], c[:, 1], s=26, facecolor=CROSS_COL,
                                edgecolor="#0d0d0d", linewidth=0.6, transform=t1, zorder=25))
        for a in drawn:                                   # the final transit ran off the sheet
            a.set_clip_path(glass)
        outline(1, ox, "#3a4a6b", 26, lw=1.0)

        cx = ox + COLW / 2
        ax.text(cx, y_top + 0.148, header, color=INK, fontsize=16.5, fontweight="bold",
                ha="center", va="baseline", zorder=30)
        ax.text(cx, y_top + 0.106, sub, color=MUTED, fontsize=11.5, ha="center",
                va="baseline", zorder=30)
        for dx, (txt, col) in zip((-0.395, 0.0, 0.375), chips):
            _chip(ax, cx + dx, y_top + 0.050, txt, col)

        # Annotate ONE real vertex: the frontmost of the centred ones. Centred so the
        # pointer falls straight and short; frontal because there is no cluster of amber
        # crossings there and it does not overlap the sheet's edge.
        mid = turns[np.abs(turns[:, 0]) <= 0.55 * xr]
        px, py = proj(*fit_xy(mid[np.argmin(mid[:, 1])][None, :], foot)[0], 1, ox)
        ax.annotate(turn_label, xy=(px, py), xytext=(px, GAP - THK - 0.062),
                    color=turn_col, fontsize=12.5, fontweight="bold", ha="center",
                    va="center", path_effects=halo, zorder=31,
                    arrowprops=dict(arrowstyle="-", color=turn_col, lw=1.1,
                                    shrinkA=7, shrinkB=3, alpha=0.9))

    OCT_TURN = "#5fbf8f"                                  # green: the turn that does NOT bleed
    pattern(OX[0], lawn, LAWN_COL, cross_l, turns_l, "CLASSICAL LAWNMOWER",
            f"parallel strips + {N_CROSS} cross-tracks",
            [(f"{len(cross_l)} self-crossings", CROSS_COL), ("90° turns", TURN_COL),
             ("4 headings", LAWN_COL)], "90° turn", TURN_COL)
    pattern(OX[1], oct_xy, OCT_COL, cross_o, corners_o, "NESTED OCTAGONS",
            "Andratx · SPARUS II",
            [(f"{len(cross_o)} self-crossings", CROSS_COL), ("45° turns", OCT_TURN),
             ("8 headings", OCT_COL)], "45° turn", OCT_TURN)

    # ── left rail: what each sheet is (once, not per column) ────────────────────
    for lvl, (head, sub) in enumerate([("PHOTOMOSAIC", "what the cameras return"),
                                       ("SURVEY PATTERN", "what the vehicle flies")]):
        yy = lvl * GAP + SY * PH * 0.40
        ax.text(-0.09, yy + 0.028, head, color=INK, fontsize=12.5, fontweight="bold",
                ha="right", va="center", zorder=30)
        ax.text(-0.09, yy - 0.014, sub, color=MUTED, fontsize=10.2, ha="right",
                va="center", zorder=30)

    # ── cards ───────────────────────────────────────────────────────────────────
    cy, ch, cw = 0.66 / figH, 2.02 / figH, 0.300
    ar = (cw * figW) / (ch * figH)
    _adv_card(fig, [0.030, cy, cw, ch], TURN_COL, "turn90", "90° turns leak heading",
              f"Reversing onto the next strip costs two 90° corners — {len(turns_l)} of them here — "
              "and the INS leaves each one with a fresh heading error. The octagon never "
              "turns more than 45° (43° measured): half the yaw step, half the drift.", ar)
    _adv_card(fig, [0.350, cy, cw, ch], CROSS_COL, "loop", "Loop closures for free",
              f"Cross-tracks buy the lawnmower {len(cross_l)} revisits, pinned to a "
              f"{N_CROSS}×{N_STRIP} grid. The octagons self-cross {len(cross_o)} times across the "
              "whole footprint, so SLAM can tie the pose graph anywhere — not only where "
              "a tie-line happens to run.", ar)
    _adv_card(fig, [0.670, cy, cw, ch], "#7db3ef", "rose", "Seen from 8 headings",
              "Each patch is imaged from up to 8 directions (45° apart) instead of 4. "
              "Multi-view geometry → robust ICP, no growing lateral bias, and the "
              "photomosaic below closes without seams.", ar)

    fig.text(0.030, 0.415 / figH, "Adjacent parallel strips still look alike to ICP: the false "
             "closures they spawn collapsed our lawnmower map 49 → 15 m.",
             color="#6f6d67", fontsize=9.5, va="center", style="italic")
    fig.text(0.030, 0.185 / figH, CREDIT, color=MUTED, fontsize=9.5, va="center")
    fig.savefig(os.path.join(out, "09_octagon_vs_lawnmower.png"), dpi=130, facecolor=PAGE)
    plt.close(fig)


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PKG_ROOT, "results")
    out = os.path.join(res, "media")
    os.makedirs(out, exist_ok=True)

    cache = os.path.join(out, ".nav_cache.npz")
    if not os.path.isfile(cache):
        print(f"[media] {cache} missing: generate it from the bag trajectories.")
        return 1

    nav = dict(np.load(cache))

    for name, fn in [("01 bathymetry", fig_bathymetry), ("02 backscatter", fig_backscatter),
                     ("03 sidescan", fig_sss), ("06 two surveys", fig_two_surveys),
                     ("08 layer stack", fig_layer_stack)]:
        print(f"[media] {name} ...")
        fn(res, out, nav)

    print("[media] 09 octagons vs lawnmower ...")
    fig_octagon_vs_lawnmower(res, out)

    print("[media] 04 reveal.gif ...")
    gif_reveal(res, out, nav)
    print("[media] 05 lightsweep.gif ...")
    gif_lightsweep(res, out)

    print(f"[media] listo -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
