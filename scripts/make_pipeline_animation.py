#!/usr/bin/env python3
"""
Pipeline story animation (matplotlib 3D), 4 stages with real mission data:

  1. AREA MATCH  : MB and SSS come from two different bagfiles. Their UTM
                   trajectories are shown overlapping in the same georeferenced
                   frame (this is the spatial match between both surveys).
  2. CLOUD       : the MB pings form a georeferenced 3D point cloud.
  3. MESH        : the cloud is reconstructed into a surface mesh.
  4. SSS TEXTURE : the SSS intensity (other bag) is projected onto the MB mesh.

Inputs  : results/anim_data/{stages.npz, footprints.npz}
Outputs : results/presentation/anim_pipeline.{mp4,gif}

Author: Antoni Martorell (SRV, UIB)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from make_media import CMAP_DEPTH

# Config (paths resolved relative to the package root, not the cwd)
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PKG_ROOT, "results", "anim_data")
OUT_DIR  = os.path.join(PKG_ROOT, "results", "presentation")
FPS      = 20
DPI      = 110

# Frame budget per stage (title hold + body). Duration = TOTAL / FPS.
# To lengthen the clip, ADD frames rather than lowering the fps: the orbit sweeps the same
# 120° of azimuth over the whole clip, so at 11-12 fps it would judder.
ST1 = 70    # area match
ST2 = 80    # point cloud
ST3 = 85    # mesh build
ST4 = 125   # sss projection (the payoff shot: longest hold)
TOTAL = ST1 + ST2 + ST3 + ST4          # 360 frames / 20 fps = 18.0 s


def load():
    s = np.load(os.path.join(DATA_DIR, "stages.npz"))
    f = np.load(os.path.join(DATA_DIR, "footprints.npz"))
    return s, f


def setup_axes(ax, V):
    # Common 3D bounds + equal aspect, looking down on the seafloor.
    xmin, xmax = V[:, 0].min(), V[:, 0].max()
    ymin, ymax = V[:, 1].min(), V[:, 1].max()
    zmin, zmax = V[:, 2].min(), V[:, 2].max()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    # zoom: without it the seafloor filled only a fifth of the canvas.
    ax.set_box_aspect((xmax - xmin, ymax - ymin, max(zmax - zmin, 1) * 3), zoom=1.28)
    ax.set_xlabel("Easting (m from survey origin)")
    ax.set_ylabel("Northing (m from survey origin)")
    ax.set_zlabel("Depth (m)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    s, f = load()
    V, T = s["V"].copy(), s["T"]
    inten, cloud = s["inten"], s["cloud"].copy()
    mb_traj, sss_traj = f["mb_traj"].copy(), f["sss_traj"].copy()

    # Everything in local metres. In UTM (446525, 4377214) matplotlib puts a "+4.465e5"
    # offset on the axis that gets clipped off the canvas and is unreadable.
    org = np.array([V[:, 0].min(), V[:, 1].min()])
    V[:, :2] -= org
    cloud[:, :2] -= org
    mb_traj -= org
    sss_traj -= org

    z0 = V[:, 2].mean()  # reference depth for the 2D-on-3D footprint stage

    # Precompute mesh triangle polygons + face colors once (stage 3-4).
    tris = V[T]                                   # (Ntri, 3, 3)
    face_z = tris[:, :, 2].mean(axis=1)
    # Stage-3 colormap: depth shading (terrain)
    zc = (face_z - face_z.min()) / (np.ptp(face_z) + 1e-9)
    # A single hue (magnitude), not cm.terrain: a rainbow invents boundaries the data
    # does not have. Same blue ramp as the rest of results/media.
    depth_colors = CMAP_DEPTH(zc)
    # Stage-4 colormap: SSS intensity per face (gray backscatter)
    face_i = inten[T].mean(axis=1)
    valid = face_i > 0
    vi = np.zeros_like(face_i)
    if valid.any():
        lo, hi = np.percentile(face_i[valid], (2, 98))
        # Stretch into a visible mid-bright range so backscatter texture pops
        vi = 0.2 + 0.8 * np.clip((face_i - lo) / (hi - lo + 1e-9), 0, 1)
    sss_colors = cm.gray(vi)
    sss_colors[~valid] = (0.16, 0.16, 0.15, 1.0)  # no SSS coverage: neutral gray,
                                                 # same as in 07_mb_sss_fusion.gif

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02)

    def title(txt, sub=""):
        ax.set_title(txt + ("\n" + sub if sub else ""), fontsize=12, fontweight="bold")

    def update(frame):
        ax.cla()
        setup_axes(ax, V)

        # Slow orbit across the whole clip
        azim = -60 + 120 * frame / TOTAL
        ax.view_init(elev=35, azim=azim)

        # -------- Stage 1: area match (two surveys, same UTM frame) --------
        if frame < ST1:
            p = frame / ST1
            # Top-down view so the spatial OVERLAP of both surveys is obvious
            ax.view_init(elev=90, azim=-90)
            ax.set_zticks([])
            ax.set_zlabel("")
            ax.zaxis.line.set_lw(0.0)
            ax.tick_params(axis="z", colors=(0, 0, 0, 0))
            title("1. Area match",
                  "Multibeam and sidescan from two different bagfiles, same UTM frame")
            zf = np.full(len(mb_traj), z0)
            n_mb = max(2, int(len(mb_traj) * min(1.0, p / 0.5)))
            ax.plot(mb_traj[:n_mb, 0], mb_traj[:n_mb, 1], zf[:n_mb],
                    color="tab:blue", lw=2.6, label="MB survey (bag A)")
            if p > 0.4:
                q = (p - 0.4) / 0.6
                n_ss = max(2, int(len(sss_traj) * min(1.0, q)))
                zfs = np.full(n_ss, z0)
                ax.plot(sss_traj[:n_ss, 0], sss_traj[:n_ss, 1], zfs,
                        color="tab:orange", lw=2.6, label="SSS survey (bag B)")
            ax.legend(loc="upper right", fontsize=9)

        # -------- Stage 2: MB georeferenced point cloud --------
        elif frame < ST1 + ST2:
            p = (frame - ST1) / ST2
            title("2. Georeferenced point cloud",
                  "Multibeam pings -> 3D points in UTM")
            n = max(100, int(len(cloud) * p))
            sub = cloud[:n]
            ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2],
                       c=sub[:, 2], cmap=CMAP_DEPTH, s=1.0, alpha=0.6)

        # -------- Stage 3: cloud -> mesh --------
        elif frame < ST1 + ST2 + ST3:
            p = (frame - ST1 - ST2) / ST3
            title("3. Surface reconstruction",
                  "Point cloud -> triangular mesh")
            # Faint cloud fading out, mesh growing in
            if p < 0.6:
                a = 0.5 * (1 - p / 0.6)
                ax.scatter(cloud[::3, 0], cloud[::3, 1], cloud[::3, 2],
                           color="0.5", s=0.6, alpha=max(a, 0.0))
            n_tri = max(50, int(len(tris) * p))
            coll = Poly3DCollection(tris[:n_tri], facecolors=depth_colors[:n_tri],
                                    edgecolors="none", linewidths=0)
            ax.add_collection3d(coll)

        # -------- Stage 4: SSS intensity projected onto the mesh --------
        else:
            p = (frame - ST1 - ST2 - ST3) / ST4
            title("4. SSS intensity projected on the MB mesh",
                  "Backscatter (bag B) textures the geometry (bag A)")
            # Blend depth coloring -> SSS coloring
            t = min(1.0, p / 0.7)
            blended = (1 - t) * depth_colors + t * sss_colors
            coll = Poly3DCollection(tris, facecolors=blended,
                                    edgecolors="none", linewidths=0)
            ax.add_collection3d(coll)

        return []

    anim = FuncAnimation(fig, update, frames=TOTAL, interval=1000 / FPS, blit=False)

    mp4 = os.path.join(OUT_DIR, "anim_pipeline.mp4")
    gif = os.path.join(OUT_DIR, "anim_pipeline.gif")

    try:
        anim.save(mp4, writer=FFMpegWriter(fps=FPS, bitrate=2400), dpi=DPI)
        print("Saved", mp4)
    except Exception as e:
        print("MP4 failed:", e)

    anim.save(gif, writer=PillowWriter(fps=FPS), dpi=int(DPI * 0.7))
    print("Saved", gif)


if __name__ == "__main__":
    main()
