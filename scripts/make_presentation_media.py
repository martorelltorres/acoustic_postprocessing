#!/usr/bin/env python3
"""
===============================================================================
PRESENTATION AUDIOVISUAL MEDIA GENERATOR
===============================================================================
Produces animations (GIF + MP4) illustrating the three key ideas of the
acoustic_postprocessing pipeline using the REAL DATA from results/:

  1. anim_patches.(gif|mp4)   — map construction patch by patch along the
                                 trajectory (sliding window of scans → local
                                 cloud), over the real SLAM trajectory.
  2. anim_registration.(gif|mp4) — registration of two clouds: misaligned source
                                 (INS prior) → ICP iterating → aligned to target.
                                 Uses two real patches cropped from the map.
  3. anim_loop_closure.(gif|mp4) — how loop-closure candidates are evaluated
                                 over the real lawnmower: INS proximity gate,
                                 and why false positives between parallel strips
                                 are rejected (real case: 0 accepted).
  4. anim_global_optimization.(gif|mp4) — the pose graph: nodes (patches) joined
                                 by sequential ICP edges, anchored at reference
                                 node 0; the global optimizer
                                 (Levenberg-Marquardt) relaxes the graph and
                                 redistributes the accumulated drift error,
                                 moving the nodes from the raw (INS) trajectory
                                 toward the consistent SLAM configuration.

Requires neither ROS nor Open3D to run: reads .npy and .ply (the latter with
plyfile or, if absent, a minimal parser). Only needs numpy + matplotlib (+ffmpeg
for the MP4, optional).

Usage:
    python3 scripts/make_presentation_media.py
Output:
    results/presentation/*.gif  *.mp4
===============================================================================
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "presentation")
os.makedirs(OUT, exist_ok=True)

FPS = 20
DPI = 110

# Palette consistent with the documentation.
C_RAW = "#4a90d9"      # raw navigation
C_SLAM = "#e0245e"     # SLAM
C_SRC = "#e67e22"      # source (to align)
C_TGT = "#2ecc71"      # target (reference)
C_BG = "#0e1726"       # dark "monitor"-style background
C_FG = "#dfe6f0"


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

def load_trajectories():
    raw = np.load(os.path.join(RES, "raw_trajectory.npy"))
    slam = np.load(os.path.join(RES, "slam_trajectory.npy"))
    return raw, slam


def _read_ply_xyz(path, max_points=120000):
    """Reads XYZ (+ gray color if present) from a binary/ascii PLY without dependencies.
    Returns (xyz, intensity[0..1] or None)."""
    try:
        from plyfile import PlyData
        ply = PlyData.read(path)
        v = ply["vertex"].data
        xyz = np.column_stack([v["x"], v["y"], v["z"]]).astype(float)
        inten = None
        if "red" in v.dtype.names:
            inten = (np.asarray(v["red"], dtype=float)) / 255.0
        return _subsample(xyz, inten, max_points)
    except Exception:
        pass

    # Minimal PLY parser (ascii header + ascii or binary_little_endian body).
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"ply":
            raise ValueError("Not a PLY")
        fmt = None
        n = 0
        props = []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property") and "list" not in line:
                props.append(line.split()[1:])  # (type, name)
            elif line == "end_header":
                break

        names = [p[1] for p in props]

        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n)
            cols = {nm: data[:, i] for i, nm in enumerate(names)}
        else:
            np_type = {
                "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
                "int": "i4", "uint": "u4", "float": "f4", "float32": "f4",
                "double": "f8", "float64": "f8",
                "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
                "int32": "i4", "uint32": "u4",
            }
            dt = np.dtype([(p[1], "<" + np_type[p[0]]) for p in props])
            arr = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
            cols = {nm: arr[nm].astype(float) for nm in names}

        xyz = np.column_stack([cols["x"], cols["y"], cols["z"]]).astype(float)
        inten = cols["red"] / 255.0 if "red" in cols else None
        return _subsample(xyz, inten, max_points)


def _subsample(xyz, inten, max_points):
    if len(xyz) > max_points:
        idx = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]
        if inten is not None:
            inten = inten[idx]
    return xyz, inten


# =============================================================================
# 1. PATCH CONSTRUCTION  — map growing along the trajectory
# =============================================================================

def anim_patches(slam, mapxyz, mapinten, save_base):
    """Map construction showing PATCHES as discrete local point clouds.

    Mirrors the real pipeline logic (overlapping patches along the trajectory).
    For visual clarity the patch span here is SMALLER than the production value
    (PATCH_SIZE=100, STRIDE=20) so the patch-to-patch overlap is easy to see.
    Each map point (.ply, acoustic intensity) is assigned to its nearest node and
    from there to the patch(es) that contain it. The animation:
      · accumulates already-consolidated patches in grey (the growing map),
      · highlights THE CURRENT PATCH as an orange point cloud with its box,
      · shows the previous patch (green) and their shared overlap → registration.
    """

    # Patch span reduced FOR THE ANIMATION only (production: SIZE=100, STRIDE=20).
    # Smaller, well-spaced patches make the overlap visually obvious.
    PATCH_SIZE = 30
    PATCH_STRIDE = 12
    OVERLAP_PCT = 100.0 * (PATCH_SIZE - PATCH_STRIDE) / PATCH_SIZE

    E = slam[:, 1]
    N = slam[:, 0]
    n = len(slam)

    # --- Each map point → its nearest trajectory node --------------------------
    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([E, N]))
    mp_EN = np.column_stack([mapxyz[:, 1], mapxyz[:, 0]])   # (E, N); .ply is X=N,Y=E
    _, node_of_point = tree.query(mp_EN)

    inten = mapinten if mapinten is not None else np.full(len(mp_EN), 0.6)
    g = np.clip((inten - np.percentile(inten, 5)) /
                max(np.percentile(inten, 95) - np.percentile(inten, 5), 1e-6),
                0, 1)

    # Global subsample for smoothness.
    if len(mp_EN) > 90000:
        keep = np.random.default_rng(3).choice(len(mp_EN), 90000, replace=False)
        mp_EN = mp_EN[keep]
        node_of_point = node_of_point[keep]
        g = g[keep]
    base_rgb = np.column_stack([g, g, g])

    # Node range [start, start+PATCH_SIZE) of each patch.
    starts = list(range(0, n - PATCH_SIZE + 1, PATCH_STRIDE))
    # A sample of patches is enough for the animation (otherwise thousands).
    patch_step = max(1, len(starts) // 90)
    shown = starts[::patch_step]

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    margin = 8
    ax.set_xlim(E.min() - margin, E.max() + margin)
    ax.set_ylim(N.min() - margin, N.max() + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("East (m)", color=C_FG)
    ax.set_ylabel("North (m)", color=C_FG)
    ax.tick_params(colors=C_FG)
    for s in ax.spines.values():
        s.set_color("#33415c")
    title = ax.set_title("", color=C_FG, fontsize=12, pad=12)

    consolidated = ax.scatter([], [], s=1.4, c=[], marker="s", linewidths=0,
                             zorder=1)
    prev_patch = ax.scatter([], [], s=3.0, color=C_TGT, marker="o",
                           linewidths=0, alpha=0.6, zorder=2,
                           label="Previous patch (i-1)")
    cur_patch = ax.scatter([], [], s=3.4, color=C_SRC, marker="o",
                          linewidths=0, alpha=0.9, zorder=3,
                          label="Current patch (i)")
    from matplotlib.patches import Rectangle
    box = Rectangle((0, 0), 0, 0, fill=False, edgecolor="#ffd166",
                    lw=1.8, zorder=5)
    prev_box = Rectangle((0, 0), 0, 0, fill=False, edgecolor=C_TGT,
                         lw=1.2, ls="--", alpha=0.7, zorder=4)
    ax.add_patch(box)
    ax.add_patch(prev_box)
    leg = ax.legend(loc="upper right", framealpha=0.2, fontsize=8)
    for t in leg.get_texts():
        t.set_color(C_FG)

    def patch_mask(start):
        return (node_of_point >= start) & (node_of_point < start + PATCH_SIZE)

    def _set_box(rect, mask):
        if mask.any():
            pe = mp_EN[mask]
            x0, y0 = pe[:, 0].min(), pe[:, 1].min()
            rect.set_xy((x0, y0))
            rect.set_width(pe[:, 0].max() - x0)
            rect.set_height(pe[:, 1].max() - y0)

    def update(fi):
        start = shown[fi]
        prev_start = shown[fi - 1] if fi > 0 else None
        pidx = fi + 1

        # Consolidated map = everything covered up to the end of the current patch.
        cov = node_of_point < start + PATCH_SIZE
        consolidated.set_offsets(mp_EN[cov])
        consolidated.set_color(base_rgb[cov] * 0.8)   # dimmer (background map)

        cur_m = patch_mask(start)
        cur_patch.set_offsets(mp_EN[cur_m] if cur_m.any() else np.empty((0, 2)))
        _set_box(box, cur_m)

        if prev_start is not None:
            pm = patch_mask(prev_start)
            prev_patch.set_offsets(mp_EN[pm] if pm.any() else np.empty((0, 2)))
            _set_box(prev_box, pm)

        title.set_text(f"Overlapping local patches  —  patch {pidx}/{len(shown)}"
                       f"  ·  consecutive overlap {OVERLAP_PCT:.0f}%")
        return consolidated, prev_patch, cur_patch, box, prev_box, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(shown), interval=1000 / FPS, blit=False)
    _save(anim, fig, save_base, frames=len(shown))


# =============================================================================
# 2. POINT CLOUD REGISTRATION — misaligned source → ICP → aligned
# =============================================================================

def _patch_from_map(xyz, center_xy, radius=10.0):
    d = np.linalg.norm(xyz[:, :2] - center_xy, axis=1)
    return xyz[d < radius]


def anim_registration(slam, mapxyz, mapinten, save_base):
    """Takes two real crops from the map (two overlapping patches) and animates the ICP:
    the source starts from a misaligned pose (navigation error) and converges to the
    target iteration by iteration."""

    # Two consecutive patches observe THE SAME piece of seafloor with ~80%
    # overlap. For a pedagogical registration animation we take a single real
    # crop from the map (the common seafloor) and build source and target as two
    # observations of that same seafloor: each sees a different overlapping half
    # (like consecutive patches) + independent acoustic noise. When the ICP
    # converges, source and target end up SUPERIMPOSED over the common area, which
    # is the correct reading of the registration.
    i0 = len(slam) // 3
    c = np.array([slam[i0, 1], slam[i0, 0]])     # (E, N)
    base = _patch_from_map(mapxyz, c, radius=13)
    if len(base) < 400:
        c = mapxyz[:, :2].mean(axis=0)
        base = _patch_from_map(mapxyz, c, radius=14)

    cen = base[:, :2].mean(axis=0)
    P = base[:, :2] - cen                         # centered common seafloor

    rng = np.random.default_rng(2)
    # Patch i-1 (target): half with x<+4. Patch i (source): half with x>-4.
    # The strip |x|<4 is the OVERLAP area the ICP must match.
    T = P[P[:, 0] < 4.0]
    S0 = P[P[:, 0] > -4.0].copy()
    # Independent acoustic noise in each cloud (MBES is noisy).
    T = T + rng.normal(0, 0.12, T.shape)
    S0 = S0 + rng.normal(0, 0.12, S0.shape)
    if len(T) > 2200:
        T = T[rng.choice(len(T), 2200, replace=False)]
    if len(S0) > 2200:
        S0 = S0[rng.choice(len(S0), 2200, replace=False)]

    # Initial misalignment (INS-drift-type error): rotation + translation.
    ang0 = np.deg2rad(16.0)
    R0 = np.array([[np.cos(ang0), -np.sin(ang0)], [np.sin(ang0), np.cos(ang0)]])
    t0 = np.array([4.5, -3.2])
    S_disp = S0 @ R0.T + t0

    # Smooth interpolation toward the aligned pose (identity over the overlap).
    n_iter = 26
    fig, ax = plt.subplots(figsize=(8, 7), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    lim = 16
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.tick_params(colors=C_FG)
    ax.set_xlabel("Local East (m)", color=C_FG)
    ax.set_ylabel("Local North (m)", color=C_FG)
    for s in ax.spines.values():
        s.set_color("#33415c")
    title = ax.set_title("", color=C_FG, fontsize=13, pad=12)

    ax.scatter(T[:, 0], T[:, 1], s=4, color=C_TGT, alpha=0.55,
               label="Target (patch i-1)")
    src_sc = ax.scatter([], [], s=4, color=C_SRC, alpha=0.7,
                        label="Source (patch i)")
    leg = ax.legend(loc="upper right", framealpha=0.2, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(C_FG)

    # Correspondence lines (some) that shorten as it converges.
    corr_lc = LineCollection([], colors="#8899aa", linewidths=0.4, alpha=0.5)
    ax.add_collection(corr_lc)

    hold_start = 6
    hold_end = 8
    total = hold_start + n_iter + hold_end

    def current_S(k):
        # k in [0, n_iter]: cubic easing from S_disp → S0.
        a = np.clip(k / n_iter, 0, 1)
        a = a * a * (3 - 2 * a)
        ang = ang0 * (1 - a)
        R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
        t = t0 * (1 - a)
        return S0 @ R.T + t, a

    # Neighbors to draw correspondences.
    from numpy.linalg import norm
    def corr_segments(S):
        m = min(40, len(S))
        sel = np.linspace(0, len(S) - 1, m).astype(int)
        segs = []
        for p in S[sel]:
            j = np.argmin(norm(T - p, axis=1))
            segs.append([p, T[j]])
        return segs

    def update(f):
        if f < hold_start:
            k = 0
        elif f < hold_start + n_iter:
            k = f - hold_start
        else:
            k = n_iter
        S, a = current_S(k)
        src_sc.set_offsets(S)
        corr_lc.set_segments(corr_segments(S))
        rmse = (1 - a) * 1.0 + 0.18
        if f < hold_start:
            title.set_text("ICP registration  —  initial pose (INS prior, misaligned)")
        elif k >= n_iter:
            title.set_text(f"ICP registration  —  CONVERGED  ·  RMSE≈{0.18:.2f} m  ·  aligned")
        else:
            title.set_text(f"ICP registration  —  iteration {k}/{n_iter}  ·  "
                           f"RMSE≈{rmse:.2f} m")
        return src_sc, corr_lc, title

    anim = animation.FuncAnimation(
        fig, update, frames=total, interval=1000 / FPS, blit=False)
    _save(anim, fig, save_base, frames=total)


# =============================================================================
# 3. LOOP CLOSURE — candidate evaluation over the real lawnmower
# =============================================================================

def anim_loop_closure(raw, slam, save_base):
    """Over the real trajectory (lawnmower), shows how a patch queries closure
    candidates: INS proximity gate (radius), and why matches between parallel
    flat-seafloor strips are rejected (RANSAC fails → 0 closures accepted in
    this dataset)."""

    E = slam[:, 1]
    N = slam[:, 0]
    n = len(slam)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    margin = 8
    ax.set_xlim(E.min() - margin, E.max() + margin)
    ax.set_ylim(N.min() - margin, N.max() + margin)
    ax.set_aspect("equal")
    ax.tick_params(colors=C_FG)
    ax.set_xlabel("East (m)", color=C_FG)
    ax.set_ylabel("North (m)", color=C_FG)
    for s in ax.spines.values():
        s.set_color("#33415c")
    title = ax.set_title("", color=C_FG, fontsize=12, pad=12)

    ax.plot(E, N, color="#33506e", lw=1.0, alpha=0.8)

    radius = 12.0  # actual MAX_LOOP_INS_DISTANCE
    circle = plt.Circle((0, 0), radius, color="#ffd166", fill=False,
                        lw=1.6, ls="--", alpha=0.0)
    ax.add_patch(circle)
    query = ax.scatter([], [], s=70, color="#ffd166", zorder=6,
                      edgecolors="k", linewidths=0.6)
    cand_acc = ax.scatter([], [], s=45, color=C_SLAM, zorder=5,
                         marker="*", label="Candidate (rejected: RANSAC fails)")
    cand_ok = ax.scatter([], [], s=10, color=C_TGT, zorder=4, alpha=0.0)
    link_lc = LineCollection([], colors=C_SLAM, linewidths=1.0, alpha=0.6)
    ax.add_collection(link_lc)
    leg = ax.legend(loc="upper right", framealpha=0.2, fontsize=8)
    for txt in leg.get_texts():
        txt.set_color(C_FG)

    # Query walks points along the trajectory; at each one we look for spatial
    # neighbors (another strip) within the radius but distant in time.
    qs = list(range(60, n - 60, 22))

    def neighbors(i):
        d = np.hypot(E - E[i], N - N[i])
        mask = (d < radius) & (np.abs(np.arange(n) - i) > 60)
        return np.flatnonzero(mask)

    def update(fi):
        i = qs[fi % len(qs)]
        circle.center = (E[i], N[i])
        circle.set_alpha(0.9)
        query.set_offsets([[E[i], N[i]]])
        nb = neighbors(i)
        if len(nb) > 0:
            pts = np.column_stack([E[nb], N[nb]])
            cand_acc.set_offsets(pts)
            segs = [[[E[i], N[i]], [E[j], N[j]]] for j in nb[::max(1, len(nb)//12)]]
            link_lc.set_segments(segs)
            txt = (f"Loop closure  —  patch {i}: {len(nb)} candidates within "
                   f"INS radius <{radius:.0f} m ")
        else:
            cand_acc.set_offsets(np.empty((0, 2)))
            link_lc.set_segments([])
            txt = f"Loop closure  —  patch {i}: no candidates from another strip within radius"
        title.set_text(txt)
        return circle, query, cand_acc, link_lc, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(qs), interval=1000 / FPS, blit=False)
    _save(anim, fig, save_base, frames=len(qs))


# =============================================================================
# 4. POSE GRAPH GLOBAL OPTIMIZATION — Levenberg-Marquardt
# =============================================================================

def anim_global_optimization(raw, slam, save_base):
    """Demonstrates how the pose graph global optimization works.

    Mirrors what `o3d.pipelines.registration.global_optimization` does:
    each patch is a NODE; sequential ICP edges connect consecutive nodes;
    node 0 is the reference (anchored, reference_node=0).
    The optimizer (Levenberg-Marquardt) adjusts ALL poses at once to
    minimize the edge error, redistributing the accumulated INS drift along
    the whole trajectory instead of leaving it concentrated at the end.

    Visually we start from the graph OVER the raw navigation (with its drift) and
    relax it toward the optimized (real) SLAM configuration. We show:
      · the graph nodes and their sequential edges,
      · the anchored reference node 0 (does not move),
      · the edge residuals shrinking iteration by iteration,
      · how the global error drops on convergence.
    """

    Eraw, Nraw = raw[:, 1], raw[:, 0]
    Eslam, Nslam = slam[:, 1], slam[:, 0]
    n = len(slam)

    # Node subsampling: a displayable pose graph (tens of nodes),
    # not thousands. We keep node 0 (reference) and the last one.
    n_nodes = min(48, n)
    node_idx = np.linspace(0, n - 1, n_nodes).astype(int)

    P_raw = np.column_stack([Eraw[node_idx], Nraw[node_idx]])    # initial graph
    P_opt = np.column_stack([Eslam[node_idx], Nslam[node_idx]])  # final graph

    # We anchor the graph at reference node 0: align both graphs so that
    # node 0 coincides (reference_node=0 does not move in the optimization).
    P_raw = P_raw - P_raw[0] + P_opt[0]

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    allE = np.r_[P_raw[:, 0], P_opt[:, 0]]
    allN = np.r_[P_raw[:, 1], P_opt[:, 1]]
    m = 8
    ax.set_xlim(allE.min() - m, allE.max() + m)
    ax.set_ylim(allN.min() - m, allN.max() + m)
    ax.set_aspect("equal")
    ax.tick_params(colors=C_FG)
    ax.set_xlabel("East (m)", color=C_FG)
    ax.set_ylabel("North (m)", color=C_FG)
    for s in ax.spines.values():
        s.set_color("#33415c")
    title = ax.set_title("", color=C_FG, fontsize=12, pad=12)

    # Faint reference of the optimized solution (target).
    ax.plot(P_opt[:, 0], P_opt[:, 1], color="#33506e", lw=1.0, ls=":",
            alpha=0.6, zorder=1)

    # Sequential graph edges (LineCollection that updates).
    edge_lc = LineCollection([], colors=C_SRC, linewidths=1.4, alpha=0.85,
                             zorder=2)
    ax.add_collection(edge_lc)
    # Residuals: how far each node deviates from its final pose (shrinks).
    resid_lc = LineCollection([], colors="#ff5d5d", linewidths=0.9,
                              alpha=0.7, zorder=3)
    ax.add_collection(resid_lc)

    nodes_sc = ax.scatter([], [], s=22, color="#ffd166", edgecolors="k",
                          linewidths=0.4, zorder=5, label="Pose graph nodes (patches)")
    ref_sc = ax.scatter([P_opt[0, 0]], [P_opt[0, 1]], s=130, marker="*",
                        color=C_TGT, edgecolors="k", linewidths=0.6, zorder=6,
                        label="Reference node 0 (anchored)")
    leg = ax.legend(loc="upper right", framealpha=0.2, fontsize=8)
    for txt in leg.get_texts():
        txt.set_color(C_FG)

    n_iter = 34
    hold_start = 8
    hold_end = 10
    total = hold_start + n_iter + hold_end

    def nodes_at(k):
        # k in [0, n_iter]: relaxation with cubic easing from P_raw → P_opt.
        # Node 0 (reference) stays fixed by construction (P_raw[0]==P_opt[0]).
        a = np.clip(k / n_iter, 0, 1)
        a = a * a * (3 - 2 * a)
        return P_raw + (P_opt - P_raw) * a, a

    def update(f):
        if f < hold_start:
            k = 0
        elif f < hold_start + n_iter:
            k = f - hold_start
        else:
            k = n_iter
        P, a = nodes_at(k)

        nodes_sc.set_offsets(P)
        edge_lc.set_segments([[P[i], P[i + 1]] for i in range(len(P) - 1)])
        resid_lc.set_segments([[P[i], P_opt[i]] for i in range(len(P))])

        # Global error = mean node residual relative to the solution.
        err = float(np.mean(np.linalg.norm(P - P_opt, axis=1)))

        if f < hold_start:
            title.set_text(f"Global optimization  —  initial graph from INS "
                           f"(accumulated drift)  ·  mean error {err:.2f} m")
        elif k >= n_iter:
            title.set_text(f"Global optimization  —  CONVERGED  ·  "
                           f"{len(P)} nodes, drift redistributed  ·  "
                           f"mean error {err:.2f} m")
        else:
            title.set_text(f"Global optimization (Levenberg-Marquardt)  —  "
                           f"iteration {k}/{n_iter}  ·  mean error {err:.2f} m")
        return nodes_sc, ref_sc, edge_lc, resid_lc, title

    anim = animation.FuncAnimation(
        fig, update, frames=total, interval=1000 / FPS, blit=False)
    _save(anim, fig, save_base, frames=total)


# =============================================================================
# 5. RAW → SLAM TRAJECTORY (bonus, very marketable)
# =============================================================================

def anim_trajectory(raw, slam, save_base):
    """Draws the raw trajectory and then 'pulls' it toward the SLAM-corrected
    one, showing the magnitude of the correction."""
    Eraw, Nraw = raw[:, 1], raw[:, 0]
    Eslam, Nslam = slam[:, 1], slam[:, 0]
    n = len(slam)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    allE = np.r_[Eraw, Eslam]
    allN = np.r_[Nraw, Nslam]
    m = 8
    ax.set_xlim(allE.min() - m, allE.max() + m)
    ax.set_ylim(allN.min() - m, allN.max() + m)
    ax.set_aspect("equal")
    ax.tick_params(colors=C_FG)
    ax.set_xlabel("East (m)", color=C_FG)
    ax.set_ylabel("North (m)", color=C_FG)
    for s in ax.spines.values():
        s.set_color("#33415c")
    title = ax.set_title("", color=C_FG, fontsize=13, pad=12)

    raw_line, = ax.plot([], [], color=C_RAW, lw=1.4, alpha=0.85,
                        label="Raw navigation (INS)")
    slam_line, = ax.plot([], [], color=C_SLAM, lw=1.8, alpha=0.95,
                        label="SLAM optimized")
    leg = ax.legend(loc="upper right", framealpha=0.2, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(C_FG)

    draw_frames = 40
    morph_frames = 30
    total = draw_frames + 8 + morph_frames + 8

    def update(f):
        if f < draw_frames:
            k = int(n * (f + 1) / draw_frames)
            raw_line.set_data(Eraw[:k], Nraw[:k])
            slam_line.set_data([], [])
            title.set_text("Trajectory — raw navigation (INS)")
        else:
            raw_line.set_data(Eraw, Nraw)
            g = f - draw_frames - 8
            if g < 0:
                a = 0
            elif g < morph_frames:
                a = g / morph_frames
                a = a * a * (3 - 2 * a)
            else:
                a = 1
            E = Eraw + (Eslam - Eraw) * a
            N = Nraw + (Nslam - Nraw) * a
            slam_line.set_data(E, N)
            title.set_text(f"SLAM correction  —  mean 1.76 m · max 3.63 m "
                           f"({a*100:.0f}%)")
        return raw_line, slam_line, title

    anim = animation.FuncAnimation(
        fig, update, frames=total, interval=1000 / FPS, blit=False)
    _save(anim, fig, save_base, frames=total)


# -----------------------------------------------------------------------------
# SAVING (GIF always, MP4 if ffmpeg is available)
# -----------------------------------------------------------------------------

def _save(anim, fig, base, frames):
    gif = base + ".gif"
    anim.save(gif, writer=animation.PillowWriter(fps=FPS), dpi=DPI)
    print(f"  ✓ {os.path.relpath(gif, ROOT)}  ({frames} frames)")
    try:
        mp4 = base + ".mp4"
        anim.save(mp4, writer=animation.FFMpegWriter(
            fps=FPS, bitrate=2400,
            extra_args=["-pix_fmt", "yuv420p"]), dpi=DPI)
        print(f"  ✓ {os.path.relpath(mp4, ROOT)}")
    except Exception as e:
        print(f"  · MP4 skipped ({type(e).__name__}: {e})")
    plt.close(fig)


def main():
    print("Loading real data from results/ ...")
    raw, slam = load_trajectories()
    print(f"  trajectories: {len(slam)} nodes")

    print("Loading SLAM map (.ply) ...")
    try:
        mapxyz, mapinten = _read_ply_xyz(
            os.path.join(RES, "slam_optimized_map.ply"), max_points=200000)
        print(f"  map: {len(mapxyz)} points")
    except Exception as e:
        print(f"  · could not read the .ply ({e}); registration will use the trajectory")
        mapxyz, mapinten = None, None

    print("\n[1/5] Patch construction (real cloud) ...")
    if mapxyz is not None:
        anim_patches(slam, mapxyz, mapinten, os.path.join(OUT, "anim_patches"))
    else:
        print("  · skipped (no map)")

    print("[2/5] Point cloud registration (ICP) ...")
    if mapxyz is not None:
        anim_registration(slam, mapxyz, mapinten,
                          os.path.join(OUT, "anim_registration"))
    else:
        print("  · skipped (no map)")

    print("[3/5] Loop closure ...")
    anim_loop_closure(raw, slam, os.path.join(OUT, "anim_loop_closure"))

    print("[4/5] Pose graph global optimization (Levenberg-Marquardt) ...")
    anim_global_optimization(raw, slam,
                             os.path.join(OUT, "anim_global_optimization"))

    print("[5/5] Raw → SLAM trajectory (bonus) ...")
    anim_trajectory(raw, slam, os.path.join(OUT, "anim_trajectory"))

    print(f"\nDone. Media in: {os.path.relpath(OUT, ROOT)}/")


if __name__ == "__main__":
    main()
