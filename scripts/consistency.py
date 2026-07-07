#!/usr/bin/env python3
"""
Consistency-based error metric for bathymetric maps (Roman & Singh, ICRA 2006).

It is the primary metric used by state-of-the-art submap bathymetric SLAM
(Palomer 2016, Barkby 2009/2011, Torroba 2020, Tan 2022) to evaluate the quality of
a map WITHOUT ground truth. We introduce it here as a MEASUREMENT INSTRUMENT for the
R0 baseline (reproducing the Torroba 2020-style submap pose-graph) and everything after.

Idea (Roman & Singh 2006, "Consistency based error evaluation for deep sea bathymetric
mapping"):

  The area is discretized into an XY grid. Each cell receives points coming from
  different passes / swaths that observe the SAME seafloor. If the map is well
  registered, all those points share the same Z depth → small vertical dispersion.
  If there is drift / bad alignment, observations from different passes
  land at different Z → the cell "thickens" (ghosting). The mean vertical dispersion
  over the cells with overlap is the "consistency error".

  consistency_error = mean over cells with ≥2 observations of  std_Z(cell)

  (Roman uses the standard deviation of depth per cell; we also report
  RMS to compare with the paper tables, e.g. Torroba 2020 "Ripples"
  1.02 → 0.57 m, Palomer 2016 ~19% improvement, Tan 2022 1.89 m.)

Why it is the correct metric for OUR scenario (parallel passes without crossings):
the relevant overlap is NOT revisits/closures, but the ADJACENT overlap between
neighboring lawnmower strips. This metric measures it directly: a cell in the overlap
zone between two parallel strips will have low dispersion if and only if both strips
are well registered with each other. It is exactly the quantity the SLAM must minimize.

The module is Open3D-agnostic at its core (works on NxT arrays), so it is
unit-testable without ROS. `consistency_error_from_patches` is the adapter for the
pipeline (patches + Open3D pose_graph).
"""

import numpy as np


# =============================================================================
# CORE — metric over points labeled by source (swath / patch)
# =============================================================================

def consistency_error(
        points,
        labels,
        cell_size=1.0,
        min_observations=2,
        min_distinct_sources=2,
        return_grid=False):
    """
    Roman 2006 consistency error over a labeled point cloud.

    Parameters
    ----------
    points : (N, 3) array
        XYZ points of the merged map, in the global frame (already transformed by their
        patch pose). Z is the depth.
    labels : (N,) int array
        Source of each point (patch / swath index). Used to require that a cell's
        dispersion comes from distinct SOURCES and not from the internal density
        of a single swath (which does not inform about registration quality).
    cell_size : float
        XY cell side of the grid, in meters. Roman/Torroba use ~0.5 m;
        Barkby reports at the mapping grid resolution. Default 1.0 m.
    min_observations : int
        Minimum points in a cell to consider it. >=2 to be able to measure
        dispersion.
    min_distinct_sources : int
        Minimum DISTINCT patches/swaths that must contribute points to the cell to
        count it. >=2 ensures we measure overlap between passes, not intra-swath thickness.
        Setting 1 measures the total dispersion (includes real seafloor roughness) — useful
        only as a diagnostic.
    return_grid : bool
        If True, also returns the per-cell maps (to visualize the error as in
        Fig. 5 of Roman/Barkby/Torroba).

    Returns
    --------
    stats : dict
        mean_std_z      : mean of the Z standard deviation over valid cells
                          (== Roman's "consistency error").
        rms_std_z       : RMS of the Z standard deviation (comparable to SOTA tables).
        median_std_z    : median (robust to outlier cells).
        p90_std_z       : 90th percentile.
        n_cells_total   : no. of occupied cells.
        n_cells_valid   : no. of cells meeting the minimums (the ones that count).
        n_points        : no. of input points.
        cell_size       : echo of the parameter.
    grid : dict (only if return_grid=True)
        ij_to_std       : {(i,j): std_z} per valid cell.
        ij_to_center_xy : {(i,j): (x,y)} center of each valid cell.
        origin_xy       : (x0, y0) corner of the grid.
    """

    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if labels.shape[0] != points.shape[0]:
        raise ValueError("labels must have N elements")

    n_points = points.shape[0]

    empty = {
        "mean_std_z": float("nan"),
        "rms_std_z": float("nan"),
        "median_std_z": float("nan"),
        "p90_std_z": float("nan"),
        "n_cells_total": 0,
        "n_cells_valid": 0,
        "n_points": int(n_points),
        "cell_size": float(cell_size),
    }

    if n_points == 0:
        return (empty, {}) if return_grid else empty

    xy = points[:, :2]
    z = points[:, 2]

    x0 = float(xy[:, 0].min())
    y0 = float(xy[:, 1].min())

    # Cell index (i, j) per point.
    ii = np.floor((xy[:, 0] - x0) / cell_size).astype(np.int64)
    jj = np.floor((xy[:, 1] - y0) / cell_size).astype(np.int64)

    # Compact cell key to group with a single sort (O(N log N)).
    # We pack (i, j) into a unique int64 using the real range of j.
    j_span = int(jj.max() - jj.min() + 1)
    jj0 = jj - int(jj.min())
    ii0 = ii - int(ii.min())
    cell_key = ii0.astype(np.int64) * j_span + jj0.astype(np.int64)

    order = np.argsort(cell_key, kind="stable")
    cell_key_sorted = cell_key[order]
    z_sorted = z[order]
    lab_sorted = labels[order]
    ii_sorted = ii[order]
    jj_sorted = jj[order]

    # Group (cell) boundaries in the sorted array.
    boundaries = np.flatnonzero(
        np.diff(cell_key_sorted)
    ) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(cell_key_sorted)]))

    stds = []
    grid_std = {}
    grid_center = {}

    for s, e in zip(starts, ends):

        count = e - s
        if count < min_observations:
            continue

        # No. of distinct sources (patches) in the cell.
        if min_distinct_sources > 1:
            n_sources = np.unique(lab_sorted[s:e]).size
            if n_sources < min_distinct_sources:
                continue

        z_cell = z_sorted[s:e]
        # ddof=0 (population deviation) as in Roman; with few samples it is the
        # standard for this metric.
        std_z = float(z_cell.std())
        stds.append(std_z)

        if return_grid:
            i_idx = int(ii_sorted[s])
            j_idx = int(jj_sorted[s])
            grid_std[(i_idx, j_idx)] = std_z
            # Cell center in real global coordinates.
            grid_center[(i_idx, j_idx)] = (
                x0 + (i_idx + 0.5) * cell_size,
                y0 + (j_idx + 0.5) * cell_size,
            )

    stds = np.asarray(stds, dtype=float)

    # no. of occupied cells (>=1 point) for context.
    n_cells_total = int(len(starts))

    if stds.size == 0:
        result = dict(empty)
        result["n_cells_total"] = n_cells_total
        return (result, {}) if return_grid else result

    stats = {
        "mean_std_z": float(stds.mean()),
        "rms_std_z": float(np.sqrt(np.mean(stds ** 2))),
        "median_std_z": float(np.median(stds)),
        "p90_std_z": float(np.percentile(stds, 90)),
        "n_cells_total": n_cells_total,
        "n_cells_valid": int(stds.size),
        "n_points": int(n_points),
        "cell_size": float(cell_size),
    }

    if return_grid:
        grid = {
            "ij_to_std": grid_std,
            "ij_to_center_xy": grid_center,
            "origin_xy": (x0, y0),
        }
        return stats, grid

    return stats


# =============================================================================
# PIPELINE ADAPTER — from patches + Open3D pose_graph
# =============================================================================

def consistency_error_from_patches(
        patches,
        pose_graph,
        cell_size=1.0,
        max_points_per_patch=20000,
        min_distinct_sources=2,
        return_grid=False,
        rng_seed=0):
    """
    Computes the consistency error by transforming each patch by its graph pose and
    labeling its points with the patch index (Roman's "source"/swath).

    Subsamples each patch to `max_points_per_patch` to bound memory/time on
    large maps (the sampling is uniform and reproducible via rng_seed). This does not bias
    the metric: per-cell dispersion is estimated the same with a uniform subset.

    Returns the same stats dict as `consistency_error` (+ grid if requested).

    Typical use (R0 baseline): compute BEFORE (poses = raw navigation) and AFTER
    (poses = optimized pose_graph) to report the reduction, like the
    Roman/Torroba tables.
    """

    rng = np.random.default_rng(rng_seed)

    all_pts = []
    all_lab = []

    n = min(len(patches), len(pose_graph.nodes))

    for idx in range(n):

        patch = patches[idx]
        if patch is None or patch.pcd is None:
            continue

        pts = np.asarray(patch.pcd.points, dtype=float)
        if pts.shape[0] == 0:
            continue

        # Transform to global by the node's current pose.
        T = np.asarray(pose_graph.nodes[idx].pose, dtype=float)
        pts_h = pts @ T[:3, :3].T + T[:3, 3]

        # Uniform subsampling per patch.
        if pts_h.shape[0] > max_points_per_patch:
            sel = rng.choice(
                pts_h.shape[0],
                size=max_points_per_patch,
                replace=False
            )
            pts_h = pts_h[sel]

        all_pts.append(pts_h)
        all_lab.append(np.full(pts_h.shape[0], idx, dtype=np.int64))

    if not all_pts:
        return consistency_error(
            np.empty((0, 3)),
            np.empty((0,), dtype=np.int64),
            cell_size=cell_size,
            min_distinct_sources=min_distinct_sources,
            return_grid=return_grid,
        )

    points = np.concatenate(all_pts, axis=0)
    labels = np.concatenate(all_lab, axis=0)

    return consistency_error(
        points,
        labels,
        cell_size=cell_size,
        min_distinct_sources=min_distinct_sources,
        return_grid=return_grid,
    )
