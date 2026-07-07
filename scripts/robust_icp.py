#!/usr/bin/env python3

import open3d as o3d
import numpy as np

from collections import OrderedDict


# Minimum correspondence count to consider an ICP result valid.
# If no scale exceeds this threshold the result is discarded.
_MIN_ICP_CORRESPONDENCES = 30


# -----------------------------------------------------------------------------
# PREPROCESSING CACHE (downsample + normals) PER CLOUD AND SCALE
# -----------------------------------------------------------------------------
# In sequential registration each patch.pcd takes part in two consecutive
# registrations (as source at idx and as target at idx+1) and the multiscale ICP
# repeats voxel_down_sample + estimate_normals on the SAME cloud at the same
# scales. Caching the result by (id(pcd), voxel) avoids ~half the preprocessing.
# The key is id(pcd): patches are persistent objects, so the id is stable for
# the whole run. The cache is bounded so it does not grow unboundedly (simple
# LRU by insertion).
# -----------------------------------------------------------------------------

_PREP_CACHE = OrderedDict()
_PREP_CACHE_MAX = 4096


def _preprocessed(pcd, voxel):
    """voxel_down_sample + estimate_normals with LRU cache by (id(pcd), voxel)."""

    key = (id(pcd), round(voxel, 4))

    cached = _PREP_CACHE.get(key)
    if cached is not None:
        # Refresh LRU position: the just-used entry becomes the most recent.
        _PREP_CACHE.move_to_end(key)
        return cached

    down = pcd.voxel_down_sample(voxel)

    if len(down.points) >= 1:
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel * 3.0,
                max_nn=30
            )
        )

    # Real LRU eviction: discard ONLY the least recently used entry, instead of
    # flushing the whole cache. In loop closure the same patch is preprocessed
    # many times; flushing the entire cache forced recomputing clouds that were
    # requested again immediately. The preprocessed result is identical.
    if len(_PREP_CACHE) >= _PREP_CACHE_MAX:
        _PREP_CACHE.popitem(last=False)

    _PREP_CACHE[key] = down
    return down


def clear_preprocess_cache():
    """Clear the preprocessing cache (call between runs if appropriate)."""
    _PREP_CACHE.clear()


def robust_icp(
        source,
        target,
        T_init,
        icp_distance=2.0,
        max_iter=60):

    # Voxel scales from largest to smallest (coarse-to-fine).
    # Start with the large voxel for global alignment
    # and refine at each step.
    voxel_scales = [1.0, 0.5, 0.25]

    # The current transform is updated only if that scale's result
    # has real correspondences (fitness > 0).
    # This way a failed step does not corrupt the next one.
    current_transform = T_init.copy()

    final_result = None

    loss = o3d.pipelines.registration.TukeyLoss(
        k=0.5
    )

    has_gicp = hasattr(
        o3d.pipelines.registration,
        "registration_generalized_icp"
    )

    for voxel_scale in voxel_scales:

        voxel = 0.5 * voxel_scale

        # Downsample + normals cached by (id(pcd), voxel): avoids reprocessing
        # the same cloud in consecutive registrations.
        s = _preprocessed(source, voxel)
        t = _preprocessed(target, voxel)

        if len(s.points) < 50:
            continue

        if len(t.points) < 50:
            continue

        if has_gicp:

            estimator = (
                o3d.pipelines.registration.
                TransformationEstimationForGeneralizedICP(loss)
            )

            result = (
                o3d.pipelines.registration.
                registration_generalized_icp(
                    s, t,
                    icp_distance,
                    current_transform,
                    estimator,
                    o3d.pipelines.registration.
                    ICPConvergenceCriteria(max_iteration=max_iter)
                )
            )

        else:

            estimator = (
                o3d.pipelines.registration.
                TransformationEstimationPointToPlane(loss)
            )

            result = (
                o3d.pipelines.registration.
                registration_icp(
                    s, t,
                    icp_distance,
                    current_transform,
                    estimator,
                    o3d.pipelines.registration.
                    ICPConvergenceCriteria(max_iteration=max_iter)
                )
            )

        # Only propagate the transform if this scale produced
        # real correspondences. If fitness=0 the returned transform
        # is invalid and must not be used as a seed.
        if len(result.correspondence_set) >= _MIN_ICP_CORRESPONDENCES:
            current_transform = result.transformation
            final_result = result

    # If the best accumulated result has fitness=0 (no scale found
    # enough correspondences) return None so the pipeline uses the
    # navigation fallback with low info.
    if final_result is None:
        return None

    if len(final_result.correspondence_set) < _MIN_ICP_CORRESPONDENCES:
        return None

    return final_result


# -----------------------------------------------------------------------------
# GEOMETRIC TEXTURE DETECTION
# -----------------------------------------------------------------------------
# Measures whether a patch pair has enough geometric relief for geometric ICP
# to constrain the XY translation. On flat bottom the normals point almost all
# upward (|nz|≈1) → geometry does not discriminate XY and ICP slides. The
# fraction of NON-vertical normals is a cheap proxy for "available geometric
# texture".
# -----------------------------------------------------------------------------

def _geometric_texture(pcd, voxel=0.5, slope_thresh=0.85):
    """
    Fraction of points whose normal is NOT nearly vertical (|nz| < slope_thresh),
    i.e. belonging to slopes/structure usable by geometric ICP. ~0 = flat bottom
    (geometry useless for XY), >0.1 = relief present.
    """

    down = _preprocessed(pcd, voxel)

    if not down.has_normals() or len(down.points) < 30:
        return 0.0

    n = np.asarray(down.normals)
    if len(n) == 0:
        return 0.0

    non_vertical = np.abs(n[:, 2]) < slope_thresh
    return float(np.mean(non_vertical))


def robust_hybrid_icp(
        source,
        target,
        T_init,
        icp_distance=2.0,
        max_iter=60,
        lambda_geometric=0.6,
        min_geometric_texture=0.08):
    """
    ADAPTIVE geometry/intensity registration.

    Strategy: use GEOMETRIC ICP by default (it works best on this dataset and
    does not introduce the noise of the color term), and fall back to Colored
    ICP (acoustic intensity) ONLY when the patch pair lacks enough geometric
    texture — the case intensity exists for: flat featureless bottom where
    geometric ICP slides.

    Per-pair decision:
      texture = fraction of non-vertical normals (usable relief).
      If texture >= min_geometric_texture  → geometric ICP (robust_icp).
      If texture <  min_geometric_texture and color present → Colored ICP.

    This way:
      - In structured areas (ships): pure geometry, precise and without the
        compression bias the color term introduced.
      - On smooth featureless bottom: intensity supplies the gradient geometry
        lacks, avoiding sliding.

    min_geometric_texture: threshold on fraction of sloped normals.
      0.08 = at least 8% of points on slopes to trust geometry.
    """

    # Texture available in both patches (use the smaller: registration is
    # limited by the patch with least relief).
    tex_s = _geometric_texture(source)
    tex_t = _geometric_texture(target)
    texture = min(tex_s, tex_t)

    has_color = source.has_colors() and target.has_colors()

    if texture >= min_geometric_texture or not has_color:
        # Usable relief present (or no intensity): pure geometry.
        return robust_icp(
            source,
            target,
            T_init,
            icp_distance=icp_distance,
            max_iter=max_iter
        )

    # Flat bottom without geometric texture: rely on intensity.
    result = robust_colored_icp(
        source,
        target,
        T_init,
        icp_distance=icp_distance,
        max_iter=max_iter,
        lambda_geometric=lambda_geometric
    )

    if result is not None:
        return result

    # If Colored ICP fails, geometric last resort.
    return robust_icp(
        source,
        target,
        T_init,
        icp_distance=icp_distance,
        max_iter=max_iter
    )


def robust_colored_icp(
        source,
        target,
        T_init,
        icp_distance=2.0,
        max_iter=60,
        lambda_geometric=0.6):
    """
    Coarse-to-fine ICP combining geometry and acoustic INTENSITY
    (Open3D Colored ICP, Park et al. 2017).

    Intensity (backscatter) travels in the cloud color channel (normalized
    gray, loaded in PatchBuilder). The optimized cost is:

        E = (1 - λ) · E_color + λ · E_geometric

    On flat bottom E_geometric has gradient ≈0 in XY (geometric ICP slides),
    but E_color DOES have gradient because intensity texture varies spatially
    → alignment stops sliding. In areas with real relief (ships) geometry still
    dominates via λ.

    lambda_geometric ∈ [0,1]: 1.0 = geometry only (≡ normal ICP),
    smaller values give more weight to intensity. 0.6 is a reasonable starting
    point for flat bottom with concentrated structure.

    Requires source and target to have `colors`. If they don't, delegates to
    robust_icp (pure geometric).
    """

    if (
        not source.has_colors()
        or not target.has_colors()
    ):
        return robust_icp(
            source,
            target,
            T_init,
            icp_distance=icp_distance,
            max_iter=max_iter
        )

    voxel_scales = [1.0, 0.5, 0.25]

    current_transform = T_init.copy()
    final_result = None

    estimator = (
        o3d.pipelines.registration.
        TransformationEstimationForColoredICP(
            lambda_geometric=lambda_geometric
        )
    )

    for voxel_scale in voxel_scales:

        voxel = 0.5 * voxel_scale

        # Cached by (id(pcd), voxel). voxel_down_sample preserves the colors
        # (intensity), which Colored ICP needs, and _preprocessed adds the
        # normals without touching the color.
        s = _preprocessed(source, voxel)
        t = _preprocessed(target, voxel)

        if len(s.points) < 50:
            continue

        if len(t.points) < 50:
            continue

        try:

            result = (
                o3d.pipelines.registration.
                registration_colored_icp(
                    s, t,
                    icp_distance,
                    current_transform,
                    estimator,
                    o3d.pipelines.registration.
                    ICPConvergenceCriteria(max_iteration=max_iter)
                )
            )

        except RuntimeError:

            # Colored ICP may raise if a scale lacks enough color gradient;
            # keep the previous transform.
            continue

        if len(result.correspondence_set) >= _MIN_ICP_CORRESPONDENCES:
            current_transform = result.transformation
            final_result = result

    if final_result is None:
        return None

    if len(final_result.correspondence_set) < _MIN_ICP_CORRESPONDENCES:
        return None

    return final_result