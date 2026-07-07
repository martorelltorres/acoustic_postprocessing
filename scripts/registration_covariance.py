#!/usr/bin/env python3
"""
Registration covariance (pICP, R1) — closed-form estimate, Censi 2007 style.

Idea (Palomer 2016 "Probabilistic Registration"; Censi 2007 "An accurate closed-form
estimate of ICP's covariance"):

  A pose-graph edge's information must NOT be isotropic (np.eye*scale). On flat
  bottom a registration is well determined ALONG-TRACK (where geometry/relief gives
  gradient) but POORLY determined CROSS-TRACK (all normals point upward → cost does
  not change when sliding laterally). Isotropic information makes the optimizer trust
  a spurious cross-track correction EQUALLY to a reliable along-track one → injects
  the lateral bias that ruins the lawnmower.

  The correct registration covariance is, at the optimum, proportional to the inverse
  of the Hessian of the point-to-plane cost w.r.t. the pose. For the 3-DoF
  gravity-constrained regime (x, y, yaw) we use:

      cost(p) = Σ_i [ n_iᵀ (R(yaw)·s_i + t − q_i) ]²

  with s_i the source point, q_i its correspondence, n_i the target normal. The
  Jacobian of each residual r_i w.r.t. (x, y, yaw) is:

      J_i = [ n_ix , n_iy , n_iᵀ (∂R/∂yaw · s_i) ]

  and the 3-DoF covariance ≈ σ² · (Σ_i J_iᵀ J_i)⁻¹ = σ² · (JᵀJ)⁻¹, with σ² the
  residual variance (RMSE²). Where there is no cross-track gradient, the corresponding
  column of J is ~0 → JᵀJ nearly singular → large covariance → low information on that
  axis. Exactly what we want.

This module works on NumPy arrays (Open3D-agnostic, unit-testable). The adapter that
connects it to an Open3D `RegistrationResult` lives in the pipeline.

R1 = analytic covariance (this module). R3 = approximate/improve it with a network
(PointNetKL) that also incorporates the backscatter channel. The interface (return a
3-DoF covariance or its 6-DoF information) is kept so R3 is drop-in.
"""

import numpy as np


def _dR_dyaw(yaw):
    """Derivative of the 2D (in-plane) rotation matrix w.r.t. yaw."""
    c = np.cos(yaw)
    s = np.sin(yaw)
    # R = [[c,-s],[s,c]]  ->  dR/dyaw = [[-s,-c],[c,-s]]
    return np.array([[-s, -c],
                     [c, -s]], dtype=float)


def registration_covariance_3dof(
        source_pts,
        target_pts,
        target_normals,
        transformation,
        residual_std=None,
        reg_eps=1e-6,
        max_cov=1e3):
    """
    3-DoF (x, y, yaw) covariance of a point-to-plane registration, Censi 2007 style.

    Parameters
    ----------
    source_pts : (M, 3)  source points IN CORRESPONDENCE (already matched).
    target_pts : (M, 3)  their correspondences in the target.
    target_normals : (M, 3)  target normals at those correspondences.
    transformation : (4,4)  estimated transform (its yaw and XY translation are used).
    residual_std : float or None
        Standard deviation of the point-to-plane residual (≈ inlier_rmse). If None,
        estimated from the current residuals.
    reg_eps : float  regularization added to the JᵀJ diagonal (avoids singularity).
    max_cov : float  per-axis covariance clamp (numerical stability).

    Returns
    -------
    cov3 : (3,3)  covariance in (x, y, yaw). Large diagonal = poorly determined axis
                  (e.g. cross-track on flat bottom).
    info : dict   diagnostics: residual_std, condition_number, n_corr, approximate
                  along/cross-track variances.
    """

    source_pts = np.asarray(source_pts, dtype=float)
    target_pts = np.asarray(target_pts, dtype=float)
    target_normals = np.asarray(target_normals, dtype=float)
    T = np.asarray(transformation, dtype=float)

    M = source_pts.shape[0]

    if M < 6:
        # Too few correspondences for a reliable covariance.
        cov3 = np.eye(3) * max_cov
        return cov3, {
            "residual_std": float("nan"),
            "condition_number": float("inf"),
            "n_corr": int(M),
        }

    yaw = np.arctan2(T[1, 0], T[0, 0])
    R2 = np.array([[np.cos(yaw), -np.sin(yaw)],
                   [np.sin(yaw), np.cos(yaw)]], dtype=float)
    t2 = T[:2, 3]
    dR = _dR_dyaw(yaw)

    s_xy = source_pts[:, :2]
    q_xy = target_pts[:, :2]
    # Normals: use their XY component for the point-to-plane residual projected to
    # the plane (the 3-DoF regime acts in XY). The normal's Z component adds no
    # gradient to (x,y,yaw).
    n_xy = target_normals[:, :2]

    # Point-to-plane residual per correspondence: r_i = n_i · (R·s_i + t − q_i)
    s_rot = s_xy @ R2.T
    diff = s_rot + t2 - q_xy                      # (M,2)
    residuals = np.einsum("ij,ij->i", n_xy, diff)  # (M,)

    if residual_std is None:
        residual_std = float(np.std(residuals))
    residual_std = max(float(residual_std), 1e-3)

    # Jacobian of r_i w.r.t. (x, y, yaw):
    #   dr/dx = n_ix ; dr/dy = n_iy ; dr/dyaw = n_i · (dR · s_i)
    dyaw_term = np.einsum("ij,ij->i", n_xy, s_xy @ dR.T)  # (M,)
    J = np.column_stack([n_xy[:, 0], n_xy[:, 1], dyaw_term])  # (M,3)

    JtJ = J.T @ J
    # Regularization (Levenberg-style) to avoid singular inverse on pure plane.
    JtJ_reg = JtJ + reg_eps * np.eye(3)

    try:
        JtJ_inv = np.linalg.inv(JtJ_reg)
    except np.linalg.LinAlgError:
        cov3 = np.eye(3) * max_cov
        return cov3, {
            "residual_std": residual_std,
            "condition_number": float("inf"),
            "n_corr": int(M),
        }

    cov3 = (residual_std ** 2) * JtJ_inv

    # Clamp for stability (an undetermined axis can give huge variances).
    diag = np.clip(np.diag(cov3), 1e-9, max_cov)
    cov3 = cov3.copy()
    np.fill_diagonal(cov3, diag)

    # Condition number of JtJ: high = strong anisotropy (a poorly determined axis,
    # typical of flat bottom). A cheap diagnostic of the degenerate regime.
    try:
        eig = np.linalg.eigvalsh(JtJ)
        eig = np.clip(eig, 1e-12, None)
        cond = float(eig.max() / eig.min())
    except np.linalg.LinAlgError:
        cond = float("inf")

    diagnostics = {
        "residual_std": residual_std,
        "condition_number": cond,
        "n_corr": int(M),
        "var_x": float(cov3[0, 0]),
        "var_y": float(cov3[1, 1]),
        "var_yaw": float(cov3[2, 2]),
    }

    return cov3, diagnostics


def intensity_informativeness(intensities, n_bins=16):
    """
    Backscatter informativeness (R3) — cheap proxy for how much XY gradient the
    acoustic intensity texture contributes to a registration.

    On FLAT bottom geometry does not determine XY (all normals vertical), but if the
    backscatter has rich TEXTURE (sediment with micro-relief, ripples, bottom-type
    changes) Colored ICP does obtain lateral gradient. This function measures that
    richness with the normalized entropy of the intensity histogram:

        0.0 = uniform/flat intensity (no gradient contributed)
        1.0 = highly textured intensity (contributes XY gradient as if relief existed)

    It is the intensity analogue of `_geometric_texture` (which measures relief). The
    NOVEL contribution of R3 is using BOTH to modulate the registration covariance:
    neither Palomer 2016 nor Torroba/Tan use backscatter in their uncertainty model.

    intensities : (M,) in [0,1] (Open3D cloud gray channel = normalized backscatter).
    """

    inten = np.asarray(intensities, dtype=float).ravel()
    inten = inten[np.isfinite(inten)]
    if inten.size < 16:
        return 0.0

    # Normalized histogram entropy: measures how spread out the dynamic range is.
    hist, _ = np.histogram(inten, bins=n_bins, range=(0.0, 1.0))
    p = hist.astype(float)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    nz = p[p > 0]
    entropy = -np.sum(nz * np.log(nz))
    entropy_norm = float(np.clip(entropy / np.log(n_bins), 0.0, 1.0))

    # SPREAD factor: a nearly constant intensity (std≈0) contributes NO XY gradient
    # even if its histogram entropy is not exactly 0 due to noise. We saturate at
    # std≈0.15 (clear texture). This distinguishes "uniform with noise" from "real
    # texture".
    spread = float(np.clip(inten.std() / 0.15, 0.0, 1.0))

    # Informativeness requires BOTH: spread range (entropy) and real dispersion.
    # The product penalizes the uniform-with-noise case (mid entropy, spread≈0).
    return float(entropy_norm * spread)


def fuse_geometry_intensity_cov(
        cov3,
        intensity_info,
        diagnostics=None,
        cross_track_floor_factor=0.05,
        intensity_gain=1.0):
    """
    Modulate the geometric 3-DoF covariance with backscatter informativeness (R3).

    When geometry poorly determines an XY axis (large var) but the backscatter is
    informative (high intensity_info), we reduce that variance: Colored ICP supplies
    the gradient geometry lacks. The reduction is proportional to intensity
    informativeness, bounded by a floor so as not to fake total certainty.

        var_fused = var_geom · (1 − g · intensity_info) ,  with floor cross_track_floor_factor·var_geom

    where g = intensity_gain. If intensity_info=0 (flat intensity) the covariance does
    not change (stays the geometric R1). If intensity_info=1 (maximum texture) the
    variance drops to the floor. Only reduces; never raises confidence above the
    geometric one when the latter is already good.

    Returns the modulated cov3 (and updates `diagnostics` if provided).
    """

    cov3 = np.asarray(cov3, dtype=float).copy()
    info = float(np.clip(intensity_info, 0.0, 1.0))

    reduction = np.clip(intensity_gain * info, 0.0, 1.0 - cross_track_floor_factor)
    factor = 1.0 - reduction  # in [floor, 1]

    # Modulate only the XY+yaw diagonal (the DoF intensity can inform).
    for k in range(3):
        cov3[k, k] = cov3[k, k] * factor

    if diagnostics is not None:
        diagnostics["intensity_info"] = info
        diagnostics["intensity_cov_factor"] = float(factor)

    return cov3


def information_6dof_from_cov3(
        cov3,
        z_info=100.0,
        roll_pitch_info=100.0,
        info_floor=1e-3,
        info_ceil=1e4):
    """
    Build the 6-DoF information Open3D expects (PoseGraphEdge.information) from the
    3-DoF registration covariance.

    DoF order in Open3D PoseGraphEdge: information is 6x6 over the twist
    (rotation 3 + translation 3). Here we populate the blocks we control —
    yaw (rot-z), x, y — with the inverse of cov3, and fix Z/roll/pitch with HIGH
    information (z_info, roll_pitch_info), because those DoF come from the reliable
    INS (gravity-constrained 3-DoF constraint, R0.3) and must not be left free.

    Open3D index convention for an edge's information (matches the twist
    parametrization [omega(3), v(3)] = [rx, ry, rz, tx, ty, tz]):
        0: roll(rx) 1: pitch(ry) 2: yaw(rz) 3: x(tx) 4: y(ty) 5: z(tz)

    cov3 is in order (x, y, yaw); we map it to (tx, ty, rz).
    """

    cov3 = np.asarray(cov3, dtype=float)

    # 3-DoF information = inverse of the covariance (regularized).
    try:
        info3 = np.linalg.inv(cov3 + 1e-9 * np.eye(3))
    except np.linalg.LinAlgError:
        info3 = np.eye(3) * info_floor

    info = np.zeros((6, 6), dtype=float)

    # Reliable INS DoF: roll(0), pitch(1), z(5) with high information.
    info[0, 0] = roll_pitch_info
    info[1, 1] = roll_pitch_info
    info[5, 5] = z_info

    # Block (x, y, yaw) = (tx=3, ty=4, rz=2) from info3 (order x,y,yaw -> 3,4,2).
    idx = [3, 4, 2]
    for a in range(3):
        for b in range(3):
            info[idx[a], idx[b]] = info3[a, b]

    # Sanitize: diagonal within [floor, ceil] and symmetry.
    info = 0.5 * (info + info.T)
    d = np.clip(np.diag(info), info_floor, info_ceil)
    np.fill_diagonal(info, d)

    return info
