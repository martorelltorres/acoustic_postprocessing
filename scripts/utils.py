#!/usr/bin/env python3

import os
import numpy as np
import tf.transformations as tr


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def wrap_angle_deg(angle):

    while angle > 180:
        angle -= 360

    while angle < -180:
        angle += 360

    return angle


def constrain_transform(
        T,
        max_yaw_deg=15.0,
        max_translation=5,
        T_ref=None):
    """
    Projects an ICP transform to a bounded 2D motion (yaw + XY,
    with Z=roll=pitch=0).

    Yaw clipping is applied as a DEVIATION from a reference rotation
    ``T_ref`` (normally T_init from the INS navigation), NOT in
    absolute value.

    Rationale: in the lawnmower turns the real relative rotation between
    consecutive patches can reach ~172°. Clipping yaw to ±15° ABSOLUTE prevented
    tracking those turns and produced a catastrophic accumulated rotational drift.
    With the reference, the real INS turn passes through intact and only the
    CORRECTION the ICP adds on top is limited (±max_yaw_deg).

    If ``T_ref is None`` the historical behavior is kept (absolute clip).
    """

    yaw = np.degrees(
        np.arctan2(
            T[1, 0],
            T[0, 0]
        )
    )

    if T_ref is None:

        # Historical behavior: absolute clip.
        yaw = np.clip(
            yaw,
            -max_yaw_deg,
            max_yaw_deg
        )

    else:

        # Clip relative to the reference rotation (INS).
        ref_yaw = np.degrees(
            np.arctan2(
                T_ref[1, 0],
                T_ref[0, 0]
            )
        )

        # ICP deviation from the reference, wrapped to [-180, 180].
        delta = wrap_angle_deg(yaw - ref_yaw)

        # Only the correction the ICP adds on top of the INS is limited.
        delta = np.clip(
            delta,
            -max_yaw_deg,
            max_yaw_deg
        )

        yaw = wrap_angle_deg(ref_yaw + delta)

    tx = np.clip(
        T[0, 3],
        -max_translation,
        max_translation
    )

    ty = np.clip(
        T[1, 3],
        -max_translation,
        max_translation
    )

    T_new = np.eye(4)

    T_new[:3, :3] = tr.euler_matrix(
        0,
        0,
        np.deg2rad(yaw)
    )[:3, :3]

    T_new[0, 3] = tx
    T_new[1, 3] = ty
    T_new[2, 3] = 0.0

    return T_new

def ins_rotation_icp_translation(
        T_icp,
        T_init,
        max_translation=5,
        min_length_ratio=None,
        max_length_ratio=None,
        anchor_scale=False,
        cross_track_gain=1.0):
    """
    Builds an SE(3) transform taking the FULL ROTATION from the
    INS navigation (``T_init``) and the TRANSLATION from the ICP result (``T_icp``),
    corrected in the travel plane.

    Rationale (rotation): on a flat, structureless seafloor, the rotation
    estimated by the ICP is random noise (std ~19°/step, mean ≈0). Integrated
    over hundreds of edges it produces a rotational random walk that drifts the
    trajectory. The INS DVL+IMU measures rotation reliably.

    Option A — full SE(3) edge (not 2D):
    The previous version rebuilt the edge as pure 2D (Z=0, roll=pitch=0,
    yaw = arctan2(T_init[1,0], T_init[0,0])). That is correct on straight, flat
    segments, but in the lawnmower TURNS the AUV has pitch: extracting yaw from
    the local frame mixes pitch+yaw and the XY projection does not preserve the
    step length → each turn compresses and rotates slightly wrong, and the error
    accumulates (drift growing with the trajectory, error-distance correlation
    0.84). Keeping the full 3D rotation of T_init and the Z component of the
    translation, the step geometry is preserved exactly in the turns.

    Translation: starts from the 3D translation of T_init (reliable INS step) and
    applies the ICP XY correction in the plane, with scale control:

    - ``anchor_scale=True``: the XY magnitude is fixed to that of the INS step and
      the ICP only provides direction (corrects the ICP compression bias).
    - Length gate (Fix A): if ``anchor_scale=False``, it is only rescaled to the
      INS when the ratio |T_icp_xy|/|T_init_xy| falls outside the band.

    Option A1 — cross-track correction damping (``cross_track_gain``):
    The ICP XY translation is decomposed into ALONG-TRACK (INS travel direction)
    and CROSS-TRACK (perpendicular). The ICP cross-track component
    introduces a systematic lateral bias (measured +23 mm/step in the turns,
    always toward +East) that shifts the lawnmower pattern and produces the
    "contracts to the left, overshoots to the right" effect. With a reliable INS
    in heading, that lateral correction only adds error. ``cross_track_gain`` scales
    ONLY the cross-track component:
      gain=1.0 → full ICP lateral correction (previous behavior)
      gain<1.0 → dampens the lateral bias
      gain=0.0 → ICP along-track + INS cross-track (no lateral bias)
    The along-track (travel scale) is kept intact.

    Output: SE(3) (4x4 matrix) with the INS 3D rotation.
    """

    icp_xy = np.array([
        T_icp[0, 3],
        T_icp[1, 3]
    ], dtype=float)

    ins_xy = np.array([
        T_init[0, 3],
        T_init[1, 3]
    ], dtype=float)

    ins_len = float(np.linalg.norm(ins_xy))
    icp_len = float(np.linalg.norm(icp_xy))

    if anchor_scale:

        # Real step magnitude from the INS, direction from the ICP.
        # Anchored to the 3D norm of T_init (preserved under the rigid
        # transform) to recover the real step length.
        ins_len_3d = float(np.linalg.norm(T_init[:3, 3]))

        if ins_len_3d > 1e-6 and icp_len > 1e-6:
            icp_xy = icp_xy * (ins_len_3d / icp_len)
        elif ins_len > 1e-6:
            # No reliable ICP direction: use the INS XY translation directly.
            icp_xy = ins_xy.copy()

    elif (
        min_length_ratio is not None
        and max_length_ratio is not None
    ):

        # Length gate (Fix A).
        if ins_len > 1e-6 and icp_len > 1e-6:

            ratio = icp_len / ins_len

            if ratio < min_length_ratio or ratio > max_length_ratio:

                # Keep the ICP direction, magnitude from the INS.
                icp_xy = icp_xy * (ins_len / icp_len)

    # Option A1 — dampen the ICP cross-track (lateral) component.
    # Decomposes the corrected translation into along-track (INS direction) and
    # cross-track (perpendicular) and rescales only the lateral one by cross_track_gain.
    if cross_track_gain != 1.0 and ins_len > 1e-6:

        ins_dir = ins_xy / ins_len
        cross_dir = np.array([-ins_dir[1], ins_dir[0]])

        along = float(np.dot(icp_xy, ins_dir))
        cross = float(np.dot(icp_xy, cross_dir))

        icp_xy = along * ins_dir + (cross_track_gain * cross) * cross_dir

    tx = np.clip(
        icp_xy[0],
        -max_translation,
        max_translation
    )

    ty = np.clip(
        icp_xy[1],
        -max_translation,
        max_translation
    )

    # Full 3D rotation of the INS (Option A): preserves real roll/pitch/yaw,
    # key to not compressing or skewing the turns.
    T_new = np.eye(4)
    T_new[:3, :3] = T_init[:3, :3]

    # Translation: ICP XY correction in the plane, Z from the INS step. Keeping
    # the Z of T_init preserves the 3D step length; the post-optimization vertical
    # restoration adjusts the absolute depth of each node.
    T_new[0, 3] = tx
    T_new[1, 3] = ty
    T_new[2, 3] = T_init[2, 3]

    return T_new


def project_to_3dof(T, vertical_ref=None):
    """
    Projects an SE(3) transform to the 3-DoF subspace (x, y, yaw), which is the
    correct regime of gravity-constrained bathymetric registration (Torroba 2020,
    Tan 2022: 3-DoF beats 6-DoF because INS roll/pitch/Z are reliable and the
    ICP rotation/Z on flat seafloor is noise).

    Unlike `constrain_transform_2d` (which sets Z = roll = pitch = 0), here the
    VERTICAL component (Z, roll, pitch) is taken from `vertical_ref` if provided — to
    keep the reliable part of the INS prior during coarse-to-fine, instead of
    discarding it. If `vertical_ref` is None, the vertical stays at 0 (equivalent to pure 2D).

    Parameters
    ----------
    T : (4,4) array        transform to project (ICP output).
    vertical_ref : (4,4) array or None
        transform from which to take Z + roll + pitch (typically T_init / INS prior).

    Returns
    --------
    (4,4) array with yaw + XY from T and Z + roll/pitch from vertical_ref (or 0).
    """

    # yaw of T (rotation in the plane).
    yaw = np.arctan2(T[1, 0], T[0, 0])

    if vertical_ref is not None:
        roll, pitch, _ = tr.euler_from_matrix(vertical_ref, axes='sxyz')
        z = float(vertical_ref[2, 3])
    else:
        roll = 0.0
        pitch = 0.0
        z = 0.0

    out = tr.euler_matrix(roll, pitch, yaw, axes='sxyz')
    out[0, 3] = T[0, 3]
    out[1, 3] = T[1, 3]
    out[2, 3] = z

    return out


def constrain_transform_2d(T):

    T2 = np.eye(4)

    # ==========================================
    # EXTRACT YAW
    # ==========================================

    yaw = np.arctan2(
        T[1, 0],
        T[0, 0]
    )

    c = np.cos(yaw)
    s = np.sin(yaw)

    T2[0, 0] = c
    T2[0, 1] = -s
    T2[1, 0] = s
    T2[1, 1] = c

    # ==========================================
    # XY ONLY
    # ==========================================

    T2[0, 3] = T[0, 3]
    T2[1, 3] = T[1, 3]

    # Z = 0
    # roll/pitch = 0

    return T2

def pose_dict_to_matrix(pose):

    T = np.eye(4)

    R = tr.euler_matrix(
        pose["roll"],
        pose["pitch"],
        pose["yaw"],
        axes='sxyz'
    )[:3, :3]

    T[:3, :3] = R

    T[0, 3] = pose["north"]
    T[1, 3] = pose["east"]
    T[2, 3] = -pose["depth"]

    return T


def expected_transform(patch_i, patch_j):

    T_i = pose_dict_to_matrix(
        patch_i.pose
    )

    T_j = pose_dict_to_matrix(
        patch_j.pose
    )

    # Correct relative transform in SE(3)
    T_rel = np.linalg.inv(T_i) @ T_j

    return T_rel
