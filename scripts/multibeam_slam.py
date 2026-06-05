#!/usr/bin/env python3

"""
===============================================================================
ADVANCED UNDERWATER MULTIBEAM SLAM
FULL METRICS + CONNECTED POSE GRAPH + RAW/SLAM MAP EXPORT
===============================================================================
"""

import os
import copy
import json
import csv

import rospy
import rosbag
import ros_numpy
import numpy as np
import open3d as o3d
import tf.transformations as tr

from tqdm import tqdm
from scipy.interpolate import interp1d

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

from utils import *
from registration import *
from robust_icp import *
from robust_ndt import *
from scan_context import *
from information_matrix import *
from visualization import *

# =============================================================================
# CONFIGURATION
# =============================================================================

PATCH_SIZE = 100
PATCH_STRIDE = 20

VOXEL_SIZE = 0.25

FINAL_DOWNSAMPLE = 0.2

ANGLE_CUTOFF_DEG = 55.0

MIN_PATCH_POINTS = 500

MAX_PATCH_EXTENT = 500.0

# =============================================================================
# ICP
# =============================================================================

ICP_DISTANCE = 2.0

ICP_MAX_ITER = 60

REGISTRATION_ALGORITHM = "icp"

NDT_RESOLUTION = 1.0

NDT_MAX_POINTS = 3000

NDT_MIN_POINTS_PER_VOXEL = 5

# -----------------------------------------------------------------------------
# FITNESS
# -----------------------------------------------------------------------------

FITNESS_THRESHOLD = 0.55

LOOP_FITNESS_THRESHOLD = 0.90

# -----------------------------------------------------------------------------
# RMSE THRESHOLDS
# -----------------------------------------------------------------------------
# MBES acoustic clouds are naturally noisy.
# RMSE values close to voxel size are completely normal.
#
# Sequential ICP:
#   More permissive because overlap is very high.
#
# Loop Closures:
#   Even more permissive due to accumulated drift.
# -----------------------------------------------------------------------------

SEQ_RMSE_THRESHOLD = VOXEL_SIZE * 1.5

LOOP_RMSE_THRESHOLD = VOXEL_SIZE * 1.5

# -----------------------------------------------------------------------------
# CORRESPONDENCES
# -----------------------------------------------------------------------------

MIN_CORRESPONDENCES = 100

# -----------------------------------------------------------------------------
# ICP–NAVIGATION CONSISTENCY GATE  (sequential registration)
# -----------------------------------------------------------------------------
# On a flat, featureless underwater seabed the GICP cost function is flat in
# XY: surface normals all point vertically, so the point-to-plane residual has
# near-zero gradient for in-plane translation.  The coarse-to-fine ICP can
# leave T_init and converge to a completely wrong local minimum — including
# reversed translations (~180° flip) or perpendicular ones.
#
# These failures pass the RMSE/fitness gates because a flat surface aligns
# well regardless of XY position.  They must be caught by comparing the ICP
# result with the navigation expectation (T_init):
#
#   MAX_SEQ_ICP_TRANSLATION_DEV  – maximum allowed difference between the ICP
#       translation and T_init translation in the local patch frame (metres).
#       Catches large-magnitude deviations.
#
#   MAX_SEQ_ICP_YAW_DEV  – maximum allowed angular deviation between the ICP
#       rotation and the T_init rotation (degrees).
#       Catches direction reversals (170°) and perpendicular divergences (90°).
#
# When either gate fires the step falls back to T_init (navigation) with low
# information weight — the same strategy used for RMSE-rejected steps.
# -----------------------------------------------------------------------------

MAX_SEQ_ICP_TRANSLATION_DEV = 0.5   # m
MAX_SEQ_ICP_YAW_DEV         = 15.0  # degrees

# -----------------------------------------------------------------------------
# HIGH-CONFIDENCE FALLBACK
# -----------------------------------------------------------------------------
# If fitness is extremely high and correspondences are massive,
# allow higher RMSE before rejecting.
# This is extremely important in flat underwater environments.
# -----------------------------------------------------------------------------

HIGH_FITNESS_THRESHOLD = 0.90

HIGH_CORRESPONDENCE_THRESHOLD = 1000

# Debe ser > 1 para tener efecto. Con 1 el bloque de alta confianza
# no relaja nada (se multiplica por 1). Valor 1.5 permite hasta
# VOXEL_SIZE * 3.0 m de RMSE en zonas de alta cobertura.
HIGH_RMSE_MULTIPLIER = 1.5

# =============================================================================
# LOOP CLOSURE
# =============================================================================

ENABLE_LOOP_CLOSURE = True

SCAN_CONTEXT_THRESHOLD = 0.15

MAX_LOOP_CANDIDATES = 3

MAX_LOOP_Z_TRANSLATION = 0.5

MAX_LOOP_XY_TRANSLATION = 8.0

MAX_LOOP_YAW_DEG = 20.0

# -----------------------------------------------------------------------------
# INS PROXIMITY GATE
# -----------------------------------------------------------------------------
# Only consider loop closure candidates whose INS positions are closer than
# this distance. Prevents false positives between far-away parts of the
# trajectory that merely look similar (e.g. parallel lawnmower strips on a
# flat, featureless seabed).
#
# Rule of thumb: set to 2-3 × the maximum expected INS drift over the mission.
# For a typical AUV DVL+IMU over 300-500 m without GPS: 8-15 m.
# -----------------------------------------------------------------------------

MAX_LOOP_INS_DISTANCE = 12.0

# -----------------------------------------------------------------------------
# ICP–INS DRIFT CONSISTENCY GATE
# -----------------------------------------------------------------------------
# For a TRUE loop closure the ICP transform corrects the accumulated INS drift,
# so the XY translation of the ICP result must be a reasonable fraction of the
# INS distance between the two nodes.
#
#   ratio = ICP_xy_translation / INS_distance
#
# True loop  → ratio ≈ 1.0 (ICP corrects exactly the drift)
# False pos. → ratio ≈ 0   (ICP finds a flat–flat coincidental alignment
#                            near identity, independent of actual drift)
#
# Threshold of 0.40 rejects loops where the ICP correction is less than 40 %
# of the INS drift — a clear sign of a non-discriminative flat-surface match.
# Only applied when INS distance > MIN_INS_DIST_FOR_RATIO_CHECK to avoid
# division instability on very close candidates.
# -----------------------------------------------------------------------------

MIN_LOOP_ICP_INS_RATIO   = 0.40
MIN_INS_DIST_FOR_RATIO_CHECK = 2.0

# =============================================================================
# MONITOR
# =============================================================================

# Frecuencia de refresco del visualizador del pose graph durante el
# registro secuencial. Cada MONITOR_UPDATE_EVERY iteraciones se
# actualiza la ventana Open3D.
#
#   1  → refresco en cada paso   (máxima frescura, más carga de CPU/GPU)
#   5  → balance recomendado     (valor por defecto)
#   10 → refresco ligero         (misiones largas, hardware limitado)
#
# El monitor SIEMPRE se actualiza al finalizar el bucle secuencial,
# en los loop closures aceptados y tras la optimización global,
# independientemente de este valor.

MONITOR_UPDATE_EVERY = 5


# =============================================================================
# PATCH
# =============================================================================

class Patch:

    def __init__(self):

        self.idx = None

        self.pcd = None

        self.center = None

        self.pose = None

        self.timestamp = None


# =============================================================================
# TF
# =============================================================================

def get_static_transform_from_tf(
        bag_file,
        parent_frame,
        child_frame):

    bag = rosbag.Bag(bag_file)

    for _, msg, _ in bag.read_messages(
            topics=['/tf_static', '/tf']):

        for transform in msg.transforms:

            if (
                transform.header.frame_id == parent_frame
                and
                transform.child_frame_id == child_frame
            ):

                q = transform.transform.rotation

                t = transform.transform.translation

                T = tr.quaternion_matrix(
                    [q.x, q.y, q.z, q.w]
                )

                T[:3, 3] = [
                    t.x,
                    t.y,
                    t.z
                ]

                bag.close()

                return T

    bag.close()

    return np.eye(4)


# =============================================================================
# NAVIGATION
# =============================================================================

class NavigationInterpolator:

    def __init__(
            self,
            bag,
            nav_topic):

        self.timestamps = []

        north = []
        east = []
        depth = []

        yaw = []
        pitch = []
        roll = []

        rospy.loginfo(
            "Loading navigation..."
        )

        for _, msg, _ in tqdm(
                bag.read_messages(
                    topics=[nav_topic]
                ),
                desc="Navigation"):

            ts = msg.header.stamp.to_sec()

            self.timestamps.append(ts)

            north.append(
                msg.position.north
            )

            east.append(
                msg.position.east
            )

            depth.append(
                msg.position.depth
            )

            yaw.append(
                msg.orientation.yaw
            )

            pitch.append(
                msg.orientation.pitch
            )

            roll.append(
                msg.orientation.roll
            )

        self.timestamps = np.array(
            self.timestamps
        )

        if len(self.timestamps) == 0:
            raise RuntimeError(
                f"No navigation messages found in topic {nav_topic}"
            )

        self.min_timestamp = float(
            np.min(self.timestamps)
        )

        self.max_timestamp = float(
            np.max(self.timestamps)
        )

        self.f_n = interp1d(
            self.timestamps,
            north,
            fill_value='extrapolate'
        )

        self.f_e = interp1d(
            self.timestamps,
            east,
            fill_value='extrapolate'
        )

        self.f_d = interp1d(
            self.timestamps,
            depth,
            fill_value='extrapolate'
        )

        self.f_y = interp1d(
            self.timestamps,
            np.unwrap(yaw),
            fill_value='extrapolate'
        )

        self.f_p = interp1d(
            self.timestamps,
            np.unwrap(pitch),
            fill_value='extrapolate'
        )

        self.f_r = interp1d(
            self.timestamps,
            np.unwrap(roll),
            fill_value='extrapolate'
        )

    def pose_values(
            self,
            ts):

        if not self.has_timestamp(ts):
            return None

        return {

            "north": float(
                self.f_n(ts)
            ),

            "east": float(
                self.f_e(ts)
            ),

            "depth": float(
                self.f_d(ts)
            ),

            "yaw": float(
                self.f_y(ts)
            ),

            "pitch": float(
                self.f_p(ts)
            ),

            "roll": float(
                self.f_r(ts)
            )
        }

    def has_timestamp(
            self,
            ts):

        return (
            self.min_timestamp <= ts <= self.max_timestamp
        )


# =============================================================================
# PATCH BUILDER
# =============================================================================

class PatchBuilder:

    def _select_scan_time_source(
            self,
            scans,
            nav):

        header_times = np.array([
            scan_entry["header_ts"]
            for scan_entry in scans
        ])

        bag_times = np.array([
            scan_entry["bag_ts"]
            for scan_entry in scans
        ])

        def overlap_ratio(times):

            valid = np.isfinite(times)

            if not np.any(valid):
                return 0.0

            inside = (
                (times[valid] >= nav.min_timestamp) &
                (times[valid] <= nav.max_timestamp)
            )

            return float(np.mean(inside))

        candidates = [
            (
                "header",
                header_times,
                0.0,
                overlap_ratio(header_times)
            ),
            (
                "bag",
                bag_times,
                0.0,
                overlap_ratio(bag_times)
            ),
        ]

        for name, times in [
                ("header_offset", header_times),
                ("bag_offset", bag_times)]:

            valid = np.isfinite(times)

            if not np.any(valid):
                continue

            offset = (
                nav.min_timestamp -
                float(np.min(times[valid]))
            )

            shifted = times + offset

            candidates.append((
                name,
                times,
                offset,
                overlap_ratio(shifted)
            ))

        best = max(
            candidates,
            key=lambda item: item[3]
        )

        name, _, offset, ratio = best

        rospy.loginfo(
            f"Scan timestamp source: {name} "
            f"(offset={offset:.6f}, "
            f"navigation overlap={ratio * 100.0:.1f}%)"
        )

        if ratio < 0.5:

            rospy.logwarn(
                "Low overlap between scan timestamps and navigation. "
                "Patch generation may skip many scans. "
                f"Navigation range=[{nav.min_timestamp:.3f}, "
                f"{nav.max_timestamp:.3f}], "
                f"scan header range=[{np.nanmin(header_times):.3f}, "
                f"{np.nanmax(header_times):.3f}], "
                f"scan bag range=[{np.nanmin(bag_times):.3f}, "
                f"{np.nanmax(bag_times):.3f}]"
            )

        def timestamp(scan_entry):

            if name.startswith("bag"):
                return scan_entry["bag_ts"] + offset

            return scan_entry["header_ts"] + offset

        return timestamp

    def build(
            self,
            bag,
            scan_topic,
            nav,
            R_sensor):

        scans = []

        rospy.loginfo(
            "Loading scans..."
        )

        for _, scan, bag_time in tqdm(
                bag.read_messages(
                    topics=[scan_topic]
                ),
                desc="MBES scans"):

            if hasattr(scan, 'header'):
                scans.append({
                    "msg": scan,
                    "header_ts": scan.header.stamp.to_sec(),
                    "bag_ts": bag_time.to_sec()
                })

        if len(scans) == 0:
            return []

        scan_timestamp = self._select_scan_time_source(
            scans,
            nav
        )

        patches = []

        rospy.loginfo(
            "Building LOCAL patches..."
        )

        for start in tqdm(
                range(
                    0,
                    len(scans) - PATCH_SIZE + 1,
                    PATCH_STRIDE
                ),
                desc="Patch generation"):

            all_points = []

            center_scan_idx = start + PATCH_SIZE // 2
            center_entry = scans[center_scan_idx]
            center_ts = scan_timestamp(center_entry)

            if not nav.has_timestamp(center_ts):

                rospy.logwarn(
                    f"Skipping patch start={start}: "
                    f"center timestamp {center_ts:.3f} outside "
                    f"navigation range "
                    f"[{nav.min_timestamp:.3f}, "
                    f"{nav.max_timestamp:.3f}]"
                )

                continue

            center_pose = nav.pose_values(center_ts)

            T_center = pose_dict_to_matrix(center_pose)
            R_center = T_center[:3, :3]
            t_center = T_center[:3, 3]

            for i in range(
                    start,
                    start + PATCH_SIZE):

                scan_entry = scans[i]
                scan = scan_entry["msg"]

                ts = scan_timestamp(scan_entry)

                if not nav.has_timestamp(ts):
                    continue

                pose = nav.pose_values(ts)

                if pose is None:
                    continue

                pc = ros_numpy.point_cloud2.pointcloud2_to_array(
                    scan
                )

                finite_mask = (
                    np.isfinite(pc['x']) &
                    np.isfinite(pc['y']) &
                    np.isfinite(pc['z'])
                )

                pc = pc[finite_mask]

                if len(pc) < 10:
                    continue

                xyz = np.column_stack((
                    pc['x'],
                    -pc['y'],
                    -pc['z']
                ))

                r_horizontal = np.sqrt(
                    xyz[:, 0]**2 +
                    xyz[:, 1]**2
                )

                depth_s = np.abs(
                    xyz[:, 2]
                )

                angles = np.degrees(
                    np.arctan2(
                        r_horizontal,
                        depth_s
                    )
                )

                xyz = xyz[
                    angles < ANGLE_CUTOFF_DEG
                ]

                if len(xyz) < 10:
                    continue

                # =====================================================
                # SENSOR FRAME
                # =====================================================

                xyz = xyz @ R_sensor.T

                # =====================================================
                # VEHICLE ROTATION
                # =====================================================

                R_vehicle = tr.euler_matrix(
                    pose["roll"],
                    pose["pitch"],
                    pose["yaw"],
                    axes='sxyz'
                )[:3, :3]

                xyz = xyz @ R_vehicle.T

                # =====================================================
                # GLOBAL INS POSITION
                # =====================================================

                xyz[:, 0] += pose["north"]
                xyz[:, 1] += pose["east"]
                xyz[:, 2] += -pose["depth"]

                # =====================================================
                # RECENTER TO PATCH CENTER FRAME
                # =====================================================

                xyz -= t_center
                xyz = xyz @ R_center

                finite_xyz = np.isfinite(xyz).all(axis=1)
                xyz = xyz[finite_xyz]

                if len(xyz) < 10:
                    continue

                all_points.append(xyz)

            if len(all_points) == 0:
                continue

            pts = np.vstack(all_points)

            pts = pts[
                np.isfinite(pts).all(axis=1)
            ]

            if len(pts) < MIN_PATCH_POINTS:
                continue

            min_bound = pts.min(axis=0)
            max_bound = pts.max(axis=0)
            extent = max_bound - min_bound

            if (
                not np.isfinite(extent).all()
                or np.max(extent) > MAX_PATCH_EXTENT
            ):

                rospy.logwarn(
                    f"Skipping patch start={start}: invalid extent "
                    f"min={min_bound.tolist()} "
                    f"max={max_bound.tolist()} "
                    f"extent={extent.tolist()}"
                )

                continue

            pcd = o3d.geometry.PointCloud()

            pcd.points = (
                o3d.utility.Vector3dVector(pts)
            )

            try:

                pcd = pcd.voxel_down_sample(
                    VOXEL_SIZE
                )

            except RuntimeError as exc:

                rospy.logwarn(
                    f"Skipping patch start={start}: "
                    f"voxel_down_sample failed with "
                    f"VOXEL_SIZE={VOXEL_SIZE}. "
                    f"min={min_bound.tolist()} "
                    f"max={max_bound.tolist()} "
                    f"extent={extent.tolist()} "
                    f"error={exc}"
                )

                continue

            if len(pcd.points) < MIN_PATCH_POINTS:
                continue

            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=VOXEL_SIZE * 3.0,
                    max_nn=30
                )
            )

            patch = Patch()

            patch.idx = len(patches)

            patch.pcd = pcd

            patch.center = np.mean(
                pts[:, :2],
                axis=0
            )

            patch.pose = center_pose

            patch.timestamp = center_ts

            patches.append(patch)

        return patches

# =============================================================================
# LOCAL REGISTRATION
# =============================================================================

def execute_local_registration(
        source,
        target,
        T_init,
        algorithm,
        icp_distance,
        max_iter,
        ndt_resolution,
        ndt_max_points,
        ndt_min_points_per_voxel):

    algorithm = algorithm.lower()

    if algorithm == "icp":

        return robust_icp(
            source,
            target,
            T_init,
            icp_distance=icp_distance,
            max_iter=max_iter
        )

    if algorithm == "ndt":

        return robust_ndt(
            source,
            target,
            T_init,
            ndt_resolution=ndt_resolution,
            max_iter=max_iter,
            max_correspondence_distance=icp_distance,
            min_points_per_voxel=ndt_min_points_per_voxel,
            max_points=ndt_max_points
        )

    raise ValueError(
        f"Unsupported registration_algorithm '{algorithm}'. "
        "Use 'icp' or 'ndt'."
    )


# =============================================================================
# METRICS PLOTS
# =============================================================================

def _save_metrics_plots(
        metrics,
        raw_trajectory,
        slam_trajectory,
        metrics_dir):

    if not _MPL:
        rospy.logwarn("matplotlib not available — skipping plots")
        return

    seq = metrics.get("sequential", [])
    loops = metrics.get("loops", [])
    traj_m = metrics.get("trajectory", {})

    # ── 1. Trajectory comparison ──────────────────────────────────────────

    if len(raw_trajectory) >= 2 and len(slam_trajectory) >= 2:

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(
            raw_trajectory[:, 1],
            raw_trajectory[:, 0],
            color='steelblue',
            alpha=0.6,
            linewidth=1.0,
            label='Raw navigation'
        )

        ax.plot(
            slam_trajectory[:, 1],
            slam_trajectory[:, 0],
            color='crimson',
            alpha=0.85,
            linewidth=1.5,
            label='SLAM optimized'
        )

        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_title('Trajectory — Raw Navigation vs SLAM Optimized')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(
            os.path.join(metrics_dir, 'trajectory_comparison.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig)

    # ── 2. Fitness + RMSE timeseries ──────────────────────────────────────

    if seq:

        indices = [m['source'] for m in seq]
        fitness_vals = [m['fitness'] for m in seq]
        rmse_vals = [m['rmse'] for m in seq]
        colors = [
            '#2ecc71' if m['accepted'] else '#e74c3c'
            for m in seq
        ]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 6), sharex=True
        )

        ax1.scatter(indices, fitness_vals, c=colors, s=8, alpha=0.7)
        ax1.axhline(
            FITNESS_THRESHOLD,
            color='orange', linestyle='--', linewidth=1.2,
            label=f'Threshold {FITNESS_THRESHOLD}'
        )
        ax1.set_ylabel('Fitness')
        ax1.set_ylim([-0.05, 1.1])
        ax1.set_title(
            'Sequential Registration Quality'
            '  (green=accepted, red=rejected)'
        )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.scatter(indices, rmse_vals, c=colors, s=8, alpha=0.7)
        ax2.axhline(
            SEQ_RMSE_THRESHOLD,
            color='orange', linestyle='--', linewidth=1.2,
            label=f'Threshold {SEQ_RMSE_THRESHOLD:.3f} m'
        )
        ax2.set_ylabel('RMSE (m)')
        ax2.set_xlabel('Patch index')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(metrics_dir, 'registration_quality.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig)

    # ── 3. SLAM correction per node ───────────────────────────────────────

    corrections = traj_m.get('corrections_xy_m', [])

    if corrections:

        corrections_arr = np.asarray(corrections)
        mean_c = float(np.mean(corrections_arr))
        max_c = float(np.max(corrections_arr))

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(
            range(len(corrections_arr)),
            corrections_arr,
            alpha=0.35, color='steelblue'
        )
        ax.plot(corrections_arr, color='steelblue', linewidth=0.8)
        ax.axhline(
            mean_c, color='crimson', linestyle='--', linewidth=1.2,
            label=f'Mean {mean_c:.2f} m'
        )
        ax.axhline(
            max_c, color='darkorange', linestyle=':', linewidth=1.0,
            label=f'Max {max_c:.2f} m'
        )
        ax.set_xlabel('Node index')
        ax.set_ylabel('Correction XY (m)')
        ax.set_title('SLAM Position Correction vs Raw Navigation')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(metrics_dir, 'slam_correction.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig)

    # ── 4. RMSE histogram (accepted) ──────────────────────────────────────

    accepted_rmse = [
        m['rmse'] for m in seq
        if m['accepted'] and m['rmse'] > 0
    ]

    if accepted_rmse:

        arr = np.asarray(accepted_rmse)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            arr, bins=40,
            color='steelblue', edgecolor='white', alpha=0.85
        )
        ax.axvline(
            float(np.mean(arr)), color='crimson',
            linestyle='--', linewidth=1.5,
            label=f'Mean {float(np.mean(arr)):.3f} m'
        )
        ax.axvline(
            SEQ_RMSE_THRESHOLD, color='orange',
            linestyle='--', linewidth=1.5,
            label=f'Threshold {SEQ_RMSE_THRESHOLD:.3f} m'
        )
        ax.set_xlabel('RMSE (m)')
        ax.set_ylabel('Count')
        ax.set_title('RMSE Distribution — Accepted Sequential Registrations')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(metrics_dir, 'rmse_histogram.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig)

    # ── 5. Loop closure quality — two panels ─────────────────────────────

    if loops:

        sc_key  = 'scan_context_score'
        ins_key = 'ins_distance_m'
        rat_key = 'icp_ins_ratio'

        loop_acc = [l for l in loops if l['accepted']]
        loop_rej = [l for l in loops if not l['accepted']]

        # panel A: SC score vs ICP fitness
        # panel B: INS distance vs ICP/INS ratio  ← new key diagnostic
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        for ax, lrej, lacc, xlabel, xkey, ylabel, ykey, thr, thr_label in [
            (
                ax1,
                [l for l in loop_rej if sc_key in l],
                [l for l in loop_acc if sc_key in l],
                'Scan Context score (lower = more similar)', sc_key,
                'ICP fitness', 'fitness',
                SCAN_CONTEXT_THRESHOLD, f'SC thr {SCAN_CONTEXT_THRESHOLD}',
            ),
        ]:
            if lrej:
                ax.scatter(
                    [l[xkey] for l in lrej],
                    [l[ykey] for l in lrej],
                    c='#e74c3c', s=12, alpha=0.5, label='Rejected', zorder=2
                )
            if lacc:
                ax.scatter(
                    [l[xkey] for l in lacc],
                    [l[ykey] for l in lacc],
                    c='#2ecc71', s=35, alpha=0.9, marker='*',
                    label='Accepted', zorder=3
                )
            ax.axvline(thr, color='orange', linestyle='--',
                       linewidth=1.2, label=thr_label)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title('SC score vs ICP fitness')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Panel B — INS distance vs ICP/INS ratio
        rej_b = [l for l in loop_rej if ins_key in l and rat_key in l]
        acc_b = [l for l in loop_acc if ins_key in l and rat_key in l]

        if rej_b:
            ax2.scatter(
                [l[ins_key] for l in rej_b],
                [l[rat_key] for l in rej_b],
                c='#e74c3c', s=12, alpha=0.5, label='Rejected', zorder=2
            )
        if acc_b:
            ax2.scatter(
                [l[ins_key] for l in acc_b],
                [l[rat_key] for l in acc_b],
                c='#2ecc71', s=35, alpha=0.9, marker='*',
                label='Accepted', zorder=3
            )

        ax2.axhline(
            MIN_LOOP_ICP_INS_RATIO, color='orange', linestyle='--',
            linewidth=1.2, label=f'Min ratio {MIN_LOOP_ICP_INS_RATIO}'
        )
        ax2.axvline(
            MAX_LOOP_INS_DISTANCE, color='steelblue', linestyle=':',
            linewidth=1.2, label=f'Max INS dist {MAX_LOOP_INS_DISTANCE}m'
        )
        ax2.set_xlabel('INS distance between patches (m)')
        ax2.set_ylabel('ICP correction / INS distance  (ideal ≈ 1.0)')
        ax2.set_title('ICP–INS Drift Consistency\n'
                      '(false positives cluster near ratio ≈ 0)')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(metrics_dir, 'loop_closure_quality.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig)

    rospy.loginfo(f"Plots saved → {metrics_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():

    rospy.init_node(
        "advanced_multibeam_slam"
    )

    bag_file = rospy.get_param(
        "~bag_file"
    )

    scan_topic = rospy.get_param(
        "~scan_topic"
    )

    nav_topic = rospy.get_param(
        "~nav_topic"
    )

    output_dir = rospy.get_param(
        "~output_dir"
    )

    registration_algorithm = rospy.get_param(
        "~registration_algorithm",
        REGISTRATION_ALGORITHM
    )
    registration_algorithm = str(
        registration_algorithm
    ).lower()

    if registration_algorithm not in ("icp", "ndt"):

        raise ValueError(
            f"Unsupported registration_algorithm "
            f"'{registration_algorithm}'. Use 'icp' or 'ndt'."
        )

    ndt_resolution = rospy.get_param(
        "~ndt_resolution",
        NDT_RESOLUTION
    )
    ndt_resolution = float(
        ndt_resolution
    )

    ndt_max_points = rospy.get_param(
        "~ndt_max_points",
        NDT_MAX_POINTS
    )
    ndt_max_points = int(
        ndt_max_points
    )

    ndt_min_points_per_voxel = rospy.get_param(
        "~ndt_min_points_per_voxel",
        NDT_MIN_POINTS_PER_VOXEL
    )
    ndt_min_points_per_voxel = int(
        ndt_min_points_per_voxel
    )

    enable_loop_closure = rospy.get_param(
        "~enable_loop_closure",
        ENABLE_LOOP_CLOSURE
    )

    global MAX_SEQ_ICP_TRANSLATION_DEV, MAX_SEQ_ICP_YAW_DEV
    global MAX_LOOP_INS_DISTANCE, MIN_LOOP_ICP_INS_RATIO, MIN_INS_DIST_FOR_RATIO_CHECK

    MAX_SEQ_ICP_TRANSLATION_DEV = float(rospy.get_param(
        "~max_seq_icp_translation_dev",
        MAX_SEQ_ICP_TRANSLATION_DEV
    ))

    MAX_SEQ_ICP_YAW_DEV = float(rospy.get_param(
        "~max_seq_icp_yaw_dev",
        MAX_SEQ_ICP_YAW_DEV
    ))

    MAX_LOOP_INS_DISTANCE = float(rospy.get_param(
        "~max_loop_ins_distance",
        MAX_LOOP_INS_DISTANCE
    ))

    MIN_LOOP_ICP_INS_RATIO = float(rospy.get_param(
        "~min_loop_icp_ins_ratio",
        MIN_LOOP_ICP_INS_RATIO
    ))

    MIN_INS_DIST_FOR_RATIO_CHECK = float(rospy.get_param(
        "~min_ins_dist_for_ratio_check",
        MIN_INS_DIST_FOR_RATIO_CHECK
    ))

    enable_monitor = rospy.get_param(
        "~enable_monitor",
        True
    )

    monitor_zoom = rospy.get_param(
        "~monitor_zoom",
        0.18
    )

    monitor_min_zoom = rospy.get_param(
        "~monitor_min_zoom",
        0.03
    )

    monitor_margin = rospy.get_param(
        "~monitor_margin",
        2.5
    )

    rospy.loginfo(
        f"Loop closure enabled: {enable_loop_closure}"
    )

    if enable_loop_closure:
        rospy.loginfo(
            f"Loop closure gates — "
            f"max_ins_dist={MAX_LOOP_INS_DISTANCE:.1f}m  "
            f"min_icp_ins_ratio={MIN_LOOP_ICP_INS_RATIO:.2f}  "
            f"min_ins_for_ratio={MIN_INS_DIST_FOR_RATIO_CHECK:.1f}m"
        )

    rospy.loginfo(
        f"Monitor enabled: {enable_monitor}"
    )

    rospy.loginfo(
        f"Monitor camera: zoom={monitor_zoom}, "
        f"min_zoom={monitor_min_zoom}, "
        f"margin={monitor_margin}"
    )

    rospy.loginfo(
        f"Registration algorithm: {registration_algorithm}"
    )

    rospy.loginfo(
        f"ICP–nav consistency gates — "
        f"max_translation_dev={MAX_SEQ_ICP_TRANSLATION_DEV:.2f}m  "
        f"max_yaw_dev={MAX_SEQ_ICP_YAW_DEV:.1f}deg"
    )

    if registration_algorithm == "ndt":

        rospy.loginfo(
            f"NDT parameters: resolution={ndt_resolution}, "
            f"max_points={ndt_max_points}, "
            f"min_points_per_voxel={ndt_min_points_per_voxel}"
        )

    ensure_dir(output_dir)

    metrics_dir = os.path.join(
        output_dir,
        "metrics"
    )

    ensure_dir(metrics_dir)

    bag = rosbag.Bag(bag_file)

    T_mb = get_static_transform_from_tf(

        bag_file,

        "sparus2/base_link",

        "sparus2/multibeam"
    )

    R_sensor = T_mb[:3, :3]

    nav = NavigationInterpolator(
        bag,
        nav_topic
    )

    builder = PatchBuilder()

    patches = builder.build(

        bag,
        scan_topic,
        nav,
        R_sensor
    )

    bag.close()

    rospy.loginfo(
        f"Generated {len(patches)} patches"
    )

    if len(patches) == 0:

        rospy.logerr(
            "No valid patches generated. "
            "Check bag topics, point density and patch filters."
        )

        return

    scan_context_manager = None

    if enable_loop_closure:

        scan_context_manager = \
            ScanContextManager()

        for patch in tqdm(
                patches,
                desc="Scan Context"):

            scan_context_manager.add_descriptor(
                patch.pcd
            )

    pose_graph = (
        o3d.pipelines.registration.PoseGraph()
    )

    T0 = pose_dict_to_matrix(
        patches[0].pose
    )

    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(
            T0
        )
    )

    odometry = T0.copy()

    monitor = PoseGraphMonitor(
        enabled=enable_monitor,
        overview_zoom=monitor_zoom,
        min_zoom=monitor_min_zoom,
        overview_margin=monitor_margin
    )

    # =========================================================================
    # METRICS
    # =========================================================================

    metrics = {

        "sequential": [],

        "loops": [],

        "summary": {}
    }

    fallback_count = 0

    accepted_seq = 0

    rejected_seq = 0

    accepted_loops = 0

    rejected_loops = 0

    # =========================================================================
    # SEQUENTIAL REGISTRATION
    # =========================================================================

    rospy.loginfo(
        f"Sequential registration ({registration_algorithm.upper()})..."
    )

    for idx in tqdm(

            range(1, len(patches)),

            desc=f"Sequential {registration_algorithm.upper()}"):

        source = patches[idx].pcd

        target = patches[idx - 1].pcd

        T_init = expected_transform(

            patches[idx - 1],

            patches[idx]
        )

        result = execute_local_registration(
            source,
            target,
            T_init,
            registration_algorithm,
            ICP_DISTANCE,
            ICP_MAX_ITER,
            ndt_resolution,
            ndt_max_points,
            ndt_min_points_per_voxel
        )

        temporal_distance = (

            patches[idx].timestamp -
            patches[idx - 1].timestamp
        )

        # =============================================================
        # ICP FAILED
        # =============================================================
        # robust_icp devuelve None si todas las escalas producen nubes
        # con menos de 50 puntos tras el downsample.
        # Se usa T_init (navegación) como transformación de fallback
        # con información muy baja para no contaminar el optimizador.
        #
        # IMPORTANTE: no hay 'continue' aquí. El flujo cae al bloque
        # común de métricas, nodo, arista y monitor al final del bucle,
        # garantizando que el pose graph siempre esté en estado
        # consistente antes de que el monitor lo visualice.
        # =============================================================

        seq_failure_reason = None

        if result is None:

            fallback_count += 1
            rejected_seq += 1
            seq_failure_reason = "registration_failed"

            rospy.logwarn(
                f"{registration_algorithm.upper()} failed "
                f"{idx}->{idx-1}"
            )

            T = T_init
            info = np.eye(6) * 0.01

            fitness = 0.0
            rmse = 0.0
            correspondences = 0
            valid_registration = False

        # =============================================================
        # ICP SUCCEEDED — validación adaptativa
        # =============================================================

        else:

            fitness = float(result.fitness)
            rmse = float(result.inlier_rmse)
            correspondences = len(result.correspondence_set)

            rospy.loginfo(
                f"[SEQ {registration_algorithm.upper()}] "
                f"{idx}->{idx-1} | "
                f"fitness={fitness:.3f} "
                f"rmse={rmse:.3f} "
                f"corr={correspondences}"
            )

            # =========================================================
            # ADAPTIVE ICP VALIDATION
            # =========================================================

            valid_registration = True

            if fitness < FITNESS_THRESHOLD:
                valid_registration = False
                seq_failure_reason = "fitness_below_threshold"

            if correspondences < MIN_CORRESPONDENCES:
                valid_registration = False
                seq_failure_reason = "insufficient_correspondences"

            adaptive_rmse_threshold = SEQ_RMSE_THRESHOLD

            # ---------------------------------------------------------
            # HIGH-CONFIDENCE FALLBACK
            # ---------------------------------------------------------
            # In underwater MBES planar regions with soft bathymetry
            # and acoustic noise: fitness ≈ 1.0, high correspondences,
            # moderate RMSE — still geometrically correct.
            # HIGH_RMSE_MULTIPLIER > 1.0 relaxes the threshold here.
            # ---------------------------------------------------------

            if (
                fitness > HIGH_FITNESS_THRESHOLD
                and
                correspondences > HIGH_CORRESPONDENCE_THRESHOLD
            ):

                adaptive_rmse_threshold *= HIGH_RMSE_MULTIPLIER

                rospy.loginfo(
                    f"[HIGH CONFIDENCE ICP] "
                    f"Relaxed RMSE threshold: "
                    f"{adaptive_rmse_threshold:.3f}"
                )

            if rmse > adaptive_rmse_threshold:
                valid_registration = False
                seq_failure_reason = "rmse_above_threshold"

            # =========================================================
            # ICP–NAVIGATION CONSISTENCY GATE
            # =========================================================
            # Flat underwater seabeds give GICP near-zero XY gradient.
            # ICP can drift to wrong local minima (reversed or
            # perpendicular steps) that still show fitness ≈ 1 / low RMSE.
            # Compare the raw ICP result with T_init before accepting.
            # =========================================================

            if valid_registration:

                _T_raw = result.transformation

                # --- translation deviation ---
                _icp_tr_dev = float(
                    np.linalg.norm(_T_raw[:2, 3] - T_init[:2, 3])
                )
                if _icp_tr_dev > MAX_SEQ_ICP_TRANSLATION_DEV:
                    valid_registration = False
                    seq_failure_reason = "icp_translation_deviation"

            if valid_registration:

                # --- translation DIRECTION deviation ---
                # arctan2(T[1,0], T[0,0]) extracts the rotation-matrix yaw,
                # which stays ≈0° even when the TRANSLATION goes in the
                # wrong direction on a flat seabed.  The correct check is
                # the angle of the TRANSLATION VECTOR T[:2,3] vs T_init[:2,3].
                _icp_len  = float(np.linalg.norm(_T_raw[:2, 3]))
                _init_len = float(np.linalg.norm(T_init[:2, 3]))

                if _icp_len > 0.05 and _init_len > 0.05:
                    _icp_dir  = np.arctan2(_T_raw[1, 3],  _T_raw[0, 3])
                    _init_dir = np.arctan2(T_init[1, 3],  T_init[0, 3])
                    _dir_dev  = float(abs(np.degrees(
                        np.arctan2(
                            np.sin(_icp_dir - _init_dir),
                            np.cos(_icp_dir - _init_dir)
                        )
                    )))
                    if _dir_dev > MAX_SEQ_ICP_YAW_DEV:
                        valid_registration = False
                        seq_failure_reason = "icp_direction_deviation"

            # =========================================================
            # VALID ICP
            # =========================================================

            if valid_registration:

                accepted_seq += 1

                # FIX 1 — Rotación INS + traslación ICP.
                # En fondo plano la rotación del ICP es ruido aleatorio
                # (std ~19°/paso) que integra en un random walk y desvía la
                # trayectoria decenas de grados. La rotación de la INS (DVL+IMU)
                # es fiable, así que la arista usa la rotación de T_init y deja
                # que el ICP solo refine la traslación XY (ya validada por los
                # gates de traslación y dirección).
                T = ins_rotation_icp_translation(
                    result.transformation,
                    T_init
                )

                info = dynamic_information_matrix(
                    result,
                    temporal_distance=temporal_distance,
                    loop=False
                )

            # =========================================================
            # FALLBACK NAVIGATION
            # =========================================================

            else:

                fallback_count += 1
                rejected_seq += 1

                rospy.logwarn(
                    f"Sequential {registration_algorithm.upper()} rejected "
                    f"{idx}->{idx-1} — {seq_failure_reason}"
                )

                T = T_init
                info = np.eye(6) * 0.01

        # =============================================================
        # MÉTRICAS — comunes a todos los casos
        # =============================================================

        metrics["sequential"].append({
            "source": idx,
            "target": idx - 1,
            "algorithm": registration_algorithm,
            "fitness": float(fitness),
            "rmse": float(rmse),
            "correspondences": int(correspondences),
            "accepted": bool(valid_registration),
            "failure_reason": seq_failure_reason,
        })

        # =============================================================
        # POSE GRAPH — nodo y arista añadidos siempre aquí
        # =============================================================
        # La odometría y los elementos del grafo se actualizan en un
        # único punto del bucle, independientemente de si el ICP
        # falló, fue rechazado o fue aceptado.
        # Esto garantiza que cuando el monitor visualiza el grafo,
        # el nodo y su arista correspondiente ya existen.
        # =============================================================

        odometry = odometry @ T

        pose_graph.nodes.append(

            o3d.pipelines.registration.
            PoseGraphNode(
                odometry
            )
        )

        pose_graph.edges.append(

            o3d.pipelines.registration.
            PoseGraphEdge(

                idx,
                idx - 1,

                T,

                info,

                uncertain=False
            )
        )

        # =============================================================
        # MONITOR — throttle unificado
        # =============================================================
        # Se ejecuta cada MONITOR_UPDATE_EVERY iteraciones para todos
        # los casos (ICP ok, rechazado o fallido).
        # El grafo ya tiene el nodo y la arista añadidos justo arriba,
        # por lo que el monitor siempre visualiza un estado consistente.
        # =============================================================

        if idx % MONITOR_UPDATE_EVERY == 0:

            rospy.loginfo(
                f"[MONITOR] "
                f"{len(pose_graph.nodes)} nodes, "
                f"{len(pose_graph.edges)} edges"
            )

            monitor.update(
                pose_graph
            )

    rospy.loginfo(
        f"PoseGraph nodes: "
        f"{len(pose_graph.nodes)}"
    )

    rospy.loginfo(
        f"PoseGraph edges: "
        f"{len(pose_graph.edges)}"
    )

    # =========================================================================
    # RAW TRAJECTORY
    # =========================================================================

    raw_trajectory = []

    for patch in patches:

        T_nav = pose_dict_to_matrix(
            patch.pose
        )

        raw_trajectory.append(
            T_nav[:3, 3]
        )

    raw_trajectory = np.array(
        raw_trajectory
    )

    # =========================================================================
    # RAW MAP
    # =========================================================================

    rospy.loginfo(
        "Building RAW map..."
    )

    raw_map = o3d.geometry.PointCloud()

    for patch in patches:

        T_nav = pose_dict_to_matrix(
            patch.pose
        )

        pcd_global = copy.deepcopy(
            patch.pcd
        )

        pcd_global.transform(T_nav)

        raw_map += pcd_global

    raw_map = raw_map.voxel_down_sample(
        FINAL_DOWNSAMPLE
    )

    # =========================================================================
    # LOOP CLOSURE
    # =========================================================================

    if enable_loop_closure:

        rospy.loginfo(
            "Searching loop closures..."
        )

        for idx in tqdm(

                range(len(patches)),

                desc="Loop closure"):

            candidates = (

                scan_context_manager.
                detect_loop_candidates(

                    idx,

                    top_k=
                    MAX_LOOP_CANDIDATES,

                    threshold=
                    SCAN_CONTEXT_THRESHOLD
                )
            )

            for cand_idx, score in candidates:

                # =============================================================
                # INS PROXIMITY GATE
                # =============================================================
                # Reject candidates whose INS positions are too far apart.
                # In flat/featureless environments, distant patches can look
                # identical (flat seabed); this gate prevents false positives
                # from parallel lawnmower strips or other repeated structures.
                # =============================================================

                ins_src = np.array([
                    patches[idx].pose['north'],
                    patches[idx].pose['east']
                ])

                ins_tgt = np.array([
                    patches[cand_idx].pose['north'],
                    patches[cand_idx].pose['east']
                ])

                ins_distance = float(
                    np.linalg.norm(ins_src - ins_tgt)
                )

                if ins_distance > MAX_LOOP_INS_DISTANCE:

                    rejected_loops += 1

                    metrics["loops"].append({
                        "source": idx,
                        "target": cand_idx,
                        "algorithm": registration_algorithm,
                        "scan_context_score": float(score),
                        "ins_distance_m": ins_distance,
                        "fitness": 0.0,
                        "rmse": 0.0,
                        "correspondences": 0,
                        "accepted": False,
                        "failure_reason": "ins_proximity_gate",
                    })

                    continue

                source = patches[idx].pcd
                target = patches[cand_idx].pcd

                ransac_result = execute_global_registration(
                    source,
                    target,
                    VOXEL_SIZE
                )

                if ransac_result is None:

                    rejected_loops += 1

                    metrics["loops"].append({
                        "source": idx,
                        "target": cand_idx,
                        "algorithm": registration_algorithm,
                        "scan_context_score": float(score),
                        "ins_distance_m": ins_distance,
                        "fitness": 0.0,
                        "rmse": 0.0,
                        "correspondences": 0,
                        "accepted": False,
                        "failure_reason": "ransac_failed",
                    })

                    continue

                T_init = ransac_result.transformation

                result = execute_local_registration(
                    source,
                    target,
                    T_init,
                    registration_algorithm,
                    ICP_DISTANCE,
                    ICP_MAX_ITER,
                    ndt_resolution,
                    ndt_max_points,
                    ndt_min_points_per_voxel
                )

                if result is None:

                    rejected_loops += 1

                    metrics["loops"].append({
                        "source": idx,
                        "target": cand_idx,
                        "algorithm": registration_algorithm,
                        "scan_context_score": float(score),
                        "ins_distance_m": ins_distance,
                        "fitness": 0.0,
                        "rmse": 0.0,
                        "correspondences": 0,
                        "accepted": False,
                        "failure_reason": "icp_failed",
                    })

                    continue

                fitness = float(result.fitness)
                rmse = float(result.inlier_rmse)
                correspondences = len(result.correspondence_set)

                # =============================================================
                # ADAPTIVE LOOP VALIDATION
                # =============================================================

                valid_loop = True
                loop_failure_reason = None

                if fitness < LOOP_FITNESS_THRESHOLD:
                    valid_loop = False
                    loop_failure_reason = "fitness_below_threshold"

                if correspondences < MIN_CORRESPONDENCES:
                    valid_loop = False
                    loop_failure_reason = "insufficient_correspondences"

                adaptive_loop_rmse = LOOP_RMSE_THRESHOLD

                if (
                    fitness > HIGH_FITNESS_THRESHOLD
                    and
                    correspondences > HIGH_CORRESPONDENCE_THRESHOLD
                ):

                    adaptive_loop_rmse *= HIGH_RMSE_MULTIPLIER

                    rospy.loginfo(
                        f"[HIGH CONFIDENCE LOOP] "
                        f"Relaxed RMSE threshold: {adaptive_loop_rmse:.3f}"
                    )

                if rmse > adaptive_loop_rmse:
                    valid_loop = False
                    loop_failure_reason = "rmse_above_threshold"

                T_raw = result.transformation

                loop_yaw_deg = np.degrees(
                    np.arctan2(T_raw[1, 0], T_raw[0, 0])
                )

                loop_xy_translation = float(
                    np.linalg.norm(T_raw[:2, 3])
                )

                loop_z_translation = float(abs(T_raw[2, 3]))

                if loop_z_translation > MAX_LOOP_Z_TRANSLATION:
                    valid_loop = False
                    loop_failure_reason = "z_translation_gate"

                if loop_xy_translation > MAX_LOOP_XY_TRANSLATION:
                    valid_loop = False
                    loop_failure_reason = "xy_translation_gate"

                if abs(loop_yaw_deg) > MAX_LOOP_YAW_DEG:
                    valid_loop = False
                    loop_failure_reason = "yaw_gate"

                # =============================================================
                # ICP–INS DRIFT CONSISTENCY GATE
                # =============================================================
                # The ICP result must correct a significant fraction of the
                # accumulated INS drift between the two patches.
                #
                # True loop:   ICP_xy ≈ INS_distance (corrects the drift)
                # False pos.:  ICP_xy ≈ 0  (flat surface aligns near identity)
                #
                # Only applied when INS distance is large enough to matter.
                # =============================================================

                if ins_distance >= MIN_INS_DIST_FOR_RATIO_CHECK:

                    icp_ins_ratio = loop_xy_translation / ins_distance

                    if icp_ins_ratio < MIN_LOOP_ICP_INS_RATIO:

                        valid_loop = False
                        loop_failure_reason = (
                            "icp_correction_too_small_for_ins_drift"
                        )

                        rospy.logdebug(
                            f"[LOOP REJECTED] {idx}<->{cand_idx}  "
                            f"ins_dist={ins_distance:.2f}m  "
                            f"icp_xy={loop_xy_translation:.3f}m  "
                            f"ratio={icp_ins_ratio:.3f} < {MIN_LOOP_ICP_INS_RATIO}"
                        )

                if valid_loop:

                    accepted_loops += 1

                    rospy.loginfo(
                        f"LOOP ACCEPTED {idx}<->{cand_idx}  "
                        f"ins={ins_distance:.2f}m  "
                        f"icp_xy={loop_xy_translation:.3f}m  "
                        f"ratio={loop_xy_translation/max(ins_distance,0.1):.2f}  "
                        f"fit={fitness:.3f}  rmse={rmse:.3f}"
                    )

                    T = constrain_transform(T_raw)

                    info = dynamic_information_matrix(
                        result,
                        temporal_distance=1.0,
                        loop=True
                    )

                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            idx,
                            cand_idx,
                            T,
                            info,
                            uncertain=True
                        )
                    )

                    monitor.update(pose_graph)

                else:

                    rejected_loops += 1

                metrics["loops"].append({
                    "source": idx,
                    "target": cand_idx,
                    "algorithm": registration_algorithm,
                    "scan_context_score": float(score),
                    "ins_distance_m": ins_distance,
                    "fitness": float(fitness),
                    "rmse": float(rmse),
                    "correspondences": int(correspondences),
                    "raw_xy_translation": float(loop_xy_translation),
                    "raw_z_translation": float(loop_z_translation),
                    "raw_yaw_deg": float(loop_yaw_deg),
                    "icp_ins_ratio": float(
                        loop_xy_translation / max(ins_distance, 0.1)
                    ),
                    "accepted": bool(valid_loop),
                    "failure_reason": loop_failure_reason,
                })

    # =========================================================================
    # GLOBAL OPTIMIZATION
    # =========================================================================

    rospy.loginfo(
        "Global pose graph optimization..."
    )

    option = (

        o3d.pipelines.registration.
        GlobalOptimizationOption(

            max_correspondence_distance=
            ICP_DISTANCE,

            edge_prune_threshold=0.25,

            reference_node=0
        )
    )

    o3d.pipelines.registration.global_optimization(

        pose_graph,

        o3d.pipelines.registration.
        GlobalOptimizationLevenbergMarquardt(),

        o3d.pipelines.registration.
        GlobalOptimizationConvergenceCriteria(),

        option
    )

    rospy.loginfo(
        "Pose graph optimization finished"
    )

    # =========================================================================
    # VERTICAL POSE RESTORATION (Z + roll + pitch)
    # =========================================================================
    # The flat underwater scene gives the Levenberg-Marquardt optimizer almost
    # no vertical constraints, so it leaves the nodes' Z translation AND their
    # roll/pitch poorly constrained. Restoring only Z is not enough: a spurious
    # roll/pitch of a few degrees, applied by build_final_map() to a patch that
    # spans 30-50 m in XY, projects the patch edges several metres up/down in Z
    # (radius·sin(θ)) — producing a thick, ghosted map (Z spread 3.2 m, points
    # up to 14 m above the seabed).
    #
    # The IMU roll/pitch and the INS depth are reliable; the optimizer's are
    # not. We therefore rebuild each node's rotation from the INS roll+pitch
    # while KEEPING the optimized yaw (the valid SLAM correction is in-plane),
    # and restore the INS Z translation. XY translation stays as optimized.
    # =========================================================================

    for i, patch in enumerate(patches):

        if i >= len(pose_graph.nodes):
            break

        T_node = pose_graph.nodes[i].pose.copy()

        # Optimized yaw (valid in-plane SLAM correction).
        opt_yaw = np.arctan2(T_node[1, 0], T_node[0, 0])

        # INS roll/pitch (reliable) + optimized yaw → clean rotation.
        R_fixed = tr.euler_matrix(
            patch.pose["roll"],
            patch.pose["pitch"],
            opt_yaw,
            axes='sxyz'
        )[:3, :3]

        T_node[:3, :3] = R_fixed

        # INS depth (reliable); XY translation stays as optimized.
        T_nav = pose_dict_to_matrix(patch.pose)
        T_node[2, 3] = T_nav[2, 3]

        pose_graph.nodes[i].pose = T_node

    rospy.loginfo(
        "Vertical pose restored from INS (Z + roll/pitch, optimized yaw kept)"
    )

    monitor.update(
        pose_graph
    )

    # =========================================================================
    # SLAM TRAJECTORY
    # =========================================================================

    slam_trajectory = []

    for node in pose_graph.nodes:

        slam_trajectory.append(
            node.pose[:3, 3]
        )

    slam_trajectory = np.array(
        slam_trajectory
    )

    # =========================================================================
    # TRAJECTORY METRICS
    # =========================================================================

    _n = min(len(raw_trajectory), len(slam_trajectory))
    _corrections_xy = np.linalg.norm(
        slam_trajectory[:_n, :2] - raw_trajectory[:_n, :2],
        axis=1
    )

    def _path_length(traj):
        return float(
            np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        )

    metrics["trajectory"] = {
        "nodes": int(_n),
        "corrections_xy_m": _corrections_xy.tolist(),
        "mean_correction_xy_m": float(np.mean(_corrections_xy)),
        "max_correction_xy_m": float(np.max(_corrections_xy)),
        "std_correction_xy_m": float(np.std(_corrections_xy)),
        "raw_path_length_m": _path_length(raw_trajectory),
        "slam_path_length_m": _path_length(slam_trajectory),
    }

    rospy.loginfo(
        f"Trajectory correction — "
        f"mean={metrics['trajectory']['mean_correction_xy_m']:.2f} m  "
        f"max={metrics['trajectory']['max_correction_xy_m']:.2f} m"
    )

    # =========================================================================
    # FINAL MAP
    # =========================================================================

    rospy.loginfo(
        "Building SLAM map..."
    )

    final_map = build_final_map(

        patches,

        pose_graph,

        voxel_size=
        FINAL_DOWNSAMPLE
    )

    # =========================================================================
    # SAVE MAPS
    # =========================================================================

    raw_map_path = os.path.join(

        output_dir,

        "raw_navigation_map.ply"
    )

    slam_map_path = os.path.join(

        output_dir,

        "slam_optimized_map.ply"
    )

    o3d.io.write_point_cloud(

        raw_map_path,

        raw_map
    )

    o3d.io.write_point_cloud(

        slam_map_path,

        final_map
    )

    rospy.loginfo(
        f"Saved RAW map: "
        f"{raw_map_path}"
    )

    rospy.loginfo(
        f"Saved SLAM map: "
        f"{slam_map_path}"
    )

    # =========================================================================
    # SAVE TRAJECTORIES
    # =========================================================================

    np.save(

        os.path.join(
            output_dir,
            "raw_trajectory.npy"
        ),

        raw_trajectory
    )

    np.save(

        os.path.join(
            output_dir,
            "slam_trajectory.npy"
        ),

        slam_trajectory
    )

    # =========================================================================
    # METRICS SUMMARY
    # =========================================================================

    _seq_accepted = [m for m in metrics["sequential"] if m["accepted"]]
    _seq_rejected = [m for m in metrics["sequential"] if not m["accepted"]]
    _loops_accepted = [m for m in metrics.get("loops", []) if m["accepted"]]
    _loops_rejected = [m for m in metrics.get("loops", []) if not m["accepted"]]

    def _safe_stats(values):
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    _seq_fitness_stats = _safe_stats(
        [m["fitness"] for m in _seq_accepted]
    )
    _seq_rmse_stats = _safe_stats(
        [m["rmse"] for m in _seq_accepted]
    )
    _seq_corr_stats = _safe_stats(
        [m["correspondences"] for m in _seq_accepted]
    )
    _loop_fitness_stats = _safe_stats(
        [m["fitness"] for m in _loops_accepted]
    )
    _loop_rmse_stats = _safe_stats(
        [m["rmse"] for m in _loops_accepted]
    )
    _loop_score_stats = _safe_stats(
        [m.get("scan_context_score", 0.0) for m in _loops_accepted]
    )
    _rej_reason_counts = {}
    for m in _seq_rejected:
        reason = m.get("failure_reason", "validation_failed")
        _rej_reason_counts[reason] = (
            _rej_reason_counts.get(reason, 0) + 1
        )
    _loop_rej_reason_counts = {}
    for m in _loops_rejected:
        reason = m.get("failure_reason", "unknown")
        _loop_rej_reason_counts[reason] = (
            _loop_rej_reason_counts.get(reason, 0) + 1
        )

    _traj = metrics.get("trajectory", {})

    metrics["summary"] = {

        # ── Graph ──────────────────────────────────────────────────────
        "registration_algorithm":
            registration_algorithm,
        "total_nodes":
            len(pose_graph.nodes),
        "total_edges":
            len(pose_graph.edges),

        # ── Sequential registration ────────────────────────────────────
        "accepted_seq":            accepted_seq,
        "rejected_seq":            rejected_seq,
        "fallback_edges":          fallback_count,
        "seq_acceptance_ratio":    accepted_seq / max(accepted_seq + rejected_seq, 1),
        "seq_fitness":             _seq_fitness_stats,
        "seq_rmse_m":              _seq_rmse_stats,
        "seq_correspondences":     _seq_corr_stats,
        "seq_rejection_reasons":   _rej_reason_counts,

        # ── Loop closure ───────────────────────────────────────────────
        "accepted_loops":          accepted_loops,
        "rejected_loops":          rejected_loops,
        "loop_acceptance_ratio":   accepted_loops / max(accepted_loops + rejected_loops, 1),
        "loop_fitness":            _loop_fitness_stats,
        "loop_rmse_m":             _loop_rmse_stats,
        "loop_scan_context_score": _loop_score_stats,
        "loop_rejection_reasons":  _loop_rej_reason_counts,

        # ── Trajectory correction ──────────────────────────────────────
        "mean_slam_correction_xy_m": _traj.get("mean_correction_xy_m", 0.0),
        "max_slam_correction_xy_m":  _traj.get("max_correction_xy_m", 0.0),
        "std_slam_correction_xy_m":  _traj.get("std_correction_xy_m", 0.0),
        "raw_path_length_m":         _traj.get("raw_path_length_m", 0.0),
        "slam_path_length_m":        _traj.get("slam_path_length_m", 0.0),
    }

    # =========================================================================
    # SAVE METRICS JSON
    # =========================================================================

    metrics_path = os.path.join(

        metrics_dir,

        "slam_metrics.json"
    )

    with open(metrics_path, "w") as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    csv_path = os.path.join(

        metrics_dir,

        "icp_stats.csv"
    )

    with open(csv_path, "w") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "source",
            "target",
            "algorithm",
            "fitness",
            "rmse",
            "correspondences",
            "accepted",
            "failure_reason",
        ])

        for m in metrics["sequential"]:

            writer.writerow([
                m["source"],
                m["target"],
                m.get("algorithm", registration_algorithm),
                m["fitness"],
                m["rmse"],
                m["correspondences"],
                m["accepted"],
                m.get("failure_reason", ""),
            ])

    loop_csv_path = os.path.join(

        metrics_dir,

        "loop_stats.csv"
    )

    with open(loop_csv_path, "w") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "source",
            "target",
            "algorithm",
            "scan_context_score",
            "ins_distance_m",
            "fitness",
            "rmse",
            "correspondences",
            "raw_xy_translation",
            "icp_ins_ratio",
            "raw_z_translation",
            "raw_yaw_deg",
            "accepted",
            "failure_reason",
        ])

        for m in metrics["loops"]:

            writer.writerow([
                m["source"],
                m["target"],
                m.get("algorithm", registration_algorithm),
                m.get("scan_context_score", ""),
                m.get("ins_distance_m", ""),
                m["fitness"],
                m["rmse"],
                m["correspondences"],
                m.get("raw_xy_translation", ""),
                m.get("icp_ins_ratio", ""),
                m.get("raw_z_translation", ""),
                m.get("raw_yaw_deg", ""),
                m["accepted"],
                m.get("failure_reason", ""),
            ])

    # =========================================================================
    # PLOTS
    # =========================================================================

    _save_metrics_plots(
        metrics,
        raw_trajectory,
        slam_trajectory,
        metrics_dir
    )

    rospy.loginfo(
        "================================================"
    )

    rospy.loginfo(
        "ADVANCED MBES SLAM FINISHED"
    )

    rospy.loginfo(
        "================================================"
    )

    monitor.close()


if __name__ == "__main__":

    main()
