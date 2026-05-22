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

from utils import *
from registration import *
from robust_icp import *
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

# =============================================================================
# ICP
# =============================================================================

ICP_DISTANCE = 2.0

ICP_MAX_ITER = 60

# -----------------------------------------------------------------------------
# FITNESS
# -----------------------------------------------------------------------------

FITNESS_THRESHOLD = 0.55

LOOP_FITNESS_THRESHOLD = 0.65

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

LOOP_RMSE_THRESHOLD = VOXEL_SIZE * 2.5

# -----------------------------------------------------------------------------
# CORRESPONDENCES
# -----------------------------------------------------------------------------

MIN_CORRESPONDENCES = 100

# -----------------------------------------------------------------------------
# HIGH-CONFIDENCE FALLBACK
# -----------------------------------------------------------------------------
# If fitness is extremely high and correspondences are massive,
# allow higher RMSE before rejecting.
# This is extremely important in flat underwater environments.
# -----------------------------------------------------------------------------

HIGH_FITNESS_THRESHOLD = 0.90

HIGH_CORRESPONDENCE_THRESHOLD = 1000

HIGH_RMSE_MULTIPLIER = 1

# =============================================================================
# LOOP CLOSURE
# =============================================================================

ENABLE_LOOP_CLOSURE = False

SCAN_CONTEXT_THRESHOLD = 0.22

MAX_LOOP_CANDIDATES = 5

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


# =============================================================================
# PATCH BUILDER
# =============================================================================

class PatchBuilder:

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

        for _, scan, _ in tqdm(
                bag.read_messages(
                    topics=[scan_topic]
                ),
                desc="MBES scans"):

            if hasattr(scan, 'header'):

                scans.append(scan)

        patches = []

        rospy.loginfo(
            "Building patches..."
        )

        for start in tqdm(

                range(
                    0,
                    len(scans) - PATCH_SIZE,
                    PATCH_STRIDE
                ),

                desc="Patch generation"):

            all_points = []

            center_pose = None

            center_ts = None

            for i in range(
                    start,
                    start + PATCH_SIZE):

                scan = scans[i]

                ts = scan.header.stamp.to_sec()

                pose = nav.pose_values(ts)

                if center_pose is None:

                    center_pose = pose

                    center_ts = ts

                pc = ros_numpy.point_cloud2.pointcloud2_to_array(
                    scan
                )

                pc = pc[
                    np.isfinite(pc['x'])
                ]

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

                xyz = xyz @ R_sensor.T

                R_vehicle = tr.euler_matrix(

                    pose["roll"],
                    pose["pitch"],
                    pose["yaw"],

                    axes='sxyz'

                )[:3, :3]

                xyz = xyz @ R_vehicle.T

                xyz[:, 0] += pose["north"]

                xyz[:, 1] += pose["east"]

                xyz[:, 2] += -pose["depth"]

                all_points.append(xyz)

            if len(all_points) == 0:
                continue

            pts = np.vstack(all_points)

            pcd = o3d.geometry.PointCloud()

            pcd.points = (

                o3d.utility.Vector3dVector(
                    pts
                )
            )

            pcd = pcd.voxel_down_sample(
                VOXEL_SIZE
            )

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

    pose_graph.nodes.append(

        o3d.pipelines.registration.PoseGraphNode(
            np.eye(4)
        )
    )

    odometry = np.eye(4)

    monitor = PoseGraphMonitor()

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
        "Sequential registration..."
    )

    for idx in tqdm(

            range(1, len(patches)),

            desc="Sequential ICP"):

        source = patches[idx].pcd

        target = patches[idx - 1].pcd

        T_init = expected_transform(

            patches[idx - 1],

            patches[idx]
        )

        result = robust_icp(

            source,
            target,
            T_init,

            icp_distance=ICP_DISTANCE,

            max_iter=ICP_MAX_ITER
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

        if result is None:

            fallback_count += 1

            rejected_seq += 1

            rospy.logwarn(
                f"ICP failed "
                f"{idx-1}->{idx}"
            )

            T = T_init

            info = np.eye(6) * 0.01

            # Variables de métricas inicializadas a cero:
            # no hay datos reales de ICP en este caso.
            fitness = 0.0
            rmse = 0.0
            correspondences = 0
            valid_registration = False

        # =============================================================
        # ICP SUCCEEDED — validación adaptativa
        # =============================================================

        else:

            fitness = float(
                result.fitness
            )

            rmse = float(
                result.inlier_rmse
            )

            correspondences = len(
                result.correspondence_set
            )

            rospy.loginfo(

                f"[SEQ] {idx-1}->{idx} | "
                f"fitness={fitness:.3f} "
                f"rmse={rmse:.3f} "
                f"corr={correspondences}"
            )

            # =========================================================
            # ADAPTIVE ICP VALIDATION
            # =========================================================

            valid_registration = True

            # ---------------------------------------------------------
            # BASIC VALIDATION
            # ---------------------------------------------------------

            if fitness < FITNESS_THRESHOLD:
                valid_registration = False

            if correspondences < MIN_CORRESPONDENCES:
                valid_registration = False

            # ---------------------------------------------------------
            # STANDARD RMSE VALIDATION
            # ---------------------------------------------------------

            adaptive_rmse_threshold = SEQ_RMSE_THRESHOLD

            # ---------------------------------------------------------
            # HIGH-CONFIDENCE FALLBACK
            # ---------------------------------------------------------
            # In underwater MBES:
            #
            # - planar regions
            # - soft bathymetry
            # - acoustic noise
            #
            # can generate:
            #
            #   fitness ≈ 1.0
            #   huge correspondences
            #   moderate RMSE
            #
            # while still being geometrically correct.
            # ---------------------------------------------------------

            if (
                fitness > HIGH_FITNESS_THRESHOLD
                and
                correspondences > HIGH_CORRESPONDENCE_THRESHOLD
            ):

                adaptive_rmse_threshold *= (
                    HIGH_RMSE_MULTIPLIER
                )

                rospy.loginfo(

                    f"[HIGH CONFIDENCE ICP] "
                    f"Relaxed RMSE threshold: "
                    f"{adaptive_rmse_threshold:.3f}"
                )

            # ---------------------------------------------------------
            # FINAL RMSE VALIDATION
            # ---------------------------------------------------------

            if rmse > adaptive_rmse_threshold:

                valid_registration = False

            # =========================================================
            # VALID ICP
            # =========================================================

            if valid_registration:

                accepted_seq += 1

                T = constrain_transform(
                    result.transformation
                )

                info = dynamic_information_matrix(

                    result,

                    temporal_distance=
                    temporal_distance,

                    loop=False
                )

            # =========================================================
            # FALLBACK NAVIGATION
            # =========================================================

            else:

                fallback_count += 1

                rejected_seq += 1

                rospy.logwarn(

                    f"Sequential ICP rejected "
                    f"{idx-1}->{idx}"
                )

                T = T_init

                info = np.eye(6) * 0.01

        # =============================================================
        # MÉTRICAS — comunes a todos los casos
        # =============================================================

        metrics["sequential"].append({

            "source": idx - 1,

            "target": idx,

            "fitness": float(fitness),

            "rmse": float(rmse),

            "correspondences": int(correspondences),

            "accepted": bool(valid_registration)
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

        pose_graph.edges.append(

            o3d.pipelines.registration.
            PoseGraphEdge(

                idx - 1,
                idx,

                T,

                info,

                uncertain=False
            )
        )

        pose_graph.nodes.append(

            o3d.pipelines.registration.
            PoseGraphNode(
                odometry
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

    for node in pose_graph.nodes:

        raw_trajectory.append(
            node.pose[:3, 3]
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

        raw_map += patch.pcd

    raw_map = raw_map.voxel_down_sample(
        FINAL_DOWNSAMPLE
    )

    # =========================================================================
    # LOOP CLOSURE
    # =========================================================================

    if ENABLE_LOOP_CLOSURE:

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

                source = patches[idx].pcd

                target = patches[cand_idx].pcd

                ransac_result = (

                    execute_global_registration(

                        source,
                        target,

                        VOXEL_SIZE
                    )
                )

                if ransac_result is None:

                    rejected_loops += 1

                    continue

                T_init = (
                    ransac_result.transformation
                )

                result = robust_icp(

                    source,
                    target,

                    T_init,

                    icp_distance=
                    ICP_DISTANCE,

                    max_iter=
                    ICP_MAX_ITER
                )

                if result is None:

                    rejected_loops += 1

                    continue

                fitness = float(
                    result.fitness
                )

                rmse = float(
                    result.inlier_rmse
                )

                correspondences = len(
                    result.correspondence_set
                )

                # =============================================================
                # ADAPTIVE LOOP VALIDATION
                # =============================================================

                valid_loop = True

                if fitness < LOOP_FITNESS_THRESHOLD:
                    valid_loop = False

                if correspondences < MIN_CORRESPONDENCES:
                    valid_loop = False

                adaptive_loop_rmse = LOOP_RMSE_THRESHOLD

                # -------------------------------------------------------------
                # HIGH-CONFIDENCE LOOP
                # -------------------------------------------------------------

                if (
                    fitness > HIGH_FITNESS_THRESHOLD
                    and
                    correspondences > HIGH_CORRESPONDENCE_THRESHOLD
                ):

                    adaptive_loop_rmse *= (
                        HIGH_RMSE_MULTIPLIER
                    )

                    rospy.loginfo(

                        f"[HIGH CONFIDENCE LOOP] "
                        f"Relaxed RMSE threshold: "
                        f"{adaptive_loop_rmse:.3f}"
                    )

                # -------------------------------------------------------------
                # FINAL LOOP VALIDATION
                # -------------------------------------------------------------

                if rmse > adaptive_loop_rmse:

                    valid_loop = False

                if valid_loop:

                    accepted_loops += 1

                    rospy.loginfo(
                        f"LOOP ACCEPTED "
                        f"{idx}<->{cand_idx}"
                    )

                    T = constrain_transform(
                        result.transformation
                    )

                    info = dynamic_information_matrix(

                        result,

                        temporal_distance=1.0,

                        loop=True
                    )

                    pose_graph.edges.append(

                        o3d.pipelines.registration.
                        PoseGraphEdge(

                            idx,
                            cand_idx,

                            T,

                            info,

                            uncertain=True
                        )
                    )

                    monitor.update(
                        pose_graph
                    )

                else:

                    rejected_loops += 1

                metrics["loops"].append({

                    "source": idx,

                    "target": cand_idx,

                    "fitness": float(fitness),

                    "rmse": float(rmse),

                    "correspondences": int(correspondences),

                    "accepted": bool(valid_loop)
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

    metrics["summary"] = {

        "total_nodes":
            len(pose_graph.nodes),

        "total_edges":
            len(pose_graph.edges),

        "accepted_seq":
            accepted_seq,

        "rejected_seq":
            rejected_seq,

        "accepted_loops":
            accepted_loops,

        "rejected_loops":
            rejected_loops,

        "fallback_edges":
            fallback_count,

        "seq_acceptance_ratio":

            accepted_seq /

            max(
                accepted_seq +
                rejected_seq,
                1
            ),

        "loop_acceptance_ratio":

            accepted_loops /

            max(
                accepted_loops +
                rejected_loops,
                1
            )
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

            "fitness",

            "rmse",

            "correspondences",

            "accepted"
        ])

        for m in metrics["sequential"]:

            writer.writerow([

                m["source"],

                m["target"],

                m["fitness"],

                m["rmse"],

                m["correspondences"],

                m["accepted"]
            ])

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