#!/usr/bin/env python3

import os
import json
import copy
import rospy
import rosbag
import ros_numpy
import numpy as np
import open3d as o3d
import tf.transformations as tr

from scipy.interpolate import interp1d
from scipy.spatial import cKDTree


# =============================================================================
# CONFIGURATION
# =============================================================================

PATCH_SIZE = 60
PATCH_STRIDE = 30

VOXEL_SIZE = 0.08

FINAL_DOWNSAMPLE = 0.05

MIN_PATCH_POINTS = 500

ANGLE_CUTOFF_DEG = 50.0

# =============================================================================
# ICP PARAMETERS
# =============================================================================

ICP_MAX_DISTANCE_COARSE = 1.5
ICP_MAX_DISTANCE_MEDIUM = 0.8
ICP_MAX_DISTANCE_FINE = 0.25

ICP_ITERATIONS_COARSE = 80
ICP_ITERATIONS_MEDIUM = 60
ICP_ITERATIONS_FINE = 40

# =============================================================================
# ICP VALIDATION
# =============================================================================

FITNESS_THRESHOLD = 0.45

RMSE_THRESHOLD = 0.20

MIN_CORRESPONDENCES = 500

# =============================================================================
# GEOMETRIC CONSTRAINTS
# =============================================================================

SPATIAL_SEARCH_RADIUS = 10.0

MAX_HEADING_DIFFERENCE_DEG = 35.0

TURN_THRESHOLD_DEG = 50.0

# =============================================================================
# CONSERVATIVE ICP CORRECTION
# =============================================================================

MAX_YAW_CORRECTION_DEG = 3.0

MAX_TRANSLATION_CORRECTION = 0.5


# =============================================================================
# UTILS
# =============================================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def wrap_angle_deg(angle):

    while angle > 180:
        angle -= 360

    while angle < -180:
        angle += 360

    return angle


def constrain_transform(T):

    yaw = np.arctan2(
        T[1, 0],
        T[0, 0]
    )

    yaw_deg = np.degrees(yaw)

    yaw_deg = np.clip(
        yaw_deg,
        -MAX_YAW_CORRECTION_DEG,
        MAX_YAW_CORRECTION_DEG
    )

    tx = np.clip(
        T[0, 3],
        -MAX_TRANSLATION_CORRECTION,
        MAX_TRANSLATION_CORRECTION
    )

    ty = np.clip(
        T[1, 3],
        -MAX_TRANSLATION_CORRECTION,
        MAX_TRANSLATION_CORRECTION
    )

    T_new = np.eye(4)

    T_new[:3, :3] = tr.euler_matrix(
        0,
        0,
        np.deg2rad(yaw_deg)
    )[:3, :3]

    T_new[0, 3] = tx
    T_new[1, 3] = ty

    return T_new


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

            if (transform.header.frame_id == parent_frame and
                    transform.child_frame_id == child_frame):

                q = transform.transform.rotation
                t = transform.transform.translation

                T = tr.quaternion_matrix(
                    [q.x, q.y, q.z, q.w]
                )

                T[:3, 3] = [t.x, t.y, t.z]

                bag.close()

                return T

    bag.close()

    return np.eye(4)


# =============================================================================
# NAVIGATION
# =============================================================================

class NavigationInterpolator:

    def __init__(self, bag, nav_topic):

        self.timestamps = []

        north = []
        east = []
        depth = []

        yaw = []
        pitch = []
        roll = []

        for _, msg, _ in bag.read_messages(
                topics=[nav_topic]):

            self.timestamps.append(
                msg.header.stamp.to_sec()
            )

            north.append(msg.position.north)
            east.append(msg.position.east)
            depth.append(msg.position.depth)

            yaw.append(msg.orientation.yaw)
            pitch.append(msg.orientation.pitch)
            roll.append(msg.orientation.roll)

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

    def pose_values(self, ts):

        return {

            "north": float(self.f_n(ts)),
            "east": float(self.f_e(ts)),
            "depth": float(self.f_d(ts)),

            "yaw": float(self.f_y(ts)),
            "pitch": float(self.f_p(ts)),
            "roll": float(self.f_r(ts))
        }


# =============================================================================
# PATCH
# =============================================================================

class Patch:

    def __init__(self):

        self.idx = None

        self.pcd = None

        self.center = None

        self.motion_heading_deg = None

        self.is_turn = False


# =============================================================================
# PATCH BUILDER
# =============================================================================

class PatchBuilder:

    def build_patches(
            self,
            bag,
            scan_topic,
            nav,
            R_sensor):

        scans = []

        for _, scan, _ in bag.read_messages(
                topics=[scan_topic]):

            if not hasattr(scan, 'header'):
                continue

            ts = scan.header.stamp.to_sec()

            scans.append((ts, scan))

        patches = []

        previous_center = None

        previous_motion_heading = None

        for start in range(
                0,
                len(scans) - PATCH_SIZE,
                PATCH_STRIDE):

            current_points = []

            for i in range(
                    start,
                    start + PATCH_SIZE):

                ts, scan = scans[i]

                pose = nav.pose_values(ts)

                pc = ros_numpy.point_cloud2.pointcloud2_to_array(
                    scan
                )

                pc = pc[np.isfinite(pc['x'])]

                if len(pc) < 10:
                    continue

                xyz = np.column_stack((

                    pc['x'],
                    -pc['y'],
                    -pc['z']
                ))

                # =====================================================
                # ANGLE FILTER
                # =====================================================

                r_horizontal = np.sqrt(
                    xyz[:, 0]**2 +
                    xyz[:, 1]**2
                )

                depth_s = np.abs(xyz[:, 2])

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
                # SENSOR TF
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
                # NAVIGATION FRAME
                # =====================================================

                xyz[:, 0] += pose["north"]
                xyz[:, 1] += pose["east"]
                xyz[:, 2] += -pose["depth"]

                current_points.append(xyz)

            if len(current_points) == 0:
                continue

            pts = np.vstack(current_points)

            patch = Patch()

            patch.idx = len(patches)

            patch.center = np.mean(
                pts[:, :2],
                axis=0
            )

            # =========================================================
            # MOTION DIRECTION
            # =========================================================

            if previous_center is not None:

                dx = (
                    patch.center[0] -
                    previous_center[0]
                )

                dy = (
                    patch.center[1] -
                    previous_center[1]
                )

                motion_heading = np.degrees(
                    np.arctan2(dy, dx)
                )

                patch.motion_heading_deg = motion_heading

                # =====================================================
                # TURN DETECTION
                # =====================================================

                if previous_motion_heading is not None:

                    delta = wrap_angle_deg(
                        motion_heading -
                        previous_motion_heading
                    )

                    if abs(delta) > TURN_THRESHOLD_DEG:

                        patch.is_turn = True

                previous_motion_heading = motion_heading

            previous_center = patch.center

            pcd = o3d.geometry.PointCloud()

            pcd.points = (
                o3d.utility.Vector3dVector(pts)
            )

            patch.pcd = pcd

            patches.append(patch)

        return patches


# =============================================================================
# PROCESSOR
# =============================================================================

class Processor:

    def preprocess(self, pcd):

        pcd = copy.deepcopy(pcd)

        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.5
        )

        pcd = pcd.voxel_down_sample(
            VOXEL_SIZE
        )

        pcd.estimate_normals(

            o3d.geometry.KDTreeSearchParamHybrid(

                radius=VOXEL_SIZE * 3.0,
                max_nn=30
            )
        )

        return pcd


# =============================================================================
# OVERLAP ESTIMATION
# =============================================================================

def estimate_overlap(
        source,
        target,
        threshold=0.5):

    src_pts = np.asarray(
        source.points
    )

    tgt_pts = np.asarray(
        target.points
    )

    if len(src_pts) == 0:
        return 0.0

    tree = cKDTree(
        tgt_pts[:, :2]
    )

    distances, _ = tree.query(
        src_pts[:, :2],
        k=1
    )

    overlap = np.mean(
        distances < threshold
    )

    return overlap


# =============================================================================
# ICP
# =============================================================================

def multiscale_icp(
        source,
        target):

    current = np.eye(4)

    scales = [

        [
            VOXEL_SIZE * 4.0,
            ICP_MAX_DISTANCE_COARSE,
            ICP_ITERATIONS_COARSE
        ],

        [
            VOXEL_SIZE * 2.0,
            ICP_MAX_DISTANCE_MEDIUM,
            ICP_ITERATIONS_MEDIUM
        ],

        [
            VOXEL_SIZE,
            ICP_MAX_DISTANCE_FINE,
            ICP_ITERATIONS_FINE
        ]
    ]

    final = None

    for voxel, dist, iterations in scales:

        source_d = source.voxel_down_sample(
            voxel
        )

        target_d = target.voxel_down_sample(
            voxel
        )

        source_d.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel * 3.0,
                max_nn=30
            )
        )

        target_d.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel * 3.0,
                max_nn=30
            )
        )

        result = (
            o3d.pipelines.registration.registration_icp(

                source_d,
                target_d,

                dist,

                current,

                o3d.pipelines.registration.
                TransformationEstimationPointToPlane(),

                o3d.pipelines.registration.
                ICPConvergenceCriteria(
                    max_iteration=iterations
                )
            )
        )

        current = result.transformation

        final = result

    return final


# =============================================================================
# MAIN
# =============================================================================

def main():

    rospy.init_node(
        "underwater_geometric_icp"
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

    # ================================================================
    # SENSOR TF
    # ================================================================

    T_mb = get_static_transform_from_tf(

        bag_file,

        "sparus2/base_link",

        "sparus2/multibeam"
    )

    R_sensor = T_mb[:3, :3]

    # ================================================================
    # NAVIGATION
    # ================================================================

    nav = NavigationInterpolator(
        bag,
        nav_topic
    )

    # ================================================================
    # BUILD PATCHES
    # ================================================================

    builder = PatchBuilder()

    patches = builder.build_patches(

        bag,
        scan_topic,
        nav,
        R_sensor
    )

    bag.close()

    rospy.loginfo(
        f"Generated {len(patches)} patches"
    )

    processor = Processor()

    raw_map = o3d.geometry.PointCloud()

    icp_map = o3d.geometry.PointCloud()

    metrics = []

    # ================================================================
    # FIRST PATCH
    # ================================================================

    first = processor.preprocess(
        patches[0].pcd
    )

    raw_map += copy.deepcopy(first)

    icp_map += copy.deepcopy(first)

    registered_patches = [

        {
            "idx": 0,
            "center": patches[0].center,
            "heading": patches[0].motion_heading_deg,
            "pcd": first
        }
    ]

    # ================================================================
    # MAIN LOOP
    # ================================================================

    for idx in range(1, len(patches)):

        patch = patches[idx]

        rospy.loginfo(
            f"Patch {idx}"
        )

        source = processor.preprocess(
            patch.pcd
        )

        if len(source.points) < MIN_PATCH_POINTS:

            rospy.logwarn(
                "Too few points"
            )

            continue

        raw_map += copy.deepcopy(source)

        # ============================================================
        # TURN DETECTION
        # ============================================================

        if patch.is_turn:

            rospy.logwarn(
                "TURN DETECTED"
            )

            icp_map += source

            # IMPORTANT:
            # STILL UPDATE SUBMAP
            registered_patches.append({

                "idx": idx,
                "center": patch.center,
                "heading": patch.motion_heading_deg,
                "pcd": source
            })

            continue

        # ============================================================
        # SEARCH COMPATIBLE PATCHES
        # ============================================================

        nearby = []

        for rp in registered_patches:

            dist = np.linalg.norm(
                patch.center -
                rp["center"]
            )

            if dist > SPATIAL_SEARCH_RADIUS:
                continue

            if patch.motion_heading_deg is not None:

                if rp["heading"] is not None:

                    heading_diff = abs(

                        wrap_angle_deg(

                            patch.motion_heading_deg -
                            rp["heading"]
                        )
                    )

                    if heading_diff > MAX_HEADING_DIFFERENCE_DEG:
                        continue

            nearby.append(rp["pcd"])

        if len(nearby) == 0:

            rospy.logwarn(
                "No compatible nearby patches"
            )

            icp_map += source

            # IMPORTANT:
            # STILL UPDATE SUBMAP
            registered_patches.append({

                "idx": idx,
                "center": patch.center,
                "heading": patch.motion_heading_deg,
                "pcd": source
            })

            continue

        # ============================================================
        # BUILD TARGET
        # ============================================================

        target = o3d.geometry.PointCloud()

        for p in nearby:
            target += p

        # ============================================================
        # OVERLAP (ONLY METRIC)
        # ============================================================

        overlap = estimate_overlap(
            source,
            target
        )

        rospy.loginfo(
            f"Overlap={overlap:.3f}"
        )

        # ============================================================
        # ICP
        # ============================================================

        result = multiscale_icp(
            source,
            target
        )

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
            f"Fitness={fitness:.3f} "
            f"RMSE={rmse:.3f} "
            f"Corr={correspondences}"
        )

        metrics.append({

            "patch_idx": idx,
            "fitness": fitness,
            "rmse": rmse,
            "correspondences": correspondences,
            "overlap": float(overlap),
            "motion_heading_deg": (
                float(patch.motion_heading_deg)
                if patch.motion_heading_deg is not None
                else None
            ),
            "turn_detected": patch.is_turn
        })

        # ============================================================
        # VALIDATION
        # ============================================================

        accepted = True

        if fitness < FITNESS_THRESHOLD:
            accepted = False

        if rmse > RMSE_THRESHOLD:
            accepted = False

        if correspondences < MIN_CORRESPONDENCES:
            accepted = False

        if accepted:

            rospy.loginfo(
                "ICP ACCEPTED"
            )

            T_corr = constrain_transform(
                result.transformation
            )

            source.transform(T_corr)

        else:

            rospy.logwarn(
                "ICP REJECTED"
            )

        # ============================================================
        # ALWAYS UPDATE SUBMAP
        # ============================================================

        icp_map += source

        registered_patches.append({

            "idx": idx,
            "center": patch.center,
            "heading": patch.motion_heading_deg,
            "pcd": source
        })

    # ================================================================
    # FINAL DOWNSAMPLE
    # ================================================================

    raw_map = raw_map.voxel_down_sample(
        FINAL_DOWNSAMPLE
    )

    icp_map = icp_map.voxel_down_sample(
        FINAL_DOWNSAMPLE
    )

    # ================================================================
    # EXPORT MAPS
    # ================================================================

    o3d.io.write_point_cloud(

        os.path.join(
            output_dir,
            "raw_navigation_map.ply"
        ),

        raw_map
    )

    o3d.io.write_point_cloud(

        os.path.join(
            output_dir,
            "icp_corrected_map.ply"
        ),

        icp_map
    )

    # ================================================================
    # EXPORT METRICS
    # ================================================================

    metrics_output = os.path.join(
        metrics_dir,
        "icp_metrics.json"
    )

    with open(metrics_output, "w") as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    rospy.loginfo("================================================")
    rospy.loginfo("PROCESS FINISHED")
    rospy.loginfo("================================================")


if __name__ == "__main__":

    main()