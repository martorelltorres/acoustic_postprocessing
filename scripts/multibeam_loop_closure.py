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

print("================================================")
print("Open3D version:", o3d.__version__)
print("================================================")

# =============================================================================
# CONFIGURATION
# =============================================================================

PATCH_SIZE = 80
PATCH_STRIDE = 10

VOXEL_SIZE = 0.30
FINAL_DOWNSAMPLE = 0.15

ANGLE_CUTOFF_DEG = 50.0

MIN_PATCH_POINTS = 500

# =============================================================================
# ICP
# =============================================================================

ICP_DISTANCE = 2.5
ICP_MAX_ITER = 50

FITNESS_THRESHOLD = 0.55
RMSE_THRESHOLD = 0.30

MIN_CORRESPONDENCES = 100

# =============================================================================
# LOOP CLOSURE
# =============================================================================

ENABLE_LOOP_CLOSURE = True

LOOP_MIN_DISTANCE = 5.0
LOOP_MAX_DISTANCE = 25.0

LOOP_MIN_TEMPORAL_DISTANCE = 30

LOOP_FITNESS_THRESHOLD = 0.70

MAX_LOOP_CANDIDATES = 5

# =============================================================================
# POSE GRAPH
# =============================================================================

POSE_GRAPH_OPTIMIZATION = True

# =============================================================================
# GEOMETRIC CONSTRAINTS
# =============================================================================

MAX_HEADING_DIFFERENCE_DEG = 35.0

MAX_YAW_CORRECTION_DEG = 5.0

MAX_TRANSLATION_CORRECTION = 1.0


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
        self.center = None
        self.heading_deg = None
        self.pcd = None


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

        for _, scan, _ in bag.read_messages(
                topics=[scan_topic]):

            if hasattr(scan, 'header'):
                scans.append(scan)

        patches = []

        previous_center = None

        for start in range(
                0,
                len(scans) - PATCH_SIZE,
                PATCH_STRIDE):

            all_points = []

            for i in range(
                    start,
                    start + PATCH_SIZE):

                scan = scans[i]

                ts = scan.header.stamp.to_sec()

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
                o3d.utility.Vector3dVector(pts)
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

            if previous_center is not None:

                dx = (
                    patch.center[0] -
                    previous_center[0]
                )

                dy = (
                    patch.center[1] -
                    previous_center[1]
                )

                patch.heading_deg = np.degrees(
                    np.arctan2(dy, dx)
                )

            previous_center = patch.center

            patches.append(patch)

        return patches


# =============================================================================
# ROBUST ICP
# =============================================================================

def robust_icp(source, target):

    voxel_scales = [1.0, 0.5, 0.25]

    max_corr = [
        ICP_DISTANCE * 2.0,
        ICP_DISTANCE,
        ICP_DISTANCE * 0.5
    ]

    current_transform = np.eye(4)

    final_result = None

    has_gicp = hasattr(
        o3d.pipelines.registration,
        'registration_generalized_icp'
    )

    for voxel, dist in zip(voxel_scales, max_corr):

        source_down = source.voxel_down_sample(voxel)
        target_down = target.voxel_down_sample(voxel)

        if len(source_down.points) < 50:
            continue

        if len(target_down.points) < 50:
            continue

        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel * 3.0,
                max_nn=30
            )
        )

        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel * 3.0,
                max_nn=30
            )
        )

        try:

            if has_gicp:

                result = (
                    o3d.pipelines.registration.
                    registration_generalized_icp(

                        source_down,
                        target_down,

                        dist,

                        current_transform,

                        o3d.pipelines.registration.
                        TransformationEstimationForGeneralizedICP(),

                        o3d.pipelines.registration.
                        ICPConvergenceCriteria(
                            max_iteration=ICP_MAX_ITER
                        )
                    )
                )

            else:

                result = (
                    o3d.pipelines.registration.
                    registration_icp(

                        source_down,
                        target_down,

                        dist,

                        current_transform,

                        o3d.pipelines.registration.
                        TransformationEstimationPointToPlane(),

                        o3d.pipelines.registration.
                        ICPConvergenceCriteria(
                            max_iteration=ICP_MAX_ITER
                        )
                    )
                )

            current_transform = result.transformation

            final_result = result

        except Exception as e:

            rospy.logwarn(str(e))

    return final_result


# =============================================================================
# MAIN
# =============================================================================

def main():

    rospy.init_node(
        "underwater_pose_graph_slam"
    )

    bag_file = rospy.get_param("~bag_file")
    scan_topic = rospy.get_param("~scan_topic")
    nav_topic = rospy.get_param("~nav_topic")
    output_dir = rospy.get_param("~output_dir")

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

    pose_graph = (
        o3d.pipelines.registration.PoseGraph()
    )

    pose_graph.nodes.append(

        o3d.pipelines.registration.PoseGraphNode(
            np.eye(4)
        )
    )

    odometry = np.eye(4)

    metrics = []

    # =========================================================================
    # SEQUENTIAL REGISTRATION
    # =========================================================================

    for idx in range(1, len(patches)):

        rospy.loginfo(
            f"Sequential ICP patch {idx}"
        )

        source = patches[idx].pcd
        target = patches[idx - 1].pcd

        result = robust_icp(
            source,
            target
        )

        # =========================================================
        # ALWAYS ADD NODE
        # =========================================================

        if result is None:

            pose_graph.nodes.append(

                o3d.pipelines.registration.PoseGraphNode(
                    np.linalg.inv(odometry)
                )
            )

            continue

        fitness = float(result.fitness)
        rmse = float(result.inlier_rmse)

        correspondences = len(
            result.correspondence_set
        )

        rospy.loginfo(

            f"Fitness={fitness:.3f} "
            f"RMSE={rmse:.3f} "
            f"Corr={correspondences}"
        )

        valid_registration = True

        if fitness < FITNESS_THRESHOLD:
            valid_registration = False

        if rmse > RMSE_THRESHOLD:
            valid_registration = False

        if correspondences < MIN_CORRESPONDENCES:
            valid_registration = False

        if valid_registration:

            T = constrain_transform(
                result.transformation
            )

            odometry = T @ odometry

            information = (
                o3d.pipelines.registration.
                get_information_matrix_from_point_clouds(

                    source,
                    target,

                    ICP_DISTANCE,

                    T
                )
            )

            pose_graph.edges.append(

                o3d.pipelines.registration.PoseGraphEdge(

                    idx - 1,
                    idx,

                    T,

                    information,

                    uncertain=False
                )
            )

        # =========================================================
        # ALWAYS APPEND NODE
        # =========================================================

        pose_graph.nodes.append(

            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(odometry)
            )
        )

        metrics.append({

            "patch_idx": idx,
            "fitness": fitness,
            "rmse": rmse,
            "correspondences": correspondences,
            "accepted": valid_registration
        })

    # =========================================================================
    # LOOP CLOSURE
    # =========================================================================

    if ENABLE_LOOP_CLOSURE:

        rospy.loginfo(
            "Searching loop closures..."
        )

        for i in range(
                0,
                len(patches),
                5):

            pi = patches[i]

            candidates = []

            for j in range(
                    i + LOOP_MIN_TEMPORAL_DISTANCE,
                    len(patches)):

                pj = patches[j]

                dist = np.linalg.norm(
                    pi.center - pj.center
                )

                if dist < LOOP_MIN_DISTANCE:
                    continue

                if dist > LOOP_MAX_DISTANCE:
                    continue

                if pi.heading_deg is not None and \
                   pj.heading_deg is not None:

                    delta_heading = abs(

                        wrap_angle_deg(

                            pi.heading_deg -
                            pj.heading_deg
                        )
                    )

                    if delta_heading > MAX_HEADING_DIFFERENCE_DEG:
                        continue

                candidates.append((j, dist))

            candidates = sorted(
                candidates,
                key=lambda x: x[1]
            )

            candidates = candidates[
                :MAX_LOOP_CANDIDATES
            ]

            rospy.loginfo(
                f"Patch {i}: "
                f"{len(candidates)} candidates"
            )

            for j, dist in candidates:

                pj = patches[j]

                rospy.loginfo(
                    f"Loop candidate {i} <-> {j}"
                )

                result = robust_icp(
                    pi.pcd,
                    pj.pcd
                )

                if result is None:
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

                if fitness < LOOP_FITNESS_THRESHOLD:
                    continue

                if rmse > RMSE_THRESHOLD:
                    continue

                if correspondences < MIN_CORRESPONDENCES:
                    continue

                rospy.loginfo(
                    f"LOOP ACCEPTED "
                    f"{i} <-> {j}"
                )

                T = constrain_transform(
                    result.transformation
                )

                information = (
                    o3d.pipelines.registration.
                    get_information_matrix_from_point_clouds(

                        pi.pcd,
                        pj.pcd,

                        ICP_DISTANCE,

                        T
                    )
                )

                pose_graph.edges.append(

                    o3d.pipelines.registration.PoseGraphEdge(

                        i,
                        j,

                        T,

                        information,

                        uncertain=True
                    )
                )

    # =========================================================================
    # GLOBAL OPTIMIZATION
    # =========================================================================

    if POSE_GRAPH_OPTIMIZATION:

        rospy.loginfo(
            "Global pose graph optimization..."
        )

        option = (

            o3d.pipelines.registration.
            GlobalOptimizationOption(

                max_correspondence_distance=ICP_DISTANCE,

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

    # =========================================================================
    # BUILD FINAL MAP
    # =========================================================================

    rospy.loginfo(
        "Building final map..."
    )

    final_map = o3d.geometry.PointCloud()
    raw_map = o3d.geometry.PointCloud()

    for idx, patch in enumerate(patches):

        raw_map += patch.pcd

        transformed = copy.deepcopy(
            patch.pcd
        )

        pose = pose_graph.nodes[idx].pose

        transformed.transform(pose)

        final_map += transformed

    raw_map = raw_map.voxel_down_sample(
        FINAL_DOWNSAMPLE
    )

    final_map = final_map.voxel_down_sample(
        FINAL_DOWNSAMPLE
    )

    raw_output = os.path.join(
        output_dir,
        "raw_navigation_map.ply"
    )

    icp_output = os.path.join(
        output_dir,
        "pose_graph_slam_map.ply"
    )

    metrics_output = os.path.join(
        metrics_dir,
        "icp_metrics.json"
    )

    rospy.loginfo(
        f"Saving RAW map: {raw_output}"
    )

    o3d.io.write_point_cloud(
        raw_output,
        raw_map
    )

    rospy.loginfo(
        f"Saving SLAM map: {icp_output}"
    )

    o3d.io.write_point_cloud(
        icp_output,
        final_map
    )

    with open(metrics_output, "w") as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    rospy.loginfo("================================================")
    rospy.loginfo("SLAM FINISHED")
    rospy.loginfo("================================================")


if __name__ == "__main__":

    main()