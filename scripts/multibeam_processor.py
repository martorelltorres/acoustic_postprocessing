#!/usr/bin/env python3
"""
Multibeam Point Cloud Processing and Surface Reconstruction
Fully aligned with sss2mosaic.py + automatic MB→SSS lever-arm correction

Author: Antoni Martorell
"""

import rospy
import rosbag
import numpy as np
import ros_numpy
import open3d as o3d
import os
import tf.transformations as tr

from scipy.interpolate import interp1d
from pyproj import Transformer
from std_msgs.msg import Bool
import time

# =========================================================
# CRS CONFIG
# =========================================================
CRS_WGS84 = "EPSG:4326"
CRS_UTM   = "EPSG:32631"
ll_to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)

# =========================================================
# TF
# =========================================================
def get_static_transform_from_tf(bag_file, parent_frame, child_frame):

    bag = rosbag.Bag(bag_file)

    for _, msg, _ in bag.read_messages(topics=['/tf_static', '/tf']):
        for transform in msg.transforms:

            if transform.header.frame_id == parent_frame and transform.child_frame_id == child_frame:

                q = transform.transform.rotation
                t = transform.transform.translation

                T = tr.quaternion_matrix([q.x, q.y, q.z, q.w])
                T[:3, 3] = [t.x, t.y, t.z]

                bag.close()
                return T

    bag.close()
    return np.identity(4)

# =========================================================
# NAV ORIGIN
# =========================================================
def get_nav_origin(bag, nav_topic):

    for _, msg, _ in bag.read_messages(topics=[nav_topic]):
        if hasattr(msg, 'origin'):
            return msg.origin.latitude, msg.origin.longitude

    raise RuntimeError("Navigation origin not found")

# =========================================================
# MAIN
# =========================================================
def main():

    rospy.init_node('multibeam_processor')

    rospy.loginfo("===== MULTIBEAM PROCESSOR STARTED =====")

    bag_file   = rospy.get_param('~bag_file')
    scan_topic = rospy.get_param('~scan_topic')
    nav_topic  = rospy.get_param('~nav_topic')
    output_dir = rospy.get_param('~output_dir')

    voxel_size = rospy.get_param('~voxel_size', 0.05)
    sor_k      = rospy.get_param('~sor_k', 50)
    sor_std    = rospy.get_param('~sor_std', 1.0)
    angle_cutoff_deg = rospy.get_param('~angle_cutoff', 60.0)

    os.makedirs(output_dir, exist_ok=True)

    bag = rosbag.Bag(bag_file)

    # =====================================================
    # ORIGIN
    # =====================================================
    lat0, lon0 = get_nav_origin(bag, nav_topic)
    X0_UTM, Y0_UTM = ll_to_utm.transform(lon0, lat0)

    rospy.loginfo(f"UTM origin: {X0_UTM:.3f}, {Y0_UTM:.3f}")

    # =====================================================
    # TF MULTIBEAM
    # =====================================================
    T_MB = get_static_transform_from_tf(
        bag_file,
        'sparus2/base_link',
        'sparus2/multibeam'
    )

    # =====================================================
    # TF SIDESCAN
    # =====================================================
    T_PORT = get_static_transform_from_tf(
        bag_file,
        'sparus2/base_link',
        'sparus2/sidescan_port'
    )

    T_STBD = get_static_transform_from_tf(
        bag_file,
        'sparus2/base_link',
        'sparus2/sidescan_starboard'
    )

    R_sensor = T_MB[:3, :3]
    sensor_offset = T_MB[:3, 3]

    # =====================================================
    # AUTOMATIC MB→SSS LEVER ARM
    # =====================================================
    sss_center = 0.5 * (T_PORT[:3, 3] + T_STBD[:3, 3])

    delta_sensor = (sss_center - sensor_offset) + np.array([0.0, -2, 0.0])

    rospy.loginfo(f"MB offset      : {sensor_offset}")
    rospy.loginfo(f"SSS center     : {sss_center}")
    rospy.loginfo(f"Lever-arm delta: {delta_sensor}")

    # =====================================================
    # NAVIGATION
    # =====================================================
    ts_nav = []
    north = []
    east = []
    depth = []
    yaw = []
    pitch = []
    roll = []

    for _, msg, _ in bag.read_messages(topics=[nav_topic]):

        ts_nav.append(msg.header.stamp.to_sec())

        north.append(msg.position.north)
        east.append(msg.position.east)
        depth.append(msg.position.depth)

        yaw.append(msg.orientation.yaw)
        pitch.append(msg.orientation.pitch)
        roll.append(msg.orientation.roll)

    ts_nav = np.array(ts_nav)

    f_n = interp1d(ts_nav, np.array(north), bounds_error=False, fill_value=np.nan)
    f_e = interp1d(ts_nav, np.array(east), bounds_error=False, fill_value=np.nan)
    f_d = interp1d(ts_nav, np.array(depth), bounds_error=False, fill_value=np.nan)

    f_y = interp1d(ts_nav, np.unwrap(np.array(yaw)), bounds_error=False, fill_value=np.nan)
    f_p = interp1d(ts_nav, np.unwrap(np.array(pitch)), bounds_error=False, fill_value=np.nan)
    f_r = interp1d(ts_nav, np.unwrap(np.array(roll)), bounds_error=False, fill_value=np.nan)

    # =====================================================
    # PROCESS
    # =====================================================
    rospy.loginfo("Processing multibeam pings...")

    buffer_points = []
    count = 0

    for _, scan, _ in bag.read_messages(topics=[scan_topic]):

        if not hasattr(scan, 'header'):
            continue

        count += 1

        if count % 100 == 0:
            rospy.loginfo(f"Pings processed: {count}")

        ts = scan.header.stamp.to_sec()

        if ts < ts_nav[0] or ts > ts_nav[-1]:
            continue

        n = float(f_n(ts))
        e = float(f_e(ts))
        d = float(f_d(ts))

        yaw_t = float(f_y(ts))
        pitch_t = float(f_p(ts))
        roll_t = float(f_r(ts))

        pc = ros_numpy.point_cloud2.pointcloud2_to_array(scan)
        pc = pc[np.isfinite(pc['x'])]

        if len(pc) < 10:
            continue

        xyz = np.column_stack((pc['x'], -pc['y'], -pc['z']))

        r_horizontal = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2)
        depth_s = np.abs(xyz[:,2])

        angles = np.degrees(np.arctan2(r_horizontal, depth_s))
        xyz = xyz[angles < angle_cutoff_deg]

        if len(xyz) < 10:
            continue

        # =================================================
        # SENSOR ROTATION
        # =================================================
        xyz = xyz @ R_sensor.T

        # =================================================
        # VEHICLE ROTATION
        # =================================================
        R_veh = tr.euler_matrix(
            roll_t,
            pitch_t,
            yaw_t,
            axes='sxyz'
        )[:3, :3]

        xyz = xyz @ R_veh.T

        # =================================================
        # OFFSET identical to SSS
        # =================================================
        offset_world = R_veh @ sensor_offset

        delta_world = R_veh @ delta_sensor

        offset_world += delta_world

        xyz[:,0] += offset_world[0]
        xyz[:,1] += offset_world[1]
        xyz[:,2] += offset_world[2]

        # =================================================
        # LOCAL WORLD
        # =================================================
        xyz[:,0] += n
        xyz[:,1] += e
        xyz[:,2] += -d

        # =================================================
        # UTM
        # =================================================
        pts_world = np.zeros_like(xyz)

        pts_world[:,0] = X0_UTM + xyz[:,1]
        pts_world[:,1] = Y0_UTM + xyz[:,0]
        pts_world[:,2] = xyz[:,2]

        buffer_points.append(pts_world)

    bag.close()

    rospy.loginfo(f"Valid pings: {len(buffer_points)}")

    if not buffer_points:
        rospy.logerr("No valid points")
        return

    # =====================================================
    # POINT CLOUD
    # =====================================================
    pts_all = np.vstack(buffer_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all)

    pcd, _ = pcd.remove_statistical_outlier(sor_k, sor_std)
    pcd = pcd.voxel_down_sample(voxel_size)

    xyz_file = os.path.join(output_dir, "mb_pointcloud.xyz")
    o3d.io.write_point_cloud(xyz_file, pcd, write_ascii=True)

    rospy.loginfo(f"XYZ saved: {xyz_file}")

    # =====================================================
    # MESH
    # =====================================================
    pts = np.asarray(pcd.points)

    centroid = pts.mean(axis=0)

    pts_centered = pts - centroid
    pcd.points = o3d.utility.Vector3dVector(pts_centered)

    normal_radius = voxel_size * 3.0

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=80
        )
    )

    pcd.orient_normals_consistent_tangent_plane(50)
    pcd.orient_normals_to_align_with_direction([0,0,1])

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=11
    )

    densities = np.asarray(densities)

    threshold = np.percentile(densities, 5)

    mesh.remove_vertices_by_mask(densities < threshold)

    vertices = np.asarray(mesh.vertices) + centroid
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    mesh.compute_vertex_normals()

    mesh_file = os.path.join(output_dir, "mb_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_file, mesh)

    rospy.loginfo(f"Mesh saved: {mesh_file}")

    pub_mb_done = rospy.Publisher('/pipeline/mb_done', Bool, queue_size=1, latch=True)

    time.sleep(0.5)
    pub_mb_done.publish(True)

    rospy.loginfo("===== MULTIBEAM FINISHED =====")

if __name__ == '__main__':
    main()