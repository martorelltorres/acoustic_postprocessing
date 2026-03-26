#!/usr/bin/env python3
"""
Multibeam Point Cloud Processing and Surface Reconstruction
UTM referenced version (compatible with sss2mosaic)

Author: Antoni Martorell
Affiliation: Systems, Robotics and Vision Group (SRV),
             University of the Balearic Islands (UIB)
"""

import rospy
import rosbag
import numpy as np
import ros_numpy
import open3d as o3d
import os
import tf.transformations as tr

from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from pyproj import Transformer   
from std_msgs.msg import Bool
import time



# =========================================================
# CRS CONFIG (MISMO QUE sss2mosaic)
# =========================================================
CRS_WGS84 = "EPSG:4326"
CRS_UTM   = "EPSG:32631"
ll_to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)


# =========================================================
# UTILS
# =========================================================
def get_transform_matrix(position, rotation_matrix):
    T = np.identity(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = position
    return T


def get_static_transform_from_tf(bag_file, parent_frame, child_frame):
    try:
        bag = rosbag.Bag(bag_file)
        for _, msg, _ in bag.read_messages(topics=['/tf_static', '/tf']):
            for transform in msg.transforms:
                if (transform.header.frame_id == parent_frame and
                        transform.child_frame_id == child_frame):

                    q = transform.transform.rotation
                    t = transform.transform.translation
                    T = tr.quaternion_matrix([q.x, q.y, q.z, q.w])
                    T[:3, 3] = [t.x, t.y, t.z]
                    bag.close()
                    return T
        bag.close()
    except Exception:
        pass
    return np.identity(4)


def get_nav_origin(bag, nav_topic):
    """MISMA lógica que sss2mosaic.py"""
    for _, msg, _ in bag.read_messages(topics=[nav_topic]):
        if hasattr(msg, 'origin'):
            lat0 = msg.origin.latitude
            lon0 = msg.origin.longitude
            rospy.loginfo(f"✔ Navigation origin: lat={lat0}, lon={lon0}")
            return lat0, lon0
    raise RuntimeError("Navigation origin not found in bag")


def save_pcd(pcd, filename, label):
    if pcd is None or len(pcd.points) == 0:
        return
    rospy.loginfo(f"Saving {label}: {filename}")
    o3d.io.write_point_cloud(filename, pcd, write_ascii=True)


# =========================================================
# MAIN
# =========================================================
def main():
    rospy.init_node('multibeam_processor', anonymous=True)

    bag_file   = rospy.get_param('~bag_file')
    scan_topic = rospy.get_param('~scan_topic')
    nav_topic  = rospy.get_param('~nav_topic')
    output_dir = rospy.get_param('~output_dir')

    base_frame   = rospy.get_param('~base_frame_id', 'sparus2/base_link')
    sensor_frame = rospy.get_param('~sensor_frame_id', 'sparus2/multibeam')

    voxel_size = rospy.get_param('~voxel_size', 0.05)
    sor_k      = rospy.get_param('~sor_k', 50)
    sor_std    = rospy.get_param('~sor_std', 1.0)
    poisson_depth = rospy.get_param('~poisson_depth', 10)

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------
    # OPEN BAG
    # -----------------------------------------------------
    bag = rosbag.Bag(bag_file)

    # -----------------------------------------------------
    # NAV ORIGIN → UTM (CLAVE)
    # -----------------------------------------------------
    lat0, lon0 = get_nav_origin(bag, nav_topic)
    X0_UTM, Y0_UTM = ll_to_utm.transform(lon0, lat0)
    rospy.loginfo(f"✔ UTM origin: X0={X0_UTM:.2f}, Y0={Y0_UTM:.2f}")

    # -----------------------------------------------------
    # SENSOR → VEHICLE TF
    # -----------------------------------------------------
    T_S2V = get_static_transform_from_tf(bag_file, base_frame, sensor_frame)

    # -----------------------------------------------------
    # LOAD NAVIGATION DATA (LOCAL NED)
    # -----------------------------------------------------
    nav_t, nav_pos, nav_quat = [], [], []

    for _, msg, t in bag.read_messages(topics=[nav_topic]):
        nav_t.append(t.to_sec())
        nav_pos.append([
            msg.position.north,
            msg.position.east,
            msg.position.depth
        ])

        r = R.from_euler(
            'xyz',
            [msg.orientation.roll,
             msg.orientation.pitch,
             msg.orientation.yaw]
        )
        nav_quat.append(r.as_quat())

    nav_t = np.array(nav_t)
    interp_pos = interp1d(nav_t, np.array(nav_pos),
                          axis=0, fill_value="extrapolate")
    slerp_rot = Slerp(nav_t, R.from_quat(nav_quat))

    buffer_points = []

    # -----------------------------------------------------
    # MULTIBEAM PROCESSING
    # -----------------------------------------------------
    for _, scan, t in bag.read_messages(topics=[scan_topic]):
        ts = t.to_sec()
        if ts < nav_t[0] or ts > nav_t[-1]:
            continue

        pc = ros_numpy.point_cloud2.pointcloud2_to_array(scan)
        pc = pc[np.isfinite(pc['x'])]

        if len(pc) < 10:
            continue

        xyz = np.column_stack((pc['x'], pc['y'], pc['z']))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # SENSOR → VEHICLE
        pcd.transform(T_S2V)

        # VEHICLE → WORLD (UTM ABSOLUTO)
        pos_local = interp_pos(ts)         # [north, east, depth]
        # rot = slerp_rot(ts).as_matrix()
        rot_nav = slerp_rot(ts).as_matrix()

        # Rotación correctora: 180 grados alrededor de Z
        R_fix = R.from_euler('z', np.pi).as_matrix()

        rot = R_fix @ rot_nav
        pos_utm = np.array([
            X0_UTM + pos_local[1],   # east  → X
            Y0_UTM + pos_local[0],   # north → Y
            -pos_local[2]            # depth → Z (ENU)
        ])

        T_world = get_transform_matrix(pos_utm, rot)
        pcd.transform(T_world)

        buffer_points.append(np.asarray(pcd.points))

    bag.close()

    # -----------------------------------------------------
    # FINAL POINT CLOUD
    # -----------------------------------------------------
    pts_all = np.vstack(buffer_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all)

    pcd, _ = pcd.remove_statistical_outlier(sor_k, sor_std)

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    xyz_file = os.path.join(output_dir, "mb_pointcloud.xyz")
    save_pcd(pcd, xyz_file, "Multibeam XYZ (UTM)")

    # -----------------------------------------------------
    # SURFACE RECONSTRUCTION
    # -----------------------------------------------------
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    rospy.loginfo(f"✔ Centering cloud at {centroid}")

    pts_centered = pts - centroid
    pcd.points = o3d.utility.Vector3dVector(pts_centered)

    rospy.loginfo("Estimating normals...")
    normal_radius = rospy.get_param('~normal_radius', 0.5)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=100
        )
    )

    # Orientar normales hacia arriba (importante para BPA en fondos marinos)
    pcd.orient_normals_to_align_with_direction([0.0, 0.0, 1.0])

    rospy.loginfo(f"Total de puntos en la nube: {len(pcd.points)}")
    
    # 1. Filtro extra para aligerar la nube antes de Poisson (opcional pero recomendado)
    rospy.loginfo("Limpiando puntos atípicos (Outliers)...")
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    rospy.loginfo(f"Puntos tras limpieza: {len(pcd.points)}")

    # 2. Reconstrucción de Poisson con depth seguro
    poisson_depth = rospy.get_param('~poisson_depth', 8) # Bajamos el depth por defecto a 8
    rospy.loginfo(f"Iniciando Poisson con depth={poisson_depth}...")
    
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
    rospy.loginfo("✔ Poisson terminado. Recortando bordes...")

    # 3. Recortar malla
    bbox = pcd.get_axis_aligned_bounding_box()
    mesh = mesh.crop(bbox)
        
    rospy.loginfo("Restaurando coordenadas originales...")
    # CORRECCIÓN: Hacemos la suma normal (no +=) para evitar el error de solo-lectura
    vertices = np.asarray(mesh.vertices) + centroid
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()

    rospy.loginfo("Guardando malla...")
    mesh_file = os.path.join(output_dir, "mb_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_file, mesh)

    pub_mb_done = rospy.Publisher('/pipeline/mb_done', Bool, queue_size=1, latch=True)
    time.sleep(0.5)
    pub_mb_done.publish(True)
    rospy.loginfo("Multibeam completion signal sent.")


if __name__ == '__main__':
    main()
