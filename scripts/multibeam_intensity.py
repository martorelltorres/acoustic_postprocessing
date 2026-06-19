#!/usr/bin/env python3
"""
Multibeam backscatter -> georeferenced mosaic + intensity-tagged cloud.
Same geometry as multibeam_processor.py (axis flip, sensor TF, MB->SSS
lever-arm), so outputs share the UTM frame and fuse with the other products.

Outputs:
  - mb_intensity.tif : top-down backscatter mosaic (GeoTIFF, UTM)
  - mb_intensity.xyz : cloud "X Y Z I" (UTM, raw intensity)

Author: Antoni Martorell (SRV, UIB)
"""

import rospy
import rosbag
import numpy as np
import ros_numpy
import os
import time

import cv2
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import interp1d
from pyproj import Transformer
import tf.transformations as tr
from std_msgs.msg import Bool

# Geographic -> UTM (zone 31N)
CRS_WGS84 = "EPSG:4326"
CRS_UTM   = "EPSG:32631"
ll_to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)


def enhance_data(img_input):
    # Normalize (2-98 pct), despeckle, CLAHE, sharpen -> 8-bit.
    if img_input is None or img_input.size == 0:
        return np.zeros_like(img_input, dtype=np.uint8)

    valid = img_input > 0
    if not np.any(valid):
        return np.zeros_like(img_input, dtype=np.uint8)

    vmin, vmax = np.percentile(img_input[valid], (2, 98))
    vmax = max(vmax, vmin + 1e-6)

    img = np.clip((img_input - vmin) * 255.0 / (vmax - vmin), 0, 255).astype(np.uint8)
    img = cv2.medianBlur(img, 5)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(img, -1, kernel)


def get_static_transform_from_tf(bag_file, parent_frame, child_frame):
    # First parent->child transform found in /tf_static or /tf (4x4).

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


def get_nav_origin(bag, nav_topic):
    # Geographic origin (lat, lon) of the local navigation frame.
    for _, msg, _ in bag.read_messages(topics=[nav_topic]):
        if hasattr(msg, 'origin'):
            return msg.origin.latitude, msg.origin.longitude

    raise RuntimeError("Navigation origin not found")


def main():

    rospy.init_node('multibeam_intensity')

    rospy.loginfo("===== MULTIBEAM INTENSITY PROCESSOR STARTED =====")

    bag_file   = rospy.get_param('~bag_file')
    scan_topic = rospy.get_param('~scan_topic')
    nav_topic  = rospy.get_param('~nav_topic')
    output_dir = rospy.get_param('~output_dir')

    mosaic_res       = rospy.get_param('~mosaic_res', 0.10)
    angle_cutoff_deg = rospy.get_param('~angle_cutoff', 60.0)
    save_cloud       = rospy.get_param('~save_cloud', True)

    os.makedirs(output_dir, exist_ok=True)

    bag = rosbag.Bag(bag_file)

    # UTM origin
    lat0, lon0 = get_nav_origin(bag, nav_topic)
    X0_UTM, Y0_UTM = ll_to_utm.transform(lon0, lat0)

    rospy.loginfo(f"UTM origin: {X0_UTM:.3f}, {Y0_UTM:.3f}")

    # TF chain (same as multibeam_processor.py)
    T_MB = get_static_transform_from_tf(
        bag_file, 'sparus2/base_link', 'sparus2/multibeam')

    T_PORT = get_static_transform_from_tf(
        bag_file, 'sparus2/base_link', 'sparus2/sidescan_port')

    T_STBD = get_static_transform_from_tf(
        bag_file, 'sparus2/base_link', 'sparus2/sidescan_starboard')

    R_sensor = T_MB[:3, :3]
    sensor_offset = T_MB[:3, 3]

    sss_center = 0.5 * (T_PORT[:3, 3] + T_STBD[:3, 3])
    delta_sensor = (sss_center - sensor_offset) + np.array([0.0, -2.0, 0.0])

    rospy.loginfo(f"Lever-arm delta: {delta_sensor}")

    # Navigation: build time interpolators for pose
    ts_nav, north, east, depth, yaw, pitch, roll = [], [], [], [], [], [], []

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
    f_e = interp1d(ts_nav, np.array(east),  bounds_error=False, fill_value=np.nan)
    f_d = interp1d(ts_nav, np.array(depth), bounds_error=False, fill_value=np.nan)

    f_y = interp1d(ts_nav, np.unwrap(np.array(yaw)),   bounds_error=False, fill_value=np.nan)
    f_p = interp1d(ts_nav, np.unwrap(np.array(pitch)), bounds_error=False, fill_value=np.nan)
    f_r = interp1d(ts_nav, np.unwrap(np.array(roll)),  bounds_error=False, fill_value=np.nan)

    # Per-ping processing: sensor frame -> vehicle -> local -> UTM
    rospy.loginfo("Processing multibeam intensity pings...")

    pts_buffer = []   # (N,3) UTM
    int_buffer = []   # (N,)  raw intensity
    count = 0

    for _, scan, _ in bag.read_messages(topics=[scan_topic]):

        if not hasattr(scan, 'header'):
            continue

        count += 1
        if count % 200 == 0:
            rospy.loginfo(f"Pings processed: {count}")

        ts = scan.header.stamp.to_sec()
        if ts < ts_nav[0] or ts > ts_nav[-1]:
            continue

        n = float(f_n(ts)); e = float(f_e(ts)); d = float(f_d(ts))
        yaw_t = float(f_y(ts)); pitch_t = float(f_p(ts)); roll_t = float(f_r(ts))

        if np.isnan(n) or np.isnan(e) or np.isnan(yaw_t):
            continue

        pc = ros_numpy.point_cloud2.pointcloud2_to_array(scan)
        pc = pc[np.isfinite(pc['x'])]
        if len(pc) < 10:
            continue

        intensity = np.array(pc['intensity'], dtype=np.float32)

        # Same axis convention as multibeam_processor.py
        xyz = np.column_stack((pc['x'], -pc['y'], -pc['z'])).astype(np.float64)

        # Angle cutoff on the across-track aperture (drop grazing outer beams)
        r_horizontal = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        depth_s = np.abs(xyz[:, 2])
        angles = np.degrees(np.arctan2(r_horizontal, depth_s))
        keep = angles < angle_cutoff_deg

        xyz = xyz[keep]
        intensity = intensity[keep]
        if len(xyz) < 10:
            continue

        # Sensor rotation
        xyz = xyz @ R_sensor.T

        # Vehicle rotation
        R_veh = tr.euler_matrix(roll_t, pitch_t, yaw_t, axes='sxyz')[:3, :3]
        xyz = xyz @ R_veh.T

        # Lever-arm offsets (identical to bathymetry pipeline)
        offset_world = R_veh @ sensor_offset
        offset_world += R_veh @ delta_sensor

        xyz[:, 0] += offset_world[0]
        xyz[:, 1] += offset_world[1]
        xyz[:, 2] += offset_world[2]

        # Local world (north, east, -depth)
        xyz[:, 0] += n
        xyz[:, 1] += e
        xyz[:, 2] += -d

        # To UTM (X=easting from local-east, Y=northing from local-north)
        pts_world = np.empty_like(xyz)
        pts_world[:, 0] = X0_UTM + xyz[:, 1]
        pts_world[:, 1] = Y0_UTM + xyz[:, 0]
        pts_world[:, 2] = xyz[:, 2]

        pts_buffer.append(pts_world)
        int_buffer.append(intensity)

    bag.close()

    if not pts_buffer:
        rospy.logerr("No valid intensity points")
        return

    pts_all = np.vstack(pts_buffer)
    int_all = np.concatenate(int_buffer)

    rospy.loginfo(f"Total intensity points: {len(pts_all)}")

    # Point cloud "X Y Z I"
    if save_cloud:
        xyz_file = os.path.join(output_dir, "mb_intensity.xyz")
        out = np.column_stack((pts_all, int_all))
        np.savetxt(xyz_file, out, fmt="%.4f %.4f %.4f %.4f")
        rospy.loginfo(f"Intensity cloud saved: {xyz_file}")

    # Georeferenced backscatter mosaic: average intensity per cell
    margin = 2.0
    x_min = pts_all[:, 0].min() - margin
    x_max = pts_all[:, 0].max() + margin
    y_min = pts_all[:, 1].min() - margin
    y_max = pts_all[:, 1].max() + margin

    width  = int(np.ceil((x_max - x_min) / mosaic_res))
    height = int(np.ceil((y_max - y_min) / mosaic_res))

    grid = np.zeros(width * height, dtype=np.float64)
    cnt  = np.zeros(width * height, dtype=np.float64)

    c = ((pts_all[:, 0] - x_min) / mosaic_res).astype(np.int32)
    r = ((y_max - pts_all[:, 1]) / mosaic_res).astype(np.int32)

    mask = (c >= 0) & (c < width) & (r >= 0) & (r < height)
    idx = r[mask] * width + c[mask]

    np.add.at(grid, idx, int_all[mask])
    np.add.at(cnt, idx, 1.0)

    img = np.zeros_like(grid)
    valid_pix = cnt > 0
    img[valid_pix] = grid[valid_pix] / cnt[valid_pix]
    img = img.reshape((height, width)).astype(np.float32)

    img8 = enhance_data(img)

    transform = from_origin(x_min, y_max, mosaic_res, mosaic_res)
    tif_file = os.path.join(output_dir, "mb_intensity.tif")

    with rasterio.open(
        tif_file, 'w', driver='GTiff',
        height=img8.shape[0], width=img8.shape[1],
        count=1, dtype=np.uint8, crs=CRS_UTM,
        transform=transform, compress='deflate'
    ) as dst:
        dst.write(img8, 1)

    rospy.loginfo(f"Backscatter mosaic saved: {tif_file}")

    # Pipeline signal
    pub = rospy.Publisher('/pipeline/mb_intensity_done', Bool, queue_size=1, latch=True)
    time.sleep(0.5)
    pub.publish(True)

    rospy.loginfo("===== MULTIBEAM INTENSITY FINISHED =====")


if __name__ == '__main__':
    main()
