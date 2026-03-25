#!/usr/bin/env python3
"""
Author: Antoni Martorell
Affiliation: Systems, Robotics and Vision Group (SRV),
             University of the Balearic Islands (UIB)
Contact: antoni.martorell@uib.es
License: This code is provided for research and academic purposes.
"""

import rospy  # ADDED FOR ROS
import rosbag
import numpy as np
import cv2
import os     # ADDED TO HANDLE PATHS
from scipy.interpolate import interp1d
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
import tf.transformations as tr   

# ================= CONFIGURATION =================
# Static paths have been removed. Now read via ROS parameters.

BASE_FRAME = 'sparus2/base_link'
SSS_FRAME  = 'sparus2/sidescan'

SONAR_RANGE = 30.0
MOSAIC_RES  = 0.07
BLIND_ZONE  = 0.2

# === Coordinate Reference Systems ===
CRS_WGS84 = "EPSG:4326"
CRS_UTM   = "EPSG:32631"

ll_to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)

# ================= TF UTIL =================

def get_static_tf(bag, parent_frame, child_frame):
    for _, msg, _ in bag.read_messages(topics=['/tf_static', '/tf']):
        for t in msg.transforms:
            if (t.header.frame_id == parent_frame and
                t.child_frame_id == child_frame):

                q = t.transform.rotation
                p = t.transform.translation

                T = tr.quaternion_matrix([q.x, q.y, q.z, q.w])
                T[:3, 3] = [p.x, p.y, p.z]
                return T

    raise RuntimeError(f"TF {parent_frame} → {child_frame} not found in bag")

# ================= IMAGE UTILITIES =================

def enhance_data(img_input):
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

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

# ================= NAVIGATION =================

def get_nav_origin(bag, nav_topic): # ADDED nav_topic ARGUMENT
    for _, msg, _ in bag.read_messages(topics=[nav_topic]):
        if hasattr(msg, 'origin'):
            lat0 = msg.origin.latitude
            lon0 = msg.origin.longitude
            rospy.loginfo(f"✔ Navigation origin detected: lat={lat0}, lon={lon0}")
            return lat0, lon0
    raise RuntimeError("Navigation geographic origin not found")


def get_nav_data(bag, nav_topic): # ADDED nav_topic ARGUMENT
    ts, north, east, yaw, alt = [], [], [], [], []

    for _, msg, _ in bag.read_messages(topics=[nav_topic]):
        ts.append(msg.header.stamp.to_sec())
        north.append(msg.position.north)
        east.append(msg.position.east)
        yaw.append(msg.orientation.yaw)
        alt.append(msg.altitude)

    ts = np.array(ts)
    idx = np.argsort(ts)

    ts    = ts[idx]
    north = np.array(north)[idx]
    east  = np.array(east)[idx]
    yaw   = np.unwrap(np.array(yaw)[idx])
    alt   = np.array(alt)[idx]

    f_n = interp1d(ts, north, bounds_error=False, fill_value=np.nan)
    f_e = interp1d(ts, east,  bounds_error=False, fill_value=np.nan)
    f_y = interp1d(ts, yaw,   bounds_error=False, fill_value=np.nan)
    f_h = interp1d(ts, alt,   bounds_error=False, fill_value=np.nan)

    return (f_n, f_e, f_y, f_h), (ts[0], ts[-1])

# ================= SSS MOSAIC =================

def process_mosaic(bag, nav, time_range, off_x, off_y):
    f_n, f_e, f_y, f_h = nav
    t0, t1 = time_range

    ts_samples = np.linspace(t0, t1, 500)
    east_samples  = f_e(ts_samples)
    north_samples = f_n(ts_samples)

    valid = ~np.isnan(east_samples) & ~np.isnan(north_samples)
    min_e, max_e = np.min(east_samples[valid]), np.max(east_samples[valid])
    min_n, max_n = np.min(north_samples[valid]), np.max(north_samples[valid])

    margin = SONAR_RANGE + 5.0
    x_min, x_max = min_e - margin, max_e + margin
    y_min, y_max = min_n - margin, max_n + margin

    width  = int(np.ceil((x_max - x_min) / MOSAIC_RES))
    height = int(np.ceil((y_max - y_min) / MOSAIC_RES))

    grid = np.zeros(width * height, dtype=np.float32)
    cnt  = np.zeros(width * height, dtype=np.float32)

    def to_idx(x, y):
        c = ((x - x_min) / MOSAIC_RES).astype(np.int32)
        r = ((y_max - y) / MOSAIC_RES).astype(np.int32)
        return c, r

    info = bag.get_type_and_topic_info()
    sss_topics = [
    t for t, v in info.topics.items()
    if "sidescan" in t.lower() and "Image" in v.msg_type
    ]

    for topic, msg, _ in bag.read_messages(topics=sss_topics):

        if not hasattr(msg, 'header') or not hasattr(msg, 'data'):
            continue

        ts = msg.header.stamp.to_sec()

        if ts < t0 or ts > t1:
            continue

        n, e = f_n(ts), f_e(ts)
        yaw, h = f_y(ts), f_h(ts)
        if np.isnan(n) or np.isnan(e) or h < 0.2:
            continue

        scan = np.frombuffer(msg.data, dtype=np.uint8).astype(np.float32)
        if "port" in topic.lower():
            scan = scan[::-1]
            sign = -1
        else:
            sign = 1

        npx = scan.size
        meters_px = SONAR_RANGE / npx
        slant = np.arange(npx) * meters_px
        ground = np.sqrt(np.maximum(slant**2 - h**2, 0.0))
        valid = ground > BLIND_ZONE
        if not np.any(valid):
            continue

        v_fwd_n =  np.cos(yaw)
        v_fwd_e =  np.sin(yaw)
        v_right_n = -np.sin(yaw)
        v_right_e =  np.cos(yaw)

        off_n = off_x * v_fwd_n + off_y * v_right_n
        off_e = off_x * v_fwd_e + off_y * v_right_e

        px_n = (n + off_n) + sign * v_right_n * ground[valid]
        px_e = (e + off_e) + sign * v_right_e * ground[valid]

        c, r = to_idx(px_e, px_n)
        mask = (c >= 0) & (c < width) & (r >= 0) & (r < height)
        idx = r[mask] * width + c[mask]

        np.add.at(grid, idx, scan[valid][mask])
        np.add.at(cnt, idx, 1)

    img = np.zeros_like(grid)
    valid = cnt > 0
    img[valid] = grid[valid] / cnt[valid]

    return img.reshape((height, width)), x_min, y_max

# ================= MAIN =================

def main():
    # INITIALIZE ROS NODE
    rospy.init_node('sss_mosaic_gen', anonymous=True)

    # READ PARAMETERS FROM LAUNCH FILE
    bag_file   = rospy.get_param('~bag_file', '')
    output_dir = rospy.get_param('~output_dir', '.')
    nav_topic  = rospy.get_param('~nav_topic', '/sparus2/navigator/navigation')

    if not bag_file:
        rospy.logerr("ERROR: 'bag_file' not provided. Aborting.")
        return

    # CREATE OUTPUT FILE PATH
    output_tiff = os.path.join(output_dir, 'sss_mosaic.tif')
    rospy.loginfo(f"Processing Sidescan from: {bag_file}")

    bag = rosbag.Bag(bag_file)

    lat0, lon0 = get_nav_origin(bag, nav_topic)
    X0_UTM, Y0_UTM = ll_to_utm.transform(lon0, lat0)

    nav, t_range = get_nav_data(bag, nav_topic)

    T_SSS = get_static_tf(bag, BASE_FRAME, SSS_FRAME)
    off_x, off_y = T_SSS[0, 3], T_SSS[1, 3]
    rospy.loginfo(f"✔ SSS offset from TF: x={off_x:.2f}, y={off_y:.2f}")

    img, x_min, y_max = process_mosaic(bag, nav, t_range, off_x, off_y)

    save_geotiff = from_origin(
        X0_UTM + x_min,
        Y0_UTM + y_max,
        MOSAIC_RES,
        MOSAIC_RES
    )

    img8 = enhance_data(img)

    with rasterio.open(
        output_tiff,
        'w',
        driver='GTiff',
        height=img8.shape[0],
        width=img8.shape[1],
        count=1,
        dtype=np.uint8,
        crs=CRS_UTM,
        transform=save_geotiff,
        compress='deflate'
    ) as dst:
        dst.write(img8, 1)

    bag.close()
    rospy.loginfo(f"✔ GeoTIFF successfully generated at: {output_tiff}")

if __name__ == "__main__":
    main()