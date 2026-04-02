#!/usr/bin/env python3
"""
Author: Antoni Martorell
Affiliation: Systems, Robotics and Vision Group (SRV),
             University of the Balearic Islands (UIB)
"""

import rospy
import rosbag
import numpy as np
import cv2
import os
from scipy.interpolate import interp1d
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
import tf.transformations as tr
from std_msgs.msg import Bool
import time

# ================= CONFIGURATION =================
SONAR_RANGE = 30.0
MOSAIC_RES  = 0.07
BLIND_ZONE  = 0.2

CRS_WGS84 = "EPSG:4326"
CRS_UTM   = "EPSG:32631"
ll_to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)

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

# ================= TF =================
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

# ================= NAVIGATION =================
def get_nav_origin(bag, nav_topic):
    for _, msg, _ in bag.read_messages(topics=[nav_topic]):
        if hasattr(msg, 'origin'):
            return msg.origin.latitude, msg.origin.longitude

    raise RuntimeError("Navigation geographic origin not found")

def get_nav_data(bag, nav_topic):
    ts, north, east, yaw, pitch, roll, alt = [], [], [], [], [], [], []

    for _, msg, _ in bag.read_messages(topics=[nav_topic]):
        ts.append(msg.header.stamp.to_sec())
        north.append(msg.position.north)
        east.append(msg.position.east)
        yaw.append(msg.orientation.yaw)
        pitch.append(msg.orientation.pitch)
        roll.append(msg.orientation.roll)
        alt.append(msg.altitude)

    ts = np.array(ts)
    idx = np.argsort(ts)

    ts = ts[idx]
    north = np.array(north)[idx]
    east = np.array(east)[idx]

    yaw   = np.unwrap(np.array(yaw)[idx])
    pitch = np.unwrap(np.array(pitch)[idx])
    roll  = np.unwrap(np.array(roll)[idx])
    alt   = np.array(alt)[idx]

    f_n = interp1d(ts, north, bounds_error=False, fill_value=np.nan)
    f_e = interp1d(ts, east,  bounds_error=False, fill_value=np.nan)
    f_y = interp1d(ts, yaw,   bounds_error=False, fill_value=np.nan)
    f_p = interp1d(ts, pitch, bounds_error=False, fill_value=np.nan)
    f_r = interp1d(ts, roll,  bounds_error=False, fill_value=np.nan)
    f_h = interp1d(ts, alt,   bounds_error=False, fill_value=np.nan)

    return (f_n, f_e, f_y, f_p, f_r, f_h), (ts[0], ts[-1])

# ================= MOSAIC =================
def process_mosaic(bag, nav, time_range, T_PORT, T_STBD):

    f_n, f_e, f_y, f_p, f_r, f_h = nav
    t0, t1 = time_range

    ts_samples = np.linspace(t0, t1, 500)

    east_samples = f_e(ts_samples)
    north_samples = f_n(ts_samples)

    valid = ~np.isnan(east_samples) & ~np.isnan(north_samples)

    margin = SONAR_RANGE + 5.0

    x_min = np.min(east_samples[valid]) - margin
    x_max = np.max(east_samples[valid]) + margin
    y_min = np.min(north_samples[valid]) - margin
    y_max = np.max(north_samples[valid]) + margin

    width = int(np.ceil((x_max - x_min) / MOSAIC_RES))
    height = int(np.ceil((y_max - y_min) / MOSAIC_RES))

    grid = np.zeros(width * height, dtype=np.float32)
    cnt  = np.zeros(width * height, dtype=np.float32)

    def to_idx(x, y):
        c = ((x - x_min) / MOSAIC_RES).astype(np.int32)
        r = ((y_max - y) / MOSAIC_RES).astype(np.int32)
        return c, r

    info = bag.get_type_and_topic_info()
    sss_topics = [t for t, v in info.topics.items()
                  if "sidescan" in t.lower() and "Image" in v.msg_type]

    for topic, msg, _ in bag.read_messages(topics=sss_topics):

        if not hasattr(msg, 'header') or not hasattr(msg, 'data'):
            continue

        ts = msg.header.stamp.to_sec()

        if ts < t0 or ts > t1:
            continue

        try:
            n = float(f_n(ts))
            e = float(f_e(ts))
            yaw = float(f_y(ts))
            pitch = float(f_p(ts))
            roll = float(f_r(ts))
            h = float(f_h(ts))
        except:
            continue

        if np.isnan(n) or np.isnan(e) or np.isnan(yaw) or np.isnan(pitch) or np.isnan(roll) or np.isnan(h):
            continue

        if h < 0.2:
            continue

        scan = np.frombuffer(msg.data, dtype=np.uint8).astype(np.float32)

        if "port" in topic.lower():
            scan = scan[::-1]
            T_sensor = T_PORT
        else:
            T_sensor = T_STBD

        npx = scan.size
        meters_px = SONAR_RANGE / npx

        slant = np.arange(npx) * meters_px
        ground = np.sqrt(np.maximum(slant**2 - h**2, 0.0))

        valid_mask = ground > BLIND_ZONE

        if not np.any(valid_mask):
            continue

        R_veh = tr.euler_matrix(roll, pitch, yaw, axes='sxyz')[:3, :3]
        R_sensor = T_sensor[:3, :3]

        R_total = R_veh @ R_sensor

        sensor_offset = T_sensor[:3, 3]
        sensor_world = R_veh @ sensor_offset

        off_n = sensor_world[0]
        off_e = sensor_world[1]

        ping_axis = R_total @ np.array([0.0, 1.0, 0.0])

        if "port" in topic.lower():
            ping_axis = -ping_axis

        v_ping_n = ping_axis[0]
        v_ping_e = ping_axis[1]

        px_n = (n + off_n) + v_ping_n * ground[valid_mask]
        px_e = (e + off_e) + v_ping_e * ground[valid_mask]

        c, r = to_idx(px_e, px_n)

        mask = (c >= 0) & (c < width) & (r >= 0) & (r < height)

        idx = r[mask] * width + c[mask]

        np.add.at(grid, idx, scan[valid_mask][mask])
        np.add.at(cnt, idx, 1)

    img = np.zeros_like(grid)

    valid_pixels = cnt > 0
    img[valid_pixels] = grid[valid_pixels] / cnt[valid_pixels]

    return img.reshape((height, width)), x_min, y_max

# ================= MAIN =================
def main():

    rospy.init_node('sss_mosaic_gen', anonymous=True)

    bag_file = rospy.get_param('~bag_file', '')
    output_dir = rospy.get_param('~output_dir', '.')
    nav_topic = rospy.get_param('~nav_topic', '/sparus2/navigator/navigation')

    if not bag_file:
        rospy.logerr("ERROR: 'bag_file' not provided.")
        return

    output_tiff = os.path.join(output_dir, 'sss_mosaic.tif')

    bag = rosbag.Bag(bag_file)

    lat0, lon0 = get_nav_origin(bag, nav_topic)
    X0_UTM, Y0_UTM = ll_to_utm.transform(lon0, lat0)

    nav, t_range = get_nav_data(bag, nav_topic)

    T_PORT = get_static_transform_from_tf(
        bag_file,
        'sparus2/base_link',
        'sparus2/port_sidescan'
    )

    T_STBD = get_static_transform_from_tf(
        bag_file,
        'sparus2/base_link',
        'sparus2/starboard_sidescan'
    )

    rospy.loginfo("✔ TF loaded from bag")

    img, x_min, y_max = process_mosaic(
        bag,
        nav,
        t_range,
        T_PORT,
        T_STBD
    )

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

    rospy.loginfo(f"✔ GeoTIFF generated: {output_tiff}")

    pub_sss_done = rospy.Publisher('/pipeline/sss_done', Bool, queue_size=1, latch=True)

    time.sleep(0.5)
    pub_sss_done.publish(True)

if __name__ == "__main__":
    main()