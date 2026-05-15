#!/usr/bin/env python3

import rospy
import rosbag
import numpy as np
import cv2
import os

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
import tf.transformations as tr

from std_msgs.msg import Bool
import time

# =========================================================
# SONAR CONFIGURATION
# =========================================================
SONAR_RANGE = 30.0                
VERTICAL_APERTURE_DEG = 45.0      
BLIND_ZONE = 1
MOSAIC_RES = 0.07

CRS_WGS84 = "EPSG:4326"
CRS_UTM = "EPSG:32631"

ll_to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)

# =========================================================
# ENHANCEMENT
# =========================================================
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

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(img, -1, kernel)

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
# NAVIGATION
# =========================================================
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

    yaw = gaussian_filter1d(np.unwrap(np.array(yaw)[idx]), sigma=2)
    pitch = np.unwrap(np.array(pitch)[idx])
    roll = np.unwrap(np.array(roll)[idx])

    alt = np.array(alt)[idx]

    f_n = interp1d(ts, north, bounds_error=False, fill_value=np.nan)
    f_e = interp1d(ts, east, bounds_error=False, fill_value=np.nan)

    f_y = interp1d(ts, yaw, bounds_error=False, fill_value=np.nan)
    f_p = interp1d(ts, pitch, bounds_error=False, fill_value=np.nan)
    f_r = interp1d(ts, roll, bounds_error=False, fill_value=np.nan)

    f_h = interp1d(ts, alt, bounds_error=False, fill_value=np.nan)

    return (f_n, f_e, f_y, f_p, f_r, f_h), (ts[0], ts[-1])

# =========================================================
# MOSAIC
# =========================================================
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
    cnt = np.zeros(width * height, dtype=np.float32)

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

        n = float(f_n(ts))
        e = float(f_e(ts))

        yaw = float(f_y(ts))
        pitch = float(f_p(ts))
        roll = float(f_r(ts))

        h = float(f_h(ts))

        if np.isnan(n) or np.isnan(e) or np.isnan(yaw):
            continue

        if h < 0.2:
            continue

        scan = np.frombuffer(msg.data, dtype=np.uint8).astype(np.float32)
        scan = gaussian_filter1d(scan, sigma=1.2)

        if "port" in topic.lower():
            scan = scan[::-1]
            T_sensor = T_PORT
        else:
            T_sensor = T_STBD


        R_sensor = T_sensor[:3, :3]
        sensor_offset = T_sensor[:3, 3]

        npx = scan.size
        slant = np.linspace(0, SONAR_RANGE, npx)

        valid_mask = slant > BLIND_ZONE

        if not np.any(valid_mask):
            continue

        # ==========================================
        # VEHICLE ROTATION
        # ==========================================
        R_veh = tr.euler_matrix(
            roll,
            pitch,
            yaw,
            axes='sxyz'
        )[:3, :3]

        # ==========================================
        # SENSOR OFFSET WORLD
        # ==========================================
        sensor_world = R_veh @ sensor_offset

        sensor_n = n + sensor_world[0]
        sensor_e = e + sensor_world[1]

        # ==========================================
        # SIDESCAN LATERAL AXIS
        # ==========================================
        if "port" in topic.lower():
            dir_n = np.sin(yaw)
            dir_e = -np.cos(yaw)
        else:
            dir_n = -np.sin(yaw)
            dir_e = np.cos(yaw)

        # ==========================================
        # SLANT TO GROUND RANGE
        # ==========================================
        mount_angle = np.deg2rad(20.0)

        effective_h = h / np.cos(mount_angle)

        ground = np.sqrt(
            np.maximum(slant[valid_mask]**2 - effective_h**2, 0.0)
        )
        # ==========================================
        # PROJECT PIXELS
        # ==========================================
        px_n = sensor_n + dir_n * ground
        px_e = sensor_e + dir_e * ground

        c, r = to_idx(px_e, px_n)

        mask = (
            (c >= 0) &
            (c < width) &
            (r >= 0) &
            (r < height)
        )

        idx = r[mask] * width + c[mask]

        np.add.at(grid, idx, scan[valid_mask][mask])
        np.add.at(cnt, idx, 1)

    img = np.zeros_like(grid)

    valid_pixels = cnt > 0
    img[valid_pixels] = grid[valid_pixels] / cnt[valid_pixels]

    return img.reshape((height, width)), x_min, y_max

# =========================================================
# MAIN
# =========================================================
def main():

    rospy.init_node('sss_mosaic_gen')

    bag_file = rospy.get_param('~bag_file', '')
    output_dir = rospy.get_param('~output_dir', '.')
    nav_topic = rospy.get_param('~nav_topic', '/sparus2/navigator/navigation')

    if not bag_file:
        rospy.logerr("bag_file missing")
        return

    output_tiff = os.path.join(output_dir, 'sss_mosaic.tif')

    bag = rosbag.Bag(bag_file)

    lat0, lon0 = get_nav_origin(bag, nav_topic)
    X0_UTM, Y0_UTM = ll_to_utm.transform(lon0, lat0)

    nav, t_range = get_nav_data(bag, nav_topic)

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

    img, x_min, y_max = process_mosaic(
        bag,
        nav,
        t_range,
        T_PORT,
        T_STBD
    )

    transform = from_origin(
        X0_UTM + x_min,
        Y0_UTM + y_max,
        MOSAIC_RES,
        MOSAIC_RES
    )

    # =========================================================
    # ENHANCE IMAGE
    # =========================================================
    img8 = enhance_data(img)

    # Mantener fondo completamente vacío
    img8[img == 0] = 0

    # =========================================================
    # TRANSPARENCY MASK
    # =========================================================
    alpha_mask = np.where(img8 > 0, 255, 0).astype(np.uint8)

    # =========================================================
    # SAVE GEOTIFF WITH TRANSPARENCY
    # =========================================================
    with rasterio.open(
        output_tiff,
        'w',
        driver='GTiff',
        height=img8.shape[0],
        width=img8.shape[1],
        count=1,
        dtype=np.uint8,
        crs=CRS_UTM,
        transform=transform,
        compress='deflate',
        nodata=0,
        photometric='MINISBLACK'

    ) as dst:

        dst.write(img8, 1)

        dst.write_mask(alpha_mask)

    bag.close()

    rospy.loginfo(f"GeoTIFF generado: {output_tiff}")

    pub = rospy.Publisher('/pipeline/sss_done', Bool, queue_size=1, latch=True)

    time.sleep(0.5)
    pub.publish(True)

if __name__ == "__main__":
    main()

