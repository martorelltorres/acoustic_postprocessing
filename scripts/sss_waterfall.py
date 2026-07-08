#!/usr/bin/env python3
"""
Sidescan -> waterfall image (ping-ordered, no GPS).
Port=red, starboard=green, black nadir strip between them.

Author: Antoni Martorell (SRV, UIB)
"""

import rospy
import rosbag
import numpy as np
import cv2
import os
from std_msgs.msg import Bool

NAV_TOPIC = '/sparus2/navigator/navigation'
SONAR_RANGE = 30.0   # m, full per-channel range
BLIND_ZONE = 0.5     # m, nadir gap to skip


def has_image_fields(msg):
    # True for image-like messages (raw byte payload).
    return (
        hasattr(msg, 'data') and
        hasattr(msg, 'header') and
        isinstance(msg.data, (bytes, bytearray)) and
        len(msg.data) > 0
    )


def enhance_data(img_gray):
    # Normalize (2-98 pct), despeckle, CLAHE, sharpen. Apply before colorizing.
    if img_gray.size == 0:
        return img_gray

    img = img_gray.copy()

    p2, p98 = np.percentile(img, (2, 98))
    denom = p98 - p2 if (p98 - p2) > 0 else 1
    img = np.clip((img - p2) * 255.0 / denom, 0, 255).astype(np.uint8)

    img = cv2.medianBlur(img, 5)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)

    return img


def main():
    rospy.init_node('sss_waterfall_gen', anonymous=True)

    bag_file   = rospy.get_param('~bag_file', '')
    output_dir = rospy.get_param('~output_dir', '.')

    if not bag_file:
        rospy.logerr("ERROR: 'bag_file' not provided. Aborting.")
        return

    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    output_img = os.path.join(images_dir, 'sss_waterfall.png')

    rospy.loginfo(f"Generating SSS Waterfall from: {bag_file}")
    bag = rosbag.Bag(bag_file)

    # Navigation: altitude only (for slant correction)
    nav_ts = []
    nav_h = []

    rospy.loginfo("Reading navigation data...")
    for _, msg, _ in bag.read_messages(topics=[NAV_TOPIC]):
        if hasattr(msg, 'header') and hasattr(msg, 'altitude'):
            nav_ts.append(msg.header.stamp.to_sec())
            nav_h.append(msg.altitude)

    nav_ts = np.array(nav_ts)
    nav_h = np.array(nav_h)

    if len(nav_ts) == 0:
        rospy.logerr("ERROR: Altitude not found in navigation topic.")
        return

    # Collect slant-corrected lines per side
    port_lines = []
    star_lines = []

    rospy.loginfo("Processing sonar lines...")
    for topic, msg, _ in bag.read_messages():

        if "sidescan" not in topic:
            continue

        if not has_image_fields(msg):
            continue

        ts = msg.header.stamp.to_sec()

        if ts < nav_ts.min() or ts > nav_ts.max():
            continue

        h = np.interp(ts, nav_ts, nav_h)
        if h < 0.2:
            continue

        scan = np.frombuffer(msg.data, dtype=np.uint8)
        if scan.size < 50:
            continue

        # Slant-range -> ground-range
        npx = scan.size
        meters_px = SONAR_RANGE / npx

        slant = np.arange(npx) * meters_px
        ground = np.sqrt(np.maximum(slant**2 - h**2, 0.0))

        valid = ground > BLIND_ZONE
        scan = scan[valid]

        if scan.size < 10:
            continue

        if "port" in topic:
            port_lines.append(scan)
        else:
            star_lines.append(scan)

    bag.close()

    if len(port_lines) == 0 or len(star_lines) == 0:
        rospy.logerr("ERROR: Not enough port/starboard lines found.")
        return

    # Crop all lines to common width and ping count
    min_len = min(
        min(len(l) for l in port_lines),
        min(len(l) for l in star_lines)
    )

    port_gray = np.array([l[:min_len] for l in port_lines], dtype=np.uint8)
    star_gray = np.array([l[:min_len] for l in star_lines], dtype=np.uint8)

    min_rows = min(port_gray.shape[0], star_gray.shape[0])
    port_gray = port_gray[:min_rows]
    star_gray = star_gray[:min_rows]

    rospy.loginfo("Applying enhancement filters...")
    port_enhanced = enhance_data(port_gray)
    star_enhanced = enhance_data(star_gray)

    # Colorize: port -> red, starboard -> green (BGR channels)
    rows, cols = port_enhanced.shape

    port_color = np.zeros((rows, cols, 3), dtype=np.uint8)
    port_color[:, :, 2] = port_enhanced

    star_color = np.zeros((rows, cols, 3), dtype=np.uint8)
    star_color[:, :, 1] = star_enhanced

    nadir = np.zeros((min_rows, 10, 3), dtype=np.uint8)

    # [red | black nadir | green]
    final_img = np.hstack((port_color, nadir, star_color))

    cv2.imwrite(output_img, final_img)
    rospy.loginfo(f"OK -> Waterfall image successfully saved to: {os.path.abspath(output_img)}")
    rospy.loginfo(f"Dimensions: {final_img.shape[1]} x {final_img.shape[0]} px")


if __name__ == "__main__":
    main()