#!/usr/bin/env python3

import rospy
import rosbag
import numpy as np
import cv2
import os

# ================= CONFIGURATION =================
NAV_TOPIC = '/sparus2/navigator/navigation'
SONAR_RANGE = 30.0
BLIND_ZONE = 0.5

def has_image_fields(msg):
    return (
        hasattr(msg, 'data') and
        hasattr(msg, 'header') and
        isinstance(msg.data, (bytes, bytearray)) and
        len(msg.data) > 0
    )

def enhance_data(img_gray):
    """
    Applies enhancement filters to a grayscale image.
    It is better to apply this BEFORE converting to color.
    """
    if img_gray.size == 0:
        return img_gray

    img = img_gray.copy()

    # 1. Robust normalization
    p2, p98 = np.percentile(img, (2, 98))
    # Avoid division by zero
    denom = p98 - p2 if (p98 - p2) > 0 else 1
    img = np.clip((img - p2) * 255.0 / denom, 0, 255).astype(np.uint8)

    # 2. Speckle reduction
    img = cv2.medianBlur(img, 5)

    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # 4. Soft enhancement / Sharpening
    kernel = np.array([[0,-1,0],
                       [-1,5,-1],
                       [0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)
    
    return img

def main():
    # INITIALIZE ROS NODE
    rospy.init_node('sss_waterfall_gen', anonymous=True)

    # READ PARAMETERS FROM THE LAUNCH FILE
    bag_file   = rospy.get_param('~bag_file', '')
    output_dir = rospy.get_param('~output_dir', '.')

    if not bag_file:
        rospy.logerr("ERROR: 'bag_file' not provided. Aborting.")
        return

    # CREATE THE OUTPUT FILE PATH
    output_img = os.path.join(output_dir, 'sss_waterfall.png')

    rospy.loginfo(f"Generating SSS Waterfall from: {bag_file}")
    bag = rosbag.Bag(bag_file)

    # ----------------- Navigation (altitude) -----------------
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

    # ----------------- SSS -----------------
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

        npx = scan.size
        meters_px = SONAR_RANGE / npx

        slant = np.arange(npx) * meters_px
        # Basic slant-range correction
        ground = np.sqrt(np.maximum(slant**2 - h**2, 0.0))

        valid = ground > BLIND_ZONE
        scan = scan[valid]

        if scan.size < 10:
            continue

        if "port" in topic:
            # Port is inverted so the nadir is in the center
            port_lines.append(scan)
        else:
            star_lines.append(scan)

    bag.close()

    # ----------------- Validations -----------------
    if len(port_lines) == 0 or len(star_lines) == 0:
        rospy.logerr("ERROR: Not enough port/starboard lines found.")
        return

    # ----------------- Size normalization -----------------
    min_len = min(
        min(len(l) for l in port_lines),
        min(len(l) for l in star_lines)
    )

    # Convert to numpy arrays
    port_gray = np.array([l[:min_len] for l in port_lines], dtype=np.uint8)
    star_gray = np.array([l[:min_len] for l in star_lines], dtype=np.uint8)

    # Equalize the number of rows (pings)
    min_rows = min(port_gray.shape[0], star_gray.shape[0])
    port_gray = port_gray[:min_rows]
    star_gray = star_gray[:min_rows]

    # ----------------- FILTERING (Before color) -----------------
    rospy.loginfo("Applying enhancement filters...")
    port_enhanced = enhance_data(port_gray)
    star_enhanced = enhance_data(star_gray)

    # ----------------- COLORIZATION -----------------
    # OpenCV uses BGR format (Blue, Green, Red)
    rows, cols = port_enhanced.shape
    
    # Create image for PORT (Red) -> Channel 2
    port_color = np.zeros((rows, cols, 3), dtype=np.uint8)
    port_color[:, :, 2] = port_enhanced  # Assign data to the RED channel

    # Create image for STARBOARD (Green) -> Channel 1
    star_color = np.zeros((rows, cols, 3), dtype=np.uint8)
    star_color[:, :, 1] = star_enhanced  # Assign data to the GREEN channel

    # Create Nadir (Black)
    nadir = np.zeros((min_rows, 10, 3), dtype=np.uint8)

    # ----------------- FUSION -----------------
    # Concatenate horizontally: [RED | BLACK | GREEN]
    final_img = np.hstack((port_color, nadir, star_color))

    # ----------------- SAVE RESULT -----------------
    cv2.imwrite(output_img, final_img)
    rospy.loginfo(f"OK -> Waterfall image successfully saved to: {os.path.abspath(output_img)}")
    rospy.loginfo(f"Dimensions: {final_img.shape[1]} x {final_img.shape[0]} px")


if __name__ == "__main__":
    main()