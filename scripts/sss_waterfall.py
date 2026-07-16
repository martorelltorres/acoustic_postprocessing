#!/usr/bin/env python3
"""
Sidescan -> waterfall image (ping-ordered, no GPS).
Port=red, starboard=green, black nadir strip between them.

Slant-range corrected against the INS altitude, with the per-channel range read from
the bag (SSSConfig) exactly like sss2mosaic.py.

Outputs:
  - images/sss_waterfall.png

Usage:
  rosrun acoustic_postprocessing sss_waterfall.py _bag_file:=<bag> _output_dir:=<dir>

Author: Antoni Martorell (SRV, UIB)
"""

import rospy
import rosbag
import numpy as np
import cv2
import os
import sys

# Shared helpers live in scripts/common.py, imported with a flat name. Under catkin the
# script runs through a devel/lib wrapper whose sys.path does NOT include this directory
# (the wrapper does point __file__ at this source file), so add it explicitly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (read_nav, get_sonar_range, enhance_data, find_sidescan_topics,
                    is_port_topic, slant_to_ground)

BLIND_ZONE = 0.5     # m, nadir gap to skip
NADIR_STRIP_PX = 10  # width of the black separator between the two channels


def main():
    rospy.init_node('sss_waterfall_gen', anonymous=True)

    bag_file   = rospy.get_param('~bag_file', '')
    output_dir = rospy.get_param('~output_dir', '.')
    nav_topic  = rospy.get_param('~nav_topic', '/sparus2/navigator/navigation')
    # 0 = read it from the bag (SSSConfig). Only set >0 for bags without SSSConfig.
    sonar_range_param = float(rospy.get_param('~sonar_range', 0.0))

    if not bag_file:
        rospy.logerr("ERROR: 'bag_file' not provided. Aborting.")
        return

    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    output_img = os.path.join(images_dir, 'sss_waterfall.png')

    rospy.loginfo(f"Generating SSS Waterfall from: {bag_file}")
    bag = rosbag.Bag(bag_file)

    # Navigation: altitude only (for the slant correction), sorted by timestamp.
    nav = read_nav(bag, nav_topic)
    nav_ts = nav['ts']
    nav_h  = nav['altitude']

    if len(nav_ts) == 0 or not np.isfinite(nav_h).any():
        rospy.logerr("ERROR: Altitude not found in navigation topic.")
        bag.close()
        return

    # Per-channel range from the bag, NOT hardcoded. The old SONAR_RANGE = 30.0 here was
    # false for the Andratx bags (real range 50.0 m on both channels): every sample got
    # placed at 60% of its true range, so the slant->ground correction and the blind-zone
    # crop both landed on the wrong sample. Same fix sss2mosaic.py already had.
    sonar_range = sonar_range_param or get_sonar_range(bag, logger=rospy.logwarn)
    rospy.loginfo(f"Per-channel range: {sonar_range:.1f} m")

    # Collect slant-corrected lines per side
    port_lines = []
    star_lines = []

    rospy.loginfo("Processing sonar lines...")
    sss_topics = find_sidescan_topics(bag)

    for topic, msg, _ in bag.read_messages(topics=sss_topics):

        if not hasattr(msg, 'header') or not hasattr(msg, 'data'):
            continue

        ts = msg.header.stamp.to_sec()

        if ts < nav_ts[0] or ts > nav_ts[-1]:
            continue

        h = float(np.interp(ts, nav_ts, nav_h))
        if not np.isfinite(h) or h < 0.2:
            continue

        scan = np.frombuffer(msg.data, dtype=np.uint8)
        if scan.size < 50:
            continue

        # The raw port ping runs FAR -> NADIR, mirrored w.r.t. starboard (verified on the
        # bag: the mean profile peaks at sample 1998/2000 for port and 1/2000 for
        # starboard; see common.is_port_topic). It MUST be reversed before the range
        # maths, or slant range 0 lands on the far-range sample and the blind-zone crop
        # eats the far end instead of the nadir. This used to be missing here while
        # sss2mosaic.py did it: the bug hid because port's nadir-last order happens to
        # put the nadir next to the centre strip, so the picture looked right.
        is_port = is_port_topic(topic)
        if is_port:
            scan = scan[::-1]

        # Slant-range -> ground-range (samples now run nadir -> far on both channels).
        ground, _ = slant_to_ground(scan.size, sonar_range, h)

        scan = scan[ground > BLIND_ZONE]

        if scan.size < 10:
            continue

        if is_port:
            port_lines.append(scan)
        else:
            star_lines.append(scan)

    bag.close()

    if len(port_lines) == 0 or len(star_lines) == 0:
        rospy.logerr("ERROR: Not enough port/starboard lines found.")
        return

    # Crop all lines to common width and ping count. Both channels now run nadir -> far,
    # so cropping to [:min_len] keeps the near range on both.
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
    # No nodata_mask: unlike the mosaics, 0 here is a REAL echo sample (the raw SSS echo
    # is saturated with zeros), so the stretch must see them. See common.enhance_data.
    port_enhanced = enhance_data(port_gray)
    star_enhanced = enhance_data(star_gray)

    # Layout [port | nadir | starboard]: port sits LEFT of the centre strip, so it is
    # flipped back to far -> nadir for DISPLAY only, putting both nadirs against the
    # strip and the far ranges at the outer edges. The maths above stays nadir -> far.
    port_enhanced = port_enhanced[:, ::-1]

    # Colorize: port -> red, starboard -> green (BGR channels)
    rows, cols = port_enhanced.shape

    port_color = np.zeros((rows, cols, 3), dtype=np.uint8)
    port_color[:, :, 2] = port_enhanced

    star_color = np.zeros((rows, cols, 3), dtype=np.uint8)
    star_color[:, :, 1] = star_enhanced

    nadir = np.zeros((min_rows, NADIR_STRIP_PX, 3), dtype=np.uint8)

    # [red | black nadir | green]
    final_img = np.hstack((port_color, nadir, star_color))

    cv2.imwrite(output_img, final_img)
    rospy.loginfo(f"OK -> Waterfall image successfully saved to: {os.path.abspath(output_img)}")
    rospy.loginfo(f"Dimensions: {final_img.shape[1]} x {final_img.shape[0]} px")


if __name__ == "__main__":
    main()
