#!/usr/bin/env python3
"""
Shared helpers for the acoustic_postprocessing scripts (bag/TF/nav/raster).

WHY THIS MODULE EXISTS: every script used to carry its own copy of these helpers
(3x get_static_transform_from_tf, 3x get_nav_origin, 3x enhance_data, 3x nav
interpolator construction). Every bug fixed here had to be fixed three times, and in
practice never was: the sonar range was read from the bag in sss2mosaic.py but stayed
hardcoded in sss_waterfall.py, the nav timestamps were sorted in sss2mosaic.py but not
in the two multibeam scripts, and the NaN guard existed only in multibeam_intensity.py.
One copy, one fix.

IMPORT: with a flat name (`from common import ...`), which assumes this directory is on
sys.path. That holds when a script is run directly from scripts/, but NOT under catkin:
catkin_install_python generates a wrapper in devel/lib/ that exec()s the source file
without its directory on sys.path. The wrapper does set __file__ to the SOURCE path, so
every importer must first do:

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

Same pattern as the sibling multibeam_SLAM package.

Author: Antoni Martorell (SRV, UIB)
"""

import numpy as np
import rosbag
import rasterio
import cv2
import tf.transformations as tr

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


# =====================================================================
# Bag: static transforms
# =====================================================================

def get_static_transforms(bag_file, pairs):
    """(parent, child) -> 4x4 transform, resolving ALL pairs in ONE pass over the bag.

    Each get_static_transform_from_tf() call used to open the bag and scan /tf_static +
    /tf from the start; multibeam_processor.py needed three transforms and so read the
    bag four times over (three TFs + nav/scan).

    Pairs with no transform in the bag get the identity, which is what the per-pair
    helper returned before.
    """
    wanted = {(p, c) for p, c in pairs}
    found = {}

    bag = rosbag.Bag(bag_file)
    try:
        for _, msg, _ in bag.read_messages(topics=['/tf_static', '/tf']):
            for transform in msg.transforms:
                key = (transform.header.frame_id, transform.child_frame_id)

                if key not in wanted or key in found:
                    continue

                q = transform.transform.rotation
                t = transform.transform.translation

                T = tr.quaternion_matrix([q.x, q.y, q.z, q.w])
                T[:3, 3] = [t.x, t.y, t.z]
                found[key] = T

                if len(found) == len(wanted):
                    return found
    finally:
        bag.close()

    for key in wanted - set(found):
        found[key] = np.identity(4)

    return found


def get_static_transform_from_tf(bag_file, parent_frame, child_frame):
    # First parent->child transform found in /tf_static or /tf (4x4); identity if absent.
    return get_static_transforms(bag_file, [(parent_frame, child_frame)])[(parent_frame, child_frame)]


# =====================================================================
# Bag: navigation
# =====================================================================

def get_nav_origin(bag, nav_topic):
    # Geographic origin (lat, lon) of the local navigation frame.
    for _, msg, _ in bag.read_messages(topics=[nav_topic]):
        if hasattr(msg, 'origin'):
            return msg.origin.latitude, msg.origin.longitude

    raise RuntimeError("Navigation origin not found")


def read_nav(bag, nav_topic):
    """Navigation arrays, SORTED and DEDUPLICATED by timestamp.

    Returns a dict with ts, north, east, depth, roll, pitch, yaw, altitude.

    WHAT THE SORT/DEDUP ACTUALLY PROTECTS (measured, because the obvious answer is
    wrong): interp1d is NOT the problem -- it defaults to assume_sorted=False and sorts
    x internally, so unordered timestamps interpolate fine. The three things that do
    break, all silently:

      - np.unwrap(yaw) is ORDER-DEPENDENT. Fed an unsorted heading it invents jumps,
        and every downstream pose inherits them.
      - np.gradient(ts) for the yaw rate goes NEGATIVE on unsorted input, so the
        attitude gate starts dropping the wrong pings.
      - DUPLICATE timestamps break interp1d itself (two y for one x: it interpolates
        against whichever landed adjacent -- measured, 64.5 where the answer was 25.0).

    The Andratx bags are clean (20 Hz, dt = 0.05 s exactly, no reordering, no
    duplicates), so this is a guard rather than a fix for them -- but it is the guard
    sss2mosaic.py already had and the two multibeam scripts did not.

    Raw yaw is returned as-is: unwrapping is the caller's business (each script wants a
    different treatment; see nav_interpolators).
    """
    ts, north, east, depth, roll, pitch, yaw, alt = [], [], [], [], [], [], [], []

    for _, msg, _ in bag.read_messages(topics=[nav_topic]):
        ts.append(msg.header.stamp.to_sec())

        north.append(msg.position.north)
        east.append(msg.position.east)
        depth.append(msg.position.depth)

        roll.append(msg.orientation.roll)
        pitch.append(msg.orientation.pitch)
        yaw.append(msg.orientation.yaw)

        alt.append(getattr(msg, 'altitude', np.nan))

    if not ts:
        raise RuntimeError(f"No navigation messages on {nav_topic}")

    ts = np.asarray(ts)
    order = np.argsort(ts, kind="stable")

    # Deduplicate AFTER sorting: a repeated timestamp gives interp1d a zero-width
    # interval. Keep the first sample of each stamp.
    ts_sorted = ts[order]
    keep = np.ones(len(ts_sorted), dtype=bool)
    keep[1:] = np.diff(ts_sorted) > 0
    idx = order[keep]

    return {
        'ts':       ts[idx],
        'north':    np.asarray(north)[idx],
        'east':     np.asarray(east)[idx],
        'depth':    np.asarray(depth)[idx],
        'roll':     np.asarray(roll)[idx],
        'pitch':    np.asarray(pitch)[idx],
        'yaw':      np.asarray(yaw)[idx],
        'altitude': np.asarray(alt)[idx],
    }


def nav_interpolators(nav, smooth_yaw_sigma=None, with_yaw_rate=False):
    """Time interpolators over the arrays from read_nav().

    Returns a dict of interp1d: n, e, d, y (yaw), p (pitch), r (roll), h (altitude), and
    yr (yaw rate, deg/s) when with_yaw_rate. All out-of-range queries give NaN, which
    callers must check.

    The options exist to keep each script's CURRENT behaviour, not to homogenise it:
      - smooth_yaw_sigma: sss2mosaic.py smooths yaw (gaussian_filter1d, sigma=2); the
        multibeam scripts do not. Applying the smoothing to the multibeam would silently
        change the cloud.
      - with_yaw_rate: only the multibeam scripts gate on yaw rate.
    """
    ts = nav['ts']

    yaw = np.unwrap(nav['yaw'])
    if smooth_yaw_sigma:
        yaw = gaussian_filter1d(yaw, sigma=smooth_yaw_sigma)

    def _i(v):
        return interp1d(ts, v, bounds_error=False, fill_value=np.nan)

    f = {
        'n': _i(nav['north']),
        'e': _i(nav['east']),
        'd': _i(nav['depth']),
        'y': _i(yaw),
        'p': _i(np.unwrap(nav['pitch'])),
        'r': _i(np.unwrap(nav['roll'])),
        'h': _i(nav['altitude']),
    }

    if with_yaw_rate:
        # Yaw rate: the direct signature of a turn. High roll comes with banking into and
        # out of it, but the swath also smears when the vehicle rotates fast in heading,
        # even flying level.
        f['yr'] = _i(np.degrees(np.gradient(yaw) / np.maximum(np.gradient(ts), 1e-3)))

    return f


# =====================================================================
# Bag: sidescan configuration
# =====================================================================

# Per-channel sidescan range (m). ONLY a fallback for bags with no SSSConfig: the real
# value is read from the bag by get_sonar_range().
SONAR_RANGE_FALLBACK = 30.0


def get_sonar_range(bag, fallback=SONAR_RANGE_FALLBACK, logger=None):
    """Per-channel sidescan range (m) read from the bag's SSSConfig; `fallback` if absent.

    NEVER hardcode this. The 30.0 that used to be a module constant was WRONG for the
    Andratx bags (real range 50.0 m, verified on both channels): it placed every sample
    at 60% of its true range, squeezing the mosaic x0.6 across-track.
    """
    cfg_topics = [
        t for t in bag.get_type_and_topic_info().topics
        if t.endswith('/sss_info')
    ]

    for _, msg, _ in bag.read_messages(topics=cfg_topics):
        if getattr(msg, 'range', 0.0) > 0.0:
            return float(msg.range)

    if logger is not None:
        logger(f"No SSSConfig in the bag; using SONAR_RANGE={fallback} m (may be wrong).")

    return float(fallback)


def find_sidescan_topics(bag):
    """Sidescan image topics: name contains 'sidescan' AND the type is an Image.

    The type check matters: filtering on the name alone also matches the .../sss_info
    (SSSConfig) topics.
    """
    info = bag.get_type_and_topic_info()
    return [
        t for t, v in info.topics.items()
        if "sidescan" in t.lower() and "Image" in v.msg_type
    ]


def is_port_topic(topic):
    """True for the port channel.

    RAW SAMPLE ORDER IS MIRRORED BETWEEN CHANNELS (verified on the Andratx bag
    10_44_27, mean profile per sample index over 400 pings):
        starboard -> peaks at sample 1/2000     = nadir FIRST  (nadir -> far)
        port      -> peaks at sample 1998/2000  = nadir LAST   (far -> nadir)
    So a port ping must be reversed before any range maths, or slant range 0 lands on
    the far-range sample. See slant_to_ground().
    """
    return "port" in topic.lower()


def slant_to_ground(n_samples, sonar_range, altitude):
    """Slant range -> ground range for one ping, assuming a flat bottom at `altitude`.

    Expects samples ordered NADIR -> FAR (reverse the port channel first, see
    is_port_topic). Returns (ground, theta_deg): the across-track distance of every
    sample and its incidence angle from vertical (0 deg at nadir, ->90 deg at far
    range), which is the variable backscatter depends on and the one AVG bins by.
    """
    slant = np.arange(n_samples) * (sonar_range / n_samples)
    ground = np.sqrt(np.maximum(slant ** 2 - altitude ** 2, 0.0))
    theta = np.degrees(np.arctan2(ground, altitude))
    return ground, theta


# =====================================================================
# Rasters and image enhancement
# =====================================================================

def enhance_data(img_input, nodata_mask=None):
    """Normalize (2-98 pct), despeckle, CLAHE, sharpen -> 8-bit.

    `nodata_mask`: boolean array, True where there is NO data. Percentiles are then
    computed over the valid pixels only.

    THE TWO SEMANTICS ARE DELIBERATE, do not collapse them:
      - MOSAICS (mb_intensity, sss2mosaic) pass a mask (empty cells are 0 = nodata).
        Stretching over the zeros would waste the ramp on background.
      - WATERFALL passes None: there, 0 is a REAL sample value, not nodata (the raw SSS
        echo is saturated with zeros -- 47.5% of samples are exactly 0), so masking them
        out would change the stretch of a legitimately dark image.

    Note the caller still owns the nodata convention of the OUTPUT: CLAHE lifts the empty
    background above 0, so mosaic callers must re-impose 0 afterwards (see the nodata
    notes in multibeam_intensity.py / sss2mosaic.py).
    """
    if img_input is None or img_input.size == 0:
        return np.zeros_like(img_input, dtype=np.uint8)

    if nodata_mask is None:
        valid = np.ones(img_input.shape, dtype=bool)
    else:
        valid = ~nodata_mask

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


def load_raster(path):
    # Read band 1 plus its geo-referencing: (data, transform, crs, res).
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.transform, src.crs, src.res
