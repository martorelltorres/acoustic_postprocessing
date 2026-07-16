#!/usr/bin/env python3
"""
Navigation cache for the media scripts: both missions' trajectories in UTM.

WHY THIS EXISTS: make_media.py and make_anim_data.py both READ
results/media/.nav_cache.npz, and until now NOTHING in the repo wrote it — the file on
disk came from an older version of make_media.py that no longer exists. From a clean
results/ the whole media suite was dead (make_media.py exits 1, make_anim_data.py raises
SystemExit). This is the missing producer.

It exists at all because extracting the tracks means reading two multi-GB bags, which is
slow and pointless to repeat for every figure.

Outputs (results/media/.nav_cache.npz), all float64 at full navigation rate (~20 Hz):
  tm, xm, ym, zm : multibeam mission — epoch seconds, UTM easting/northing, z = -depth
  ts, xs, ys, zs : sidescan mission  — idem

Note the two products come from DIFFERENT missions at different times, which is exactly
what figure 06 ("two surveys") is about. Both are converted to absolute UTM through their
OWN nav origin, so they share a frame and can be plotted together.

Usage:
  python3 make_nav_cache.py [results_dir] [--mb-bag <bag>] [--sss-bag <bag>]

The defaults are the bags the media suite is built around — the 13:52 lawnmower and the
10:44 sidescan run, which is what the hardcoded labels in make_media.py say. They are NOT
the acoustic_pipeline.launch defaults (that launch runs the 12:52 octagon).

Author: Antoni Martorell (SRV, UIB)
"""

import os
import sys

import numpy as np
import rosbag
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import get_nav_origin, read_nav

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NAV_TOPIC = '/sparus2/navigator/navigation'

BAGS = "/media/uib/Datos/bagfiles/andratx/download_2026-07-07_08-06-28/2026_07_02"
MB_BAG_DEFAULT  = f"{BAGS}/13_52_22/sparus2_multibeam_2026-07-02-11-52-22_0.bag"
SSS_BAG_DEFAULT = f"{BAGS}/10_44_27/sparus2_sidescan_2026-07-02-08-44-27_0.bag"

CRS_WGS84 = "EPSG:4326"
CRS_UTM   = "EPSG:32631"
ll_to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)


def track_utm(bag_file):
    """(t, x, y, z) of one mission in absolute UTM."""
    if not os.path.isfile(bag_file):
        raise SystemExit(f"Bag not found: {bag_file}")

    bag = rosbag.Bag(bag_file)
    try:
        lat0, lon0 = get_nav_origin(bag, NAV_TOPIC)
        x0, y0 = ll_to_utm.transform(lon0, lat0)
        nav = read_nav(bag, NAV_TOPIC)
    finally:
        bag.close()

    # Local (north, east, depth) -> UTM. X = easting <- east, Y = northing <- north:
    # the same axis mapping the projection scripts use, or the tracks would not sit on
    # top of the rasters they are drawn over.
    return nav['ts'], x0 + nav['east'], y0 + nav['north'], -nav['depth']


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    res = args[0] if args else os.path.join(PKG_ROOT, "results")

    def opt(name, default):
        return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default

    mb_bag  = opt("--mb-bag", MB_BAG_DEFAULT)
    sss_bag = opt("--sss-bag", SSS_BAG_DEFAULT)

    out = os.path.join(res, "media")
    os.makedirs(out, exist_ok=True)

    print(f"[nav-cache] multibeam: {mb_bag}")
    tm, xm, ym, zm = track_utm(mb_bag)
    print(f"[nav-cache]   {len(tm):,} fixes over {(tm[-1] - tm[0]) / 60:.1f} min")

    print(f"[nav-cache] sidescan : {sss_bag}")
    ts, xs, ys, zs = track_utm(sss_bag)
    print(f"[nav-cache]   {len(ts):,} fixes over {(ts[-1] - ts[0]) / 60:.1f} min")

    cache = os.path.join(out, ".nav_cache.npz")
    np.savez(cache, tm=tm, xm=xm, ym=ym, zm=zm, ts=ts, xs=xs, ys=ys, zs=zs)

    print(f"[nav-cache] done -> {cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
