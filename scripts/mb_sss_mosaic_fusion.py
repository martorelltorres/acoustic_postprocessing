#!/usr/bin/env python3
"""
Fuse the multibeam backscatter mosaic with the sidescan mosaic.

Both inputs are UTM GeoTIFFs (same CRS). They are resampled onto a common
grid (union extent, finest resolution) and blended. Output: mb_sss_mosaic.tif.

Author: Antoni Martorell (SRV, UIB)
"""

import rospy
import os
import time
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from std_msgs.msg import Bool

# Pipeline completion flags
mb_finished = False
sss_finished = False


def mb_callback(msg):
    global mb_finished
    if msg.data:
        mb_finished = True
        rospy.loginfo("MB intensity signal received.")


def sss_callback(msg):
    global sss_finished
    if msg.data:
        sss_finished = True
        rospy.loginfo("Sidescan signal received.")


def load_raster(path):
    # Read band 1 plus its geo-referencing.
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32), src.transform, src.crs, src.res


def resample_to(data, src_transform, src_crs, dst_transform, dst_crs, shape):
    # Warp a raster onto the target grid (nearest, preserves intensity).
    out = np.zeros(shape, dtype=np.float32)
    reproject(
        source=data,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
    )
    return out


def main():
    rospy.init_node('mb_sss_mosaic_fusion', anonymous=True)

    mb_tif     = rospy.get_param('~mb_tif')
    sss_tif    = rospy.get_param('~sss_tif')
    output_dir = rospy.get_param('~output_dir', '.')
    mode       = rospy.get_param('~mode', 'max')   # max | mean | mb | sss
    timeout    = rospy.get_param('~wait_timeout', 600.0)

    tif_dir = os.path.join(output_dir, 'tif')
    os.makedirs(tif_dir, exist_ok=True)

    out_tif = os.path.join(tif_dir, 'mb_sss_mosaic.tif')

    # Wait for both producers to finish (signals guarantee the .tif are fully
    # written). Falls back to files on disk on timeout. Wall-clock based, so it
    # works even without /clock (use_sim_time).
    rospy.Subscriber('/pipeline/mb_intensity_done', Bool, mb_callback)
    rospy.Subscriber('/pipeline/sss_done', Bool, sss_callback)

    # Checked before the loop: `timeout > 0` used to live inside the break condition,
    # so timeout=0 made the wait infinite instead of disabling it.
    if timeout <= 0:
        rospy.loginfo("wait_timeout<=0: skipping signals, reading the .tif from disk.")
    else:
        rospy.loginfo("Waiting for MB intensity + SSS mosaics (timeout %.0fs)..." % timeout)

        t0 = time.time()
        while not (mb_finished and sss_finished):
            if rospy.is_shutdown():
                return
            if (time.time() - t0) > timeout:
                rospy.logwarn("Timeout waiting for signals. Using files on disk if present.")
                break
            time.sleep(0.5)

    if not (os.path.isfile(mb_tif) and os.path.isfile(sss_tif)):
        rospy.logerr("Missing input mosaics, aborting.")
        return

    mb, mb_T, mb_crs, mb_res = load_raster(mb_tif)
    sss, sss_T, sss_crs, _   = load_raster(sss_tif)

    if str(mb_crs) != str(sss_crs):
        rospy.logwarn(f"CRS mismatch ({mb_crs} vs {sss_crs}); assuming compatible.")

    # Common grid: union of both extents at the finest resolution.
    res = min(mb_res[0], mb_res[1])

    def bounds(T, shape):
        # (left, bottom, right, top) from an affine transform + raster shape.
        h, w = shape
        left, top = T.c, T.f
        return left, top - h * abs(T.e), left + w * abs(T.a), top

    mb_l, mb_b, mb_r, mb_t = bounds(mb_T, mb.shape)
    ss_l, ss_b, ss_r, ss_t = bounds(sss_T, sss.shape)

    left   = min(mb_l, ss_l)
    right  = max(mb_r, ss_r)
    bottom = min(mb_b, ss_b)
    top    = max(mb_t, ss_t)

    width  = int(np.ceil((right - left) / res))
    height = int(np.ceil((top - bottom) / res))
    dst_T  = from_origin(left, top, res, res)
    shape  = (height, width)

    mb_g  = resample_to(mb,  mb_T,  mb_crs,  dst_T, mb_crs, shape)
    sss_g = resample_to(sss, sss_T, sss_crs, dst_T, mb_crs, shape)

    # Blend on the common grid.
    mb_v, sss_v = mb_g > 0, sss_g > 0
    fused = np.zeros(shape, dtype=np.float32)

    if mode == 'mb':
        fused = mb_g
    elif mode == 'sss':
        fused = sss_g
    elif mode == 'mean':
        both = mb_v & sss_v
        fused[both] = 0.5 * (mb_g[both] + sss_g[both])
        fused[mb_v & ~sss_v] = mb_g[mb_v & ~sss_v]
        fused[sss_v & ~mb_v] = sss_g[sss_v & ~mb_v]
    else:  # 'max': keep the stronger return per pixel
        fused = np.maximum(mb_g, sss_g)

    fused8 = np.clip(fused, 0, 255).astype(np.uint8)

    with rasterio.open(
        out_tif, 'w', driver='GTiff',
        height=height, width=width, count=1, dtype=np.uint8,
        crs=mb_crs, transform=dst_T, compress='deflate'
    ) as dst:
        dst.write(fused8, 1)

    rospy.loginfo(f"Fused mosaic saved: {out_tif}")

    pub = rospy.Publisher('/pipeline/mosaic_fusion_done', Bool, queue_size=1, latch=True)
    time.sleep(0.5)
    pub.publish(True)


if __name__ == '__main__':
    main()
