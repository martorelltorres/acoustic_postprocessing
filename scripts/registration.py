#!/usr/bin/env python3

import open3d as o3d
import numpy as np

from collections import OrderedDict


# -----------------------------------------------------------------------------
# PERF: FPFH CACHE PER CLOUD
# -----------------------------------------------------------------------------
# In loop closure each patch takes part as source or target in MANY candidate
# pairs, and preprocess_pointcloud recomputes downsample + normals + FPFH (the
# most expensive: max_nn=100) on the SAME cloud each time. A patch's FPFH is
# invariant (does not depend on the pair), so it is cached by (id(pcd), voxel)
# and reused. RANSAC receives exactly the same features → identical result,
# with a fraction of the compute. The id(pcd) key is stable because patches are
# persistent objects for the whole run.
# -----------------------------------------------------------------------------

_FPFH_CACHE = OrderedDict()
_FPFH_CACHE_MAX = 4096


def clear_fpfh_cache():
    """Clear the FPFH cache (call between runs if appropriate)."""
    _FPFH_CACHE.clear()


def preprocess_pointcloud(
        pcd,
        voxel_size):

    key = (id(pcd), round(voxel_size, 4))

    cached = _FPFH_CACHE.get(key)
    if cached is not None:
        _FPFH_CACHE.move_to_end(key)
        return cached

    pcd_down = pcd.voxel_down_sample(
        voxel_size
    )

    pcd_down.estimate_normals(

        o3d.geometry.KDTreeSearchParamHybrid(

            radius=voxel_size * 2.0,
            max_nn=30
        )
    )

    fpfh = (
        o3d.pipelines.registration.
        compute_fpfh_feature(

            pcd_down,

            o3d.geometry.KDTreeSearchParamHybrid(

                radius=voxel_size * 5.0,
                max_nn=100
            )
        )
    )

    if len(_FPFH_CACHE) >= _FPFH_CACHE_MAX:
        _FPFH_CACHE.popitem(last=False)

    _FPFH_CACHE[key] = (pcd_down, fpfh)

    return pcd_down, fpfh

def execute_global_registration(
        source,
        target,
        voxel_size,
        min_fitness=0.25):          # minimum acceptance threshold

    source_down, source_fpfh = preprocess_pointcloud(source, voxel_size)
    target_down, target_fpfh = preprocess_pointcloud(target, voxel_size)

    distance_threshold = voxel_size * 1.5   # was: * 2.0
                                             # stricter reduces false positives

    result = (
        o3d.pipelines.registration.
        registration_ransac_based_on_feature_matching(

            source_down,
            target_down,
            source_fpfh,
            target_fpfh,

            mutual_filter=True,

            max_correspondence_distance=distance_threshold,

            estimation_method=(
                o3d.pipelines.registration.
                TransformationEstimationPointToPoint(False)
            ),

            ransac_n=3,             # was: 4
                                    # 3 points define a plane: minimum needed
                                    # for 3D transform, more efficient

            checkers=[
                o3d.pipelines.registration.
                CorrespondenceCheckerBasedOnEdgeLength(
                    0.8             # was: 0.9 — less restrictive
                ),
                o3d.pipelines.registration.
                CorrespondenceCheckerBasedOnDistance(
                    distance_threshold
                ),
            ],

            criteria=(
                o3d.pipelines.registration.
                RANSACConvergenceCriteria(
                    100000,         # PERF: lowered from 4,000,000.
                                    # On flat bottom FPFH does not discriminate
                                    # and RANSAC fails almost always (see
                                    # report_ransac.md), so 4M iterations were
                                    # wasted compute. 100k is enough where there
                                    # is structure and is much faster where not.
                    0.999
                )
            )
        )
    )

    if result.fitness < min_fitness:
        return None

    return result
