#!/usr/bin/env python3

import open3d as o3d
import numpy as np
from sklearn.preprocessing import scale


def robust_icp(
        source,
        target,
        T_init,
        icp_distance=2.0,
        max_iter=60):

    voxel_scales = [1.0, 0.5, 0.25]

    current_transform = T_init.copy()

    final_result = None

    loss = o3d.pipelines.registration.TukeyLoss(
        k=0.5
    )

    has_gicp = hasattr(

        o3d.pipelines.registration,
        "registration_generalized_icp"
    )

    for scale in voxel_scales:

        base_voxel = 0.5

        voxel = base_voxel * scale

        s = source.voxel_down_sample(voxel)
        t = target.voxel_down_sample(voxel)

        if len(s.points) < 50:
            continue

        if len(t.points) < 50:
            continue

        s.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel * 3.0,
                max_nn=30
            )
        )

        t.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel * 3.0,
                max_nn=30
            )
        )

        if has_gicp:

            estimator = (

                o3d.pipelines.registration.
                TransformationEstimationForGeneralizedICP(
                    loss
                )
            )

            result = (

                o3d.pipelines.registration.
                registration_generalized_icp(

                    s,
                    t,

                    icp_distance,

                    current_transform,

                    estimator,

                    o3d.pipelines.registration.
                    ICPConvergenceCriteria(
                        max_iteration=max_iter
                    )
                )
            )

        else:

            estimator = (

                o3d.pipelines.registration.
                TransformationEstimationPointToPlane(
                    loss
                )
            )

            result = (

                o3d.pipelines.registration.
                registration_icp(

                    s,
                    t,

                    icp_distance,

                    current_transform,

                    estimator,

                    o3d.pipelines.registration.
                    ICPConvergenceCriteria(
                        max_iteration=max_iter
                    )
                )
            )

        current_transform = result.transformation

        final_result = result

    return final_result