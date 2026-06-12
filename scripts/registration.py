#!/usr/bin/env python3

import open3d as o3d
import numpy as np


def preprocess_pointcloud(
        pcd,
        voxel_size):

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

    return pcd_down, fpfh

def execute_global_registration(
        source,
        target,
        voxel_size,
        min_fitness=0.25):          # umbral mínimo de aceptación

    source_down, source_fpfh = preprocess_pointcloud(source, voxel_size)
    target_down, target_fpfh = preprocess_pointcloud(target, voxel_size)

    distance_threshold = voxel_size * 1.5   # antes: * 2.0
                                             # más estricto reduce falsos positivos

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

            ransac_n=3,             # antes: 4
                                    # 3 puntos definen un plano: mínimo necesario
                                    # para transformación 3D, más eficiente

            checkers=[
                o3d.pipelines.registration.
                CorrespondenceCheckerBasedOnEdgeLength(
                    0.8             # antes: 0.9 — menos restrictivo
                ),
                o3d.pipelines.registration.
                CorrespondenceCheckerBasedOnDistance(
                    distance_threshold
                ),
            ],

            criteria=(
                o3d.pipelines.registration.
                RANSACConvergenceCriteria(
                    100000,         # PERF: bajado de 4.000.000.
                                    # En fondo plano FPFH no discrimina y RANSAC
                                    # falla casi siempre (ver report_ransac.md),
                                    # así que 4M iteraciones eran cómputo
                                    # malgastado. 100k basta donde sí hay
                                    # estructura y acelera mucho donde no.
                    0.999
                )
            )
        )
    )

    if result.fitness < min_fitness:
        return None

    return result
