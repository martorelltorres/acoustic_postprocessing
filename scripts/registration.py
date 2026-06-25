#!/usr/bin/env python3

import open3d as o3d
import numpy as np

from collections import OrderedDict


# -----------------------------------------------------------------------------
# PERF: CACHÉ DE FPFH POR NUBE
# -----------------------------------------------------------------------------
# En el loop closure cada patch participa como source o target en MUCHOS pares
# candidatos, y preprocess_pointcloud recalcula downsample + normales + FPFH
# (lo más caro: max_nn=100) sobre la MISMA nube cada vez. El FPFH de un patch es
# invariante (no depende del par), así que se cachea por (id(pcd), voxel) y se
# reutiliza. El RANSAC recibe exactamente las mismas features → resultado
# idéntico, con una fracción del cómputo. La clave id(pcd) es estable porque los
# patches son objetos persistentes durante todo el run.
# -----------------------------------------------------------------------------

_FPFH_CACHE = OrderedDict()
_FPFH_CACHE_MAX = 4096


def clear_fpfh_cache():
    """Limpia el caché de FPFH (llamar entre runs si procede)."""
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
