#!/usr/bin/env python3

import open3d as o3d
import numpy as np


# Umbral mínimo de correspondencias para considerar un resultado ICP válido.
# Si ninguna escala supera este umbral el resultado se descarta.
_MIN_ICP_CORRESPONDENCES = 30


def robust_icp(
        source,
        target,
        T_init,
        icp_distance=2.0,
        max_iter=60):

    # Escalas de voxel de mayor a menor (coarse-to-fine).
    # Se empieza con el voxel grande para alinear globalmente
    # y se refina en cada paso.
    voxel_scales = [1.0, 0.5, 0.25]

    # La transformación actual solo se actualiza si el resultado
    # de esa escala tiene correspondencias reales (fitness > 0).
    # De este modo un paso fallido no corrompe el siguiente.
    current_transform = T_init.copy()

    final_result = None

    loss = o3d.pipelines.registration.TukeyLoss(
        k=0.5
    )

    has_gicp = hasattr(
        o3d.pipelines.registration,
        "registration_generalized_icp"
    )

    for voxel_scale in voxel_scales:

        voxel = 0.5 * voxel_scale

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
                TransformationEstimationForGeneralizedICP(loss)
            )

            result = (
                o3d.pipelines.registration.
                registration_generalized_icp(
                    s, t,
                    icp_distance,
                    current_transform,
                    estimator,
                    o3d.pipelines.registration.
                    ICPConvergenceCriteria(max_iteration=max_iter)
                )
            )

        else:

            estimator = (
                o3d.pipelines.registration.
                TransformationEstimationPointToPlane(loss)
            )

            result = (
                o3d.pipelines.registration.
                registration_icp(
                    s, t,
                    icp_distance,
                    current_transform,
                    estimator,
                    o3d.pipelines.registration.
                    ICPConvergenceCriteria(max_iteration=max_iter)
                )
            )

        # Solo propagar la transformación si esta escala produjo
        # correspondencias reales. Si fitness=0 la transformación
        # devuelta es inválida y no debe usarse como semilla.
        if len(result.correspondence_set) >= _MIN_ICP_CORRESPONDENCES:
            current_transform = result.transformation
            final_result = result

    # Si el mejor resultado acumulado tiene fitness=0 (ninguna escala
    # encontró correspondencias suficientes) se devuelve None para que
    # el pipeline use el fallback de navegación con info baja.
    if final_result is None:
        return None

    if len(final_result.correspondence_set) < _MIN_ICP_CORRESPONDENCES:
        return None

    return final_result