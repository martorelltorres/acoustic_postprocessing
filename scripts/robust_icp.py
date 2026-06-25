#!/usr/bin/env python3

import open3d as o3d
import numpy as np

from collections import OrderedDict


# Umbral mínimo de correspondencias para considerar un resultado ICP válido.
# Si ninguna escala supera este umbral el resultado se descarta.
_MIN_ICP_CORRESPONDENCES = 30


# -----------------------------------------------------------------------------
# CACHÉ DE PREPROCESADO (downsample + normales) POR NUBE Y ESCALA
# -----------------------------------------------------------------------------
# En el registro secuencial cada patch.pcd participa en dos registros
# consecutivos (como source en idx y como target en idx+1) y el ICP multiescala
# repite voxel_down_sample + estimate_normals sobre la MISMA nube a las mismas
# escalas. Cacheando el resultado por (id(pcd), voxel) se evita ~la mitad del
# preprocesado. La clave es id(pcd): los patches son objetos persistentes, así
# que el id es estable durante todo el run. El caché se acota para no crecer
# sin límite (LRU simple por inserción).
# -----------------------------------------------------------------------------

_PREP_CACHE = OrderedDict()
_PREP_CACHE_MAX = 4096


def _preprocessed(pcd, voxel):
    """voxel_down_sample + estimate_normals con caché LRU por (id(pcd), voxel)."""

    key = (id(pcd), round(voxel, 4))

    cached = _PREP_CACHE.get(key)
    if cached is not None:
        # Refresca la posición LRU: lo recién usado pasa a ser lo más reciente.
        _PREP_CACHE.move_to_end(key)
        return cached

    down = pcd.voxel_down_sample(voxel)

    if len(down.points) >= 1:
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel * 3.0,
                max_nn=30
            )
        )

    # Evicción LRU real: descarta SOLO la entrada menos usada recientemente, en
    # vez de vaciar todo el caché. En loop closure un mismo patch se preprocesa
    # muchas veces; vaciar el caché entero obligaba a recalcular nubes que se
    # volvían a pedir de inmediato. El resultado preprocesado es idéntico.
    if len(_PREP_CACHE) >= _PREP_CACHE_MAX:
        _PREP_CACHE.popitem(last=False)

    _PREP_CACHE[key] = down
    return down


def clear_preprocess_cache():
    """Limpia el caché de preprocesado (llamar entre runs si procede)."""
    _PREP_CACHE.clear()


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

        # Downsample + normales cacheados por (id(pcd), voxel): evita reprocesar
        # la misma nube en registros consecutivos.
        s = _preprocessed(source, voxel)
        t = _preprocessed(target, voxel)

        if len(s.points) < 50:
            continue

        if len(t.points) < 50:
            continue

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


# -----------------------------------------------------------------------------
# DETECCIÓN DE TEXTURA GEOMÉTRICA
# -----------------------------------------------------------------------------
# Mide si un par de patches tiene suficiente relieve geométrico para que el ICP
# geométrico restrinja la traslación XY. En fondo plano las normales apuntan
# casi todas hacia arriba (|nz|≈1) → la geometría no discrimina XY y el ICP
# desliza. La fracción de normales NO verticales es un proxy barato de "textura
# geométrica disponible".
# -----------------------------------------------------------------------------

def _geometric_texture(pcd, voxel=0.5, slope_thresh=0.85):
    """
    Fracción de puntos cuya normal NO es casi vertical (|nz| < slope_thresh),
    es decir, que pertenecen a pendientes/estructura aprovechable por el ICP
    geométrico. ~0 = fondo plano (geometría inútil para XY), >0.1 = hay relieve.
    """

    down = _preprocessed(pcd, voxel)

    if not down.has_normals() or len(down.points) < 30:
        return 0.0

    n = np.asarray(down.normals)
    if len(n) == 0:
        return 0.0

    non_vertical = np.abs(n[:, 2]) < slope_thresh
    return float(np.mean(non_vertical))


def robust_hybrid_icp(
        source,
        target,
        T_init,
        icp_distance=2.0,
        max_iter=60,
        lambda_geometric=0.6,
        min_geometric_texture=0.08):
    """
    Registro ADAPTATIVO geometría/intensidad.

    Estrategia: usar ICP GEOMÉTRICO por defecto (es el que mejor funciona en
    este dataset y no introduce el ruido del término de color), y recurrir al
    Colored ICP (intensidad acústica) SOLO cuando el par de patches carece de
    textura geométrica suficiente — el caso para el que la intensidad existe:
    fondo plano sin relieve donde el ICP geométrico desliza.

    Decisión por par:
      texture = fracción de normales no verticales (relieve aprovechable).
      Si texture >= min_geometric_texture  → ICP geométrico (robust_icp).
      Si texture <  min_geometric_texture y hay color → Colored ICP.

    De este modo:
      - En zonas con estructura (barcos): geometría pura, precisa y sin el
        sesgo de compresión que el color introducía.
      - En fondo liso sin features: la intensidad aporta el gradiente que la
        geometría no tiene, evitando el deslizamiento.

    min_geometric_texture: umbral de fracción de normales con pendiente.
      0.08 = al menos un 8% de puntos en pendiente para fiarse de la geometría.
    """

    # Textura disponible en ambos patches (se usa la menor: el registro está
    # limitado por el patch con menos relieve).
    tex_s = _geometric_texture(source)
    tex_t = _geometric_texture(target)
    texture = min(tex_s, tex_t)

    has_color = source.has_colors() and target.has_colors()

    if texture >= min_geometric_texture or not has_color:
        # Hay relieve aprovechable (o no hay intensidad): geometría pura.
        return robust_icp(
            source,
            target,
            T_init,
            icp_distance=icp_distance,
            max_iter=max_iter
        )

    # Fondo plano sin textura geométrica: apoyarse en la intensidad.
    result = robust_colored_icp(
        source,
        target,
        T_init,
        icp_distance=icp_distance,
        max_iter=max_iter,
        lambda_geometric=lambda_geometric
    )

    if result is not None:
        return result

    # Si el Colored ICP falla, último recurso geométrico.
    return robust_icp(
        source,
        target,
        T_init,
        icp_distance=icp_distance,
        max_iter=max_iter
    )


def robust_colored_icp(
        source,
        target,
        T_init,
        icp_distance=2.0,
        max_iter=60,
        lambda_geometric=0.6):
    """
    ICP coarse-to-fine que combina geometría e INTENSIDAD acústica
    (Colored ICP de Open3D, Park et al. 2017).

    La intensidad (backscatter) viaja en el canal de color de las nubes
    (gris normalizado, cargado en PatchBuilder). El coste optimizado es:

        E = (1 - λ) · E_color + λ · E_geometric

    En fondo plano E_geometric tiene gradiente ≈0 en XY (el ICP geométrico
    desliza), pero E_color SÍ tiene gradiente porque la textura de intensidad
    varía espacialmente → la alineación deja de deslizar. En zonas con relieve
    real (barcos) la geometría sigue dominando vía λ.

    lambda_geometric ∈ [0,1]: 1.0 = solo geometría (≡ ICP normal),
    valores menores dan más peso a la intensidad. 0.6 es un punto de partida
    razonable para fondo plano con estructura concentrada.

    Requiere que source y target tengan `colors`. Si no los tienen, se delega
    en robust_icp (geométrico puro).
    """

    if (
        not source.has_colors()
        or not target.has_colors()
    ):
        return robust_icp(
            source,
            target,
            T_init,
            icp_distance=icp_distance,
            max_iter=max_iter
        )

    voxel_scales = [1.0, 0.5, 0.25]

    current_transform = T_init.copy()
    final_result = None

    estimator = (
        o3d.pipelines.registration.
        TransformationEstimationForColoredICP(
            lambda_geometric=lambda_geometric
        )
    )

    for voxel_scale in voxel_scales:

        voxel = 0.5 * voxel_scale

        # Cacheado por (id(pcd), voxel). voxel_down_sample conserva los colores
        # (la intensidad), que el Colored ICP necesita, y _preprocessed añade
        # las normales sin tocar el color.
        s = _preprocessed(source, voxel)
        t = _preprocessed(target, voxel)

        if len(s.points) < 50:
            continue

        if len(t.points) < 50:
            continue

        try:

            result = (
                o3d.pipelines.registration.
                registration_colored_icp(
                    s, t,
                    icp_distance,
                    current_transform,
                    estimator,
                    o3d.pipelines.registration.
                    ICPConvergenceCriteria(max_iteration=max_iter)
                )
            )

        except RuntimeError:

            # Colored ICP puede lanzar si una escala carece de gradiente de
            # color suficiente; se conserva la transformada previa.
            continue

        if len(result.correspondence_set) >= _MIN_ICP_CORRESPONDENCES:
            current_transform = result.transformation
            final_result = result

    if final_result is None:
        return None

    if len(final_result.correspondence_set) < _MIN_ICP_CORRESPONDENCES:
        return None

    return final_result