#!/usr/bin/env python3

import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAS_KDTREE = True
except ImportError:
    _HAS_KDTREE = False


# =============================================================================
# SCAN CONTEXT DESCRIPTOR
# =============================================================================

def compute_scan_context(
        points,
        num_rings=20,
        num_sectors=60,
        max_radius=40.0):

    desc = np.full(
        (num_rings, num_sectors),
        -np.inf,
        dtype=np.float32
    )

    if len(points) == 0:
        return np.zeros(
            (num_rings, num_sectors),
            dtype=np.float32
        )

    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]

    if len(points) == 0:
        return np.zeros(
            (num_rings, num_sectors),
            dtype=np.float32
        )

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    r = np.sqrt(x**2 + y**2)

    theta = np.degrees(
        np.arctan2(y, x)
    )

    theta[theta < 0] += 360

    ring_idx = np.clip(
        (r / max_radius * num_rings).astype(int),
        0,
        num_rings - 1
    )

    sector_idx = np.clip(
        (theta / 360.0 * num_sectors).astype(int),
        0,
        num_sectors - 1
    )

    np.maximum.at(
        desc,
        (ring_idx, sector_idx),
        z
    )

    desc[~np.isfinite(desc)] = 0.0

    return desc


# =============================================================================
# SCAN CONTEXT DISTANCE
# =============================================================================
# Distancia basada en similitud coseno entre descriptores aplanados.
# Incluye búsqueda de alineación rotacional por columnas (sector shift)
# para ser invariante a la orientación del vehículo.
# =============================================================================

def scan_context_distance(desc1, desc2):

    # Ring key: media de cada columna → vector 1D compacto para preselección
    rk1 = desc1.mean(axis=0)
    rk2 = desc2.mean(axis=0)

    n1 = np.linalg.norm(rk1)
    n2 = np.linalg.norm(rk2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 1.0

    # Encontrar el desplazamiento de columna óptimo vía correlación circular
    # Equivalente eficiente a probar todos los shifts pero en O(S log S)
    corr = np.fft.ifft(
        np.fft.fft(rk1) * np.conj(np.fft.fft(rk2))
    ).real

    best_shift = int(np.argmax(corr))

    # Evaluar distancia coseno con el shift óptimo
    desc2_shifted = np.roll(desc2, best_shift, axis=1)

    d1 = desc1.flatten()
    d2 = desc2_shifted.flatten()

    n1f = np.linalg.norm(d1)
    n2f = np.linalg.norm(d2)

    if n1f < 1e-6 or n2f < 1e-6:
        return 1.0

    similarity = np.dot(d1, d2) / (n1f * n2f)

    return 1.0 - float(similarity)


# =============================================================================
# SCAN CONTEXT MANAGER
# =============================================================================
# Gestiona la base de datos de descriptores y la búsqueda de candidatos
# a cierre de bucle.
#
# Interfaz esperada por multibeam_slam.py:
#
#   manager = ScanContextManager(min_temporal_gap=25)
#   manager.add_descriptor(pcd)               # Open3D PointCloud
#   candidates = manager.detect_loop_candidates(
#       idx, top_k=5, threshold=0.22
#   )  → lista de (cand_idx, score) ordenada por score ascendente
#
# =============================================================================

class ScanContextManager:

    def __init__(
            self,
            num_rings=20,
            num_sectors=60,
            max_radius=40.0,
            min_temporal_gap=25):

        # Parámetros del descriptor
        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self.max_radius = max_radius

        # Mínima separación temporal (en índices) para considerar un
        # candidato como cierre de bucle y no como arista secuencial
        self.min_temporal_gap = min_temporal_gap

        # Base de datos de descriptores: lista de arrays (num_rings, num_sectors)
        self.descriptors = []

        # Ring keys precalculadas para búsqueda rápida por columnas
        self._ring_keys = []

        # Posiciones INS (norte, este) por patch, para el pre-filtro espacial.
        # Opcional: si no se proporcionan, la búsqueda recae al O(N²) clásico.
        self._ins_xy = []
        self._kdtree = None

    # -------------------------------------------------------------------------
    # add_descriptor
    # -------------------------------------------------------------------------

    def add_descriptor(self, pcd, ins_xy=None):
        """
        Calcula el Scan Context de un Open3D PointCloud y lo añade a la BD.

        ins_xy: (norte, este) opcional de la pose INS del patch. Si se aporta
        para todos los patches, detect_loop_candidates pre-filtra por proximidad
        espacial con un KD-tree (de O(N²) a O(N log N + N·vecinos)).
        """

        points = np.asarray(pcd.points)

        desc = compute_scan_context(
            points,
            num_rings=self.num_rings,
            num_sectors=self.num_sectors,
            max_radius=self.max_radius
        )

        self.descriptors.append(desc)

        # Ring key: media por columna (sector) → vector 1D para preselección
        self._ring_keys.append(desc.mean(axis=0))

        if ins_xy is not None:
            self._ins_xy.append(
                np.asarray(ins_xy, dtype=float)
            )

        # El KD-tree se invalida; se reconstruye perezosamente al buscar.
        self._kdtree = None

    # -------------------------------------------------------------------------
    # detect_loop_candidates
    # -------------------------------------------------------------------------

    def detect_loop_candidates(
            self,
            query_idx,
            top_k=5,
            threshold=0.22,
            max_ins_distance=None):
        """
        Busca los top_k candidatos más similares al descriptor query_idx
        que estén separados al menos min_temporal_gap posiciones.

        Si hay posiciones INS para todos los patches y se pasa max_ins_distance,
        solo se evalúan los descriptores de patches espacialmente cercanos
        (pre-filtro con KD-tree). Esto evita ~1456² comparaciones con FFT,
        que es el principal cuello de botella del loop closure.

        Retorna lista de tuplas (cand_idx, distancia) ordenada por distancia
        ascendente, filtrada por el umbral threshold.
        """

        if query_idx >= len(self.descriptors):
            return []

        query_desc = self.descriptors[query_idx]

        # -- Conjunto de candidatos a evaluar --------------------------------
        use_spatial = (
            _HAS_KDTREE
            and max_ins_distance is not None
            and len(self._ins_xy) == len(self.descriptors)
        )

        if use_spatial:

            if self._kdtree is None:
                self._kdtree = cKDTree(np.vstack(self._ins_xy))

            # Vecinos espaciales del query dentro de max_ins_distance.
            cand_indices = self._kdtree.query_ball_point(
                self._ins_xy[query_idx],
                r=max_ins_distance
            )

        else:

            cand_indices = range(len(self.descriptors))

        distances = []

        for i in cand_indices:

            # Excluir vecinos temporales próximos
            if abs(i - query_idx) < self.min_temporal_gap:
                continue

            dist = scan_context_distance(
                query_desc,
                self.descriptors[i]
            )

            distances.append((i, dist))

        if len(distances) == 0:
            return []

        # Ordenar por distancia ascendente y filtrar por umbral
        distances.sort(key=lambda x: x[1])

        candidates = [
            (idx, dist)
            for idx, dist in distances[:top_k]
            if dist < threshold
        ]

        return candidates
