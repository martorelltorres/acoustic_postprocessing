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
# Distance based on cosine similarity between flattened descriptors.
# Includes a rotational alignment search by columns (sector shift)
# to be invariant to vehicle orientation.
# =============================================================================

def scan_context_distance(desc1, desc2, rk1_fft=None, rk2_fft=None):

    # Ring key: mean of each column → compact 1D vector for preselection
    rk1 = desc1.mean(axis=0)
    rk2 = desc2.mean(axis=0)

    n1 = np.linalg.norm(rk1)
    n2 = np.linalg.norm(rk2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 1.0

    # Find the optimal column shift via circular correlation
    # Efficient equivalent to trying all shifts but in O(S log S).
    # PERF: the ring-key FFT is invariant per patch; if the caller passes it
    # precomputed (rk1_fft, rk2_fft) we avoid recomputing the SAME FFT on each
    # comparison. The result (the correlation) is identical.
    if rk1_fft is None:
        rk1_fft = np.fft.fft(rk1)
    if rk2_fft is None:
        rk2_fft = np.fft.fft(rk2)

    corr = np.fft.ifft(
        rk1_fft * np.conj(rk2_fft)
    ).real

    best_shift = int(np.argmax(corr))

    # Evaluate cosine distance with the optimal shift
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
# Manages the descriptor database and the search for loop-closure candidates.
#
# Interface expected by multibeam_slam.py:
#
#   manager = ScanContextManager(min_temporal_gap=25)
#   manager.add_descriptor(pcd)               # Open3D PointCloud
#   candidates = manager.detect_loop_candidates(
#       idx, top_k=5, threshold=0.22
#   )  → list of (cand_idx, score) sorted by ascending score
#
# =============================================================================

class ScanContextManager:

    def __init__(
            self,
            num_rings=20,
            num_sectors=60,
            max_radius=40.0,
            min_temporal_gap=25):

        # Descriptor parameters
        self.num_rings = num_rings
        self.num_sectors = num_sectors
        self.max_radius = max_radius

        # Minimum temporal separation (in indices) to consider a
        # candidate a loop closure and not a sequential edge
        self.min_temporal_gap = min_temporal_gap

        # Descriptor database: list of (num_rings, num_sectors) arrays
        self.descriptors = []

        # Precomputed ring keys for fast column search
        self._ring_keys = []

        # PERF: ring-key FFT precomputed per patch. The circular correlation in
        # scan_context_distance reuses it on each comparison instead of
        # recomputing two FFTs per pair (same result, much less compute).
        self._ring_key_ffts = []

        # INS positions (north, east) per patch, for the spatial pre-filter.
        # Optional: if not provided, the search falls back to the classic O(N²).
        self._ins_xy = []
        self._kdtree = None

    # -------------------------------------------------------------------------
    # add_descriptor
    # -------------------------------------------------------------------------

    def add_descriptor(self, pcd, ins_xy=None):
        """
        Compute the Scan Context of an Open3D PointCloud and add it to the DB.

        ins_xy: optional (north, east) of the patch's INS pose. If provided for
        all patches, detect_loop_candidates pre-filters by spatial proximity
        with a KD-tree (from O(N²) to O(N log N + N·neighbors)).
        """

        points = np.asarray(pcd.points)

        desc = compute_scan_context(
            points,
            num_rings=self.num_rings,
            num_sectors=self.num_sectors,
            max_radius=self.max_radius
        )

        self.descriptors.append(desc)

        # Ring key: mean per column (sector) → 1D vector for preselection
        ring_key = desc.mean(axis=0)
        self._ring_keys.append(ring_key)
        self._ring_key_ffts.append(np.fft.fft(ring_key))

        if ins_xy is not None:
            self._ins_xy.append(
                np.asarray(ins_xy, dtype=float)
            )

        # The KD-tree is invalidated; rebuilt lazily on search.
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
        Search the top_k candidates most similar to descriptor query_idx
        that are separated by at least min_temporal_gap positions.

        If there are INS positions for all patches and max_ins_distance is
        passed, only the descriptors of spatially nearby patches are evaluated
        (KD-tree pre-filter). This avoids ~1456² FFT comparisons, which is the
        main loop-closure bottleneck.

        Returns a list of (cand_idx, distance) tuples sorted by ascending
        distance, filtered by the threshold.
        """

        if query_idx >= len(self.descriptors):
            return []

        query_desc = self.descriptors[query_idx]
        query_fft = self._ring_key_ffts[query_idx]

        # -- Candidate set to evaluate ---------------------------------------
        use_spatial = (
            _HAS_KDTREE
            and max_ins_distance is not None
            and len(self._ins_xy) == len(self.descriptors)
        )

        if use_spatial:

            if self._kdtree is None:
                self._kdtree = cKDTree(np.vstack(self._ins_xy))

            # Spatial neighbors of the query within max_ins_distance.
            cand_indices = self._kdtree.query_ball_point(
                self._ins_xy[query_idx],
                r=max_ins_distance
            )

        else:

            cand_indices = range(len(self.descriptors))

        distances = []

        for i in cand_indices:

            # Exclude nearby temporal neighbors
            if abs(i - query_idx) < self.min_temporal_gap:
                continue

            dist = scan_context_distance(
                query_desc,
                self.descriptors[i],
                rk1_fft=query_fft,
                rk2_fft=self._ring_key_ffts[i]
            )

            distances.append((i, dist))

        if len(distances) == 0:
            return []

        # Sort by ascending distance and filter by threshold
        distances.sort(key=lambda x: x[1])

        candidates = [
            (idx, dist)
            for idx, dist in distances[:top_k]
            if dist < threshold
        ]

        return candidates
