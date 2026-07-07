#!/usr/bin/env python3

"""
===============================================================================
ADVANCED UNDERWATER MULTIBEAM SLAM
FULL METRICS + CONNECTED POSE GRAPH + RAW/SLAM MAP EXPORT
===============================================================================
"""

import os

# -----------------------------------------------------------------------------
# PERF: thread-pool wait policy (OpenMP / OpenBLAS).
# Open3D and NumPy/SciPy each load their own OpenMP + OpenBLAS runtime. By default
# idle threads BUSY-WAIT (spin) on a futex between calls, saturating ~all cores
# with no useful work ("373% phantom CPU" effect).
#   - OMP_WAIT_POLICY=passive  -> idle threads SLEEP instead of spinning.
#   - *_NUM_THREADS bounded     -> prevents the two pools fighting over cores.
# This does NOT alter any numeric result: real parallel work is unchanged, only
# the wasted spin is removed. MUST run before importing open3d/numpy.
# -----------------------------------------------------------------------------
os.environ.setdefault("OMP_WAIT_POLICY", "passive")
_n_threads = str(max(1, (os.cpu_count() or 4) // 2))
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, _n_threads)

import copy
import json
import csv
import time

import rospy
import rosbag
import ros_numpy
import numpy as np
import open3d as o3d
import tf.transformations as tr

# PERF: bound Open3D's internal thread pool to the same level as the others
# (reinforces OMP_NUM_THREADS via the API itself). Does not change results.
try:
    o3d.utility.set_num_threads(int(_n_threads))
except (AttributeError, ValueError):
    pass

from tqdm import tqdm
from scipy.interpolate import interp1d

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False

# Sibling modules (utils, registration, ...) are imported with a flat name,
# which assumes the script directory is on sys.path. True when run directly from
# scripts/, but NOT when catkin installs the script (catkin_install_python
# generates a wrapper that runs the .py from another path via exec(), without the
# module directory on sys.path -> ModuleNotFoundError). Explicitly adding this
# file's directory makes the flat imports work in both cases (direct execution
# and via the catkin wrapper).
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import *
from registration import *
from robust_icp import *
# _geometric_texture is not exported by `import *` (leading underscore); imported
# explicitly to modulate the sequential consistency gates by the pair's geometric
# texture (see SEQ_GATE_TEXTURE_* and the ICP-navigation gate).
from robust_icp import _geometric_texture
from robust_ndt import *
from scan_context import *
from information_matrix import *
from visualization import *
# Roman 2006 consistency metric (consistency error): primary measurement
# instrument of the submap bathymetric SLAM state of the art. Measures the
# vertical dispersion in overlap regions (incl. adjacent overlap between parallel
# strips), exactly what SLAM must minimize. See scripts/consistency.py.
from consistency import consistency_error_from_patches
# Registration covariance (pICP, R1 — Palomer 2016 / Censi 2007): ANISOTROPIC
# edge information derived from the pair's real geometry, not isotropic. On flat
# seabed gives low cross-track information (where registration is undetermined)
# and high along-track. See scripts/registration_covariance.py.
from registration_covariance import (
    registration_covariance_3dof,
    information_6dof_from_cov3,
    intensity_informativeness,
    fuse_geometry_intensity_cov,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

PATCH_SIZE = 100
PATCH_STRIDE = 20

VOXEL_SIZE = 0.25

FINAL_DOWNSAMPLE = 0.2

# Cell side (m) of the Roman 2006 consistency-error grid. ~0.5-1.0 m is typical in
# the SOTA (Roman/Torroba use ~0.5 m). See scripts/consistency.py.
CONSISTENCY_CELL_SIZE = 1.0

ANGLE_CUTOFF_DEG = 55.0

MIN_PATCH_POINTS = 500

MAX_PATCH_EXTENT = 500.0

# =============================================================================
# ICP
# =============================================================================

ICP_DISTANCE = 2.0

ICP_MAX_ITER = 60

REGISTRATION_ALGORITHM = "icp"

# Geometric weight of Colored ICP (registration_algorithm="colored_icp").
# 1.0 = geometry only (== icp); lower values give more weight to acoustic
# intensity. 0.6 = starting point for flat seabed with concentrated structure.
COLORED_ICP_LAMBDA = 0.6

# "hybrid" mode: geometric texture threshold to choose geometry vs intensity.
# Minimum fraction of sloped (non-vertical) normals to trust the geometric ICP;
# below it, the pair is registered with Colored ICP (intensity).
# 0.08 = at least 8% relief to use pure geometry.
HYBRID_MIN_TEXTURE = 0.08

NDT_RESOLUTION = 1.0

NDT_MAX_POINTS = 3000

NDT_MIN_POINTS_PER_VOXEL = 5

# -----------------------------------------------------------------------------
# FITNESS
# -----------------------------------------------------------------------------

FITNESS_THRESHOLD = 0.55

LOOP_FITNESS_THRESHOLD = 0.90

# -----------------------------------------------------------------------------
# RMSE THRESHOLDS
# -----------------------------------------------------------------------------
# MBES acoustic clouds are naturally noisy.
# RMSE values close to voxel size are completely normal.
#
# Sequential ICP:
#   More permissive because overlap is very high.
#
# Loop Closures:
#   Even more permissive due to accumulated drift.
# -----------------------------------------------------------------------------

SEQ_RMSE_THRESHOLD = VOXEL_SIZE * 1.5

LOOP_RMSE_THRESHOLD = VOXEL_SIZE * 1.5

# -----------------------------------------------------------------------------
# CORRESPONDENCES
# -----------------------------------------------------------------------------

MIN_CORRESPONDENCES = 100

# -----------------------------------------------------------------------------
# ICP–NAVIGATION CONSISTENCY GATE  (sequential registration)
# -----------------------------------------------------------------------------
# On a flat, featureless underwater seabed the GICP cost function is flat in
# XY: surface normals all point vertically, so the point-to-plane residual has
# near-zero gradient for in-plane translation.  The coarse-to-fine ICP can
# leave T_init and converge to a completely wrong local minimum — including
# reversed translations (~180° flip) or perpendicular ones.
#
# These failures pass the RMSE/fitness gates because a flat surface aligns
# well regardless of XY position.  They must be caught by comparing the ICP
# result with the navigation expectation (T_init):
#
#   MAX_SEQ_ICP_TRANSLATION_DEV  – maximum allowed difference between the ICP
#       translation and T_init translation in the local patch frame (metres).
#       Catches large-magnitude deviations.
#
#   MAX_SEQ_ICP_YAW_DEV  – maximum allowed angular deviation between the ICP
#       rotation and the T_init rotation (degrees).
#       Catches direction reversals (170°) and perpendicular divergences (90°).
#
# When either gate fires the step falls back to T_init (navigation) with low
# information weight — the same strategy used for RMSE-rejected steps.
# -----------------------------------------------------------------------------

MAX_SEQ_ICP_TRANSLATION_DEV = 0.5   # m
MAX_SEQ_ICP_YAW_DEV         = 15.0  # degrees

# El umbral de desviación de traslación se aplicaba como cota ABSOLUTA (0.5 m).
# Pero la desviación admisible del ICP debe escalar con el PASO INS del par: con
# patches poco solapados (p.ej. patch_stride=80 -> paso ~0.55 m) una desviación
# de 0.5 m es una fracción enorme del paso y rechaza correcciones válidas (medido:
# 407 rechazos icp_translation_deviation, 26% de los registros). El umbral efectivo
# pasa a ser  max(MAX_SEQ_ICP_TRANSLATION_DEV, SEQ_TR_DEV_STEP_GAIN · paso_INS),
# de modo que en pasos largos se permite proporcionalmente más deriva del ICP sin
# aflojar el suelo mínimo de 0.5 m en pasos cortos. gain=0 recupera el umbral fijo.
SEQ_TR_DEV_STEP_GAIN = 0.5

# -----------------------------------------------------------------------------
# GATE MODULATION BY GEOMETRIC TEXTURE  (sequential registration)
# -----------------------------------------------------------------------------
# The consistency gates (translation/direction deviation vs INS) exist to curb
# ICP slide on FLAT SEABED, where geometry does not constrain XY and the ICP
# converges to false minima even at fitness~1. But with a FIXED low threshold
# (0.5 m / 15°) they also reject VALID ICP corrections in areas with real RELIEF,
# where geometry is reliable (measured: 62% of steps fall back to INS despite
# fitness~1 on a dataset with texture 0.49).
#
# Fix: modulate the threshold by the pair's geometric texture (fraction of
# non-vertical normals, the same proxy the hybrid uses). Where there is NO relief
# (texture->0) the gate is left intact (anti-flat-seabed protection preserved);
# where there IS relief (high texture) it relaxes up to SEQ_GATE_TEXTURE_RELAX x.
#
#   relax = 1 + (SEQ_GATE_TEXTURE_RELAX − 1) · clip(texture/SEQ_GATE_TEXTURE_FULL, 0, 1)
#   effective_threshold = base_threshold · relax
#
# SEQ_GATE_TEXTURE_RELAX = 1.0 -> disables modulation (previous behavior).
# SEQ_GATE_TEXTURE_FULL  = texture at which full relaxation is reached.
# With RELAX=2.0, FULL=0.30: texture 0 -> gate 0.5 m/15°; texture >=0.30 -> 1.0 m/30°.
# -----------------------------------------------------------------------------

SEQ_GATE_TEXTURE_RELAX = 2.0
SEQ_GATE_TEXTURE_FULL  = 0.30

# -----------------------------------------------------------------------------
# ICP TRANSLATION LENGTH-RATIO GATE  (sequential registration)
# -----------------------------------------------------------------------------
# With 80% patch overlap (PATCH_SIZE=100, PATCH_STRIDE=20) the ICP translation
# is biased toward the centroid of the overlapping region, so accepted steps
# are systematically ~8% SHORTER than the true INS motion. Integrated over
# ~1100 edges this compresses the SLAM path by ~7% and the position lags
# progressively behind the INS → XY error that grows with distance.
#
# The INS step length (DVL+IMU baseline) is reliable. We therefore compare the
# ICP translation magnitude with the INS-expected magnitude (|T_init[:2,3]|):
#
#   ratio = |T_icp_xy| / |T_init_xy|
#
# If the ratio falls outside [MIN, MAX], the ICP has mis-scaled the step
# (overlap-slide or spurious stretch) and we keep the ICP DIRECTION but rescale
# the translation magnitude to the INS length. The ICP rotation is already
# replaced by the INS rotation (Fix 1), so this confines the ICP contribution
# to a reliable, scale-correct in-plane refinement.
#
# Band [0.85, 1.15] on the derelictes mission cuts the path deficit 7.1% → 3.3%.
# -----------------------------------------------------------------------------

MIN_SEQ_ICP_LENGTH_RATIO = 0.85
MAX_SEQ_ICP_LENGTH_RATIO = 1.15

# -----------------------------------------------------------------------------
# TRANSLATION SCALE ANCHORING  (sequential registration)
# -----------------------------------------------------------------------------
# With SEQ_ANCHOR_SCALE=True each step's magnitude is fixed to the INS prior and
# the ICP supplies only the direction. Flattened the ICP peaks (max 15->9.7m) but
# introduced a systematic bias that RAISED the mean correction (4.14->4.85m) and
# linearized the drift (error-distance correlation 0.59->0.69). Disabled by
# default: the length gate (Fix A), band [0.85,1.15], is used instead — the state
# that gave the best baseline (mean 4.14m).
# -----------------------------------------------------------------------------

SEQ_ANCHOR_SCALE = False

# -----------------------------------------------------------------------------
# CROSS-TRACK GAIN OF THE ICP CORRECTION  (Option A1)
# -----------------------------------------------------------------------------
# The ICP XY translation decomposes into along-track (forward, INS direction) and
# cross-track (lateral, perpendicular). The ICP cross-track component injects a
# systematic lateral bias (+23 mm/step at turns, always toward +East) that shifts
# the lawnmower pattern: contracts left and overshoots right, an error that grows
# with the mission. With the INS reliable in heading, that lateral correction only
# adds error.
#   1.0 = full ICP lateral correction
#   0.0 = ICP along-track + INS cross-track (no lateral bias)
# The ICP along-track is kept (refines the forward scale where there is
# structure); only the lateral is damped.
SEQ_CROSS_TRACK_GAIN = 0.0

# -----------------------------------------------------------------------------
# HIGH-CONFIDENCE FALLBACK
# -----------------------------------------------------------------------------
# If fitness is extremely high and correspondences are massive,
# allow higher RMSE before rejecting.
# This is extremely important in flat underwater environments.
# -----------------------------------------------------------------------------

HIGH_FITNESS_THRESHOLD = 0.90

HIGH_CORRESPONDENCE_THRESHOLD = 1000

# Must be > 1 to have an effect. At 1 the high-confidence block relaxes
# nothing (multiply by 1). Value 1.5 allows up to VOXEL_SIZE * 3.0 m of
# RMSE in high-coverage areas.
HIGH_RMSE_MULTIPLIER = 1.5

# =============================================================================
# LOOP CLOSURE
# =============================================================================

ENABLE_LOOP_CLOSURE = True

SCAN_CONTEXT_THRESHOLD = 0.15

# -----------------------------------------------------------------------------
# VOXEL DEL REGISTRO GLOBAL DE LOOP CLOSURE (RANSAC-FPFH)  — DESACOPLADO
# -----------------------------------------------------------------------------
# execute_global_registration recibía VOXEL_SIZE (el voxel del PATCH). El voxel
# del patch se ajusta por ALTITUD (fino en vuelo rasante: 0.10 m a 3 m), pero el
# RANSAC-FPFH necesita un voxel más GRUESO para que el descriptor FPFH tenga
# soporte y sea discriminativo, y para que el umbral de correspondencia
# (voxel·1.5) no sea absurdamente estricto. Con voxel 0.10 el umbral caía a
# 0.15 m y RANSAC fallaba 5362/5380 veces PESE A HABER RELIEVE (mediana 4.4 m de
# rango Z por celda de 5 m) — no era fondo plano, era el voxel demasiado fino.
# Se fija un voxel propio para el loop closure, independiente del voxel del patch.
# None -> usa max(VOXEL_SIZE, LOOP_RANSAC_MIN_VOXEL).
LOOP_RANSAC_VOXEL = 0.30
LOOP_RANSAC_MIN_VOXEL = 0.25

# Poda pre-RANSAC: si AMBOS patches del par son geométricamente planos
# (textura < umbral, casi todas las normales verticales) FPFH no discrimina y
# RANSAC no converge — saltar el par ahorra el cómputo. CONSERVADOR: solo se poda
# cuando el par NO tiene relieve por ninguno de los dos lados; con relieve real
# (este dataset: mediana 4.4 m de rango Z / celda) no se poda nada. 0.0 -> desactiva.
LOOP_MIN_TEXTURE_FOR_RANSAC = 0.05

MAX_LOOP_CANDIDATES = 3

MAX_LOOP_Z_TRANSLATION = 0.5

MAX_LOOP_XY_TRANSLATION = 8.0

MAX_LOOP_YAW_DEG = 20.0

# -----------------------------------------------------------------------------
# MINIMUM TEMPORAL GAP  (loop closure)
# -----------------------------------------------------------------------------
# Minimum temporal separation (in patch indices) between two patches for them to
# be loop-closure candidates. Baseline value (4.14m).
# -----------------------------------------------------------------------------

MIN_LOOP_TEMPORAL_GAP = 25

# -----------------------------------------------------------------------------
# INS PROXIMITY GATE
# -----------------------------------------------------------------------------
# Only consider loop closure candidates whose INS positions are closer than
# this distance. Prevents false positives between far-away parts of the
# trajectory that merely look similar (e.g. parallel lawnmower strips on a
# flat, featureless seabed).
#
# Rule of thumb: set to 2-3 × the maximum expected INS drift over the mission.
# For a typical AUV DVL+IMU over 300-500 m without GPS: 8-15 m.
# -----------------------------------------------------------------------------

MAX_LOOP_INS_DISTANCE = 12.0

# -----------------------------------------------------------------------------
# INS<->LOOP CONSISTENCY GATE  (redesigned — 3 AND conditions)
# -----------------------------------------------------------------------------
# A closure must distinguish a REAL REVISIT (the AUV returns to the same area; the
# INS has drift the ICP corrects) from a PARALLEL-STRIP FALSE POSITIVE (two
# adjacent lawnmower passes, laterally separated, whose seabed looks alike because
# the wide MBES beam overlaps them). All THREE are required (AND):
#
#   1) MAX_LOOP_REVISIT_INS_DIST — ins_distance <= this. A real closure is between
#      points the INS places CLOSE; parallel strips (4-5 m) fall outside.
#      Default 2.5 m (~ max drift between two passes over the same point).
#
#   2) MIN_LOOP_ICP_INS_RATIO — |T_raw_xy|/ins_distance >= this (only if
#      ins_distance >= MIN_INS_DIST_FOR_RATIO_CHECK). Rejects |T_raw|~0, the
#      signature of the similar-seabed false positive (ICP "fits" without moving).
#
#   3) MAX_LOOP_INS_DISCREPANCY — ||T_raw_xy − T_ins_rel_xy|| <= this. Anti-nonsense
#      upper bound (T_raw incoherent with the INS).
#
# HISTORY: the ORIGINAL gate was only (2). It was "reformulated" to only (3),
# believing (2) rejected good closures — WRONG: with only (3), 321 parallel-strip
# false positives were accepted and the optimizer COLLAPSED the lawnmower (width
# 49 m -> 15 m, mean correction 1.3 -> 16.8 m). Condition (2) WAS the correct
# protection; the redesign restores it and combines it with (1) and (3).
# See results/metrics/ANALISIS_RESULTADOS.md §2.
#
# On a single-pass lawnmower (no revisits) this gives ~0 closures = correct.
# -----------------------------------------------------------------------------

MAX_LOOP_REVISIT_INS_DIST = 2.5
MIN_LOOP_ICP_INS_RATIO   = 0.40
MIN_INS_DIST_FOR_RATIO_CHECK = 2.0
MAX_LOOP_INS_DISCREPANCY = 8.0

# =============================================================================
# CROSS-TRACK OVERLAP CONSTRAINTS (R0.2 — Torroba 2020 style)
# =============================================================================
# On a lawnmower WITHOUT crossings, the drift between parallel strips is NOT
# corrected by loop closures (no revisits: see ANALISIS_RESULTADOS §2f). What DOES
# exist is the LATERAL OVERLAP between adjacent strips (one line's MBES beam sees
# part of the seabed the neighboring line sees). Torroba 2020 ("Industrial-Scale
# Bathymetric Surveying") exploits exactly that adjacent overlap as a graph
# constraint, and reduces consistency error ~44% on crossing-free datasets
# ("Ripples").
#
# KEY distinction from loop closure:
#   - Loop closure: looks for a REVISIT (small ins_distance, ~0). None on single pass.
#   - Adjacent overlap: looks for NEIGHBORING STRIPS (ins_distance ~ line spacing,
#     typically half the swath width). No revisit required.
#
# Registration is seeded with the INS PRIOR (not RANSAC): between neighboring
# strips the INS gets heading and forward motion right; only the lateral
# (cross-track) offset drifts — exactly what registration must estimate. The edge
# is accepted if the correction is coherent with the prior (bounded discrepancy):
# it does NOT merge strips (the info matrix keeps them at their separation), only
# corrects their relative drift.
ENABLE_CROSS_TRACK_EDGES = True

# R1 (pICP) — anisotropic edge information from the registration's real covariance
# (Censi/Palomer). True = anisotropic information; False = prior isotropic
# (dynamic_information_matrix). Allows R0 (False) vs R1 (True) ablation.
USE_REGISTRATION_COVARIANCE = True

# R2 (robust back-end) — Choi 2015 line process (~ Switchable Constraints).
# edge_prune_threshold: line-process threshold below which an uncertain edge is
# deemed spurious and pruned (higher = more aggressive). preference_loop_closure:
# relative trust in uncertain edges (loop/cross-track) vs odometry; <1.0 = back-end
# more skeptical of them (recommended: protects from parallel-strip false
# positives). Only affects uncertain=True edges; odometry is respected.
EDGE_PRUNE_THRESHOLD = 0.25
PREFERENCE_LOOP_CLOSURE = 0.6

# -----------------------------------------------------------------------------
# GATE ANTI-DIVERGENCIA DEL OPTIMIZADOR GLOBAL (post-optimización)
# -----------------------------------------------------------------------------
# El line process actúa por arista (residuo individual) y NO detecta una
# divergencia COLECTIVA: un conjunto de aristas individualmente plausibles puede
# empujar al optimizador LM a una solución con nodos catapultados decenas de metros
# (medido en Andratx octógonos: correcciones de hasta 89 m sobre un mapa de 55×36 m,
# path SLAM 2032 m vs 1034 m raw). La corrección XY del SLAM respecto al INS debe ser
# físicamente acotada: el INS deriva como mucho unos metros en una misión compacta.
# Tras optimizar, si un nodo se ha desplazado más de MAX_NODE_CORRECTION_M respecto a
# su posición INS, es una divergencia: se revierte su XY a la navegación bruta (mismo
# principio que el fallback secuencial). 0.0 desactiva el gate.
MAX_NODE_CORRECTION_M = 10.0

# R3 (novel contribution) — use the BACKSCATTER (intensity channel) in the
# uncertainty model: where geometry is flat but intensity has rich texture, the
# cross-track uncertainty is reduced (Colored ICP supplies XY gradient). Neither
# Palomer nor Torroba/Tan use intensity in their covariance. intensity_gain scales
# the max reduction (1.0 = can drop to the floor when intensity texture is
# maximal). This is the ANALYTIC component of R3; PointNetKL (a network that learns
# the covariance including backscatter) is the evolution, with the same
# edge_information interface.
USE_INTENSITY_IN_COVARIANCE = True
INTENSITY_COV_GAIN = 1.0
# INS distance band (m) to consider two patches "neighboring strips". The minimum
# excludes the same strip / nearly overlapping patches; the maximum excludes
# non-adjacent strips. Tune to the lawnmower's real line spacing.
XTRACK_MIN_INS_DIST = 1.5
XTRACK_MAX_INS_DIST = 8.0
# Minimum temporal separation (nb of patches) for a pair to count as inter-strip
# and not as sequential neighborhood (already covered by sequential registration).
XTRACK_MIN_TEMPORAL_GAP = 40
# -----------------------------------------------------------------------------
# AUTO-CALIBRACIÓN DE LA BANDA CROSS-TRACK  (por geometría real de la misión)
# -----------------------------------------------------------------------------
# La banda [min,max] y el gap temporal DEPENDEN de la misión (lawnmower de 23 m
# en Cabrera, octógonos de 10-22 m en Andratx a 3 m de altitud...). Ajustarla a
# mano es frágil: la banda [12,35] de Cabrera generó 3595 candidatos y solo 3
# aristas útiles en los octógonos. Con XTRACK_AUTO_TUNE=True se mide la separación
# real entre pasadas vecinas del propio recorrido INS: para cada patch, el vecino
# espacial más cercano que NO es secuencial (|Δidx| >= gap temporal); la MEDIANA
# de esas distancias es la separación típica entre pasadas. La banda se centra en
# ella con una tolerancia relativa, y el gap temporal se deriva de cuántos patches
# transcurren de media hasta revisitar esa vecindad. Los valores fijados a mano
# quedan como respaldo si el auto-tune se desactiva o no encuentra estructura.
XTRACK_AUTO_TUNE = True
# Semi-anchura relativa de la banda alrededor de la mediana de separación:
# [mediana·(1-tol), mediana·(1+tol)]. 0.5 -> ±50%.
XTRACK_AUTO_BAND_TOL = 0.5
# Max overlap edges per patch (nearest in INS first), to bound cost.
XTRACK_MAX_EDGES_PER_PATCH = 2
# Minimum registration quality to accept the edge.
XTRACK_MIN_FITNESS = 0.30
# Max XY discrepancy (m) between the registration correction and the INS prior.
# Coherence bound: rejects spurious alignments that would move the strip absurdly.
XTRACK_MAX_INS_DISCREPANCY = 3.0

# =============================================================================
# MONITOR
# =============================================================================

# Refresh frequency of the pose-graph visualizer during sequential
# registration. The Open3D window updates every MONITOR_UPDATE_EVERY
# iterations.
#
#   1  -> refresh every step      (max freshness, more CPU/GPU load)
#   5  -> recommended balance      (default)
#   10 -> light refresh            (long missions, limited hardware)
#
# The monitor ALWAYS updates at the end of the sequential loop, on
# accepted loop closures, and after global optimization, regardless
# of this value.

MONITOR_UPDATE_EVERY = 5


# =============================================================================
# PATCH
# =============================================================================

class Patch:

    def __init__(self):

        self.idx = None

        self.pcd = None

        self.center = None

        self.pose = None

        self.timestamp = None


# =============================================================================
# TF
# =============================================================================

def get_static_transform_from_tf(
        bag_file,
        parent_frame,
        child_frame):

    bag = rosbag.Bag(bag_file)

    for _, msg, _ in bag.read_messages(
            topics=['/tf_static', '/tf']):

        for transform in msg.transforms:

            if (
                transform.header.frame_id == parent_frame
                and
                transform.child_frame_id == child_frame
            ):

                q = transform.transform.rotation

                t = transform.transform.translation

                T = tr.quaternion_matrix(
                    [q.x, q.y, q.z, q.w]
                )

                T[:3, 3] = [
                    t.x,
                    t.y,
                    t.z
                ]

                bag.close()

                return T

    bag.close()

    return np.eye(4)


# =============================================================================
# NAVIGATION
# =============================================================================

class NavigationInterpolator:

    def __init__(
            self,
            bag,
            nav_topic):

        self.timestamps = []

        north = []
        east = []
        depth = []

        yaw = []
        pitch = []
        roll = []

        rospy.loginfo(
            "Loading navigation..."
        )

        for _, msg, _ in tqdm(
                bag.read_messages(
                    topics=[nav_topic]
                ),
                desc="Navigation"):

            ts = msg.header.stamp.to_sec()

            self.timestamps.append(ts)

            north.append(
                msg.position.north
            )

            east.append(
                msg.position.east
            )

            depth.append(
                msg.position.depth
            )

            yaw.append(
                msg.orientation.yaw
            )

            pitch.append(
                msg.orientation.pitch
            )

            roll.append(
                msg.orientation.roll
            )

        self.timestamps = np.array(
            self.timestamps
        )

        if len(self.timestamps) == 0:
            raise RuntimeError(
                f"No navigation messages found in topic {nav_topic}"
            )

        self.min_timestamp = float(
            np.min(self.timestamps)
        )

        self.max_timestamp = float(
            np.max(self.timestamps)
        )

        self.f_n = interp1d(
            self.timestamps,
            north,
            fill_value='extrapolate'
        )

        self.f_e = interp1d(
            self.timestamps,
            east,
            fill_value='extrapolate'
        )

        self.f_d = interp1d(
            self.timestamps,
            depth,
            fill_value='extrapolate'
        )

        self.f_y = interp1d(
            self.timestamps,
            np.unwrap(yaw),
            fill_value='extrapolate'
        )

        self.f_p = interp1d(
            self.timestamps,
            np.unwrap(pitch),
            fill_value='extrapolate'
        )

        self.f_r = interp1d(
            self.timestamps,
            np.unwrap(roll),
            fill_value='extrapolate'
        )

    def pose_values(
            self,
            ts):

        if not self.has_timestamp(ts):
            return None

        return {

            "north": float(
                self.f_n(ts)
            ),

            "east": float(
                self.f_e(ts)
            ),

            "depth": float(
                self.f_d(ts)
            ),

            "yaw": float(
                self.f_y(ts)
            ),

            "pitch": float(
                self.f_p(ts)
            ),

            "roll": float(
                self.f_r(ts)
            )
        }

    def has_timestamp(
            self,
            ts):

        return (
            self.min_timestamp <= ts <= self.max_timestamp
        )


# =============================================================================
# PATCH BUILDER
# =============================================================================

class PatchBuilder:

    def __init__(self):
        # AVG banding-correction profile by incidence angle.
        # Filled in build() via _build_avg_profile(); None = no correction.
        self._avg_centers = None
        self._avg_gain = None

    def _select_scan_time_source(
            self,
            scans,
            nav):

        header_times = np.array([
            scan_entry["header_ts"]
            for scan_entry in scans
        ])

        bag_times = np.array([
            scan_entry["bag_ts"]
            for scan_entry in scans
        ])

        def overlap_ratio(times):

            valid = np.isfinite(times)

            if not np.any(valid):
                return 0.0

            inside = (
                (times[valid] >= nav.min_timestamp) &
                (times[valid] <= nav.max_timestamp)
            )

            return float(np.mean(inside))

        candidates = [
            (
                "header",
                header_times,
                0.0,
                overlap_ratio(header_times)
            ),
            (
                "bag",
                bag_times,
                0.0,
                overlap_ratio(bag_times)
            ),
        ]

        for name, times in [
                ("header_offset", header_times),
                ("bag_offset", bag_times)]:

            valid = np.isfinite(times)

            if not np.any(valid):
                continue

            offset = (
                nav.min_timestamp -
                float(np.min(times[valid]))
            )

            shifted = times + offset

            candidates.append((
                name,
                times,
                offset,
                overlap_ratio(shifted)
            ))

        best = max(
            candidates,
            key=lambda item: item[3]
        )

        name, _, offset, ratio = best

        rospy.loginfo(
            f"Scan timestamp source: {name} "
            f"(offset={offset:.6f}, "
            f"navigation overlap={ratio * 100.0:.1f}%)"
        )

        if ratio < 0.5:

            rospy.logwarn(
                "Low overlap between scan timestamps and navigation. "
                "Patch generation may skip many scans. "
                f"Navigation range=[{nav.min_timestamp:.3f}, "
                f"{nav.max_timestamp:.3f}], "
                f"scan header range=[{np.nanmin(header_times):.3f}, "
                f"{np.nanmax(header_times):.3f}], "
                f"scan bag range=[{np.nanmin(bag_times):.3f}, "
                f"{np.nanmax(bag_times):.3f}]"
            )

        def timestamp(scan_entry):

            if name.startswith("bag"):
                return scan_entry["bag_ts"] + offset

            return scan_entry["header_ts"] + offset

        return timestamp

    def _build_avg_profile(self, scans):
        """
        Compute the gain profile by incidence angle (AVG, Angle Varying Gain)
        from ALL scans.

        MBES intensity is dominated by the incidence angle: symmetric bell
        shape, peak ~45 near nadir, dropping to ~10 at the edges (±50°). That
        banding is tied to the vehicle pose, not the seabed, and ruins Colored
        ICP (aligns the bands instead of the seabed).

        The profile is the MEDIAN intensity per angular bin (robust to seabed
        structure and outliers). Then, at each point:

            I_corrected = I / gain(incidence_angle)

        leaves intensity ~1 on average at any angle, keeping only the seabed's
        real texture (sediment/rock signature). Reduces banding ~99% and
        preserves the seabed signal.

        Returns (centers, gain) or (None, None) if no usable intensity.
        """

        # ABSOLUTE-angle bins (0..60°) so the profile matches the
        # `angles = arctan2(r_horizontal, depth)` (always >=0) used in build().
        # Banding is symmetric about nadir, so |angle| is the correct variable
        # and doubles the samples per bin.
        bins = np.arange(0.0, 61.0, 2.0)
        centers = (bins[:-1] + bins[1:]) / 2.0
        per_bin = [[] for _ in range(len(centers))]

        sampled = 0
        for entry in scans:

            scan = entry["msg"]

            pc = ros_numpy.point_cloud2.pointcloud2_to_array(scan)

            if 'intensity' not in pc.dtype.names:
                return None, None

            fin = (
                np.isfinite(pc['x']) &
                np.isfinite(pc['z']) &
                np.isfinite(pc['intensity'])
            )
            pc = pc[fin]
            if len(pc) < 10:
                continue

            x = np.asarray(pc['x'], dtype=float)
            z = np.asarray(pc['z'], dtype=float)
            inten = np.asarray(pc['intensity'], dtype=float)

            # ABSOLUTE incidence angle from nadir (x = across-track,
            # z = depth). |angle| because banding is symmetric.
            ang = np.abs(np.degrees(np.arctan2(x, np.abs(z))))

            idx = np.clip(
                np.digitize(ang, bins) - 1,
                0,
                len(centers) - 1
            )

            for b, iv in zip(idx, inten):
                if len(per_bin[b]) < 4000:      # per-bin cap
                    per_bin[b].append(iv)

            sampled += 1

        gain = np.full(len(centers), np.nan)
        for i in range(len(centers)):
            if len(per_bin[i]) >= 50:
                gain[i] = np.median(per_bin[i])

        valid = np.isfinite(gain)
        if valid.sum() < 3:
            return None, None

        centers_v = centers[valid]
        gain_v = np.maximum(gain[valid], 1e-3)

        rospy.loginfo(
            f"AVG intensity profile built from {sampled} scans: "
            f"gain {gain_v.min():.1f}–{gain_v.max():.1f} "
            f"(banding swing {gain_v.max()-gain_v.min():.1f})"
        )

        return centers_v, gain_v

    def build(
            self,
            bag,
            scan_topic,
            nav,
            R_sensor):

        scans = []

        rospy.loginfo(
            "Loading scans..."
        )

        for _, scan, bag_time in tqdm(
                bag.read_messages(
                    topics=[scan_topic]
                ),
                desc="MBES scans"):

            if hasattr(scan, 'header'):
                scans.append({
                    "msg": scan,
                    "header_ts": scan.header.stamp.to_sec(),
                    "bag_ts": bag_time.to_sec()
                })

        if len(scans) == 0:
            return []

        scan_timestamp = self._select_scan_time_source(
            scans,
            nav
        )

        # AVG banding-correction profile by incidence angle.
        # Computed once over all scans and applied to each point's intensity
        # during patch construction.
        rospy.loginfo(
            "Building AVG intensity profile (incidence-angle correction)..."
        )
        self._avg_centers, self._avg_gain = self._build_avg_profile(scans)
        if self._avg_centers is None:
            rospy.logwarn(
                "No usable intensity for AVG correction; "
                "intensity used uncorrected (colored_icp may band)."
            )

        patches = []

        rospy.loginfo(
            "Building LOCAL patches..."
        )

        for start in tqdm(
                range(
                    0,
                    len(scans) - PATCH_SIZE + 1,
                    PATCH_STRIDE
                ),
                desc="Patch generation"):

            all_points = []
            all_intensity = []

            center_scan_idx = start + PATCH_SIZE // 2
            center_entry = scans[center_scan_idx]
            center_ts = scan_timestamp(center_entry)

            if not nav.has_timestamp(center_ts):

                rospy.logwarn(
                    f"Skipping patch start={start}: "
                    f"center timestamp {center_ts:.3f} outside "
                    f"navigation range "
                    f"[{nav.min_timestamp:.3f}, "
                    f"{nav.max_timestamp:.3f}]"
                )

                continue

            center_pose = nav.pose_values(center_ts)

            T_center = pose_dict_to_matrix(center_pose)
            R_center = T_center[:3, :3]
            t_center = T_center[:3, 3]

            for i in range(
                    start,
                    start + PATCH_SIZE):

                scan_entry = scans[i]
                scan = scan_entry["msg"]

                ts = scan_timestamp(scan_entry)

                if not nav.has_timestamp(ts):
                    continue

                pose = nav.pose_values(ts)

                if pose is None:
                    continue

                pc = ros_numpy.point_cloud2.pointcloud2_to_array(
                    scan
                )

                finite_mask = (
                    np.isfinite(pc['x']) &
                    np.isfinite(pc['y']) &
                    np.isfinite(pc['z'])
                )

                # Acoustic intensity (backscatter) is carried alongside xyz for
                # use in Colored ICP. If the sensor does not publish it, it is
                # filled with zeros (the pcd has no usable texture).
                if 'intensity' in pc.dtype.names:
                    finite_mask = finite_mask & np.isfinite(pc['intensity'])

                pc = pc[finite_mask]

                if len(pc) < 10:
                    continue

                xyz = np.column_stack((
                    pc['x'],
                    -pc['y'],
                    -pc['z']
                ))

                if 'intensity' in pc.dtype.names:
                    intensity = np.asarray(pc['intensity'], dtype=float)
                else:
                    intensity = np.zeros(len(xyz), dtype=float)

                r_horizontal = np.sqrt(
                    xyz[:, 0]**2 +
                    xyz[:, 1]**2
                )

                depth_s = np.abs(
                    xyz[:, 2]
                )

                angles = np.degrees(
                    np.arctan2(
                        r_horizontal,
                        depth_s
                    )
                )

                # AVG correction: divide intensity by the expected gain at its
                # incidence angle, removing the beam banding and leaving only the
                # seabed's real texture. The profile and `angles` both use the
                # absolute angle from nadir.
                if self._avg_centers is not None:
                    gain = np.interp(
                        angles,
                        self._avg_centers,
                        self._avg_gain
                    )
                    intensity = intensity / np.maximum(gain, 1e-3)

                angle_keep = angles < ANGLE_CUTOFF_DEG

                xyz = xyz[angle_keep]
                intensity = intensity[angle_keep]

                if len(xyz) < 10:
                    continue

                # =====================================================
                # SENSOR FRAME
                # =====================================================

                xyz = xyz @ R_sensor.T

                # =====================================================
                # VEHICLE ROTATION
                # =====================================================

                R_vehicle = tr.euler_matrix(
                    pose["roll"],
                    pose["pitch"],
                    pose["yaw"],
                    axes='sxyz'
                )[:3, :3]

                xyz = xyz @ R_vehicle.T

                # =====================================================
                # GLOBAL INS POSITION
                # =====================================================

                xyz[:, 0] += pose["north"]
                xyz[:, 1] += pose["east"]
                xyz[:, 2] += -pose["depth"]

                # =====================================================
                # RECENTER TO PATCH CENTER FRAME
                # =====================================================

                xyz -= t_center
                xyz = xyz @ R_center

                finite_xyz = np.isfinite(xyz).all(axis=1)
                xyz = xyz[finite_xyz]
                intensity = intensity[finite_xyz]

                if len(xyz) < 10:
                    continue

                all_points.append(xyz)
                all_intensity.append(intensity)

            if len(all_points) == 0:
                continue

            pts = np.vstack(all_points)
            inten = np.concatenate(all_intensity)

            keep_finite = np.isfinite(pts).all(axis=1)
            pts = pts[keep_finite]
            inten = inten[keep_finite]

            if len(pts) < MIN_PATCH_POINTS:
                continue

            min_bound = pts.min(axis=0)
            max_bound = pts.max(axis=0)
            extent = max_bound - min_bound

            if (
                not np.isfinite(extent).all()
                or np.max(extent) > MAX_PATCH_EXTENT
            ):

                rospy.logwarn(
                    f"Skipping patch start={start}: invalid extent "
                    f"min={min_bound.tolist()} "
                    f"max={max_bound.tolist()} "
                    f"extent={extent.tolist()}"
                )

                continue

            pcd = o3d.geometry.PointCloud()

            pcd.points = (
                o3d.utility.Vector3dVector(pts)
            )

            # Acoustic intensity -> normalized gray color [0,1].
            # Robust percentile normalization (2-98) to use the range well and
            # not let backscatter outliers crush the signal.
            # voxel_down_sample averages colors per voxel, so intensity is
            # preserved coherently after the downsample.
            if np.ptp(inten) > 1e-6:
                lo = np.percentile(inten, 2)
                hi = np.percentile(inten, 98)
                inten_n = np.clip(
                    (inten - lo) / max(hi - lo, 1e-6),
                    0.0,
                    1.0
                )
            else:
                inten_n = np.zeros(len(inten), dtype=float)

            pcd.colors = (
                o3d.utility.Vector3dVector(
                    np.tile(inten_n[:, None], (1, 3))
                )
            )

            try:

                pcd = pcd.voxel_down_sample(
                    VOXEL_SIZE
                )

            except RuntimeError as exc:

                rospy.logwarn(
                    f"Skipping patch start={start}: "
                    f"voxel_down_sample failed with "
                    f"VOXEL_SIZE={VOXEL_SIZE}. "
                    f"min={min_bound.tolist()} "
                    f"max={max_bound.tolist()} "
                    f"extent={extent.tolist()} "
                    f"error={exc}"
                )

                continue

            if len(pcd.points) < MIN_PATCH_POINTS:
                continue

            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=VOXEL_SIZE * 3.0,
                    max_nn=30
                )
            )

            patch = Patch()

            patch.idx = len(patches)

            patch.pcd = pcd

            patch.center = np.mean(
                pts[:, :2],
                axis=0
            )

            patch.pose = center_pose

            patch.timestamp = center_ts

            patches.append(patch)

        return patches

# =============================================================================
# LOCAL REGISTRATION
# =============================================================================

def execute_local_registration(
        source,
        target,
        T_init,
        algorithm,
        icp_distance,
        max_iter,
        ndt_resolution,
        ndt_max_points,
        ndt_min_points_per_voxel,
        colored_icp_lambda=0.6,
        hybrid_min_texture=0.08):

    algorithm = algorithm.lower()

    if algorithm == "icp":

        return robust_icp(
            source,
            target,
            T_init,
            icp_distance=icp_distance,
            max_iter=max_iter
        )

    if algorithm == "colored_icp":

        return robust_colored_icp(
            source,
            target,
            T_init,
            icp_distance=icp_distance,
            max_iter=max_iter,
            lambda_geometric=colored_icp_lambda
        )

    if algorithm == "hybrid":

        # Adaptive: geometry where there is relief, intensity where there is not.
        return robust_hybrid_icp(
            source,
            target,
            T_init,
            icp_distance=icp_distance,
            max_iter=max_iter,
            lambda_geometric=colored_icp_lambda,
            min_geometric_texture=hybrid_min_texture
        )

    if algorithm == "ndt":

        return robust_ndt(
            source,
            target,
            T_init,
            ndt_resolution=ndt_resolution,
            max_iter=max_iter,
            max_correspondence_distance=icp_distance,
            min_points_per_voxel=ndt_min_points_per_voxel,
            max_points=ndt_max_points
        )

    raise ValueError(
        f"Unsupported registration_algorithm '{algorithm}'. "
        "Use 'icp', 'colored_icp', 'hybrid' or 'ndt'."
    )


def edge_information(
        result,
        source_pcd,
        target_pcd,
        temporal_distance=1.0,
        loop=False,
        use_covariance=None):
    """
    Pose-graph edge information (pICP, R1).

    If `use_covariance` and there is enough data, computes ANISOTROPIC 6-DoF
    information from the registration's real covariance (Censi/Palomer): low
    cross-track on flat seabed, high where relief determines the pose. Z/roll/pitch
    keep high information (reliable INS DoF, coherent with R0.3's 3-DoF).

    If not possible (few correspondences, no normals on the target), falls back to
    the prior isotropic `dynamic_information_matrix` — never breaks the graph.

    Returns (info_6x6, diag) where diag is the covariance diagnostic (or None).
    """

    if use_covariance is None:
        use_covariance = USE_REGISTRATION_COVARIANCE

    fallback = dynamic_information_matrix(
        result,
        temporal_distance=temporal_distance,
        loop=loop,
    )

    if not use_covariance or result is None:
        return fallback, None

    try:
        corr = np.asarray(result.correspondence_set)
        if corr.shape[0] < 6:
            return fallback, None

        # The target must have normals for the point-to-plane residual.
        if not target_pcd.has_normals():
            target_pcd.estimate_normals()

        src = np.asarray(source_pcd.points)
        tgt = np.asarray(target_pcd.points)
        nrm = np.asarray(target_pcd.normals)

        s_idx = corr[:, 0]
        t_idx = corr[:, 1]

        cov3, diag = registration_covariance_3dof(
            src[s_idx],
            tgt[t_idx],
            nrm[t_idx],
            result.transformation,
            residual_std=float(result.inlier_rmse),
        )

        # R3 — modulate the geometric covariance by the BACKSCATTER informativeness.
        # Backscatter travels in the cloud's color (gray) channel. Where geometry
        # is flat (large cov XY) but intensity has rich texture, Colored ICP
        # supplies XY gradient -> we reduce the cross-track uncertainty.
        # Novel contribution: uncertainty integrating geometry AND backscatter
        # (Palomer and Torroba/Tan do not use intensity in their uncertainty model).
        if USE_INTENSITY_IN_COVARIANCE and target_pcd.has_colors():
            colors = np.asarray(target_pcd.colors)
            if colors.shape[0] == tgt.shape[0]:
                inten = colors[t_idx, 0]  # gray channel = normalized backscatter
                i_info = intensity_informativeness(inten)
                cov3 = fuse_geometry_intensity_cov(
                    cov3, i_info, diagnostics=diag,
                    intensity_gain=INTENSITY_COV_GAIN,
                )

        info = information_6dof_from_cov3(cov3)
        return info, diag

    except Exception:
        # Any problem -> safe isotropic information.
        return fallback, None


# =============================================================================
# METRICS PLOTS
# =============================================================================

def _save_metrics_plots(
        metrics,
        raw_trajectory,
        slam_trajectory,
        metrics_dir):

    if not _MPL:
        rospy.logwarn("matplotlib not available — skipping plots")
        return

    seq = metrics.get("sequential", [])
    loops = metrics.get("loops", [])
    traj_m = metrics.get("trajectory", {})

    # ── 1. Trajectory comparison ──────────────────────────────────────────

    if len(raw_trajectory) >= 2 and len(slam_trajectory) >= 2:

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(
            raw_trajectory[:, 1],
            raw_trajectory[:, 0],
            color='steelblue',
            alpha=0.6,
            linewidth=1.0,
            label='Raw navigation'
        )

        ax.plot(
            slam_trajectory[:, 1],
            slam_trajectory[:, 0],
            color='crimson',
            alpha=0.85,
            linewidth=1.5,
            label='SLAM optimized'
        )

        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_title('Trajectory — Raw Navigation vs SLAM Optimized')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(
            os.path.join(metrics_dir, 'trajectory_comparison.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig)

    # ── 2. Fitness + RMSE timeseries ──────────────────────────────────────

    if seq:

        indices = [m['source'] for m in seq]
        fitness_vals = [m['fitness'] for m in seq]
        rmse_vals = [m['rmse'] for m in seq]
        colors = [
            '#2ecc71' if m['accepted'] else '#e74c3c'
            for m in seq
        ]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 6), sharex=True
        )

        ax1.scatter(indices, fitness_vals, c=colors, s=8, alpha=0.7)
        ax1.axhline(
            FITNESS_THRESHOLD,
            color='orange', linestyle='--', linewidth=1.2,
            label=f'Threshold {FITNESS_THRESHOLD}'
        )
        ax1.set_ylabel('Fitness')
        ax1.set_ylim([-0.05, 1.1])
        ax1.set_title(
            'Sequential Registration Quality'
            '  (green=accepted, red=rejected)'
        )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.scatter(indices, rmse_vals, c=colors, s=8, alpha=0.7)
        ax2.axhline(
            SEQ_RMSE_THRESHOLD,
            color='orange', linestyle='--', linewidth=1.2,
            label=f'Threshold {SEQ_RMSE_THRESHOLD:.3f} m'
        )
        ax2.set_ylabel('RMSE (m)')
        ax2.set_xlabel('Patch index')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(metrics_dir, 'registration_quality.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig)

    # ── 3. SLAM correction per node ───────────────────────────────────────

    corrections = traj_m.get('corrections_xy_m', [])

    if corrections:

        corrections_arr = np.asarray(corrections)
        mean_c = float(np.mean(corrections_arr))
        max_c = float(np.max(corrections_arr))

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(
            range(len(corrections_arr)),
            corrections_arr,
            alpha=0.35, color='steelblue'
        )
        ax.plot(corrections_arr, color='steelblue', linewidth=0.8)
        ax.axhline(
            mean_c, color='crimson', linestyle='--', linewidth=1.2,
            label=f'Mean {mean_c:.2f} m'
        )
        ax.axhline(
            max_c, color='darkorange', linestyle=':', linewidth=1.0,
            label=f'Max {max_c:.2f} m'
        )
        ax.set_xlabel('Node index')
        ax.set_ylabel('Correction XY (m)')
        ax.set_title('SLAM Position Correction vs Raw Navigation')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(metrics_dir, 'slam_correction.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig)

    # ── 4. RMSE histogram (accepted) ──────────────────────────────────────

    accepted_rmse = [
        m['rmse'] for m in seq
        if m['accepted'] and m['rmse'] > 0
    ]

    if accepted_rmse:

        arr = np.asarray(accepted_rmse)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            arr, bins=40,
            color='steelblue', edgecolor='white', alpha=0.85
        )
        ax.axvline(
            float(np.mean(arr)), color='crimson',
            linestyle='--', linewidth=1.5,
            label=f'Mean {float(np.mean(arr)):.3f} m'
        )
        ax.axvline(
            SEQ_RMSE_THRESHOLD, color='orange',
            linestyle='--', linewidth=1.5,
            label=f'Threshold {SEQ_RMSE_THRESHOLD:.3f} m'
        )
        ax.set_xlabel('RMSE (m)')
        ax.set_ylabel('Count')
        ax.set_title('RMSE Distribution — Accepted Sequential Registrations')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(metrics_dir, 'rmse_histogram.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig)

    # ── 5. Loop closure quality — two panels ─────────────────────────────

    if loops:

        sc_key  = 'scan_context_score'
        ins_key = 'ins_distance_m'
        rat_key = 'icp_ins_ratio'

        loop_acc = [l for l in loops if l['accepted']]
        loop_rej = [l for l in loops if not l['accepted']]

        # panel A: SC score vs ICP fitness
        # panel B: INS distance vs ICP/INS ratio  ← new key diagnostic
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        for ax, lrej, lacc, xlabel, xkey, ylabel, ykey, thr, thr_label in [
            (
                ax1,
                [l for l in loop_rej if sc_key in l],
                [l for l in loop_acc if sc_key in l],
                'Scan Context score (lower = more similar)', sc_key,
                'ICP fitness', 'fitness',
                SCAN_CONTEXT_THRESHOLD, f'SC thr {SCAN_CONTEXT_THRESHOLD}',
            ),
        ]:
            if lrej:
                ax.scatter(
                    [l[xkey] for l in lrej],
                    [l[ykey] for l in lrej],
                    c='#e74c3c', s=12, alpha=0.5, label='Rejected', zorder=2
                )
            if lacc:
                ax.scatter(
                    [l[xkey] for l in lacc],
                    [l[ykey] for l in lacc],
                    c='#2ecc71', s=35, alpha=0.9, marker='*',
                    label='Accepted', zorder=3
                )
            ax.axvline(thr, color='orange', linestyle='--',
                       linewidth=1.2, label=thr_label)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title('SC score vs ICP fitness')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Panel B — INS distance vs ICP/INS ratio
        rej_b = [l for l in loop_rej if ins_key in l and rat_key in l]
        acc_b = [l for l in loop_acc if ins_key in l and rat_key in l]

        if rej_b:
            ax2.scatter(
                [l[ins_key] for l in rej_b],
                [l[rat_key] for l in rej_b],
                c='#e74c3c', s=12, alpha=0.5, label='Rejected', zorder=2
            )
        if acc_b:
            ax2.scatter(
                [l[ins_key] for l in acc_b],
                [l[rat_key] for l in acc_b],
                c='#2ecc71', s=35, alpha=0.9, marker='*',
                label='Accepted', zorder=3
            )

        ax2.axhline(
            MIN_LOOP_ICP_INS_RATIO, color='orange', linestyle='--',
            linewidth=1.2, label=f'Min ratio {MIN_LOOP_ICP_INS_RATIO}'
        )
        ax2.axvline(
            MAX_LOOP_INS_DISTANCE, color='steelblue', linestyle=':',
            linewidth=1.2, label=f'Max INS dist {MAX_LOOP_INS_DISTANCE}m'
        )
        ax2.set_xlabel('INS distance between patches (m)')
        ax2.set_ylabel('ICP correction / INS distance  (ideal ≈ 1.0)')
        ax2.set_title('ICP–INS Drift Consistency\n'
                      '(false positives cluster near ratio ≈ 0)')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(metrics_dir, 'loop_closure_quality.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close(fig)

    rospy.loginfo(f"Plots saved → {metrics_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():

    rospy.init_node(
        "advanced_multibeam_slam"
    )

    bag_file = rospy.get_param(
        "~bag_file"
    )

    scan_topic = rospy.get_param(
        "~scan_topic"
    )

    nav_topic = rospy.get_param(
        "~nav_topic"
    )

    output_dir = rospy.get_param(
        "~output_dir"
    )

    registration_algorithm = rospy.get_param(
        "~registration_algorithm",
        REGISTRATION_ALGORITHM
    )
    registration_algorithm = str(
        registration_algorithm
    ).lower()

    if registration_algorithm not in ("icp", "colored_icp", "hybrid", "ndt"):

        raise ValueError(
            f"Unsupported registration_algorithm "
            f"'{registration_algorithm}'. "
            f"Use 'icp', 'colored_icp', 'hybrid' or 'ndt'."
        )

    colored_icp_lambda = float(rospy.get_param(
        "~colored_icp_lambda",
        COLORED_ICP_LAMBDA
    ))

    hybrid_min_texture = float(rospy.get_param(
        "~hybrid_min_texture",
        HYBRID_MIN_TEXTURE
    ))

    ndt_resolution = rospy.get_param(
        "~ndt_resolution",
        NDT_RESOLUTION
    )
    ndt_resolution = float(
        ndt_resolution
    )

    ndt_max_points = rospy.get_param(
        "~ndt_max_points",
        NDT_MAX_POINTS
    )
    ndt_max_points = int(
        ndt_max_points
    )

    ndt_min_points_per_voxel = rospy.get_param(
        "~ndt_min_points_per_voxel",
        NDT_MIN_POINTS_PER_VOXEL
    )
    ndt_min_points_per_voxel = int(
        ndt_min_points_per_voxel
    )

    enable_loop_closure = rospy.get_param(
        "~enable_loop_closure",
        ENABLE_LOOP_CLOSURE
    )

    global MAX_SEQ_ICP_TRANSLATION_DEV, MAX_SEQ_ICP_YAW_DEV
    global SEQ_GATE_TEXTURE_RELAX, SEQ_GATE_TEXTURE_FULL
    global MIN_SEQ_ICP_LENGTH_RATIO, MAX_SEQ_ICP_LENGTH_RATIO
    global SEQ_ANCHOR_SCALE, SEQ_CROSS_TRACK_GAIN
    global MAX_LOOP_INS_DISTANCE, MIN_LOOP_ICP_INS_RATIO, MIN_INS_DIST_FOR_RATIO_CHECK
    global MAX_LOOP_INS_DISCREPANCY, MAX_LOOP_REVISIT_INS_DIST
    global MIN_LOOP_TEMPORAL_GAP
    global ENABLE_CROSS_TRACK_EDGES, XTRACK_MIN_INS_DIST, XTRACK_MAX_INS_DIST
    global XTRACK_MIN_TEMPORAL_GAP, XTRACK_MAX_EDGES_PER_PATCH
    global XTRACK_MIN_FITNESS, XTRACK_MAX_INS_DISCREPANCY
    global USE_REGISTRATION_COVARIANCE
    global EDGE_PRUNE_THRESHOLD, PREFERENCE_LOOP_CLOSURE
    global MAX_NODE_CORRECTION_M
    global USE_INTENSITY_IN_COVARIANCE, INTENSITY_COV_GAIN
    # Patch / preprocessing and quality thresholds, now configurable from the
    # launch. They were module-level constants used directly by PatchBuilder and
    # the registration loops; reassigned here to keep their usage unchanged.
    global PATCH_SIZE, PATCH_STRIDE, VOXEL_SIZE, FINAL_DOWNSAMPLE, ANGLE_CUTOFF_DEG
    global CONSISTENCY_CELL_SIZE
    global FITNESS_THRESHOLD, SEQ_RMSE_THRESHOLD, MIN_CORRESPONDENCES
    global SCAN_CONTEXT_THRESHOLD, MAX_LOOP_CANDIDATES
    global LOOP_FITNESS_THRESHOLD, LOOP_RMSE_THRESHOLD
    global MAX_LOOP_Z_TRANSLATION, MAX_LOOP_XY_TRANSLATION, MAX_LOOP_YAW_DEG
    global LOOP_RANSAC_VOXEL, LOOP_RANSAC_MIN_VOXEL, LOOP_MIN_TEXTURE_FOR_RANSAC
    global SEQ_TR_DEV_STEP_GAIN
    global XTRACK_AUTO_TUNE, XTRACK_AUTO_BAND_TOL
    global MONITOR_UPDATE_EVERY

    PATCH_SIZE = max(1, int(rospy.get_param("~patch_size", PATCH_SIZE)))
    PATCH_STRIDE = max(1, int(rospy.get_param("~patch_stride", PATCH_STRIDE)))
    VOXEL_SIZE = float(rospy.get_param("~patch_voxel_size", VOXEL_SIZE))
    FINAL_DOWNSAMPLE = float(rospy.get_param("~final_downsample", FINAL_DOWNSAMPLE))
    CONSISTENCY_CELL_SIZE = float(
        rospy.get_param("~consistency_cell_size", CONSISTENCY_CELL_SIZE)
    )
    ANGLE_CUTOFF_DEG = float(rospy.get_param("~angle_cutoff_deg", ANGLE_CUTOFF_DEG))

    FITNESS_THRESHOLD = float(rospy.get_param("~fitness_threshold", FITNESS_THRESHOLD))
    SEQ_RMSE_THRESHOLD = float(rospy.get_param("~seq_rmse_threshold", SEQ_RMSE_THRESHOLD))
    MIN_CORRESPONDENCES = int(rospy.get_param("~min_correspondences", MIN_CORRESPONDENCES))

    SCAN_CONTEXT_THRESHOLD = float(rospy.get_param("~scan_context_threshold", SCAN_CONTEXT_THRESHOLD))
    LOOP_RANSAC_VOXEL = float(rospy.get_param("~loop_ransac_voxel", LOOP_RANSAC_VOXEL))
    LOOP_RANSAC_MIN_VOXEL = float(rospy.get_param("~loop_ransac_min_voxel", LOOP_RANSAC_MIN_VOXEL))
    LOOP_MIN_TEXTURE_FOR_RANSAC = float(rospy.get_param("~loop_min_texture_for_ransac", LOOP_MIN_TEXTURE_FOR_RANSAC))
    MAX_LOOP_CANDIDATES = int(rospy.get_param("~max_loop_candidates", MAX_LOOP_CANDIDATES))
    LOOP_FITNESS_THRESHOLD = float(rospy.get_param("~loop_fitness_threshold", LOOP_FITNESS_THRESHOLD))
    LOOP_RMSE_THRESHOLD = float(rospy.get_param("~loop_rmse_threshold", LOOP_RMSE_THRESHOLD))
    MAX_LOOP_Z_TRANSLATION = float(rospy.get_param("~max_loop_z_translation", MAX_LOOP_Z_TRANSLATION))
    MAX_LOOP_XY_TRANSLATION = float(rospy.get_param("~max_loop_xy_translation", MAX_LOOP_XY_TRANSLATION))
    MAX_LOOP_YAW_DEG = float(rospy.get_param("~max_loop_yaw_deg", MAX_LOOP_YAW_DEG))

    # max(1, ...): avoids division by zero in `idx % MONITOR_UPDATE_EVERY`.
    MONITOR_UPDATE_EVERY = max(1, int(rospy.get_param("~monitor_update_every", MONITOR_UPDATE_EVERY)))

    MAX_SEQ_ICP_TRANSLATION_DEV = float(rospy.get_param(
        "~max_seq_icp_translation_dev",
        MAX_SEQ_ICP_TRANSLATION_DEV
    ))

    MAX_SEQ_ICP_YAW_DEV = float(rospy.get_param(
        "~max_seq_icp_yaw_dev",
        MAX_SEQ_ICP_YAW_DEV
    ))

    SEQ_GATE_TEXTURE_RELAX = float(rospy.get_param(
        "~seq_gate_texture_relax",
        SEQ_GATE_TEXTURE_RELAX
    ))

    SEQ_GATE_TEXTURE_FULL = float(rospy.get_param(
        "~seq_gate_texture_full",
        SEQ_GATE_TEXTURE_FULL
    ))

    MIN_SEQ_ICP_LENGTH_RATIO = float(rospy.get_param(
        "~min_seq_icp_length_ratio",
        MIN_SEQ_ICP_LENGTH_RATIO
    ))

    MAX_SEQ_ICP_LENGTH_RATIO = float(rospy.get_param(
        "~max_seq_icp_length_ratio",
        MAX_SEQ_ICP_LENGTH_RATIO
    ))

    SEQ_ANCHOR_SCALE = bool(rospy.get_param(
        "~seq_anchor_scale",
        SEQ_ANCHOR_SCALE
    ))

    SEQ_CROSS_TRACK_GAIN = float(rospy.get_param(
        "~seq_cross_track_gain",
        SEQ_CROSS_TRACK_GAIN
    ))

    MAX_LOOP_INS_DISTANCE = float(rospy.get_param(
        "~max_loop_ins_distance",
        MAX_LOOP_INS_DISTANCE
    ))

    MAX_LOOP_INS_DISCREPANCY = float(rospy.get_param(
        "~max_loop_ins_discrepancy",
        MAX_LOOP_INS_DISCREPANCY
    ))

    MAX_LOOP_REVISIT_INS_DIST = float(rospy.get_param(
        "~max_loop_revisit_ins_dist",
        MAX_LOOP_REVISIT_INS_DIST
    ))

    MIN_LOOP_ICP_INS_RATIO = float(rospy.get_param(
        "~min_loop_icp_ins_ratio",
        MIN_LOOP_ICP_INS_RATIO
    ))

    MIN_INS_DIST_FOR_RATIO_CHECK = float(rospy.get_param(
        "~min_ins_dist_for_ratio_check",
        MIN_INS_DIST_FOR_RATIO_CHECK
    ))

    MIN_LOOP_TEMPORAL_GAP = int(rospy.get_param(
        "~min_loop_temporal_gap",
        MIN_LOOP_TEMPORAL_GAP
    ))

    # Cross-track overlap (R0.2)
    ENABLE_CROSS_TRACK_EDGES = bool(rospy.get_param(
        "~enable_cross_track_edges", ENABLE_CROSS_TRACK_EDGES
    ))
    XTRACK_MIN_INS_DIST = float(rospy.get_param(
        "~xtrack_min_ins_dist", XTRACK_MIN_INS_DIST
    ))
    XTRACK_MAX_INS_DIST = float(rospy.get_param(
        "~xtrack_max_ins_dist", XTRACK_MAX_INS_DIST
    ))
    XTRACK_MIN_TEMPORAL_GAP = int(rospy.get_param(
        "~xtrack_min_temporal_gap", XTRACK_MIN_TEMPORAL_GAP
    ))
    XTRACK_AUTO_TUNE = bool(rospy.get_param(
        "~xtrack_auto_tune", XTRACK_AUTO_TUNE
    ))
    XTRACK_AUTO_BAND_TOL = float(rospy.get_param(
        "~xtrack_auto_band_tol", XTRACK_AUTO_BAND_TOL
    ))
    XTRACK_MAX_EDGES_PER_PATCH = int(rospy.get_param(
        "~xtrack_max_edges_per_patch", XTRACK_MAX_EDGES_PER_PATCH
    ))
    XTRACK_MIN_FITNESS = float(rospy.get_param(
        "~xtrack_min_fitness", XTRACK_MIN_FITNESS
    ))
    XTRACK_MAX_INS_DISCREPANCY = float(rospy.get_param(
        "~xtrack_max_ins_discrepancy", XTRACK_MAX_INS_DISCREPANCY
    ))

    # pICP (R1)
    USE_REGISTRATION_COVARIANCE = bool(rospy.get_param(
        "~use_registration_covariance", USE_REGISTRATION_COVARIANCE
    ))

    # Robust back-end (R2)
    EDGE_PRUNE_THRESHOLD = float(rospy.get_param(
        "~edge_prune_threshold", EDGE_PRUNE_THRESHOLD
    ))
    PREFERENCE_LOOP_CLOSURE = float(rospy.get_param(
        "~preference_loop_closure", PREFERENCE_LOOP_CLOSURE
    ))
    MAX_NODE_CORRECTION_M = float(rospy.get_param(
        "~max_node_correction_m", MAX_NODE_CORRECTION_M
    ))

    # Backscatter in the covariance (R3)
    USE_INTENSITY_IN_COVARIANCE = bool(rospy.get_param(
        "~use_intensity_in_covariance", USE_INTENSITY_IN_COVARIANCE
    ))
    INTENSITY_COV_GAIN = float(rospy.get_param(
        "~intensity_cov_gain", INTENSITY_COV_GAIN
    ))

    enable_monitor = rospy.get_param(
        "~enable_monitor",
        True
    )

    monitor_zoom = rospy.get_param(
        "~monitor_zoom",
        0.18
    )

    monitor_min_zoom = rospy.get_param(
        "~monitor_min_zoom",
        0.03
    )

    monitor_margin = rospy.get_param(
        "~monitor_margin",
        2.5
    )

    monitor_show_clouds = rospy.get_param(
        "~monitor_show_clouds",
        True
    )

    monitor_cloud_window = int(rospy.get_param(
        "~monitor_cloud_window",
        15
    ))

    monitor_cloud_voxel = float(rospy.get_param(
        "~monitor_cloud_voxel",
        0.4
    ))

    rospy.loginfo(
        f"Loop closure enabled: {enable_loop_closure}"
    )

    if enable_loop_closure:
        rospy.loginfo(
            f"Loop closure gates (real revisit, AND) — "
            f"max_revisit_ins_dist={MAX_LOOP_REVISIT_INS_DIST:.1f}m  "
            f"min_icp_ins_ratio={MIN_LOOP_ICP_INS_RATIO:.2f}  "
            f"max_ins_loop_discrepancy={MAX_LOOP_INS_DISCREPANCY:.1f}m  "
            f"(candidates pre-filtered to <{MAX_LOOP_INS_DISTANCE:.0f}m)"
        )

    rospy.loginfo(
        f"Monitor enabled: {enable_monitor}"
    )

    rospy.loginfo(
        f"Monitor camera: zoom={monitor_zoom}, "
        f"min_zoom={monitor_min_zoom}, "
        f"margin={monitor_margin}"
    )

    rospy.loginfo(
        f"Registration algorithm: {registration_algorithm}"
    )

    if registration_algorithm == "colored_icp":
        rospy.loginfo(
            f"Colored ICP (acoustic intensity) — "
            f"lambda_geometric={colored_icp_lambda:.2f}"
        )

    if registration_algorithm == "hybrid":
        rospy.loginfo(
            f"Hybrid registration — geometry where relief exists, "
            f"intensity where flat (min_texture={hybrid_min_texture:.2f}, "
            f"lambda_geometric={colored_icp_lambda:.2f})"
        )

    rospy.loginfo(
        f"Patch/preprocess — size={PATCH_SIZE} stride={PATCH_STRIDE} "
        f"(overlap={100.0 * (1.0 - PATCH_STRIDE / max(PATCH_SIZE, 1)):.0f}%)  "
        f"voxel={VOXEL_SIZE:.2f}m  final_downsample={FINAL_DOWNSAMPLE:.2f}m  "
        f"angle_cutoff={ANGLE_CUTOFF_DEG:.1f}deg"
    )

    rospy.loginfo(
        f"Sequential quality gates — fitness>={FITNESS_THRESHOLD:.2f}  "
        f"rmse<={SEQ_RMSE_THRESHOLD:.3f}m  "
        f"min_correspondences={MIN_CORRESPONDENCES}"
    )

    rospy.loginfo(
        f"ICP–nav consistency gates — "
        f"max_translation_dev={MAX_SEQ_ICP_TRANSLATION_DEV:.2f}m  "
        f"max_yaw_dev={MAX_SEQ_ICP_YAW_DEV:.1f}deg  "
        f"length_ratio_band=[{MIN_SEQ_ICP_LENGTH_RATIO:.2f}, "
        f"{MAX_SEQ_ICP_LENGTH_RATIO:.2f}]"
    )

    if SEQ_GATE_TEXTURE_RELAX > 1.0 and SEQ_GATE_TEXTURE_FULL > 0.0:
        rospy.loginfo(
            f"Consistency gate texture modulation ENABLED — "
            f"relax up to {SEQ_GATE_TEXTURE_RELAX:.1f}x at texture "
            f">={SEQ_GATE_TEXTURE_FULL:.2f} "
            f"(flat seabed keeps {MAX_SEQ_ICP_TRANSLATION_DEV:.2f}m/"
            f"{MAX_SEQ_ICP_YAW_DEV:.0f}deg; full relief allows "
            f"{MAX_SEQ_ICP_TRANSLATION_DEV*SEQ_GATE_TEXTURE_RELAX:.2f}m/"
            f"{MAX_SEQ_ICP_YAW_DEV*SEQ_GATE_TEXTURE_RELAX:.0f}deg)"
        )

    if SEQ_ANCHOR_SCALE:
        rospy.loginfo(
            "Scale anchoring ENABLED — step magnitude from INS, "
            "direction from ICP (fixes systematic ICP compression)"
        )

    if SEQ_CROSS_TRACK_GAIN != 1.0:
        rospy.loginfo(
            f"Cross-track damping ENABLED — ICP lateral correction gain="
            f"{SEQ_CROSS_TRACK_GAIN:.2f} (0=INS lateral; fixes east drift)"
        )

    if registration_algorithm == "ndt":

        rospy.loginfo(
            f"NDT parameters: resolution={ndt_resolution}, "
            f"max_points={ndt_max_points}, "
            f"min_points_per_voxel={ndt_min_points_per_voxel}"
        )

    ensure_dir(output_dir)

    metrics_dir = os.path.join(
        output_dir,
        "metrics"
    )

    ensure_dir(metrics_dir)

    bag = rosbag.Bag(bag_file)

    T_mb = get_static_transform_from_tf(

        bag_file,

        "sparus2/base_link",

        "sparus2/multibeam"
    )

    R_sensor = T_mb[:3, :3]

    nav = NavigationInterpolator(
        bag,
        nav_topic
    )

    # Per-stage timing (PERF). Shows where time goes.
    stage_times = {}
    _t_stage = time.time()

    builder = PatchBuilder()

    patches = builder.build(

        bag,
        scan_topic,
        nav,
        R_sensor
    )

    bag.close()

    stage_times["patch_building"] = time.time() - _t_stage
    _t_stage = time.time()

    rospy.loginfo(
        f"Generated {len(patches)} patches"
    )

    if len(patches) == 0:

        rospy.logerr(
            "No valid patches generated. "
            "Check bag topics, point density and patch filters."
        )

        return

    scan_context_manager = None

    if enable_loop_closure:

        scan_context_manager = \
            ScanContextManager(
                min_temporal_gap=MIN_LOOP_TEMPORAL_GAP
            )

        for patch in tqdm(
                patches,
                desc="Scan Context"):

            # The patch's INS position (north, east) is supplied to enable the
            # spatial KD-tree pre-filter in detect_loop_candidates.
            scan_context_manager.add_descriptor(
                patch.pcd,
                ins_xy=(
                    patch.pose['north'],
                    patch.pose['east']
                )
            )

    stage_times["scan_context"] = time.time() - _t_stage
    _t_stage = time.time()

    pose_graph = (
        o3d.pipelines.registration.PoseGraph()
    )

    T0 = pose_dict_to_matrix(
        patches[0].pose
    )

    pose_graph.nodes.append(
        o3d.pipelines.registration.PoseGraphNode(
            T0
        )
    )

    odometry = T0.copy()

    monitor = PoseGraphMonitor(
        enabled=enable_monitor,
        overview_zoom=monitor_zoom,
        min_zoom=monitor_min_zoom,
        overview_margin=monitor_margin,
        show_clouds=monitor_show_clouds,
        cloud_window=monitor_cloud_window,
        cloud_voxel=monitor_cloud_voxel
    )

    # =========================================================================
    # METRICS
    # =========================================================================

    metrics = {

        "sequential": [],

        "loops": [],

        "summary": {}
    }

    fallback_count = 0

    accepted_seq = 0

    rejected_seq = 0

    accepted_loops = 0

    rejected_loops = 0

    # =========================================================================
    # SEQUENTIAL REGISTRATION
    # =========================================================================

    rospy.loginfo(
        f"Sequential registration ({registration_algorithm.upper()})..."
    )

    for idx in tqdm(

            range(1, len(patches)),

            desc=f"Sequential {registration_algorithm.upper()}"):

        source = patches[idx].pcd

        target = patches[idx - 1].pcd

        T_init = expected_transform(

            patches[idx - 1],

            patches[idx]
        )

        result = execute_local_registration(
            source,
            target,
            T_init,
            registration_algorithm,
            ICP_DISTANCE,
            ICP_MAX_ITER,
            ndt_resolution,
            ndt_max_points,
            ndt_min_points_per_voxel,
            colored_icp_lambda=colored_icp_lambda,
            hybrid_min_texture=hybrid_min_texture
        )

        temporal_distance = (

            patches[idx].timestamp -
            patches[idx - 1].timestamp
        )

        # =============================================================
        # ICP FAILED
        # =============================================================
        # robust_icp returns None if all scales produce clouds with
        # fewer than 50 points after the downsample.
        # T_init (navigation) is used as the fallback transform with
        # very low information so as not to contaminate the optimizer.
        #
        # IMPORTANT: no 'continue' here. Flow falls through to the
        # common metrics/node/edge/monitor block at the end of the loop,
        # guaranteeing the pose graph is always in a consistent state
        # before the monitor visualizes it.
        # =============================================================

        seq_failure_reason = None

        # Consistency-gate diagnostic (filled in the success branch; initialized
        # here so the metrics always have these fields).
        _texture = None
        _icp_tr_dev = None
        _dir_dev = None
        _tr_dev_thr = None
        _yaw_dev_thr = None

        if result is None:

            fallback_count += 1
            rejected_seq += 1
            seq_failure_reason = "registration_failed"

            rospy.logwarn(
                f"{registration_algorithm.upper()} failed "
                f"{idx}->{idx-1}"
            )

            T = T_init
            info = np.eye(6) * 0.01

            fitness = 0.0
            rmse = 0.0
            correspondences = 0
            valid_registration = False

        # =============================================================
        # ICP SUCCEEDED — adaptive validation
        # =============================================================

        else:

            fitness = float(result.fitness)
            rmse = float(result.inlier_rmse)
            correspondences = len(result.correspondence_set)

            rospy.loginfo(
                f"[SEQ {registration_algorithm.upper()}] "
                f"{idx}->{idx-1} | "
                f"fitness={fitness:.3f} "
                f"rmse={rmse:.3f} "
                f"corr={correspondences}"
            )

            # =========================================================
            # ADAPTIVE ICP VALIDATION
            # =========================================================

            valid_registration = True

            if fitness < FITNESS_THRESHOLD:
                valid_registration = False
                seq_failure_reason = "fitness_below_threshold"

            if correspondences < MIN_CORRESPONDENCES:
                valid_registration = False
                seq_failure_reason = "insufficient_correspondences"

            adaptive_rmse_threshold = SEQ_RMSE_THRESHOLD

            # ---------------------------------------------------------
            # HIGH-CONFIDENCE FALLBACK
            # ---------------------------------------------------------
            # In underwater MBES planar regions with soft bathymetry
            # and acoustic noise: fitness ≈ 1.0, high correspondences,
            # moderate RMSE — still geometrically correct.
            # HIGH_RMSE_MULTIPLIER > 1.0 relaxes the threshold here.
            # ---------------------------------------------------------

            if (
                fitness > HIGH_FITNESS_THRESHOLD
                and
                correspondences > HIGH_CORRESPONDENCE_THRESHOLD
            ):

                adaptive_rmse_threshold *= HIGH_RMSE_MULTIPLIER

                rospy.loginfo(
                    f"[HIGH CONFIDENCE ICP] "
                    f"Relaxed RMSE threshold: "
                    f"{adaptive_rmse_threshold:.3f}"
                )

            if rmse > adaptive_rmse_threshold:
                valid_registration = False
                seq_failure_reason = "rmse_above_threshold"

            # =========================================================
            # ICP–NAVIGATION CONSISTENCY GATE  (texture-modulated)
            # =========================================================
            # Flat underwater seabeds give GICP near-zero XY gradient.
            # ICP can drift to wrong local minima (reversed or
            # perpendicular steps) that still show fitness ≈ 1 / low RMSE.
            # Compare the raw ICP result with T_init before accepting.
            #
            # The gate threshold is MODULATED by the pair's geometric texture
            # (fraction of non-vertical normals): stays strict on flat seabed
            # (where ICP slides) and relaxes where there is real relief (where ICP
            # is reliable and the fixed gate rejected valid corrections). See
            # SEQ_GATE_TEXTURE_*.
            # =========================================================

            # Pair texture: the lesser of both clouds (registration is limited by
            # the patch with less relief). _preprocessed caches the computation, so
            # it reuses the work already done by the hybrid/ICP.
            _texture = min(
                _geometric_texture(source),
                _geometric_texture(target)
            )

            if SEQ_GATE_TEXTURE_RELAX > 1.0 and SEQ_GATE_TEXTURE_FULL > 0.0:
                _relax = 1.0 + (SEQ_GATE_TEXTURE_RELAX - 1.0) * float(
                    np.clip(_texture / SEQ_GATE_TEXTURE_FULL, 0.0, 1.0)
                )
            else:
                _relax = 1.0

            # Umbral de desviación de traslación relativo al PASO INS del par: la
            # deriva admisible del ICP escala con lo que se ha movido el vehículo
            # (paso = ||T_init[:2,3]||). Suelo mínimo MAX_SEQ_ICP_TRANSLATION_DEV
            # para no ser demasiado laxo en pasos cortos. gain=0 -> umbral fijo.
            _ins_step = float(np.linalg.norm(T_init[:2, 3]))
            _tr_dev_base = max(
                MAX_SEQ_ICP_TRANSLATION_DEV,
                SEQ_TR_DEV_STEP_GAIN * _ins_step
            )
            _tr_dev_thr  = _tr_dev_base * _relax
            _yaw_dev_thr = MAX_SEQ_ICP_YAW_DEV * _relax

            _icp_tr_dev = None
            _dir_dev = None

            if valid_registration:

                _T_raw = result.transformation

                # --- translation deviation ---
                _icp_tr_dev = float(
                    np.linalg.norm(_T_raw[:2, 3] - T_init[:2, 3])
                )
                if _icp_tr_dev > _tr_dev_thr:
                    valid_registration = False
                    seq_failure_reason = "icp_translation_deviation"

            if valid_registration:

                # --- translation DIRECTION deviation ---
                # arctan2(T[1,0], T[0,0]) extracts the rotation-matrix yaw,
                # which stays ≈0° even when the TRANSLATION goes in the
                # wrong direction on a flat seabed.  The correct check is
                # the angle of the TRANSLATION VECTOR T[:2,3] vs T_init[:2,3].
                _icp_len  = float(np.linalg.norm(_T_raw[:2, 3]))
                _init_len = float(np.linalg.norm(T_init[:2, 3]))

                if _icp_len > 0.05 and _init_len > 0.05:
                    _icp_dir  = np.arctan2(_T_raw[1, 3],  _T_raw[0, 3])
                    _init_dir = np.arctan2(T_init[1, 3],  T_init[0, 3])
                    _dir_dev  = float(abs(np.degrees(
                        np.arctan2(
                            np.sin(_icp_dir - _init_dir),
                            np.cos(_icp_dir - _init_dir)
                        )
                    )))
                    if _dir_dev > _yaw_dev_thr:
                        valid_registration = False
                        seq_failure_reason = "icp_direction_deviation"

            # =========================================================
            # VALID ICP
            # =========================================================

            if valid_registration:

                accepted_seq += 1

                # FIX 1 — INS rotation + ICP translation.
                # On flat seabed the ICP rotation is random noise
                # (std ~19°/step) that integrates into a random walk and deviates
                # the trajectory tens of degrees. The INS rotation (DVL+IMU) is
                # reliable, so the edge uses the rotation of T_init.
                #
                # Option A — the edge is full SE(3) (3D rotation + INS Z), not
                # pure 2D. Avoids the compression/torsion of turns (where the AUV
                # has pitch) that made drift grow with the trajectory. The XY
                # translation carries the ICP correction.
                #
                # Option A1 — SEQ_CROSS_TRACK_GAIN damps the ICP lateral
                # (cross-track) component, which injected a systematic bias toward
                # +East (contracts left, overshoots right).
                T = ins_rotation_icp_translation(
                    result.transformation,
                    T_init,
                    min_length_ratio=MIN_SEQ_ICP_LENGTH_RATIO,
                    max_length_ratio=MAX_SEQ_ICP_LENGTH_RATIO,
                    anchor_scale=SEQ_ANCHOR_SCALE,
                    cross_track_gain=SEQ_CROSS_TRACK_GAIN
                )

                # R1 — anisotropic information from the registration covariance
                # (Censi/Palomer): low cross-track on flat seabed, high where
                # relief determines the pose. Falls back to isotropic if not
                # computable.
                info, _cov_diag = edge_information(
                    result,
                    source,
                    target,
                    temporal_distance=temporal_distance,
                    loop=False,
                )

            # =========================================================
            # FALLBACK NAVIGATION
            # =========================================================

            else:

                fallback_count += 1
                rejected_seq += 1

                rospy.logwarn(
                    f"Sequential {registration_algorithm.upper()} rejected "
                    f"{idx}->{idx-1} — {seq_failure_reason}"
                )

                T = T_init
                info = np.eye(6) * 0.01

        # =============================================================
        # METRICS — common to all cases
        # =============================================================

        metrics["sequential"].append({
            "source": idx,
            "target": idx - 1,
            "algorithm": registration_algorithm,
            "fitness": float(fitness),
            "rmse": float(rmse),
            "correspondences": int(correspondences),
            "accepted": bool(valid_registration),
            "failure_reason": seq_failure_reason,
            # Texture-modulated gate diagnostic (None if the ICP failed).
            "geometric_texture": (
                float(_texture) if _texture is not None else None
            ),
            "icp_translation_dev_m": (
                float(_icp_tr_dev) if _icp_tr_dev is not None else None
            ),
            "icp_direction_dev_deg": (
                float(_dir_dev) if _dir_dev is not None else None
            ),
            "translation_dev_threshold_m": (
                float(_tr_dev_thr) if _tr_dev_thr is not None else None
            ),
            "yaw_dev_threshold_deg": (
                float(_yaw_dev_thr) if _yaw_dev_thr is not None else None
            ),
        })

        # =============================================================
        # POSE GRAPH — node and edge always added here
        # =============================================================
        # Odometry and the graph elements are updated at a single
        # point of the loop, regardless of whether the ICP failed,
        # was rejected, or was accepted.
        # This guarantees that when the monitor visualizes the graph,
        # the node and its corresponding edge already exist.
        # =============================================================

        odometry = odometry @ T

        pose_graph.nodes.append(

            o3d.pipelines.registration.
            PoseGraphNode(
                odometry
            )
        )

        pose_graph.edges.append(

            o3d.pipelines.registration.
            PoseGraphEdge(

                idx,
                idx - 1,

                T,

                info,

                uncertain=False
            )
        )

        # =============================================================
        # MONITOR — unified throttle
        # =============================================================
        # Runs every MONITOR_UPDATE_EVERY iterations for all cases
        # (ICP ok, rejected, or failed).
        # The graph already has the node and edge added just above,
        # so the monitor always visualizes a consistent state.
        # =============================================================

        if idx % MONITOR_UPDATE_EVERY == 0:

            rospy.loginfo(
                f"[MONITOR] "
                f"{len(pose_graph.nodes)} nodes, "
                f"{len(pose_graph.edges)} edges"
            )

            monitor.update(
                pose_graph,
                patches=patches
            )

    rospy.loginfo(
        f"PoseGraph nodes: "
        f"{len(pose_graph.nodes)}"
    )

    rospy.loginfo(
        f"PoseGraph edges: "
        f"{len(pose_graph.edges)}"
    )

    # =========================================================================
    # RAW TRAJECTORY
    # =========================================================================

    raw_trajectory = []

    for patch in patches:

        T_nav = pose_dict_to_matrix(
            patch.pose
        )

        raw_trajectory.append(
            T_nav[:3, 3]
        )

    raw_trajectory = np.array(
        raw_trajectory
    )

    # =========================================================================
    # RAW MAP
    # =========================================================================

    rospy.loginfo(
        "Building RAW map..."
    )

    raw_map = o3d.geometry.PointCloud()

    for patch in patches:

        T_nav = pose_dict_to_matrix(
            patch.pose
        )

        pcd_global = copy.deepcopy(
            patch.pcd
        )

        pcd_global.transform(T_nav)

        raw_map += pcd_global

    raw_map = raw_map.voxel_down_sample(
        FINAL_DOWNSAMPLE
    )

    stage_times["sequential_and_raw_map"] = time.time() - _t_stage
    _t_stage = time.time()

    # =========================================================================
    # LOOP CLOSURE
    # =========================================================================

    if enable_loop_closure:

        # Voxel del RANSAC-FPFH del loop closure, DESACOPLADO del voxel del patch:
        # el patch puede ser muy fino (0.10 m en vuelo rasante) pero FPFH necesita
        # soporte grueso para discriminar. Sin esto, RANSAC falla casi siempre
        # aunque haya relieve (ver LOOP_RANSAC_VOXEL arriba).
        _loop_ransac_voxel = max(VOXEL_SIZE, LOOP_RANSAC_VOXEL, LOOP_RANSAC_MIN_VOXEL)

        rospy.loginfo(
            "Searching loop closures... "
            f"(RANSAC-FPFH voxel={_loop_ransac_voxel:.2f} m, "
            f"patch voxel={VOXEL_SIZE:.2f} m)"
        )

        for idx in tqdm(

                range(len(patches)),

                desc="Loop closure"):

            candidates = (

                scan_context_manager.
                detect_loop_candidates(

                    idx,

                    top_k=
                    MAX_LOOP_CANDIDATES,

                    threshold=
                    SCAN_CONTEXT_THRESHOLD,

                    # Spatial pre-filter: only descriptors of patches within the
                    # INS proximity gate are evaluated. Since the gate rejected
                    # ~99% of candidates afterward, their FFTs are now not even
                    # computed -> loop closure from O(N²) to nearly linear.
                    max_ins_distance=MAX_LOOP_INS_DISTANCE
                )
            )

            for cand_idx, score in candidates:

                # =============================================================
                # INS PROXIMITY GATE
                # =============================================================
                # Reject candidates whose INS positions are too far apart.
                # In flat/featureless environments, distant patches can look
                # identical (flat seabed); this gate prevents false positives
                # from parallel lawnmower strips or other repeated structures.
                # =============================================================

                ins_src = np.array([
                    patches[idx].pose['north'],
                    patches[idx].pose['east']
                ])

                ins_tgt = np.array([
                    patches[cand_idx].pose['north'],
                    patches[cand_idx].pose['east']
                ])

                ins_distance = float(
                    np.linalg.norm(ins_src - ins_tgt)
                )

                if ins_distance > MAX_LOOP_INS_DISTANCE:

                    rejected_loops += 1

                    metrics["loops"].append({
                        "source": idx,
                        "target": cand_idx,
                        "algorithm": registration_algorithm,
                        "scan_context_score": float(score),
                        "ins_distance_m": ins_distance,
                        "fitness": 0.0,
                        "rmse": 0.0,
                        "correspondences": 0,
                        "accepted": False,
                        "failure_reason": "ins_proximity_gate",
                        "loop_seed": None,
                    })

                    continue

                source = patches[idx].pcd
                target = patches[cand_idx].pcd

                # Poda pre-RANSAC por textura geométrica: si NINGUNO de los dos
                # patches tiene relieve, FPFH no discrimina y RANSAC no convergerá.
                # Se salta el par (ahorra el registro caro). Con relieve real en
                # cualquiera de los dos, NO se poda (umbral bajo, conservador).
                if LOOP_MIN_TEXTURE_FOR_RANSAC > 0.0:
                    _tex_pair = max(
                        _geometric_texture(source),
                        _geometric_texture(target)
                    )
                    if _tex_pair < LOOP_MIN_TEXTURE_FOR_RANSAC:
                        rejected_loops += 1
                        metrics["loops"].append({
                            "source": idx,
                            "target": cand_idx,
                            "algorithm": registration_algorithm,
                            "scan_context_score": float(score),
                            "ins_distance_m": ins_distance,
                            "fitness": 0.0,
                            "rmse": 0.0,
                            "correspondences": 0,
                            "accepted": False,
                            "failure_reason": "flat_pair_skipped_pre_ransac",
                            "loop_seed": None,
                        })
                        continue

                ransac_result = execute_global_registration(
                    source,
                    target,
                    _loop_ransac_voxel
                )

                # RANSAC-only (baseline): if RANSAC fails the candidate is
                # discarded. Seeding the ICP with the INS prior when RANSAC fails
                # (Fix B) worsened the result — accepted loops were nearly
                # consecutive patches that deformed the graph. See report.md.
                if ransac_result is None:

                    rejected_loops += 1

                    metrics["loops"].append({
                        "source": idx,
                        "target": cand_idx,
                        "algorithm": registration_algorithm,
                        "scan_context_score": float(score),
                        "ins_distance_m": ins_distance,
                        "fitness": 0.0,
                        "rmse": 0.0,
                        "correspondences": 0,
                        "accepted": False,
                        "failure_reason": "ransac_failed",
                        "loop_seed": None,
                    })

                    continue

                T_init = ransac_result.transformation
                loop_seed = "ransac"

                result = execute_local_registration(
                    source,
                    target,
                    T_init,
                    registration_algorithm,
                    ICP_DISTANCE,
                    ICP_MAX_ITER,
                    ndt_resolution,
                    ndt_max_points,
                    ndt_min_points_per_voxel,
                    colored_icp_lambda=colored_icp_lambda,
                    hybrid_min_texture=hybrid_min_texture
                )

                if result is None:

                    rejected_loops += 1

                    metrics["loops"].append({
                        "source": idx,
                        "target": cand_idx,
                        "algorithm": registration_algorithm,
                        "scan_context_score": float(score),
                        "ins_distance_m": ins_distance,
                        "fitness": 0.0,
                        "rmse": 0.0,
                        "correspondences": 0,
                        "accepted": False,
                        "failure_reason": "icp_failed",
                        "loop_seed": loop_seed,
                    })

                    continue

                fitness = float(result.fitness)
                rmse = float(result.inlier_rmse)
                correspondences = len(result.correspondence_set)

                # =============================================================
                # ADAPTIVE LOOP VALIDATION
                # =============================================================

                valid_loop = True
                loop_failure_reason = None

                if fitness < LOOP_FITNESS_THRESHOLD:
                    valid_loop = False
                    loop_failure_reason = "fitness_below_threshold"

                if correspondences < MIN_CORRESPONDENCES:
                    valid_loop = False
                    loop_failure_reason = "insufficient_correspondences"

                adaptive_loop_rmse = LOOP_RMSE_THRESHOLD

                if (
                    fitness > HIGH_FITNESS_THRESHOLD
                    and
                    correspondences > HIGH_CORRESPONDENCE_THRESHOLD
                ):

                    adaptive_loop_rmse *= HIGH_RMSE_MULTIPLIER

                    rospy.loginfo(
                        f"[HIGH CONFIDENCE LOOP] "
                        f"Relaxed RMSE threshold: {adaptive_loop_rmse:.3f}"
                    )

                if rmse > adaptive_loop_rmse:
                    valid_loop = False
                    loop_failure_reason = "rmse_above_threshold"

                T_raw = result.transformation

                loop_yaw_deg = np.degrees(
                    np.arctan2(T_raw[1, 0], T_raw[0, 0])
                )

                loop_xy_translation = float(
                    np.linalg.norm(T_raw[:2, 3])
                )

                loop_z_translation = float(abs(T_raw[2, 3]))

                if loop_z_translation > MAX_LOOP_Z_TRANSLATION:
                    valid_loop = False
                    loop_failure_reason = "z_translation_gate"

                if loop_xy_translation > MAX_LOOP_XY_TRANSLATION:
                    valid_loop = False
                    loop_failure_reason = "xy_translation_gate"

                if abs(loop_yaw_deg) > MAX_LOOP_YAW_DEG:
                    valid_loop = False
                    loop_failure_reason = "yaw_gate"

                # =============================================================
                # INS<->LOOP CONSISTENCY GATE  (redesigned — 3 AND conditions)
                # =============================================================
                # A closure is VALID only if it distinguishes a REAL REVISIT (the
                # AUV returns to the same area; the INS has drift the ICP
                # corrects) from a PARALLEL-STRIP FALSE POSITIVE (two adjacent
                # lawnmower passes, laterally separated, whose seabed looks alike
                # because the wide MBES beam overlaps them).
                #
                # Lesson from the run that collapsed the lawnmower (49 m -> 15 m):
                # a gate based ONLY on the discrepancy |T_raw − T_ins_rel| accepts
                # the false positives, because on a parallel strip the ICP aligns
                # with |T_raw|~0 (spurious coincidence) and the discrepancy ~ the
                # INS separation (4-5 m), which passes any reasonable bound. The
                # optimizer then SUPERPOSES strips that should stay separate.
                # (321 false positives, 0 real revisits.)
                #
                # Physical distinction (real revisit vs parallel strip):
                #   - REVISIT: small INS separation (the AUV came back close) AND
                #     the ICP produces a correction |T_raw| of the order of the
                #     drift (ratio = |T_raw_xy|/ins_distance is not ~0).
                #   - PARALLEL STRIP: INS separation several meters (not the same
                #     track) and/or |T_raw|~0 (ratio->0, spurious alignment).
                #
                # All THREE conditions are required (AND):
                #   1) Revisit proximity: ins_distance <= MAX_LOOP_REVISIT_INS_DIST
                #      (a real closure is between points the INS places close;
                #      parallel strips fall outside).
                #   2) Correction ratio: |T_raw_xy| / ins_distance >=
                #      MIN_LOOP_ICP_INS_RATIO  (rejects |T_raw|~0: the signature of
                #      the similar-seabed false positive). Restored from the
                #      original gate, which DID capture this protection.
                #   3) Bounded discrepancy: ||T_raw_xy − T_ins_rel_xy|| <=
                #      MAX_LOOP_INS_DISCREPANCY  (anti-nonsense upper bound).
                #
                # On a single-pass lawnmower (no revisits) this gives ~0 closures,
                # which is correct. On datasets WITH revisits, it lets the
                # legitimate ones through. T_ins_rel = expected_transform(target,
                # source) = inv(T_target)·T_source, in the same frame as T_raw.
                # =============================================================

                T_ins_rel = expected_transform(
                    patches[cand_idx],   # target (reference frame of T_raw)
                    patches[idx]         # source
                )

                ins_loop_discrepancy = float(
                    np.linalg.norm(T_raw[:2, 3] - T_ins_rel[:2, 3])
                )

                icp_ins_ratio = loop_xy_translation / max(ins_distance, 0.1)

                # Solo se evalúa si los gates previos (fitness/rmse/z/xy/yaw) no
                # rechazaron ya el cierre, para no pisar su razón de rechazo.
                if valid_loop:

                    # 1) Proximidad de revisita (franjas paralelas quedan fuera).
                    if ins_distance > MAX_LOOP_REVISIT_INS_DIST:
                        valid_loop = False
                        loop_failure_reason = "not_a_revisit_ins_too_far"

                    # 2) El ICP debe corregir deriva real, no |T_raw|≈0 espurio.
                    elif (
                        ins_distance >= MIN_INS_DIST_FOR_RATIO_CHECK
                        and icp_ins_ratio < MIN_LOOP_ICP_INS_RATIO
                    ):
                        valid_loop = False
                        loop_failure_reason = "icp_translation_near_zero_vs_ins"

                    # 3) Cota superior anti-disparate.
                    elif ins_loop_discrepancy > MAX_LOOP_INS_DISCREPANCY:
                        valid_loop = False
                        loop_failure_reason = "ins_loop_discrepancy_too_large"

                    if not valid_loop:
                        rospy.logdebug(
                            f"[LOOP REJECTED] {idx}<->{cand_idx}  "
                            f"ins_dist={ins_distance:.2f}m  "
                            f"icp_xy={loop_xy_translation:.3f}m  "
                            f"ratio={icp_ins_ratio:.3f}  "
                            f"discrepancy={ins_loop_discrepancy:.3f}m  "
                            f"reason={loop_failure_reason}"
                        )

                if valid_loop:

                    accepted_loops += 1

                    rospy.loginfo(
                        f"LOOP ACCEPTED {idx}<->{cand_idx}  "
                        f"ins={ins_distance:.2f}m  "
                        f"icp_xy={loop_xy_translation:.3f}m  "
                        f"ins_loop_discrepancy={ins_loop_discrepancy:.3f}m  "
                        f"fit={fitness:.3f}  rmse={rmse:.3f}"
                    )

                    T = constrain_transform(T_raw)

                    # R1 — información anisótropa del registro (loop closure).
                    info, _cov_diag = edge_information(
                        result,
                        source,
                        target,
                        temporal_distance=1.0,
                        loop=True,
                    )

                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            idx,
                            cand_idx,
                            T,
                            info,
                            uncertain=True
                        )
                    )

                    monitor.update(pose_graph, patches=patches)

                else:

                    rejected_loops += 1

                metrics["loops"].append({
                    "source": idx,
                    "target": cand_idx,
                    "algorithm": registration_algorithm,
                    "scan_context_score": float(score),
                    "ins_distance_m": ins_distance,
                    "fitness": float(fitness),
                    "rmse": float(rmse),
                    "correspondences": int(correspondences),
                    "raw_xy_translation": float(loop_xy_translation),
                    "raw_z_translation": float(loop_z_translation),
                    "raw_yaw_deg": float(loop_yaw_deg),
                    # Nueva métrica del gate reformulado: discrepancia XY entre la
                    # pose relativa medida por el cierre y la predicha por el INS.
                    "ins_loop_discrepancy_m": float(ins_loop_discrepancy),
                    # icp_ins_ratio se conserva solo para diagnóstico/compatibilidad
                    # con los plots; YA NO es criterio de aceptación.
                    "icp_ins_ratio": float(
                        loop_xy_translation / max(ins_distance, 0.1)
                    ),
                    "accepted": bool(valid_loop),
                    "failure_reason": loop_failure_reason,
                    "loop_seed": loop_seed,
                })

    stage_times["loop_closure"] = time.time() - _t_stage
    _t_stage = time.time()

    # =========================================================================
    # CROSS-TRACK OVERLAP CONSTRAINTS (R0.2 — estilo Torroba 2020)
    # =========================================================================
    # Añade aristas entre franjas ADYACENTES del lawnmower (espacialmente vecinas,
    # temporalmente lejanas), explotando el solape lateral que SÍ existe aunque no
    # haya cruces. Corrige la deriva entre líneas sin fusionarlas. Ver el bloque de
    # constantes XTRACK_* para el porqué de cada umbral.
    accepted_xtrack = 0
    rejected_xtrack = 0
    metrics["cross_track"] = []

    if ENABLE_CROSS_TRACK_EDGES and len(patches) > XTRACK_MIN_TEMPORAL_GAP:

        # Posiciones INS (north, east) de cada patch para el KD-tree espacial.
        ins_xy = np.array([
            [p.pose['north'], p.pose['east']] for p in patches
        ])

        from scipy.spatial import cKDTree
        xtrack_tree = cKDTree(ins_xy)

        # ---------------------------------------------------------------------
        # AUTO-CALIBRACIÓN de la banda [min,max] y del gap temporal por la
        # geometría REAL de la misión (ver XTRACK_AUTO_TUNE arriba).
        # Para cada patch buscamos el vecino espacial más cercano que NO sea
        # secuencial (|Δidx| >= gap temporal base): esa distancia es la separación
        # a la pasada vecina. La MEDIANA de esas separaciones centra la banda.
        # ---------------------------------------------------------------------
        if XTRACK_AUTO_TUNE:
            # Objetivo: la banda debe rodear la separación a la PASADA VECINA que el
            # loop closure NO cubre. El loop closure ya empareja revisitas cercanas
            # (< MAX_LOOP_INS_DISTANCE); por debajo de ese suelo el par es "revisita",
            # no "franja adyacente", y meterlo como cross-track duplica trabajo. Por eso
            # el suelo de medida es _floor = MAX_LOOP_INS_DISTANCE (evita el colapso a
            # ~0 cuando la trayectoria se auto-cruza mucho, como en octógonos densos).
            _base_gap = max(2, XTRACK_MIN_TEMPORAL_GAP)
            _floor = max(1.0, MAX_LOOP_INS_DISTANCE)
            # Radio de búsqueda amplio: la separación esperada más un margen.
            _search_r = max(XTRACK_MAX_INS_DIST, 2.0 * _floor)
            _sep = []
            _revisit_gaps = []
            for _i in range(len(patches)):
                # TODOS los vecinos no-secuenciales dentro del radio; nos quedamos con
                # el más cercano que supere el suelo de revisita (la franja adyacente).
                _cand = xtrack_tree.query_ball_point(ins_xy[_i], _search_r)
                _best = None
                for _jj in _cand:
                    if _jj == _i or abs(int(_jj) - _i) < _base_gap:
                        continue
                    _dd = float(np.linalg.norm(ins_xy[_i] - ins_xy[_jj]))
                    if _dd < _floor or not np.isfinite(_dd):
                        continue
                    if _best is None or _dd < _best[0]:
                        _best = (_dd, abs(int(_jj) - _i))
                if _best is not None:
                    _sep.append(_best[0])
                    _revisit_gaps.append(_best[1])
            if len(_sep) >= 10:
                _med = float(np.median(_sep))
                XTRACK_MIN_INS_DIST = max(_floor, _med * (1.0 - XTRACK_AUTO_BAND_TOL))
                XTRACK_MAX_INS_DIST = max(_med * (1.0 + XTRACK_AUTO_BAND_TOL),
                                          XTRACK_MIN_INS_DIST + 2.0)
                # Gap temporal: percentil 10 de los saltos de índice hasta revisitar
                # la vecindad, con suelo (no lo relajamos por encima del manual).
                _auto_gap = int(np.percentile(_revisit_gaps, 10))
                XTRACK_MIN_TEMPORAL_GAP = max(10, min(XTRACK_MIN_TEMPORAL_GAP, _auto_gap))
                rospy.loginfo(
                    "Cross-track AUTO-TUNE: separación mediana a la pasada vecina "
                    f"(>{_floor:.0f} m) = {_med:.1f} m (n={len(_sep)}) -> banda "
                    f"[{XTRACK_MIN_INS_DIST:.1f}, {XTRACK_MAX_INS_DIST:.1f}] m, "
                    f"gap temporal >= {XTRACK_MIN_TEMPORAL_GAP}"
                )
            else:
                rospy.logwarn(
                    "Cross-track AUTO-TUNE: sin pasadas vecinas claras por encima de "
                    f"{_floor:.0f} m (n={len(_sep)}); la revisita la cubre el loop "
                    f"closure. Se mantiene la banda manual "
                    f"[{XTRACK_MIN_INS_DIST:.1f}, {XTRACK_MAX_INS_DIST:.1f}] m."
                )

        rospy.loginfo(
            "Cross-track overlap constraints: searching adjacent-strip pairs "
            f"(INS dist in [{XTRACK_MIN_INS_DIST:.1f}, {XTRACK_MAX_INS_DIST:.1f}] m, "
            f"temporal gap >= {XTRACK_MIN_TEMPORAL_GAP})..."
        )

        for idx in tqdm(range(len(patches)), desc="Cross-track"):

            # Vecinos espaciales dentro del radio máximo (incluye el propio idx y
            # vecinos secuenciales, que filtramos por gap temporal abajo).
            neigh = xtrack_tree.query_ball_point(
                ins_xy[idx], XTRACK_MAX_INS_DIST
            )

            # Candidatos: franja vecina (gap temporal grande, distancia en banda),
            # y solo j < idx para no duplicar la arista (i,j)/(j,i).
            cand = []
            for j in neigh:
                if j >= idx:
                    continue
                if abs(idx - j) < XTRACK_MIN_TEMPORAL_GAP:
                    continue
                d = float(np.linalg.norm(ins_xy[idx] - ins_xy[j]))
                if d < XTRACK_MIN_INS_DIST or d > XTRACK_MAX_INS_DIST:
                    continue
                cand.append((d, j))

            # Las franjas vecinas más cercanas primero; limita el nº de aristas.
            cand.sort(key=lambda t: t[0])
            cand = cand[:XTRACK_MAX_EDGES_PER_PATCH]

            for d_ins, cand_idx in cand:

                source = patches[idx].pcd
                target = patches[cand_idx].pcd

                # Prior INS relativo (inv(T_cand) @ T_idx): entre franjas vecinas el
                # INS acierta rumbo/avance; solo deriva el offset lateral. Es una
                # semilla mucho mejor que RANSAC para este caso.
                T_prior = expected_transform(patches[cand_idx], patches[idx])

                result = execute_local_registration(
                    source,
                    target,
                    T_prior,
                    registration_algorithm,
                    ICP_DISTANCE,
                    ICP_MAX_ITER,
                    ndt_resolution,
                    ndt_max_points,
                    ndt_min_points_per_voxel,
                    colored_icp_lambda=colored_icp_lambda,
                    hybrid_min_texture=hybrid_min_texture
                )

                fitness = float(result.fitness) if result is not None else 0.0
                rmse = (
                    float(result.inlier_rmse) if result is not None else 0.0
                )

                # Discrepancia XY entre la corrección medida y el prior INS: cota de
                # coherencia (no debe mover la franja absurdamente).
                if result is not None:
                    t_meas = result.transformation[:2, 3]
                    t_prior = T_prior[:2, 3]
                    discrepancy = float(np.linalg.norm(t_meas - t_prior))
                else:
                    discrepancy = float("inf")

                valid = (
                    result is not None
                    and fitness >= XTRACK_MIN_FITNESS
                    and discrepancy <= XTRACK_MAX_INS_DISCREPANCY
                )

                if valid:

                    accepted_xtrack += 1

                    # Arista 3-DoF nativa (R0.3): yaw + XY del registro, Z + roll/pitch
                    # del prior INS (fiables). Es coherente con la arista secuencial
                    # (ins_rotation_icp_translation, que también conserva la vertical
                    # del INS) y con el constraint gravity-constrained de Torroba 2020 /
                    # Tan 2022 (3-DoF > 6-DoF). Conservar la vertical del INS — en vez de
                    # ponerla a 0 — evita introducir un escalón de Z entre franjas.
                    T_edge = project_to_3dof(
                        result.transformation,
                        vertical_ref=T_prior
                    )

                    # R1 — información anisótropa del registro (cross-track).
                    info, _cov_diag = edge_information(
                        result,
                        source,
                        target,
                        temporal_distance=1.0,
                        loop=True,
                    )

                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(
                            idx,
                            cand_idx,
                            T_edge,
                            info,
                            uncertain=True
                        )
                    )

                else:
                    rejected_xtrack += 1

                metrics["cross_track"].append({
                    "source": idx,
                    "target": cand_idx,
                    "ins_distance_m": d_ins,
                    "fitness": fitness,
                    "rmse": rmse,
                    "ins_discrepancy_m": discrepancy,
                    "accepted": bool(valid),
                })

        rospy.loginfo(
            f"Cross-track overlap: {accepted_xtrack} edges accepted, "
            f"{rejected_xtrack} rejected"
        )
        monitor.update(pose_graph, patches=patches)

    stage_times["cross_track"] = time.time() - _t_stage
    _t_stage = time.time()

    # =========================================================================
    # GLOBAL OPTIMIZATION
    # =========================================================================

    rospy.loginfo(
        "Global pose graph optimization (robust back-end)..."
    )

    # =========================================================================
    # R2 — BACK-END ROBUSTO
    # =========================================================================
    # Open3D optimiza el pose-graph con el LINE PROCESS de Choi et al. 2015 ("Robust
    # Reconstruction of Indoor Scenes"), que es el equivalente práctico a las
    # Switchable Constraints de Sünderhauf 2012 / Dynamic Covariance Scaling: a cada
    # arista marcada uncertain=True (loop closure y cross-track) le asocia una variable
    # de "line process" l∈[0,1] que el optimizador estima junto con las poses. Una
    # arista cuyo residuo es inconsistente con el resto del grafo ve su l→0: queda
    # AUTOMÁTICAMENTE DESACTIVADA, sin gates manuales. Esto ataca de raíz la trampa de
    # las franjas paralelas (ANALISIS_RESULTADOS §2f): si una arista cross-track
    # alinea espuriamente dos franjas vecinas, el back-end la apaga en vez de colapsar
    # el lawnmower.
    #
    # Palancas:
    #  - edge_prune_threshold: por debajo de este valor de line-process la arista se
    #    poda. Más alto = más agresivo descartando aristas dudosas.
    #  - preference_loop_closure: confianza relativa en las aristas uncertain frente a
    #    la odometría. <1.0 = back-end más escéptico con loops/cross-track (recomendado
    #    aquí, donde pueden colarse falsos positivos de franja paralela). =1.0 neutro.
    #  Las aristas de ODOMETRÍA (secuencial, uncertain=False) NO se ven afectadas por
    #  el line process: la cadena de odometría se respeta siempre.
    option = (

        o3d.pipelines.registration.
        GlobalOptimizationOption(

            max_correspondence_distance=
            ICP_DISTANCE,

            edge_prune_threshold=
            EDGE_PRUNE_THRESHOLD,

            preference_loop_closure=
            PREFERENCE_LOOP_CLOSURE,

            reference_node=0
        )
    )

    # Conteo de aristas uncertain ANTES de optimizar, para reportar cuántas sobreviven
    # (las podadas por el line process son las que el back-end consideró espurias).
    _n_uncertain_before = sum(
        1 for e in pose_graph.edges if e.uncertain
    )

    o3d.pipelines.registration.global_optimization(

        pose_graph,

        o3d.pipelines.registration.
        GlobalOptimizationLevenbergMarquardt(),

        o3d.pipelines.registration.
        GlobalOptimizationConvergenceCriteria(),

        option
    )

    rospy.loginfo(
        "Pose graph optimization finished"
    )

    # R2 — diagnóstico del line process: cuántas aristas uncertain quedaron con
    # confianza baja (el back-end las consideró espurias y las atenuó/desactivó).
    # Open3D escribe el valor del line process en edge.confidence tras optimizar.
    _n_uncertain_off = sum(
        1 for e in pose_graph.edges
        if e.uncertain and float(e.confidence) < EDGE_PRUNE_THRESHOLD
    )
    metrics["robust_backend"] = {
        "uncertain_edges": int(_n_uncertain_before),
        "deactivated_by_line_process": int(_n_uncertain_off),
        "edge_prune_threshold": float(EDGE_PRUNE_THRESHOLD),
        "preference_loop_closure": float(PREFERENCE_LOOP_CLOSURE),
    }
    rospy.loginfo(
        f"Robust back-end (line process): {_n_uncertain_off}/"
        f"{_n_uncertain_before} uncertain edges deactivated as spurious"
    )

    stage_times["global_optimization"] = time.time() - _t_stage

    # =========================================================================
    # VERTICAL POSE RESTORATION (Z + roll + pitch)
    # =========================================================================
    # The flat underwater scene gives the Levenberg-Marquardt optimizer almost
    # no vertical constraints, so it leaves the nodes' Z translation AND their
    # roll/pitch poorly constrained. Restoring only Z is not enough: a spurious
    # roll/pitch of a few degrees, applied by build_final_map() to a patch that
    # spans 30-50 m in XY, projects the patch edges several metres up/down in Z
    # (radius·sin(θ)) — producing a thick, ghosted map (Z spread 3.2 m, points
    # up to 14 m above the seabed).
    #
    # The IMU roll/pitch and the INS depth are reliable; the optimizer's are
    # not. We therefore rebuild each node's rotation from the INS roll+pitch
    # while KEEPING the optimized yaw (the valid SLAM correction is in-plane),
    # and restore the INS Z translation. XY translation stays as optimized.
    #
    # R0.3 — Esta restauración ES el componente 3-DoF (gravity-constrained) del
    # back-end: como Open3D optimiza en SE(3) completo (6-DoF) y el fondo plano no
    # restringe la vertical, proyectamos cada nodo al subespacio (x, y, yaw) tras
    # optimizar, fijando Z/roll/pitch del INS. Es COHERENTE con las aristas, que ya
    # son 3-DoF (secuencial: ins_rotation_icp_translation; cross-track: project_to_3dof
    # con vertical del INS). El estado del arte (Torroba 2020, Tan 2022) muestra que
    # este 3-DoF gravity-constrained SUPERA al 6-DoF libre. La forma idiomática sería
    # optimizar nativamente en SE(2)+altura; con Open3D, esta proyección post-opt es
    # el equivalente práctico.
    # =========================================================================

    _n_diverged = 0
    for i, patch in enumerate(patches):

        if i >= len(pose_graph.nodes):
            break

        T_node = pose_graph.nodes[i].pose.copy()

        # INS depth (reliable); XY translation stays as optimized (por defecto).
        T_nav = pose_dict_to_matrix(patch.pose)

        # ---------------------------------------------------------------------
        # GATE ANTI-DIVERGENCIA: si el optimizador desplazó este nodo en XY más de
        # MAX_NODE_CORRECTION_M respecto al INS, es una divergencia (el line process
        # no detecta la degeneración colectiva). Se revierte su XY a la navegación
        # bruta — cae a INS, igual que un registro secuencial rechazado. También se
        # descarta el yaw optimizado (viene de la misma solución divergente).
        # ---------------------------------------------------------------------
        _xy_corr = float(np.linalg.norm(T_node[:2, 3] - T_nav[:2, 3]))
        if MAX_NODE_CORRECTION_M > 0.0 and _xy_corr > MAX_NODE_CORRECTION_M:
            T_node[:2, 3] = T_nav[:2, 3]
            opt_yaw = patch.pose["yaw"]
            _n_diverged += 1
        else:
            # Optimized yaw (valid in-plane SLAM correction).
            opt_yaw = np.arctan2(T_node[1, 0], T_node[0, 0])

        # INS roll/pitch (reliable) + yaw (optimizado, o INS si divergió) → rotación.
        R_fixed = tr.euler_matrix(
            patch.pose["roll"],
            patch.pose["pitch"],
            opt_yaw,
            axes='sxyz'
        )[:3, :3]

        T_node[:3, :3] = R_fixed

        # INS depth (reliable).
        T_node[2, 3] = T_nav[2, 3]

        pose_graph.nodes[i].pose = T_node

    rospy.loginfo(
        "Vertical pose restored from INS (Z + roll/pitch, optimized yaw kept)"
    )
    if _n_diverged > 0:
        rospy.logwarn(
            f"Anti-divergence gate: {_n_diverged}/{len(patches)} nodos revertidos "
            f"a INS por corrección XY > {MAX_NODE_CORRECTION_M:.0f} m "
            "(el optimizador global divergió en esos nodos)."
        )

    # NOTA: como consecuencia de esta restauración, el SLAM NO corrige la
    # profundidad — la Z de cada nodo es exactamente la del INS, así que la
    # corrección en Z respecto a la navegación bruta es 0 por diseño (la métrica
    # de corrección Z saldrá ~0). La corrección SLAM válida vive en el plano XY
    # (yaw + traslación), donde el fondo sí aporta restricciones. Si en algún
    # dataset la profundidad del INS NO fuera fiable, habría que reconsiderar
    # esta restauración (p. ej. conservar la Z optimizada, no la del INS).

    monitor.update(
        pose_graph,
        patches=patches
    )

    # =========================================================================
    # SLAM TRAJECTORY
    # =========================================================================

    slam_trajectory = []

    for node in pose_graph.nodes:

        slam_trajectory.append(
            node.pose[:3, 3]
        )

    slam_trajectory = np.array(
        slam_trajectory
    )

    # =========================================================================
    # TRAJECTORY METRICS
    # =========================================================================

    _n = min(len(raw_trajectory), len(slam_trajectory))
    _corrections_xy = np.linalg.norm(
        slam_trajectory[:_n, :2] - raw_trajectory[:_n, :2],
        axis=1
    )

    def _path_length(traj):
        return float(
            np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
        )

    metrics["trajectory"] = {
        "nodes": int(_n),
        "corrections_xy_m": _corrections_xy.tolist(),
        "mean_correction_xy_m": float(np.mean(_corrections_xy)),
        "max_correction_xy_m": float(np.max(_corrections_xy)),
        "std_correction_xy_m": float(np.std(_corrections_xy)),
        "raw_path_length_m": _path_length(raw_trajectory),
        "slam_path_length_m": _path_length(slam_trajectory),
    }

    rospy.loginfo(
        f"Trajectory correction — "
        f"mean={metrics['trajectory']['mean_correction_xy_m']:.2f} m  "
        f"max={metrics['trajectory']['max_correction_xy_m']:.2f} m"
    )

    # =========================================================================
    # FINAL MAP
    # =========================================================================

    rospy.loginfo(
        "Building SLAM map..."
    )

    final_map = build_final_map(

        patches,

        pose_graph,

        voxel_size=
        FINAL_DOWNSAMPLE
    )

    # =========================================================================
    # CONSISTENCY ERROR (Roman 2006) — métrica primaria SOTA
    # =========================================================================
    # Dispersión vertical en las zonas de solape (incl. solape ADYACENTE entre
    # franjas paralelas del lawnmower). Se calcula ANTES (poses = navegación
    # bruta) y DESPUÉS (poses = grafo optimizado) para reportar la reducción,
    # igual que las tablas de Roman/Torroba/Palomer. NO necesita ground truth ni
    # cruces de trayectoria: es el control medible de la línea R0.
    #
    # El "antes" usa un pose-graph trivial cuyos nodos llevan la pose de
    # navegación bruta de cada patch (mismo adaptador, distintas poses).
    class _RawGraph:
        pass

    _raw_graph = _RawGraph()
    _raw_graph.nodes = [
        type("N", (), {"pose": pose_dict_to_matrix(p.pose)})()
        for p in patches
    ]

    try:
        cons_before = consistency_error_from_patches(
            patches, _raw_graph,
            cell_size=CONSISTENCY_CELL_SIZE,
            min_distinct_sources=2,
        )
        cons_after = consistency_error_from_patches(
            patches, pose_graph,
            cell_size=CONSISTENCY_CELL_SIZE,
            min_distinct_sources=2,
        )

        _mb = cons_before["mean_std_z"]
        _ma = cons_after["mean_std_z"]
        _improv = (
            100.0 * (_mb - _ma) / _mb
            if _mb and np.isfinite(_mb) and _mb > 0 else float("nan")
        )

        metrics["consistency"] = {
            "cell_size_m": CONSISTENCY_CELL_SIZE,
            "raw_navigation": cons_before,
            "slam_optimized": cons_after,
            "improvement_pct": _improv,
        }

        rospy.loginfo(
            f"Consistency error (Roman 2006, cell={CONSISTENCY_CELL_SIZE} m): "
            f"raw={_mb:.3f} m -> slam={_ma:.3f} m  "
            f"({_improv:+.1f}% , valid cells "
            f"{cons_after['n_cells_valid']})"
        )
    except Exception as exc:
        rospy.logwarn(f"Consistency error computation failed: {exc}")
        metrics["consistency"] = {"error": str(exc)}

    # =========================================================================
    # SAVE MAPS
    # =========================================================================

    raw_map_path = os.path.join(

        output_dir,

        "raw_navigation_map.ply"
    )

    slam_map_path = os.path.join(

        output_dir,

        "slam_optimized_map.ply"
    )

    o3d.io.write_point_cloud(

        raw_map_path,

        raw_map
    )

    o3d.io.write_point_cloud(

        slam_map_path,

        final_map
    )

    rospy.loginfo(
        f"Saved RAW map: "
        f"{raw_map_path}"
    )

    rospy.loginfo(
        f"Saved SLAM map: "
        f"{slam_map_path}"
    )

    # =========================================================================
    # SAVE TRAJECTORIES
    # =========================================================================

    np.save(

        os.path.join(
            output_dir,
            "raw_trajectory.npy"
        ),

        raw_trajectory
    )

    np.save(

        os.path.join(
            output_dir,
            "slam_trajectory.npy"
        ),

        slam_trajectory
    )

    # =========================================================================
    # METRICS SUMMARY
    # =========================================================================

    _seq_accepted = [m for m in metrics["sequential"] if m["accepted"]]
    _seq_rejected = [m for m in metrics["sequential"] if not m["accepted"]]
    _loops_accepted = [m for m in metrics.get("loops", []) if m["accepted"]]
    _loops_rejected = [m for m in metrics.get("loops", []) if not m["accepted"]]

    def _safe_stats(values):
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    _seq_fitness_stats = _safe_stats(
        [m["fitness"] for m in _seq_accepted]
    )
    _seq_rmse_stats = _safe_stats(
        [m["rmse"] for m in _seq_accepted]
    )
    _seq_corr_stats = _safe_stats(
        [m["correspondences"] for m in _seq_accepted]
    )
    _loop_fitness_stats = _safe_stats(
        [m["fitness"] for m in _loops_accepted]
    )
    _loop_rmse_stats = _safe_stats(
        [m["rmse"] for m in _loops_accepted]
    )
    _loop_score_stats = _safe_stats(
        [m.get("scan_context_score", 0.0) for m in _loops_accepted]
    )
    _rej_reason_counts = {}
    for m in _seq_rejected:
        reason = m.get("failure_reason", "validation_failed")
        _rej_reason_counts[reason] = (
            _rej_reason_counts.get(reason, 0) + 1
        )
    _loop_rej_reason_counts = {}
    for m in _loops_rejected:
        reason = m.get("failure_reason", "unknown")
        _loop_rej_reason_counts[reason] = (
            _loop_rej_reason_counts.get(reason, 0) + 1
        )

    _traj = metrics.get("trajectory", {})

    metrics["summary"] = {

        # ── Graph ──────────────────────────────────────────────────────
        "registration_algorithm":
            registration_algorithm,
        "total_nodes":
            len(pose_graph.nodes),
        "total_edges":
            len(pose_graph.edges),

        # ── Sequential registration ────────────────────────────────────
        "accepted_seq":            accepted_seq,
        "rejected_seq":            rejected_seq,
        "fallback_edges":          fallback_count,
        "seq_acceptance_ratio":    accepted_seq / max(accepted_seq + rejected_seq, 1),
        "seq_fitness":             _seq_fitness_stats,
        "seq_rmse_m":              _seq_rmse_stats,
        "seq_correspondences":     _seq_corr_stats,
        "seq_rejection_reasons":   _rej_reason_counts,

        # ── Loop closure ───────────────────────────────────────────────
        "accepted_loops":          accepted_loops,
        "rejected_loops":          rejected_loops,
        "loop_acceptance_ratio":   accepted_loops / max(accepted_loops + rejected_loops, 1),
        "loop_fitness":            _loop_fitness_stats,
        "loop_rmse_m":             _loop_rmse_stats,
        "loop_scan_context_score": _loop_score_stats,
        "loop_rejection_reasons":  _loop_rej_reason_counts,

        # ── Trajectory correction ──────────────────────────────────────
        "mean_slam_correction_xy_m": _traj.get("mean_correction_xy_m", 0.0),
        "max_slam_correction_xy_m":  _traj.get("max_correction_xy_m", 0.0),
        "std_slam_correction_xy_m":  _traj.get("std_correction_xy_m", 0.0),
        "raw_path_length_m":         _traj.get("raw_path_length_m", 0.0),
        "slam_path_length_m":        _traj.get("slam_path_length_m", 0.0),

        # ── Anti-divergence gate ───────────────────────────────────────
        "nodes_reverted_divergence": int(_n_diverged),
        "max_node_correction_gate_m": float(MAX_NODE_CORRECTION_M),
    }

    # =========================================================================
    # SAVE METRICS JSON
    # =========================================================================

    metrics_path = os.path.join(

        metrics_dir,

        "slam_metrics.json"
    )

    with open(metrics_path, "w") as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    # =========================================================================
    # SAVE CSV
    # =========================================================================

    csv_path = os.path.join(

        metrics_dir,

        "icp_stats.csv"
    )

    with open(csv_path, "w") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "source",
            "target",
            "algorithm",
            "fitness",
            "rmse",
            "correspondences",
            "accepted",
            "failure_reason",
            "geometric_texture",
            "icp_translation_dev_m",
            "icp_direction_dev_deg",
            "translation_dev_threshold_m",
            "yaw_dev_threshold_deg",
        ])

        for m in metrics["sequential"]:

            writer.writerow([
                m["source"],
                m["target"],
                m.get("algorithm", registration_algorithm),
                m["fitness"],
                m["rmse"],
                m["correspondences"],
                m["accepted"],
                m.get("failure_reason", ""),
                m.get("geometric_texture", ""),
                m.get("icp_translation_dev_m", ""),
                m.get("icp_direction_dev_deg", ""),
                m.get("translation_dev_threshold_m", ""),
                m.get("yaw_dev_threshold_deg", ""),
            ])

    loop_csv_path = os.path.join(

        metrics_dir,

        "loop_stats.csv"
    )

    with open(loop_csv_path, "w") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "source",
            "target",
            "algorithm",
            "scan_context_score",
            "ins_distance_m",
            "fitness",
            "rmse",
            "correspondences",
            "raw_xy_translation",
            "ins_loop_discrepancy_m",
            "icp_ins_ratio",
            "raw_z_translation",
            "raw_yaw_deg",
            "accepted",
            "failure_reason",
            "loop_seed",
        ])

        for m in metrics["loops"]:

            writer.writerow([
                m["source"],
                m["target"],
                m.get("algorithm", registration_algorithm),
                m.get("scan_context_score", ""),
                m.get("ins_distance_m", ""),
                m["fitness"],
                m["rmse"],
                m["correspondences"],
                m.get("raw_xy_translation", ""),
                m.get("ins_loop_discrepancy_m", ""),
                m.get("icp_ins_ratio", ""),
                m.get("raw_z_translation", ""),
                m.get("raw_yaw_deg", ""),
                m["accepted"],
                m.get("failure_reason", ""),
                m.get("loop_seed", ""),
            ])

    # =========================================================================
    # PLOTS
    # =========================================================================

    _save_metrics_plots(
        metrics,
        raw_trajectory,
        slam_trajectory,
        metrics_dir
    )

    # =========================================================================
    # STAGE TIMING REPORT (PERF)
    # =========================================================================
    total = sum(stage_times.values())
    rospy.loginfo(
        "================================================"
    )
    rospy.loginfo("STAGE TIMING (PERF):")
    for name, secs in stage_times.items():
        pct = 100.0 * secs / max(total, 1e-6)
        rospy.loginfo(f"  {name:24s}: {secs:8.1f} s  ({pct:4.1f}%)")
    rospy.loginfo(f"  {'TOTAL (timed stages)':24s}: {total:8.1f} s")

    # Persistir tiempos junto a las métricas para comparar runs.
    try:
        with open(os.path.join(metrics_dir, "stage_times.json"), "w") as f:
            json.dump(
                {**stage_times, "total_s": total},
                f,
                indent=4
            )
    except Exception as exc:
        rospy.logwarn(f"Could not save stage_times.json: {exc}")

    rospy.loginfo(
        "================================================"
    )

    rospy.loginfo(
        "ADVANCED MBES SLAM FINISHED"
    )

    rospy.loginfo(
        "================================================"
    )

    monitor.close()


if __name__ == "__main__":

    main()
