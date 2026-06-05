#!/usr/bin/env python3

import numpy as np
import open3d as o3d

from scipy.optimize import least_squares
from scipy.spatial import cKDTree


_MIN_NDT_CORRESPONDENCES = 30


class NDTRegistrationResult:

    def __init__(
            self,
            transformation,
            fitness,
            inlier_rmse,
            correspondence_set):

        self.transformation = transformation
        self.fitness = fitness
        self.inlier_rmse = inlier_rmse
        self.correspondence_set = correspondence_set


def _transform_from_params(params):

    tx, ty, yaw = params

    c = np.cos(yaw)
    s = np.sin(yaw)

    T = np.eye(4)
    T[0, 0] = c
    T[0, 1] = -s
    T[1, 0] = s
    T[1, 1] = c
    T[0, 3] = tx
    T[1, 3] = ty

    return T


def _params_from_transform(T):

    yaw = np.arctan2(
        T[1, 0],
        T[0, 0]
    )

    return np.array([
        T[0, 3],
        T[1, 3],
        yaw
    ], dtype=float)


def _transform_points_2d(points, params):

    tx, ty, yaw = params

    c = np.cos(yaw)
    s = np.sin(yaw)

    out = points.copy()

    x = points[:, 0]
    y = points[:, 1]

    out[:, 0] = c * x - s * y + tx
    out[:, 1] = s * x + c * y + ty

    return out


def _pointcloud_points(pcd, voxel_size, max_points):

    down = pcd.voxel_down_sample(
        voxel_size
    )

    points = np.asarray(
        down.points
    )

    points = points[
        np.isfinite(points).all(axis=1)
    ]

    if len(points) > max_points:

        idx = np.linspace(
            0,
            len(points) - 1,
            max_points
        ).astype(int)

        points = points[idx]

    return points


def _build_ndt_grid(
        target_points,
        resolution,
        min_points_per_voxel):

    voxel_points = {}

    voxel_idx = np.floor(
        target_points / resolution
    ).astype(np.int64)

    for idx, point in zip(voxel_idx, target_points):

        key = tuple(idx.tolist())

        voxel_points.setdefault(
            key,
            []
        ).append(point)

    means = []
    sqrt_infos = []

    regularization = (
        resolution * 0.05
    ) ** 2

    for points in voxel_points.values():

        if len(points) < min_points_per_voxel:
            continue

        pts = np.asarray(
            points,
            dtype=float
        )

        mean = pts.mean(axis=0)

        cov = np.cov(
            pts.T
        )

        cov += np.eye(3) * regularization

        try:

            inv_cov = np.linalg.inv(
                cov
            )

            sqrt_info = np.linalg.cholesky(
                inv_cov
            )

        except np.linalg.LinAlgError:

            continue

        means.append(mean)
        sqrt_infos.append(sqrt_info)

    if len(means) == 0:
        return None, None, None

    means = np.asarray(
        means
    )

    sqrt_infos = np.asarray(
        sqrt_infos
    )

    tree = cKDTree(
        means
    )

    return means, sqrt_infos, tree


def robust_ndt(
        source,
        target,
        T_init,
        ndt_resolution=1.0,
        max_iter=60,
        max_correspondence_distance=2.0,
        min_points_per_voxel=5,
        max_points=3000):

    sample_voxel = max(
        ndt_resolution * 0.5,
        0.05
    )

    source_points = _pointcloud_points(
        source,
        sample_voxel,
        max_points
    )

    target_points = _pointcloud_points(
        target,
        sample_voxel,
        max_points * 2
    )

    if len(source_points) < 50:
        return None

    if len(target_points) < 50:
        return None

    means, sqrt_infos, tree = _build_ndt_grid(
        target_points,
        ndt_resolution,
        min_points_per_voxel
    )

    if tree is None:
        return None

    target_tree = cKDTree(
        target_points
    )

    miss_penalty = max(
        max_correspondence_distance,
        ndt_resolution
    )

    def residual(params):

        transformed = _transform_points_2d(
            source_points,
            params
        )

        distances, indices = tree.query(
            transformed,
            distance_upper_bound=max_correspondence_distance
        )

        residuals = np.zeros(
            (len(transformed), 3),
            dtype=float
        )

        valid = np.isfinite(distances)

        if np.any(valid):

            diffs = (
                transformed[valid] -
                means[indices[valid]]
            )

            residuals[valid] = np.einsum(
                "nij,nj->ni",
                sqrt_infos[indices[valid]],
                diffs
            )

        residuals[~valid, 0] = miss_penalty

        return residuals.ravel()

    initial_params = _params_from_transform(
        T_init
    )

    try:

        opt = least_squares(
            residual,
            initial_params,
            loss="soft_l1",
            f_scale=1.0,
            max_nfev=max_iter,
            xtol=1e-4,
            ftol=1e-4,
            gtol=1e-4
        )

    except Exception:

        return None

    T = _transform_from_params(
        opt.x
    )

    transformed = _transform_points_2d(
        source_points,
        opt.x
    )

    distances, indices = target_tree.query(
        transformed,
        distance_upper_bound=max_correspondence_distance
    )

    valid = np.isfinite(distances)

    correspondences = np.flatnonzero(
        valid
    )

    if len(correspondences) < _MIN_NDT_CORRESPONDENCES:
        return None

    diffs = (
        transformed[valid] -
        target_points[indices[valid]]
    )

    rmse = float(
        np.sqrt(
            np.mean(
                np.sum(diffs ** 2, axis=1)
            )
        )
    )

    fitness = float(
        len(correspondences) /
        max(len(source_points), 1)
    )

    correspondence_set = [
        (int(src_idx), int(indices[src_idx]))
        for src_idx in correspondences
    ]

    return NDTRegistrationResult(
        T,
        fitness,
        rmse,
        correspondence_set
    )
