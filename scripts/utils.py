#!/usr/bin/env python3

import os
import numpy as np
import tf.transformations as tr


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def wrap_angle_deg(angle):

    while angle > 180:
        angle -= 360

    while angle < -180:
        angle += 360

    return angle


def constrain_transform(
        T,
        max_yaw_deg=15.0,
        max_translation=5):

    yaw = np.degrees(
        np.arctan2(
            T[1, 0],
            T[0, 0]
        )
    )

    yaw = np.clip(
        yaw,
        -max_yaw_deg,
        max_yaw_deg
    )

    tx = np.clip(
        T[0, 3],
        -max_translation,
        max_translation
    )

    ty = np.clip(
        T[1, 3],
        -max_translation,
        max_translation
    )

    T_new = np.eye(4)

    T_new[:3, :3] = tr.euler_matrix(
        0,
        0,
        np.deg2rad(yaw)
    )[:3, :3]

    T_new[0, 3] = tx
    T_new[1, 3] = ty
    T_new[2, 3] = T[2, 3]

    return T_new

def expected_transform(patch_i, patch_j):

    dx = patch_j.pose["north"] - patch_i.pose["north"]
    dy = patch_j.pose["east"]  - patch_i.pose["east"]
    dyaw_rad = patch_j.pose["yaw"] - patch_i.pose["yaw"]
    dyaw_rad = (dyaw_rad + np.pi) % (2 * np.pi) - np.pi  # wrap a [-π, π]

    T = np.eye(4)

    c = np.cos(dyaw_rad)
    s = np.sin(dyaw_rad)

    T[0, 0] =  c;  T[0, 1] = -s
    T[1, 0] =  s;  T[1, 1] =  c
    T[0, 3] = dx
    T[1, 3] = dy

    return T