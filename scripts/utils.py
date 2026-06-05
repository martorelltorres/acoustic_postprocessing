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
        max_translation=5,
        T_ref=None):
    """
    Proyecta una transformación ICP a un movimiento 2D acotado (yaw + XY,
    con Z=roll=pitch=0).

    El recorte de yaw se aplica como DESVIACIÓN respecto a una rotación de
    referencia ``T_ref`` (normalmente T_init de la navegación INS), NO en
    valor absoluto.

    Motivo: en los giros del lawnmower la rotación relativa real entre patches
    consecutivos puede llegar a ~172°. Recortar el yaw a ±15° ABSOLUTO impedía
    seguir esos giros y producía una deriva rotacional acumulada catastrófica.
    Con la referencia, el giro real de la INS pasa íntegro y solo se limita la
    CORRECCIÓN que el ICP añade encima (±max_yaw_deg).

    Si ``T_ref is None`` se mantiene el comportamiento histórico (clip absoluto).
    """

    yaw = np.degrees(
        np.arctan2(
            T[1, 0],
            T[0, 0]
        )
    )

    if T_ref is None:

        # Comportamiento histórico: recorte absoluto.
        yaw = np.clip(
            yaw,
            -max_yaw_deg,
            max_yaw_deg
        )

    else:

        # Recorte relativo a la rotación de referencia (INS).
        ref_yaw = np.degrees(
            np.arctan2(
                T_ref[1, 0],
                T_ref[0, 0]
            )
        )

        # Desviación del ICP respecto a la referencia, envuelta a [-180, 180].
        delta = wrap_angle_deg(yaw - ref_yaw)

        # Solo se limita la corrección que el ICP añade sobre la INS.
        delta = np.clip(
            delta,
            -max_yaw_deg,
            max_yaw_deg
        )

        yaw = wrap_angle_deg(ref_yaw + delta)

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
    T_new[2, 3] = 0.0

    return T_new

def ins_rotation_icp_translation(
        T_icp,
        T_init,
        max_translation=5):
    """
    Construye una transformación 2D tomando la ROTACIÓN de la navegación INS
    (``T_init``) y la TRASLACIÓN del resultado ICP (``T_icp``).

    Motivo: en un fondo marino plano y sin estructura, la rotación estimada por
    el ICP es ruido aleatorio (std ~19°/paso, media ≈0). Integrada sobre cientos
    de aristas produce un random walk rotacional que desvía la trayectoria SLAM
    decenas de grados respecto a la navegación. El DVL+IMU de la INS mide la
    rotación con fiabilidad, así que se usa la rotación INS y se deja que el ICP
    solo refine la traslación XY (que sí aporta valor cuando pasa los gates).

    Salida: SE(3) 2D con yaw de la INS, traslación XY del ICP (acotada),
    y Z = roll = pitch = 0.
    """

    yaw = np.arctan2(
        T_init[1, 0],
        T_init[0, 0]
    )

    tx = np.clip(
        T_icp[0, 3],
        -max_translation,
        max_translation
    )

    ty = np.clip(
        T_icp[1, 3],
        -max_translation,
        max_translation
    )

    T_new = np.eye(4)

    c = np.cos(yaw)
    s = np.sin(yaw)

    T_new[0, 0] = c
    T_new[0, 1] = -s
    T_new[1, 0] = s
    T_new[1, 1] = c

    T_new[0, 3] = tx
    T_new[1, 3] = ty
    T_new[2, 3] = 0.0

    return T_new


def constrain_transform_2d(T):

    T2 = np.eye(4)

    # ==========================================
    # EXTRAER YAW
    # ==========================================

    yaw = np.arctan2(
        T[1, 0],
        T[0, 0]
    )

    c = np.cos(yaw)
    s = np.sin(yaw)

    T2[0, 0] = c
    T2[0, 1] = -s
    T2[1, 0] = s
    T2[1, 1] = c

    # ==========================================
    # SOLO XY
    # ==========================================

    T2[0, 3] = T[0, 3]
    T2[1, 3] = T[1, 3]

    # Z = 0
    # roll/pitch = 0

    return T2

def pose_dict_to_matrix(pose):

    T = np.eye(4)

    R = tr.euler_matrix(
        pose["roll"],
        pose["pitch"],
        pose["yaw"],
        axes='sxyz'
    )[:3, :3]

    T[:3, :3] = R

    T[0, 3] = pose["north"]
    T[1, 3] = pose["east"]
    T[2, 3] = -pose["depth"]

    return T


def expected_transform(patch_i, patch_j):

    T_i = pose_dict_to_matrix(
        patch_i.pose
    )

    T_j = pose_dict_to_matrix(
        patch_j.pose
    )

    # Transform relativo correcto en SE(3)
    T_rel = np.linalg.inv(T_i) @ T_j

    return T_rel
