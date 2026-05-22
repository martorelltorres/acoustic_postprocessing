#!/usr/bin/env python3

import numpy as np


def compute_scan_context(
        points,
        num_rings=20,
        num_sectors=60,
        max_radius=40.0):

    desc = np.zeros(
        (num_rings, num_sectors),
        dtype=np.float32
    )

    if len(points) == 0:
        return desc

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

    for i in range(len(points)):

        rid = ring_idx[i]
        sid = sector_idx[i]

        desc[rid, sid] = max(
            desc[rid, sid],
            z[i]
        )

    return desc


def scan_context_distance(desc1, desc2):

    d1 = desc1.flatten()
    d2 = desc2.flatten()

    n1 = np.linalg.norm(d1)
    n2 = np.linalg.norm(d2)

    if n1 < 1e-6 or n2 < 1e-6:
        return 1.0

    similarity = np.dot(d1, d2) / (n1 * n2)

    return 1.0 - similarity