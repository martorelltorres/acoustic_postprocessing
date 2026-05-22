#!/usr/bin/env python3

import numpy as np


def dynamic_information_matrix(
        result,
        base_scale=100.0,
        temporal_distance=1.0,
        loop=False):

    fitness = max(
        float(result.fitness),
        1e-6
    )

    rmse = max(
        float(result.inlier_rmse),
        1e-6
    )

    confidence = np.clip(fitness / rmse, 0.0, 1000.0)

    if loop:

        scale = (
            base_scale *
            confidence *
            5.0
        )

    else:

        scale = (
            base_scale *
            confidence /
            temporal_distance
        )

    info = np.eye(6) * scale

    return info