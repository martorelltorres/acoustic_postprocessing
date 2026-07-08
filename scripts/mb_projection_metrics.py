#!/usr/bin/env python3
"""
Métricas de calidad de la PROYECCIÓN de multibeam (nube + malla + mosaico de
backscatter). No es SLAM: evalúa el producto cartográfico directo de
multibeam_processor.py y multibeam_intensity.py.

Genera results/metrics/:
  - mb_projection_metrics.json  (resumen numérico)
  - mb_zconsistency.png         (mapa de rugosidad/consistencia vertical por celda)
  - mb_intensity_hist.png       (histograma de backscatter: destapa el banding)
  - mb_coverage.png             (mapa de densidad de puntos por celda)

Uso:
  rosrun acoustic_postprocessing mb_projection_metrics.py            # usa results/ por defecto
  python3 mb_projection_metrics.py <results_dir>

Solo numpy/matplotlib/(rasterio opcional): no necesita ROS ni Open3D.
"""

import os
import sys
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _grid_stats(x, y, z, cell):
    """Estadística por celda XY: media, std (rugosidad) y conteo de Z."""
    xi = np.floor((x - x.min()) / cell).astype(np.int64)
    yi = np.floor((y - y.min()) / cell).astype(np.int64)
    W = xi.max() + 1
    key = yi * W + xi

    # Vectorizado (31M+ puntos): media y std por celda vía sumas segmentadas,
    # sin bucle Python. std = sqrt(E[z^2] - E[z]^2).
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    z_s = z[order]

    cell_keys, starts, counts = np.unique(key_s, return_index=True, return_counts=True)
    csum = np.concatenate(([0.0], np.cumsum(z_s)))
    csum2 = np.concatenate(([0.0], np.cumsum(z_s * z_s)))
    ends = starts + counts
    s1 = csum[ends] - csum[starts]
    s2 = csum2[ends] - csum2[starts]
    means = s1 / counts
    var = np.maximum(s2 / counts - means * means, 0.0)
    stds = np.sqrt(var)

    cy = (cell_keys // W)
    cx = (cell_keys % W)
    return cx, cy, means, stds, counts.astype(np.int64), W


def _load_xyz(path):
    """Carga rápida de .xyz ASCII (loadtxt es lentísimo para 30M+ filas)."""
    try:
        import pandas as pd
        arr = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1, 2],
                          dtype=np.float64).to_numpy()
        return arr
    except Exception:
        return np.loadtxt(path)


def analyze_cloud(xyz_path, cell=1.0, min_pts_cell=5):
    pts = _load_xyz(xyz_path)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    area = float(x.ptp() * y.ptp())
    cx, cy, means, stds, counts, W = _grid_stats(x, y, z, cell)

    valid = counts >= min_pts_cell
    rough = stds[valid]

    zmed = np.median(z)
    zstd = float(z.std())
    outliers = int((np.abs(z - zmed) > 3.0 * zstd).sum())

    metrics = {
        "n_points": int(len(pts)),
        "footprint_x_m": float(x.ptp()),
        "footprint_y_m": float(y.ptp()),
        "footprint_area_m2": area,
        "density_pts_per_m2": float(len(pts) / area) if area > 0 else 0.0,
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "z_span_m": float(z.ptp()),
        "z_std_m": zstd,
        "z_outliers_3sigma": outliers,
        "z_outlier_fraction": float(outliers / len(pts)),
        # Consistencia vertical (Roman 2006): dispersión Z por celda en solape.
        "consistency_cell_size_m": cell,
        "cells_total": int(valid.sum()),
        "roughness_mean_std_z_m": float(rough.mean()) if rough.size else None,
        "roughness_median_std_z_m": float(np.median(rough)) if rough.size else None,
        "roughness_p90_std_z_m": float(np.percentile(rough, 90)) if rough.size else None,
        # Cobertura: celdas de la rejilla con al menos 1 punto / total del bbox.
        "coverage_filled_cells": int(len(counts)),
        "coverage_bbox_cells": int(np.ceil(x.ptp() / cell) * np.ceil(y.ptp() / cell)),
    }
    metrics["coverage_fraction"] = float(
        metrics["coverage_filled_cells"] / max(metrics["coverage_bbox_cells"], 1)
    )
    grids = {"cx": cx, "cy": cy, "std": stds, "cnt": counts, "W": W, "valid": valid}
    return metrics, grids


def analyze_intensity(tif_path):
    try:
        import rasterio
    except ImportError:
        return None, None
    with rasterio.open(tif_path) as src:
        a = src.read(1)
        res = src.res
    nod = 0  # uint8 mosaic: 0 = sin dato
    valid = a[a != nod].astype(float)
    if valid.size == 0:
        return None, None
    m = {
        "raster_w": int(a.shape[1]),
        "raster_h": int(a.shape[0]),
        "res_m": float(res[0]),
        "coverage_fraction": float((a != nod).mean()),
        "intensity_min": float(valid.min()),
        "intensity_p10": float(np.percentile(valid, 10)),
        "intensity_median": float(np.median(valid)),
        "intensity_mean": float(valid.mean()),
        "intensity_p90": float(np.percentile(valid, 90)),
        "intensity_max": float(valid.max()),
        "intensity_std": float(valid.std()),
        # Banding / rango dinámico: fracción casi-negra vs saturada. Un mosaico con
        # >~40% de píxeles casi-negros delata el banding por ángulo sin corregir.
        "frac_near_black_le2": float((valid <= 2).mean()),
        "frac_saturated_ge254": float((valid >= 254).mean()),
        "bimodality_dark_bright_ratio": float((valid <= 8).sum() / max((valid >= 64).sum(), 1)),
    }
    return m, valid


def make_plots(grids, intensity_vals, cloud_m, out_dir):
    # 1) Mapa de rugosidad (std Z por celda) — consistencia vertical
    cx, cy, std, cnt, W, valid = (grids["cx"], grids["cy"], grids["std"],
                                  grids["cnt"], grids["W"], grids["valid"])
    H = cy.max() + 1
    img = np.full((H, W), np.nan)
    img[cy[valid], cx[valid]] = std[valid]
    plt.figure(figsize=(7, 7))
    plt.imshow(np.flipud(img), cmap="inferno", vmin=0,
               vmax=np.nanpercentile(img, 95))
    plt.colorbar(label="std Z por celda (m)")
    plt.title("Consistencia vertical (rugosidad) — menor = mejor")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mb_zconsistency.png"), dpi=120)
    plt.close()

    # 2) Mapa de densidad (cobertura)
    img_c = np.full((H, W), np.nan)
    img_c[cy, cx] = cnt
    plt.figure(figsize=(7, 7))
    plt.imshow(np.flipud(img_c), cmap="viridis",
               vmax=np.nanpercentile(img_c, 98))
    plt.colorbar(label="puntos por celda")
    plt.title("Densidad de puntos / cobertura")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mb_coverage.png"), dpi=120)
    plt.close()

    # 3) Histograma de backscatter (banding)
    if intensity_vals is not None:
        plt.figure(figsize=(7, 4))
        plt.hist(intensity_vals, bins=64, color="steelblue")
        plt.axvline(np.median(intensity_vals), color="r", ls="--",
                    label=f"mediana={np.median(intensity_vals):.0f}")
        plt.xlabel("backscatter (0-255)")
        plt.ylabel("nº píxeles")
        plt.title("Distribución de backscatter (bimodal = banding por ángulo)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mb_intensity_hist.png"), dpi=120)
        plt.close()


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    out_dir = os.path.join(results_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)

    report = {"results_dir": results_dir}

    # Layout por tipo de producto (ver results/README.md).
    xyz = os.path.join(results_dir, "pointcloud", "mb_pointcloud.xyz")
    grids = None
    if os.path.isfile(xyz):
        print(f"[mb-metrics] analizando nube {xyz} ...")
        cloud_m, grids = analyze_cloud(xyz)
        report["cloud"] = cloud_m
    else:
        print(f"[mb-metrics] AVISO: no existe {xyz}")

    # analyze_intensity lee la banda 1 (valor de backscatter). El .tif lleva una
    # paleta viridis embebida, pero read(1) sigue devolviendo el valor, no el RGB.
    tif = os.path.join(results_dir, "tif", "mb_intensity.tif")
    intensity_vals = None
    if os.path.isfile(tif):
        print(f"[mb-metrics] analizando mosaico {tif} ...")
        int_m, intensity_vals = analyze_intensity(tif)
        report["intensity"] = int_m
    else:
        print(f"[mb-metrics] AVISO: no existe {tif}")

    mesh = os.path.join(results_dir, "mesh", "mb_mesh.ply")
    report["mesh_present"] = os.path.isfile(mesh)

    if grids is not None:
        make_plots(grids, intensity_vals, report.get("cloud"), out_dir)

    json_path = os.path.join(out_dir, "mb_projection_metrics.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[mb-metrics] escrito {json_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
