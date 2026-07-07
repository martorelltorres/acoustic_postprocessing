#!/usr/bin/env python3
"""
R0 → R3 ablation study for the SOTA baseline (see background/PROPUESTA_SLAM_SOTA.md).

Launches the `acoustic_pipeline.launch` pipeline several times, incrementally enabling
the improvements via the launch flags, and collects Roman's consistency error
(primary metric) plus auxiliary metrics into a comparison table.

Configurations (ablation steps):

  R0   baseline         : isotropic information, no backscatter in cov, neutral back-end,
                          WITH adjacent-overlap edges (R0.2 is already structural).
  R1   +pICP cov        : ANISOTROPIC edge information (registration covariance).
  R1R2 +robust back-end : adds skeptical line process (preference_loop_closure<1).
  FULL +backscatter     : adds backscatter to the uncertainty model (R3).

Each configuration runs on the SAME bag, and its `slam_metrics.json` is saved under
`results/ablation/<config>/` to avoid clobbering. At the end it prints the table and writes
`results/ablation/ablation_summary.csv`.

Usage:
    rosrun acoustic_postprocessing run_ablation.py            # all configs
    rosrun acoustic_postprocessing run_ablation.py --configs R0 FULL
    rosrun acoustic_postprocessing run_ablation.py --bag /path/to/bag.bag
    rosrun acoustic_postprocessing run_ablation.py --dry-run  # only show the commands

Overlap sweep (orthogonal to R0→R3): fixes a base config and varies patch_size/stride
to see how along-track overlap between patches affects registration. Writes its own
table and `results/ablation/overlap_sweep_summary.csv`:

    rosrun acoustic_postprocessing run_ablation.py --sweep-overlap --bag /path/to/bag.bag
    rosrun acoustic_postprocessing run_ablation.py --sweep-overlap --sweep-config R0
    rosrun acoustic_postprocessing run_ablation.py --sweep-overlap --patch-grid 100:10 80:20

Notes:
  - Requires an accessible `roscore` (roslaunch starts one if none exists).
  - The Open3D monitor is disabled (enable_monitor:=false) to run headless.
  - Does NOT re-run a config whose folder already has slam_metrics.json unless --force.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time


# Flags per configuration. Each dict holds overrides of launch args (passed as
# arg:=value to roslaunch). Anything unspecified stays at the launch default.
# Keep R0.2 (cross-track) ALWAYS on: it is structural, not part of the uncertainty
# ablation. To isolate R0.2 too, add a separate config.
CONFIGS = {
    # Baseline R0: everything "classic", without the uncertainty improvements.
    "R0": {
        "use_registration_covariance": "false",
        "use_intensity_in_covariance": "false",
        "preference_loop_closure": "1.0",   # neutral back-end
        "enable_cross_track_edges": "true",
    },
    # + R1: anisotropic edge information (registration covariance).
    "R1": {
        "use_registration_covariance": "true",
        "use_intensity_in_covariance": "false",
        "preference_loop_closure": "1.0",
        "enable_cross_track_edges": "true",
    },
    # + R2: robust back-end (skeptical line process over loops/cross-track).
    "R1R2": {
        "use_registration_covariance": "true",
        "use_intensity_in_covariance": "false",
        "preference_loop_closure": "0.6",
        "enable_cross_track_edges": "true",
    },
    # + R3: backscatter in the uncertainty model (full config).
    "FULL": {
        "use_registration_covariance": "true",
        "use_intensity_in_covariance": "true",
        "preference_loop_closure": "0.6",
        "enable_cross_track_edges": "true",
    },
}

# Presentation order (incremental).
CONFIG_ORDER = ["R0", "R1", "R1R2", "FULL"]


# Overlap-sweep grid (--sweep-overlap). Each pair is (patch_size, patch_stride).
# Along-track overlap between consecutive patches is 1 - stride/size:
#   (100, 10) → 90%   (large size, small stride: max overlap and edge count)
#   (100, 20) → 80%   (what the code comments and the length-ratio gate assume)
#   ( 80, 20) → 75%
#   ( 50, 20) → 60%   (current launch default)
# Order: most overlapped to least, to see the maximum-common-area regime first.
# More overlap helps the ICP latch on (more correspondences) but amplifies the
# "short step" bias (see gate MIN/MAX_SEQ_ICP_LENGTH_RATIO in multibeam_slam.py).
PATCH_GRID = [
    (100, 10),   # 90%
    (100, 20),   # 80%
    (80, 20),    # 75%
    (50, 20),    # 60%
]


def overlap_pct(size, stride):
    """Along-track overlap (%) between consecutive patches."""
    return 100.0 * (1.0 - stride / float(size))


def sweep_config_name(base_config, size, stride):
    """Folder name for a sweep point (unique per overlap and base config)."""
    return f"sweep_{base_config}_o{int(round(overlap_pct(size, stride)))}_s{size}x{stride}"


def find_package_dir():
    """Package directory (this script lives in scripts/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_roslaunch_cmd(config_name, overrides, bag, output_dir):
    cmd = [
        "roslaunch",
        "acoustic_postprocessing",
        "acoustic_pipeline.launch",
        "enable_monitor:=false",          # headless for the ablation
        f"output_dir:={output_dir}",
    ]
    if bag is not None:
        cmd.append(f"mb_bagfile:={bag}")
    for k, v in overrides.items():
        cmd.append(f"{k}:={v}")
    return cmd


def run_config(config_name, overrides, bag, ablation_root, force, dry_run, timeout):
    """Launches a configuration and returns the path to its slam_metrics.json (or None).

    The mb_slam node is NOT `required` in the launch and `main()` returns without spin(),
    so roslaunch does not exit on its own when the node finishes. To avoid waiting the
    timeout on each config, we launch roslaunch with Popen and watch for the appearance/
    update of slam_metrics.json (signal that the node finished its work); as soon as it is
    there, we stop roslaunch cleanly. No need to touch the user's launch.
    """

    out_dir = os.path.join(ablation_root, config_name)
    metrics_path = os.path.join(out_dir, "metrics", "slam_metrics.json")

    if os.path.exists(metrics_path) and not force:
        print(f"[{config_name}] metrics already exist → reused (use --force to redo)")
        return metrics_path

    # If redoing, delete the previous metrics to detect the new one unambiguously.
    if os.path.exists(metrics_path):
        try:
            os.remove(metrics_path)
        except OSError:
            pass

    os.makedirs(out_dir, exist_ok=True)

    cmd = build_roslaunch_cmd(config_name, overrides, bag, out_dir)
    print(f"\n[{config_name}] {' '.join(cmd)}")

    if dry_run:
        return None

    t0 = time.time()
    # start_new_session so we can kill the whole roslaunch process group.
    proc = subprocess.Popen(cmd, start_new_session=True)

    metrics_seen = False
    try:
        while True:
            # did roslaunch exit on its own? (e.g. launch error)
            if proc.poll() is not None:
                break
            # did slam_metrics.json appear? → the node finished its work.
            if os.path.exists(metrics_path):
                # Brief wait to ensure the file is complete (json.dump).
                time.sleep(3.0)
                metrics_seen = True
                break
            if time.time() - t0 > timeout:
                print(f"[{config_name}] TIMEOUT after {timeout}s — aborted")
                break
            time.sleep(2.0)
    finally:
        _terminate_process_group(proc)

    dt = time.time() - t0
    print(f"[{config_name}] finished in {dt:.0f}s "
          f"({'metrics OK' if metrics_seen else 'no metrics'})")

    if not os.path.exists(metrics_path):
        print(f"[{config_name}] ⚠ {metrics_path} was not generated")
        return None

    return metrics_path


def _terminate_process_group(proc):
    """Cleanly kills the roslaunch process group (SIGINT → SIGTERM)."""
    import signal
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        # SIGINT first (roslaunch handles it as Ctrl-C, clean node shutdown).
        os.killpg(pgid, signal.SIGINT)
        for _ in range(10):
            if proc.poll() is not None:
                return
            time.sleep(1.0)
        # If still alive, SIGTERM.
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(2.0)
        if proc.poll() is None:
            os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def extract_row(config_name, metrics_path):
    """Extracts the key metrics from a config's slam_metrics.json."""

    row = {"config": config_name}

    if metrics_path is None or not os.path.exists(metrics_path):
        row["status"] = "missing"
        return row

    try:
        with open(metrics_path) as f:
            m = json.load(f)
    except Exception as exc:
        row["status"] = f"parse_error: {exc}"
        return row

    row["status"] = "ok"

    cons = m.get("consistency", {})
    raw = cons.get("raw_navigation", {})
    slam = cons.get("slam_optimized", {})
    row["cons_raw_m"] = raw.get("mean_std_z")
    row["cons_slam_m"] = slam.get("mean_std_z")
    row["cons_improv_pct"] = cons.get("improvement_pct")
    row["cons_cells"] = slam.get("n_cells_valid")
    row["cons_rms_slam_m"] = slam.get("rms_std_z")

    summ = m.get("summary", {})
    row["seq_accept_ratio"] = summ.get("seq_acceptance_ratio")
    row["accepted_loops"] = summ.get("accepted_loops")
    row["mean_corr_xy_m"] = summ.get("mean_slam_correction_xy_m")

    xt = m.get("cross_track", [])
    row["xtrack_total"] = len(xt)
    row["xtrack_accepted"] = sum(1 for e in xt if e.get("accepted"))

    rb = m.get("robust_backend", {})
    row["uncertain_edges"] = rb.get("uncertain_edges")
    row["edges_deactivated"] = rb.get("deactivated_by_line_process")

    return row


def fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def print_table(rows):
    cols = [
        ("config", "config", 6),
        ("cons_raw_m", "raw", 7),
        ("cons_slam_m", "slam", 7),
        ("cons_improv_pct", "improv%", 8),
        ("cons_cells", "cells", 6),
        ("seq_accept_ratio", "seq_acc", 8),
        ("xtrack_accepted", "xtrack", 7),
        ("edges_deactivated", "deact", 6),
        ("mean_corr_xy_m", "corrXY", 7),
    ]

    header = "  ".join(f"{label:>{w}}" for _, label, w in cols)
    print("\n" + "=" * len(header))
    print("ABLATION — Roman's consistency error (mean_std_z, lower = better)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['config']:>6}  [{r.get('status')}]")
            continue
        line = "  ".join(
            f"{fmt(r.get(key)):>{w}}" for key, _, w in cols
        )
        print(line)
    print("=" * len(header))
    print("raw/slam = consistency error before/after (m); improv% = reduction;")
    print("cells = valid overlap cells; seq_acc = sequential acceptance ratio;")
    print("xtrack = accepted adjacent-overlap edges; deact = edges turned off")
    print("by the robust back-end; corrXY = mean trajectory correction (m).")


def print_sweep_table(rows, base_config):
    """Overlap-sweep table: one row per (patch_size, stride)."""
    cols = [
        ("overlap_pct", "overlap%", 9),
        ("patch_size", "size", 5),
        ("patch_stride", "stride", 7),
        ("seq_accept_ratio", "seq_acc", 8),
        ("cons_slam_m", "slam", 7),
        ("cons_improv_pct", "improv%", 8),
        ("xtrack_accepted", "xtrack", 7),
        ("mean_corr_xy_m", "corrXY", 7),
    ]
    header = "  ".join(f"{label:>{w}}" for _, label, w in cols)
    print("\n" + "=" * len(header))
    print(f"OVERLAP SWEEP — base config '{base_config}' "
          f"(higher seq_acc = more edges; higher improv% = better map)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("status") != "ok":
            print(f"{fmt(r.get('overlap_pct'),0):>9}  [{r.get('status')}]")
            continue
        line = "  ".join(f"{fmt(r.get(key)):>{w}}" for key, _, w in cols)
        print(line)
    print("=" * len(header))
    print("overlap% = 1 - stride/size; seq_acc = accepted sequential edge ratio;")
    print("slam = consistency error after SLAM (m); improv% = reduction vs raw nav;")
    print("xtrack = accepted adjacent-overlap edges; corrXY = mean correction (m).")


def run_overlap_sweep(base_config, grid, bag, ablation_root, force, dry_run, timeout):
    """Sweeps the (patch_size, stride) grid over a single config and returns the rows."""
    base_overrides = CONFIGS[base_config]
    rows = []
    for size, stride in grid:
        name = sweep_config_name(base_config, size, stride)
        overrides = dict(base_overrides)
        overrides["patch_size"] = str(size)
        overrides["patch_stride"] = str(stride)
        mp = run_config(name, overrides, bag, ablation_root, force, dry_run, timeout)
        if dry_run:
            continue
        row = extract_row(name, mp)
        # Record the sweep parameters for the table/CSV.
        row["patch_size"] = size
        row["patch_stride"] = stride
        row["overlap_pct"] = overlap_pct(size, stride)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="R0→R3 ablation of the bathymetric SLAM")
    parser.add_argument("--configs", nargs="+", default=CONFIG_ORDER,
                        help="subset of configs to run")
    parser.add_argument("--bag", default=None,
                        help="path to the bag (otherwise uses the launch default)")
    parser.add_argument("--force", action="store_true",
                        help="re-run even if slam_metrics.json already exists")
    parser.add_argument("--dry-run", action="store_true",
                        help="only show the roslaunch commands")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="per-config timeout in seconds (default 3600)")
    parser.add_argument("--sweep-overlap", action="store_true",
                        help="instead of the R0→R3 ablation, sweep patch_size/stride "
                             "(overlap) over ONE base config (--sweep-config)")
    parser.add_argument("--sweep-config", default="FULL",
                        help="base config for the overlap sweep (default FULL)")
    parser.add_argument("--patch-grid", nargs="+", default=None,
                        metavar="SIZE:STRIDE",
                        help="sweep points as size:stride "
                             "(e.g. 100:10 100:20 80:20); otherwise uses PATCH_GRID")
    args = parser.parse_args()

    pkg = find_package_dir()
    ablation_root = os.path.join(pkg, "results", "ablation")
    os.makedirs(ablation_root, exist_ok=True)

    # ---- Overlap sweep mode -----------------------------------------------------
    if args.sweep_overlap:
        if args.sweep_config not in CONFIGS:
            print(f"--sweep-config '{args.sweep_config}' unknown "
                  f"(options: {list(CONFIGS)})")
            return 1
        if args.patch_grid:
            try:
                grid = [tuple(int(x) for x in p.split(":")) for p in args.patch_grid]
            except ValueError:
                print("⚠ malformed --patch-grid; use size:stride (e.g. 100:10)")
                return 1
        else:
            grid = PATCH_GRID

        print(f"Overlap sweep over '{args.sweep_config}'")
        print(f"Grid (size, stride): {grid}")
        print(f"Results in: {ablation_root}")
        if args.bag:
            print(f"Bag: {args.bag}")

        rows = run_overlap_sweep(
            args.sweep_config, grid, args.bag, ablation_root,
            args.force, args.dry_run, args.timeout,
        )
        if args.dry_run:
            print("\n(dry-run: nothing was executed)")
            return 0

        print_sweep_table(rows, args.sweep_config)

        csv_path = os.path.join(ablation_root, "overlap_sweep_summary.csv")
        if rows:
            keys = sorted({k for r in rows for k in r.keys()})
            keys = ["config"] + [k for k in keys if k != "config"]
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            print(f"\nCSV written → {csv_path}")
        return 0
    # -----------------------------------------------------------------------------

    configs = [c for c in CONFIG_ORDER if c in args.configs]
    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        print(f"⚠ unknown configs ignored: {unknown}")
    if not configs:
        print("No valid configs to run.")
        return 1

    print(f"Ablation: {configs}")
    print(f"Results in: {ablation_root}")
    if args.bag:
        print(f"Bag: {args.bag}")

    rows = []
    for name in configs:
        mp = run_config(
            name, CONFIGS[name], args.bag, ablation_root,
            args.force, args.dry_run, args.timeout,
        )
        if not args.dry_run:
            rows.append(extract_row(name, mp))

    if args.dry_run:
        print("\n(dry-run: nothing was executed)")
        return 0

    print_table(rows)

    # Summary CSV.
    csv_path = os.path.join(ablation_root, "ablation_summary.csv")
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        # 'config' first.
        keys = ["config"] + [k for k in keys if k != "config"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nCSV written → {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
