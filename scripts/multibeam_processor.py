#!/usr/bin/env python3
"""
Multibeam point cloud -> georeferenced cloud + Poisson surface mesh.
Shares geometry with multibeam_intensity.py (same axis flip, sensor TF and
MB->SSS lever-arm) so all products live in the same UTM frame.

Author: Antoni Martorell (SRV, UIB)
"""

import rospy
import rosbag
import numpy as np
import ros_numpy
import open3d as o3d
import os
import sys
import cv2
import rasterio
from rasterio.transform import from_origin
import tf.transformations as tr

from pyproj import Transformer
from std_msgs.msg import Bool
import time

# Shared helpers live in scripts/common.py, imported with a flat name. Under catkin the
# script runs through a devel/lib wrapper whose sys.path does NOT include this directory
# (the wrapper does point __file__ at this source file), so add it explicitly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import get_static_transforms, get_nav_origin, read_nav, nav_interpolators

# Geographic -> UTM (zone 31N)
CRS_WGS84 = "EPSG:4326"
CRS_UTM   = "EPSG:32631"
ll_to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)


def _cell_median(pts, cell):
    """Per-cell median Z and MAD, broadcast back to each point."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    xi = np.floor((x - x.min()) / cell).astype(np.int64)
    yi = np.floor((y - y.min()) / cell).astype(np.int64)
    W = xi.max() + 1
    key = yi * W + xi

    order = np.argsort(key, kind="stable")
    ks = key[order]
    zs = z[order]
    uniq, starts, counts = np.unique(ks, return_index=True, return_counts=True)

    med = np.empty(len(uniq))
    mad = np.empty(len(uniq))
    for i, (s, c) in enumerate(zip(starts, counts)):
        seg = zs[s:s + c]
        m = np.median(seg)
        med[i] = m
        mad[i] = np.median(np.abs(seg - m)) * 1.4826 if c > 2 else 0.0

    cell_of = np.searchsorted(uniq, key)
    return med[cell_of], mad[cell_of]


def surface_relative_filter(pts, cell=0.5, n_mad=3.0, floor=0.3, iters=2):
    """
    Outlier filter RELATIVE TO THE LOCAL SURFACE (not k-NN).

    Statistical SOR looks at the distance to the k neighbours, so it does NOT catch
    the vertical spikes of the multibeam: stray beams with a wrong range (multipath,
    fish, noise) and grazing-beam noise, which project points metres above/below the
    seafloor and show up as spurious "mountains" in CloudCompare.

    Points are gridded in XY at `cell` m and dropped when their Z departs more than
    n_mad * MAD(cell) from the cell MEDIAN (with a `floor` minimum). It is ADAPTIVE:
    on rough bottom the MAD is high and real relief survives. It ITERATES because a
    cluster of spikes biases the median/MAD of its own cell on the first pass.
    n_mad <= 0 disables it.
    """
    if n_mad <= 0 or len(pts) < 10:
        return pts

    for _ in range(max(1, iters)):
        med, mad = _cell_median(pts, cell)
        thr = np.maximum(n_mad * mad, floor)
        keep = np.abs(pts[:, 2] - med) <= thr
        if keep.all():
            break
        pts = pts[keep]
    return pts


def main():

    rospy.init_node('multibeam_processor')

    rospy.loginfo("===== MULTIBEAM PROCESSOR STARTED =====")

    bag_file   = rospy.get_param('~bag_file')
    scan_topic = rospy.get_param('~scan_topic')
    nav_topic  = rospy.get_param('~nav_topic')
    output_dir = rospy.get_param('~output_dir')

    voxel_size = rospy.get_param('~voxel_size', 0.05)
    sor_k      = rospy.get_param('~sor_k', 50)
    sor_std    = rospy.get_param('~sor_std', 1.0)
    # 55.0 to match the launch's <arg>: running the script standalone used to apply a
    # different cutoff (60) than the pipeline.
    angle_cutoff_deg = rospy.get_param('~angle_cutoff', 55.0)
    poisson_depth = int(rospy.get_param('~poisson_depth', 10))
    mosaic_res    = float(rospy.get_param('~dem_res', 0.10))   # DEM/TIF cell size
    # Surface-relative outlier filter (removes spurious vertical spikes).
    surf_filter_cell  = float(rospy.get_param('~surf_filter_cell', 0.5))
    surf_filter_nmad  = float(rospy.get_param('~surf_filter_nmad', 3.0))
    surf_filter_floor = float(rospy.get_param('~surf_filter_floor', 0.3))
    surf_filter_iters = int(rospy.get_param('~surf_filter_iters', 2))
    # ATTITUDE gating: drops the whole ping while the vehicle is turning. Must match
    # multibeam_intensity.py (same launch args) or the intensity and bathymetric clouds
    # would stop sharing points. <=0 disables.
    max_roll_deg       = float(rospy.get_param('~max_roll_deg', 5.0))
    max_yaw_rate_dps   = float(rospy.get_param('~max_yaw_rate_deg_s', 8.0))
    # float() is required: roslaunch delivers "NaN" as a STRING (its 'auto' conversion
    # only tries float when the value contains a '.').
    roll_bias_deg      = float(rospy.get_param('~roll_bias_deg', float('nan')))
    angle_cutoff_frame = str(rospy.get_param('~angle_cutoff_frame', 'world')).lower()

    if angle_cutoff_frame not in ('sensor', 'world'):
        rospy.logwarn(f"angle_cutoff_frame='{angle_cutoff_frame}' invalid; using 'world'.")
        angle_cutoff_frame = 'world'

    # results/ layout: one folder per product. See results/README.md.
    tif_dir    = os.path.join(output_dir, "tif")
    images_dir = os.path.join(output_dir, "images")
    cloud_dir  = os.path.join(output_dir, "pointcloud")
    mesh_dir   = os.path.join(output_dir, "mesh")

    for d in (output_dir, tif_dir, images_dir, cloud_dir, mesh_dir):
        os.makedirs(d, exist_ok=True)

    bag = rosbag.Bag(bag_file)

    # UTM origin
    lat0, lon0 = get_nav_origin(bag, nav_topic)
    X0_UTM, Y0_UTM = ll_to_utm.transform(lon0, lat0)

    rospy.loginfo(f"UTM origin: {X0_UTM:.3f}, {Y0_UTM:.3f}")

    # Sensor TFs (multibeam + sidescan, for the lever-arm), all three in ONE pass over
    # the bag: resolving them one by one re-scanned /tf_static + /tf from the start each
    # time, so the bag was opened four times over.
    TFS = get_static_transforms(bag_file, [
        ('sparus2/base_link', 'sparus2/multibeam'),
        ('sparus2/base_link', 'sparus2/sidescan_port'),
        ('sparus2/base_link', 'sparus2/sidescan_starboard'),
    ])

    T_MB   = TFS[('sparus2/base_link', 'sparus2/multibeam')]
    T_PORT = TFS[('sparus2/base_link', 'sparus2/sidescan_port')]
    T_STBD = TFS[('sparus2/base_link', 'sparus2/sidescan_starboard')]

    R_sensor = T_MB[:3, :3]
    sensor_offset = T_MB[:3, 3]

    # MB->SSS lever-arm: aligns the cloud onto the sidescan mosaic frame. The extra
    # `mb_sss_extra_offset` term (empirical, -2 m in Y) is a parameter rather than a
    # constant: in a georeferenced product a fixed 2 m shifts EVERYTHING, so it has to
    # be explicit. Set it to 0 for no shift towards the SSS frame.
    sss_center = 0.5 * (T_PORT[:3, 3] + T_STBD[:3, 3])

    mb_sss_extra_offset = float(rospy.get_param('~mb_sss_extra_offset_y', -2.0))
    delta_sensor = (sss_center - sensor_offset) + np.array([0.0, mb_sss_extra_offset, 0.0])

    rospy.loginfo(f"MB offset      : {sensor_offset}")
    rospy.loginfo(f"SSS center     : {sss_center}")
    rospy.loginfo(f"Lever-arm delta: {delta_sensor}")

    # Navigation: read (sorted + deduplicated by timestamp) and build the interpolators.
    # No yaw smoothing here: that is a sidescan-only treatment (see common.py).
    nav = read_nav(bag, nav_topic)
    ts_nav = nav['ts']

    f = nav_interpolators(nav, with_yaw_rate=True)
    f_n, f_e, f_d = f['n'], f['e'], f['d']
    f_y, f_p, f_r = f['y'], f['p'], f['r']
    f_yr = f['yr']

    # Roll bias (mounting trim): median roll is ~+2° in these bags, not 0. The gate
    # measures the EXCURSION about that bias, not absolute roll — a constant 2° roll is
    # modelled fine by the rotation matrix, while thresholding raw |roll| cuts
    # asymmetrically (r=+0.46 with raw |roll| vs r=+0.54 with |roll - bias|).
    # Computed on RAW roll (not unwrapped), as the gate compares against raw roll too.
    if not np.isfinite(roll_bias_deg):
        roll_bias_deg = float(np.degrees(np.median(nav['roll'])))

    rospy.loginfo(
        f"Attitude: roll bias {roll_bias_deg:+.2f}° | gate |roll-bias|<={max_roll_deg}° "
        f"and |yaw_rate|<={max_yaw_rate_dps}°/s | cutoff {angle_cutoff_deg}° (frame {angle_cutoff_frame})"
    )

    # Per-ping processing: sensor frame -> vehicle -> local -> UTM
    rospy.loginfo("Processing multibeam pings...")

    buffer_points = []
    count = 0
    n_skip_roll = 0
    n_skip_yaw = 0

    for _, scan, _ in bag.read_messages(topics=[scan_topic]):

        if not hasattr(scan, 'header'):
            continue

        count += 1

        if count % 100 == 0:
            rospy.loginfo(f"Pings processed: {count}")

        ts = scan.header.stamp.to_sec()

        if ts < ts_nav[0] or ts > ts_nav[-1]:
            continue

        n = float(f_n(ts))
        e = float(f_e(ts))
        d = float(f_d(ts))

        yaw_t = float(f_y(ts))
        pitch_t = float(f_p(ts))
        roll_t = float(f_r(ts))

        # A gap in the INS inside the nav time span interpolates to NaN, which would
        # otherwise propagate silently into the cloud (the time-range check above only
        # catches pings OUTSIDE the span). Same guard as multibeam_intensity.py.
        if np.isnan(n) or np.isnan(e) or np.isnan(yaw_t):
            continue

        # Attitude gate: in a turn the swath projects as a tilted fan that does not match
        # the neighbouring passes, and there is no post-hoc fix without registering swaths
        # (that is the SLAM), so the WHOLE ping is dropped. The fraction of points falling
        # off the surface grows from 2.1% (|roll-bias|<1°) to 33.7% (>12°).
        if max_roll_deg > 0 and abs(np.degrees(roll_t) - roll_bias_deg) > max_roll_deg:
            n_skip_roll += 1
            continue

        yaw_rate_t = float(f_yr(ts))

        if max_yaw_rate_dps > 0 and abs(yaw_rate_t) > max_yaw_rate_dps:
            n_skip_yaw += 1
            continue

        pc = ros_numpy.point_cloud2.pointcloud2_to_array(scan)
        pc = pc[np.isfinite(pc['x'])]

        if len(pc) < 10:
            continue

        # Flip to match the sensor TF convention (seafloor stays below)
        xyz = np.column_stack((pc['x'], -pc['y'], -pc['z']))

        # Beam angle from the SENSOR vertical (what the cutoff used to measure).
        ang_sensor = np.degrees(np.arctan2(
            np.sqrt(xyz[:,0]**2 + xyz[:,1]**2),
            np.abs(xyz[:,2])
        ))

        # Sensor rotation
        xyz = xyz @ R_sensor.T

        # Vehicle rotation
        R_veh = tr.euler_matrix(
            roll_t,
            pitch_t,
            yaw_t,
            axes='sxyz'
        )[:3, :3]

        xyz = xyz @ R_veh.T

        # Drop grazing outer beams. Mind the frame: cutting in the SENSOR frame lets
        # through beams that, with the vehicle rolled, point well past the real cutoff
        # (cutoff 55° + roll 10° = 65° from TRUE vertical). In world frame the cutoff
        # means what it says.
        ang_world = np.degrees(np.arctan2(
            np.sqrt(xyz[:,0]**2 + xyz[:,1]**2),
            np.abs(xyz[:,2])
        ))

        angles = ang_world if angle_cutoff_frame == 'world' else ang_sensor
        xyz = xyz[angles < angle_cutoff_deg]

        if len(xyz) < 10:
            continue

        # Lever-arm offsets (same as intensity pipeline)
        offset_world = R_veh @ sensor_offset

        delta_world = R_veh @ delta_sensor

        offset_world += delta_world

        xyz[:,0] += offset_world[0]
        xyz[:,1] += offset_world[1]
        xyz[:,2] += offset_world[2]

        # Local world (north, east, -depth)
        xyz[:,0] += n
        xyz[:,1] += e
        xyz[:,2] += -d

        # To UTM (X=easting, Y=northing)
        pts_world = np.zeros_like(xyz)

        pts_world[:,0] = X0_UTM + xyz[:,1]
        pts_world[:,1] = Y0_UTM + xyz[:,0]
        pts_world[:,2] = xyz[:,2]

        buffer_points.append(pts_world)

    bag.close()

    rospy.loginfo(
        f"Pings: {count} read, {n_skip_roll} dropped by roll, "
        f"{n_skip_yaw} by yaw-rate, {len(buffer_points)} valid "
        f"({(n_skip_roll + n_skip_yaw) / max(count, 1) * 100:.2f}% dropped by attitude)"
    )

    if not buffer_points:
        rospy.logerr("No valid points")
        return

    # Point cloud: voxel downsample -> outlier removal. THE ORDER MATTERS: downsample
    # first, denoise after. The other way round, SOR builds its KD-tree over the ~31 M
    # raw points and is both very slow and a RAM spike. Nothing is lost — 1 cm is far
    # below the useful MBES resolution at these flight altitudes.
    pts_all = np.vstack(buffer_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all)

    n_raw = len(pcd.points)
    pcd = pcd.voxel_down_sample(voxel_size)
    n_voxel = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(sor_k, sor_std)
    n_sor = len(pcd.points)

    # Surface-relative filter: removes the vertical spikes (bad-range beams, edge curl)
    # that the k-NN SOR misses and that produce the spurious "mountains".
    filtered = surface_relative_filter(
        np.asarray(pcd.points),
        cell=surf_filter_cell,
        n_mad=surf_filter_nmad,
        floor=surf_filter_floor,
        iters=surf_filter_iters
    )
    pcd.points = o3d.utility.Vector3dVector(filtered)

    rospy.loginfo(
        f"Cloud: {n_raw} raw -> {n_voxel} voxel({voxel_size} m) "
        f"-> {n_sor} SOR -> {len(pcd.points)} after surface filter"
    )

    xyz_file = os.path.join(cloud_dir, "mb_pointcloud.xyz")
    o3d.io.write_point_cloud(xyz_file, pcd, write_ascii=True)

    rospy.loginfo(f"XYZ saved: {xyz_file}")

    # =====================================================================
    # Rasterized bathymetric DEM: mb_pointcloud.tif + JPG in images/
    # =====================================================================
    # The 3D cloud is not a raster: for the .tif it is projected onto an XY grid holding
    # the MEDIAN Z per cell, i.e. a georeferenced digital terrain model.
    dem_pts = np.asarray(pcd.points)
    dxmin = dem_pts[:, 0].min(); dxmax = dem_pts[:, 0].max()
    dymin = dem_pts[:, 1].min(); dymax = dem_pts[:, 1].max()
    dW = int(np.ceil((dxmax - dxmin) / mosaic_res))
    dH = int(np.ceil((dymax - dymin) / mosaic_res))

    cc = ((dem_pts[:, 0] - dxmin) / mosaic_res).astype(np.int64)
    rr = ((dymax - dem_pts[:, 1]) / mosaic_res).astype(np.int64)
    m = (cc >= 0) & (cc < dW) & (rr >= 0) & (rr < dH)
    cell = rr[m] * dW + cc[m]
    zval = dem_pts[m, 2]

    order = np.argsort(cell, kind="stable")
    cell_s = cell[order]; z_s = zval[order]
    uniq, starts, counts = np.unique(cell_s, return_index=True, return_counts=True)
    dem = np.full(dW * dH, np.nan, dtype=np.float32)
    for u, s, cnt_u in zip(uniq, starts, counts):
        dem[u] = np.median(z_s[s:s + cnt_u])
    dem = dem.reshape((dH, dW))

    dem_transform = from_origin(dxmin, dymax, mosaic_res, mosaic_res)
    dem_tif = os.path.join(tif_dir, "mb_pointcloud.tif")
    with rasterio.open(
        dem_tif, 'w', driver='GTiff',
        height=dH, width=dW, count=1, dtype=np.float32,
        crs="EPSG:32631", transform=dem_transform,
        nodata=np.nan, compress='deflate'
    ) as dst:
        dst.write(dem, 1)
    rospy.loginfo(f"DEM TIF saved: {dem_tif}")

    # DEM JPG (depth -> colormap).
    valid = np.isfinite(dem)
    if valid.any():
        vmin, vmax = np.percentile(dem[valid], (2, 98))
        norm = np.zeros_like(dem, dtype=np.float32)
        norm[valid] = np.clip(
            (dem[valid] - vmin) / max(vmax - vmin, 1e-6), 0, 1
        )
        gray = (norm * 255).astype(np.uint8)
        gray[~valid] = 0
        color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        color[~valid] = 0
        dem_jpg = os.path.join(images_dir, "mb_pointcloud.jpg")
        cv2.imwrite(dem_jpg, color, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        rospy.loginfo(f"DEM JPG saved: {dem_jpg}")

    # Mesh: center for numerical stability, then Poisson reconstruction
    pts = np.asarray(pcd.points)

    centroid = pts.mean(axis=0)

    pts_centered = pts - centroid
    pcd.points = o3d.utility.Vector3dVector(pts_centered)

    normal_radius = voxel_size * 3.0

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius,
            max_nn=80
        )
    )

    pcd.orient_normals_consistent_tangent_plane(50)
    pcd.orient_normals_to_align_with_direction([0,0,1])

    # Poisson depth is a parameter: depth=11 on dense clouds blows up RAM (55 GB with
    # 31 M points, which is what caused the OOM-kill). At voxel 0.05-0.10 m, depth 9-10
    # is plenty.
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=poisson_depth
    )

    # Trim low-density (extrapolated) vertices
    densities = np.asarray(densities)

    threshold = np.percentile(densities, 5)

    mesh.remove_vertices_by_mask(densities < threshold)

    # Back to absolute UTM coordinates
    vertices = np.asarray(mesh.vertices) + centroid
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    mesh.compute_vertex_normals()

    # Poisson returns an ALL-ZERO per-vertex color array when the input cloud has no
    # color, and write_triangle_mesh writes it into the PLY: CloudCompare honours vertex
    # color and painted the whole mesh black. Dropping the array lets it shade by normals
    # instead. Color belongs in mb_textured_sss.ply (sss_mb_fusion.py).
    if mesh.has_vertex_colors() and not np.asarray(mesh.vertex_colors).any():
        mesh.vertex_colors = o3d.utility.Vector3dVector()
        rospy.loginfo("Uncolored mesh: dropped Poisson's all-black vertex color array.")

    mesh_file = os.path.join(mesh_dir, "mb_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_file, mesh)

    rospy.loginfo(f"Mesh saved: {mesh_file}")

    pub_mb_done = rospy.Publisher('/pipeline/mb_done', Bool, queue_size=1, latch=True)

    time.sleep(0.5)
    pub_mb_done.publish(True)

    rospy.loginfo("===== MULTIBEAM FINISHED =====")

if __name__ == '__main__':
    main()