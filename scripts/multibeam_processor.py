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
import cv2
import rasterio
from rasterio.transform import from_origin
import tf.transformations as tr

from scipy.interpolate import interp1d
from pyproj import Transformer
from std_msgs.msg import Bool
import time

# Geographic -> UTM (zone 31N)
CRS_WGS84 = "EPSG:4326"
CRS_UTM   = "EPSG:32631"
ll_to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)


def get_static_transform_from_tf(bag_file, parent_frame, child_frame):
    # First parent->child transform found in /tf_static or /tf (4x4).

    bag = rosbag.Bag(bag_file)

    for _, msg, _ in bag.read_messages(topics=['/tf_static', '/tf']):
        for transform in msg.transforms:

            if transform.header.frame_id == parent_frame and transform.child_frame_id == child_frame:

                q = transform.transform.rotation
                t = transform.transform.translation

                T = tr.quaternion_matrix([q.x, q.y, q.z, q.w])
                T[:3, 3] = [t.x, t.y, t.z]

                bag.close()
                return T

    bag.close()
    return np.identity(4)


def get_nav_origin(bag, nav_topic):
    # Geographic origin (lat, lon) of the local navigation frame.
    for _, msg, _ in bag.read_messages(topics=[nav_topic]):
        if hasattr(msg, 'origin'):
            return msg.origin.latitude, msg.origin.longitude

    raise RuntimeError("Navigation origin not found")


def _cell_median(pts, cell):
    """(median_z_por_celda, indice_de_celda_de_cada_punto, MAD_por_celda)."""
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
    Filtro de outliers RELATIVO A LA SUPERFICIE local (no k-NN).

    El SOR estadístico (remove_statistical_outlier) mira la distancia a los k
    vecinos, así que NO caza los picos verticales del multihaz: haces sueltos con
    rango erróneo (multipath, peces, ruido) y el ruido de los haces rasantes que,
    a 3 m de altitud y con el vehículo cabeceando (pitch mediana -6°, hasta -17°),
    proyectan puntos varios metros por encima/debajo del fondo. Eso es lo que
    produce las "montañas" y los radios espurios que se ven en CloudCompare.

    Se rejilla en XY a `cell` m y se descarta cada punto cuya Z se aparte más de
    n_mad * MAD(celda) de la MEDIANA de su celda (suelo mínimo `floor` m). Es
    ADAPTATIVO: en fondo rugoso real el MAD es alto y respeta el relieve; en fondo
    liso caza los picos. Se ITERA (`iters`) porque un cúmulo de picos sesga la
    mediana/MAD de su propia celda en la primera pasada; al quitar los peores, la
    segunda pasada afina sobre una superficie ya más limpia. n_mad<=0 lo desactiva.
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
    angle_cutoff_deg = rospy.get_param('~angle_cutoff', 60.0)
    poisson_depth = int(rospy.get_param('~poisson_depth', 10))
    mosaic_res    = float(rospy.get_param('~dem_res', 0.10))   # celda del DEM/TIF
    # Filtro de outliers relativo a la superficie (quita picos verticales espurios).
    surf_filter_cell  = float(rospy.get_param('~surf_filter_cell', 0.5))
    surf_filter_nmad  = float(rospy.get_param('~surf_filter_nmad', 3.0))
    surf_filter_floor = float(rospy.get_param('~surf_filter_floor', 0.3))
    surf_filter_iters = int(rospy.get_param('~surf_filter_iters', 2))
    # Gating por ACTITUD: descarta el ping entero cuando el vehículo está virando.
    # Idénticos en multibeam_intensity.py (mismos args del launch), o la nube de
    # intensidad y la batimétrica dejarían de compartir puntos. <=0 desactiva.
    max_roll_deg       = float(rospy.get_param('~max_roll_deg', 5.0))
    max_yaw_rate_dps   = float(rospy.get_param('~max_yaw_rate_deg_s', 8.0))
    # OJO al float(): roslaunch entrega "NaN" como STRING (su conversión 'auto' solo
    # intenta float si el valor lleva un '.'), así que hay que forzarlo aquí.
    roll_bias_deg      = float(rospy.get_param('~roll_bias_deg', float('nan')))
    angle_cutoff_frame = str(rospy.get_param('~angle_cutoff_frame', 'world')).lower()

    if angle_cutoff_frame not in ('sensor', 'world'):
        rospy.logwarn(f"angle_cutoff_frame='{angle_cutoff_frame}' no válido; uso 'world'.")
        angle_cutoff_frame = 'world'

    # Layout de results/: cada producto en su carpeta. Ver results/README.md.
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

    # Sensor TFs (multibeam + sidescan, for the lever-arm)
    T_MB = get_static_transform_from_tf(
        bag_file,
        'sparus2/base_link',
        'sparus2/multibeam'
    )

    T_PORT = get_static_transform_from_tf(
        bag_file,
        'sparus2/base_link',
        'sparus2/sidescan_port'
    )

    T_STBD = get_static_transform_from_tf(
        bag_file,
        'sparus2/base_link',
        'sparus2/sidescan_starboard'
    )

    R_sensor = T_MB[:3, :3]
    sensor_offset = T_MB[:3, 3]

    # MB->SSS lever-arm: align cloud onto the sidescan mosaic frame.
    # El término extra `mb_sss_extra_offset` (por defecto -2 m en Y, el ajuste
    # empírico previo para cuadrar con el mosaico SSS) se expone como parámetro en
    # vez de estar hardcodeado: en una proyección georreferenciada 2 m fijos
    # desplazan TODO el producto, así que debe ser explícito y ajustable. Ponlo a
    # 0 si no quieres el corrimiento hacia el marco del SSS.
    sss_center = 0.5 * (T_PORT[:3, 3] + T_STBD[:3, 3])

    mb_sss_extra_offset = float(rospy.get_param('~mb_sss_extra_offset_y', -2.0))
    delta_sensor = (sss_center - sensor_offset) + np.array([0.0, mb_sss_extra_offset, 0.0])

    rospy.loginfo(f"MB offset      : {sensor_offset}")
    rospy.loginfo(f"SSS center     : {sss_center}")
    rospy.loginfo(f"Lever-arm delta: {delta_sensor}")

    # Navigation: build time interpolators for pose
    ts_nav = []
    north = []
    east = []
    depth = []
    yaw = []
    pitch = []
    roll = []

    for _, msg, _ in bag.read_messages(topics=[nav_topic]):

        ts_nav.append(msg.header.stamp.to_sec())

        north.append(msg.position.north)
        east.append(msg.position.east)
        depth.append(msg.position.depth)

        yaw.append(msg.orientation.yaw)
        pitch.append(msg.orientation.pitch)
        roll.append(msg.orientation.roll)

    ts_nav = np.array(ts_nav)
    yaw_unwrapped = np.unwrap(np.array(yaw))

    f_n = interp1d(ts_nav, np.array(north), bounds_error=False, fill_value=np.nan)
    f_e = interp1d(ts_nav, np.array(east), bounds_error=False, fill_value=np.nan)
    f_d = interp1d(ts_nav, np.array(depth), bounds_error=False, fill_value=np.nan)

    f_y = interp1d(ts_nav, yaw_unwrapped, bounds_error=False, fill_value=np.nan)
    f_p = interp1d(ts_nav, np.unwrap(np.array(pitch)), bounds_error=False, fill_value=np.nan)
    f_r = interp1d(ts_nav, np.unwrap(np.array(roll)), bounds_error=False, fill_value=np.nan)

    # Velocidad de guiñada: es la firma directa del viraje. El roll alto llega con
    # el alabeo de entrada/salida del giro, pero el barrido también se emborrona
    # cuando el vehículo rota rápido en rumbo, aunque vaya plano.
    yaw_rate_dps = np.degrees(
        np.gradient(yaw_unwrapped) / np.maximum(np.gradient(ts_nav), 1e-3)
    )
    f_yr = interp1d(ts_nav, yaw_rate_dps, bounds_error=False, fill_value=np.nan)

    # Sesgo de roll (trim de montaje): la mediana del roll es ~+2° en estos bags,
    # no 0. El gate mide la EXCURSIÓN respecto a ese sesgo, no el roll absoluto:
    # un roll constante de 2° lo modela bien la matriz de rotación y no estropea
    # nada, mientras que umbralar |roll| crudo corta asimétricamente (medido sobre
    # el bag 13_52_22: correlación con el error de proyección r=+0.46 con |roll|
    # crudo frente a r=+0.54 con |roll - sesgo|).
    if not np.isfinite(roll_bias_deg):
        roll_bias_deg = float(np.degrees(np.median(np.array(roll))))

    rospy.loginfo(
        f"Actitud: sesgo roll {roll_bias_deg:+.2f}° | gate |roll-sesgo|<={max_roll_deg}° "
        f"y |yaw_rate|<={max_yaw_rate_dps}°/s | cutoff {angle_cutoff_deg}° (frame {angle_cutoff_frame})"
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

        # Gate por actitud: en viraje la franja se proyecta como un abanico inclinado
        # que no casa con la de las pasadas vecinas. Descartamos el ping ENTERO en vez
        # de intentar salvarlo: no hay corrección geométrica posible a posteriori sin
        # registrar franjas (eso es el SLAM). Medido en el bag 13_52_22, la fracción de
        # puntos que caen fuera de la superficie pasa de 2.1% (|roll-sesgo|<1°) a 33.7%
        # (>12°); el yaw-rate sube de 2.4% (<1°/s) a ~18% (10-15°/s).
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

        # Ángulo del haz respecto a la vertical DEL SENSOR (el que usaba el cutoff).
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

        # Drop grazing outer beams. OJO al frame: recortar en el frame del SENSOR
        # (lo que se hacía) deja pasar haces que, con el vehículo alabeado, apuntan
        # muy por debajo del corte real. Con cutoff=55° y roll de 10°, el haz exterior
        # de sotavento sale a 65° de la vertical VERDADERA. Medido en 13_52_22:
        # 548.600 puntos (2.26%) superan los 55° reales pese a "pasar" el cutoff, y
        # su tasa de error es 13.6%; los que pasan de 60° reales, 47.4%. Recortando
        # en frame mundo el cutoff significa lo que dice.
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
        f"Pings: {count} leídos, {n_skip_roll} descartados por roll, "
        f"{n_skip_yaw} por yaw-rate, {len(buffer_points)} válidos "
        f"({(n_skip_roll + n_skip_yaw) / max(count, 1) * 100:.2f}% descartado por actitud)"
    )

    if not buffer_points:
        rospy.logerr("No valid points")
        return

    # Point cloud: voxel downsample -> outlier removal.
    # ORDEN IMPORTANTE: primero se submuestrea al voxel y LUEGO se filtra el ruido.
    # Antes se hacía al revés: el SOR (que construye un KD-tree) corría sobre los
    # ~31 M de puntos crudos (voxel 0.01 m), lentísimo y con un pico de RAM enorme.
    # Submuestrear primero deja ~1-3 M puntos y el SOR es casi instantáneo. El
    # producto batimétrico del fondo no pierde nada: 1 cm es muy por debajo de la
    # resolución útil del MBES a estas alturas de vuelo.
    pts_all = np.vstack(buffer_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_all)

    n_raw = len(pcd.points)
    pcd = pcd.voxel_down_sample(voxel_size)
    n_voxel = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(sor_k, sor_std)
    n_sor = len(pcd.points)

    # Filtro relativo a la superficie: elimina los picos verticales (haces con
    # rango erróneo / curl de bordes) que el SOR k-NN no caza y que producen las
    # "montañas" espurias en la nube. Adaptativo por MAD de celda.
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
        f"-> {n_sor} SOR -> {len(pcd.points)} tras filtro de superficie"
    )

    xyz_file = os.path.join(cloud_dir, "mb_pointcloud.xyz")
    o3d.io.write_point_cloud(xyz_file, pcd, write_ascii=True)

    rospy.loginfo(f"XYZ saved: {xyz_file}")

    # =====================================================================
    # DEM (batimetría) rasterizado: mb_pointcloud.tif + JPG en images/
    # =====================================================================
    # La nube 3D no es un ráster; para el .tif se proyecta a una rejilla XY con la
    # Z (profundidad) MEDIANA por celda -> modelo digital del terreno georreferenciado.
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

    # JPG del DEM (profundidad -> mapa de color, con hillshade suave para el relieve).
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

    # depth de Poisson parametrizable: depth=11 sobre nubes densas dispara la RAM
    # (fue lo que provocó el OOM-kill: 55 GB con 31 M puntos). A voxel 0.05-0.10 m,
    # depth 9-10 da resolución de sobra sin reventar memoria.
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

    # Poisson devuelve un array de colores por vértice TODO A CERO cuando la nube de
    # entrada no tiene color, y write_triangle_mesh los escribe en el PLY. CloudCompare
    # respeta el color del vértice, así que pintaba la malla entera de negro. Sin el
    # array de colores usa su sombreado por normales y se ve el relieve. El color va en
    # mb_textured_sss.ply, que es el producto texturizado (sss_mb_fusion.py).
    if mesh.has_vertex_colors() and not np.asarray(mesh.vertex_colors).any():
        mesh.vertex_colors = o3d.utility.Vector3dVector()
        rospy.loginfo("Malla sin color: descartado el array de vértices negros de Poisson.")

    mesh_file = os.path.join(mesh_dir, "mb_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_file, mesh)

    rospy.loginfo(f"Mesh saved: {mesh_file}")

    pub_mb_done = rospy.Publisher('/pipeline/mb_done', Bool, queue_size=1, latch=True)

    time.sleep(0.5)
    pub_mb_done.publish(True)

    rospy.loginfo("===== MULTIBEAM FINISHED =====")

if __name__ == '__main__':
    main()