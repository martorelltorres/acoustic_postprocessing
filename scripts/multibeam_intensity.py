#!/usr/bin/env python3
"""
Multibeam backscatter -> georeferenced mosaic + intensity-tagged cloud.
Same geometry as multibeam_processor.py (axis flip, sensor TF, MB->SSS
lever-arm), so outputs share the UTM frame and fuse with the other products.

Outputs:
  - tif/mb_intensity.tif        : top-down backscatter mosaic (GeoTIFF, UTM, paleta viridis)
  - images/mb_intensity.jpg     : el mismo mosaico como JPG
  - pointcloud/mb_intensity.xyz : cloud "X Y Z I" (UTM, raw intensity)

Author: Antoni Martorell (SRV, UIB)
"""

import rospy
import rosbag
import numpy as np
import ros_numpy
import os
import time

import cv2
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import interp1d
from pyproj import Transformer
import tf.transformations as tr
from std_msgs.msg import Bool

# Geographic -> UTM (zone 31N)
CRS_WGS84 = "EPSG:4326"
CRS_UTM   = "EPSG:32631"
ll_to_utm = Transformer.from_crs(CRS_WGS84, CRS_UTM, always_xy=True)


def enhance_data(img_input):
    # Normalize (2-98 pct), despeckle, CLAHE, sharpen -> 8-bit.
    if img_input is None or img_input.size == 0:
        return np.zeros_like(img_input, dtype=np.uint8)

    valid = img_input > 0
    if not np.any(valid):
        return np.zeros_like(img_input, dtype=np.uint8)

    vmin, vmax = np.percentile(img_input[valid], (2, 98))
    vmax = max(vmax, vmin + 1e-6)

    img = np.clip((img_input - vmin) * 255.0 / (vmax - vmin), 0, 255).astype(np.uint8)
    img = cv2.medianBlur(img, 5)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(img, -1, kernel)


def viridis_colormap():
    """Tabla de color 0-255 -> RGBA para embeber en el GeoTIFF.

    Es la MISMA rampa que el JPG (cv2.COLORMAP_VIRIDIS), así que .tif y .jpg se ven
    idénticos. Se embebe como paleta en vez de escribir 3 bandas RGB a propósito: el
    .tif sigue siendo 1 banda con el VALOR de backscatter, que es lo que necesita
    mb_sss_mosaic_fusion.py (hace src.read(1) y fusiona valores, no canales). Con RGB
    leería el canal rojo del viridis como si fuera intensidad. QGIS/GDAL y la mayoría
    de visores respetan la paleta y lo pintan en color.
    """
    ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
    bgr = cv2.applyColorMap(ramp, cv2.COLORMAP_VIRIDIS).reshape(256, 3)
    cmap = {i: (int(px[2]), int(px[1]), int(px[0]), 255) for i, px in enumerate(bgr)}
    cmap[0] = (0, 0, 0, 0)   # 0 = nodata -> transparente (ver la nota del nodata en main)
    return cmap


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


def main():

    rospy.init_node('multibeam_intensity')

    rospy.loginfo("===== MULTIBEAM INTENSITY PROCESSOR STARTED =====")

    bag_file   = rospy.get_param('~bag_file')
    scan_topic = rospy.get_param('~scan_topic')
    nav_topic  = rospy.get_param('~nav_topic')
    output_dir = rospy.get_param('~output_dir')

    mosaic_res       = rospy.get_param('~mosaic_res', 0.10)
    angle_cutoff_deg = rospy.get_param('~angle_cutoff', 60.0)
    save_cloud       = rospy.get_param('~save_cloud', True)
    apply_avg        = rospy.get_param('~apply_avg', True)   # corrección AVG por ángulo
    # Gating por ACTITUD: idéntico a multibeam_processor.py (mismos args del launch).
    # Ver allí la nota larga. <=0 desactiva cada gate.
    max_roll_deg       = float(rospy.get_param('~max_roll_deg', 5.0))
    max_yaw_rate_dps   = float(rospy.get_param('~max_yaw_rate_deg_s', 8.0))
    # OJO al float(): roslaunch entrega "NaN" como STRING (ver multibeam_processor.py).
    roll_bias_deg      = float(rospy.get_param('~roll_bias_deg', float('nan')))
    angle_cutoff_frame = str(rospy.get_param('~angle_cutoff_frame', 'world')).lower()

    if angle_cutoff_frame not in ('sensor', 'world'):
        rospy.logwarn(f"angle_cutoff_frame='{angle_cutoff_frame}' no válido; uso 'world'.")
        angle_cutoff_frame = 'world'

    # Layout de results/: cada producto en su carpeta. Ver results/README.md.
    tif_dir    = os.path.join(output_dir, "tif")
    images_dir = os.path.join(output_dir, "images")
    cloud_dir  = os.path.join(output_dir, "pointcloud")

    for d in (output_dir, tif_dir, images_dir, cloud_dir):
        os.makedirs(d, exist_ok=True)

    bag = rosbag.Bag(bag_file)

    # UTM origin
    lat0, lon0 = get_nav_origin(bag, nav_topic)
    X0_UTM, Y0_UTM = ll_to_utm.transform(lon0, lat0)

    rospy.loginfo(f"UTM origin: {X0_UTM:.3f}, {Y0_UTM:.3f}")

    # TF chain (same as multibeam_processor.py)
    T_MB = get_static_transform_from_tf(
        bag_file, 'sparus2/base_link', 'sparus2/multibeam')

    T_PORT = get_static_transform_from_tf(
        bag_file, 'sparus2/base_link', 'sparus2/sidescan_port')

    T_STBD = get_static_transform_from_tf(
        bag_file, 'sparus2/base_link', 'sparus2/sidescan_starboard')

    R_sensor = T_MB[:3, :3]
    sensor_offset = T_MB[:3, 3]

    # Lever-arm MB->SSS. El offset extra en Y (por defecto -2 m, ajuste empírico
    # para cuadrar con el mosaico SSS) se parametriza en vez de hardcodearse: debe
    # coincidir con el de multibeam_processor.py para que nube y mosaico compartan
    # marco. Ver la nota en multibeam_processor.py.
    sss_center = 0.5 * (T_PORT[:3, 3] + T_STBD[:3, 3])
    mb_sss_extra_offset = float(rospy.get_param('~mb_sss_extra_offset_y', -2.0))
    delta_sensor = (sss_center - sensor_offset) + np.array([0.0, mb_sss_extra_offset, 0.0])

    rospy.loginfo(f"Lever-arm delta: {delta_sensor}")

    # Navigation: build time interpolators for pose
    ts_nav, north, east, depth, yaw, pitch, roll = [], [], [], [], [], [], []

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
    f_e = interp1d(ts_nav, np.array(east),  bounds_error=False, fill_value=np.nan)
    f_d = interp1d(ts_nav, np.array(depth), bounds_error=False, fill_value=np.nan)

    f_y = interp1d(ts_nav, yaw_unwrapped,              bounds_error=False, fill_value=np.nan)
    f_p = interp1d(ts_nav, np.unwrap(np.array(pitch)), bounds_error=False, fill_value=np.nan)
    f_r = interp1d(ts_nav, np.unwrap(np.array(roll)),  bounds_error=False, fill_value=np.nan)

    # Yaw-rate y sesgo de roll para el gate de actitud (ver multibeam_processor.py).
    yaw_rate_dps = np.degrees(
        np.gradient(yaw_unwrapped) / np.maximum(np.gradient(ts_nav), 1e-3)
    )
    f_yr = interp1d(ts_nav, yaw_rate_dps, bounds_error=False, fill_value=np.nan)

    if not np.isfinite(roll_bias_deg):
        roll_bias_deg = float(np.degrees(np.median(np.array(roll))))

    rospy.loginfo(
        f"Actitud: sesgo roll {roll_bias_deg:+.2f}° | gate |roll-sesgo|<={max_roll_deg}° "
        f"y |yaw_rate|<={max_yaw_rate_dps}°/s | cutoff {angle_cutoff_deg}° (frame {angle_cutoff_frame})"
    )

    # Per-ping processing: sensor frame -> vehicle -> local -> UTM
    rospy.loginfo("Processing multibeam intensity pings...")

    pts_buffer = []   # (N,3) UTM
    int_buffer = []   # (N,)  raw intensity
    ang_buffer = []   # (N,)  ángulo de incidencia (para la corrección AVG)
    count = 0
    n_skip_roll = 0
    n_skip_yaw = 0

    for _, scan, _ in bag.read_messages(topics=[scan_topic]):

        if not hasattr(scan, 'header'):
            continue

        count += 1
        if count % 200 == 0:
            rospy.loginfo(f"Pings processed: {count}")

        ts = scan.header.stamp.to_sec()
        if ts < ts_nav[0] or ts > ts_nav[-1]:
            continue

        n = float(f_n(ts)); e = float(f_e(ts)); d = float(f_d(ts))
        yaw_t = float(f_y(ts)); pitch_t = float(f_p(ts)); roll_t = float(f_r(ts))

        if np.isnan(n) or np.isnan(e) or np.isnan(yaw_t):
            continue

        # Gate por actitud (ping entero). Mismos umbrales que la nube batimétrica.
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

        intensity = np.array(pc['intensity'], dtype=np.float32)

        # Same axis convention as multibeam_processor.py
        xyz = np.column_stack((pc['x'], -pc['y'], -pc['z'])).astype(np.float64)

        # Ángulo del haz respecto a la vertical DEL SENSOR.
        ang_sensor = np.degrees(np.arctan2(
            np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2),
            np.abs(xyz[:, 2])
        ))

        # Sensor rotation
        xyz = xyz @ R_sensor.T

        # Vehicle rotation
        R_veh = tr.euler_matrix(roll_t, pitch_t, yaw_t, axes='sxyz')[:3, :3]
        xyz = xyz @ R_veh.T

        # Ángulo respecto a la vertical REAL, ya con roll/pitch aplicados. Ver la nota
        # en multibeam_processor.py: en frame sensor el cutoff deja pasar haces que
        # apuntan mucho más rasantes de lo que el umbral promete.
        ang_world = np.degrees(np.arctan2(
            np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2),
            np.abs(xyz[:, 2])
        ))

        # El AVG se bina con el MISMO ángulo con el que se recorta. En frame mundo eso
        # es el ángulo de incidencia sobre fondo plano, que es la variable física de la
        # que depende el backscatter; el ángulo de haz en frame sensor solo coincide con
        # ella cuando el roll es cero (aquí difieren p90=5.4°, y el AVG bina a 2°).
        angles = ang_world if angle_cutoff_frame == 'world' else ang_sensor
        keep = angles < angle_cutoff_deg

        xyz = xyz[keep]
        intensity = intensity[keep]
        ang_keep = angles[keep]     # ángulo de incidencia por punto (para AVG)
        if len(xyz) < 10:
            continue

        # Lever-arm offsets (identical to bathymetry pipeline)
        offset_world = R_veh @ sensor_offset
        offset_world += R_veh @ delta_sensor

        xyz[:, 0] += offset_world[0]
        xyz[:, 1] += offset_world[1]
        xyz[:, 2] += offset_world[2]

        # Local world (north, east, -depth)
        xyz[:, 0] += n
        xyz[:, 1] += e
        xyz[:, 2] += -d

        # To UTM (X=easting from local-east, Y=northing from local-north)
        pts_world = np.empty_like(xyz)
        pts_world[:, 0] = X0_UTM + xyz[:, 1]
        pts_world[:, 1] = Y0_UTM + xyz[:, 0]
        pts_world[:, 2] = xyz[:, 2]

        pts_buffer.append(pts_world)
        int_buffer.append(intensity)
        ang_buffer.append(ang_keep)

    bag.close()

    if not pts_buffer:
        rospy.logerr("No valid intensity points")
        return

    pts_all = np.vstack(pts_buffer)
    int_all = np.concatenate(int_buffer)
    ang_all = np.concatenate(ang_buffer)

    rospy.loginfo(
        f"Pings: {count} leídos, {n_skip_roll} descartados por roll, "
        f"{n_skip_yaw} por yaw-rate ({(n_skip_roll + n_skip_yaw) / max(count, 1) * 100:.2f}% "
        f"descartado por actitud)"
    )
    rospy.loginfo(f"Total intensity points: {len(pts_all)}")

    # =====================================================================
    # CORRECCIÓN AVG (Angle Varying Gain) — portada del SLAM
    # =====================================================================
    # El backscatter del MBES está dominado por el ángulo de incidencia: forma
    # de campana, brillante cerca del nadir y oscuro en los haces rasantes. Ese
    # banding va con la pose del vehículo, no con el fondo, y arruina el mosaico
    # (la distribución de intensidad quedaba bimodal: ~53% de píxeles casi negros).
    #
    # Perfil = MEDIANA de intensidad por bin angular (robusto a la estructura del
    # fondo y a outliers). Luego  I_corr = I / gain(angulo), que deja la intensidad
    # ~1 de media a cualquier ángulo y conserva solo la textura real del fondo
    # (firma sedimento/roca). Reduce el banding ~99%.
    if apply_avg:
        bins = np.arange(0.0, angle_cutoff_deg + 1.0, 2.0)
        centers = (bins[:-1] + bins[1:]) / 2.0
        idx = np.clip(np.digitize(ang_all, bins) - 1, 0, len(centers) - 1)
        gain = np.full(len(centers), np.nan)
        for b in range(len(centers)):
            sel = int_all[idx == b]
            if sel.size >= 50:
                gain[b] = np.median(sel)
        valid_bins = np.isfinite(gain)
        if valid_bins.sum() >= 3:
            gain_v = np.maximum(gain[valid_bins], 1e-3)
            centers_v = centers[valid_bins]
            # Ganancia interpolada por punto y normalizada a la mediana global,
            # para no cambiar la escala absoluta del backscatter.
            g_pts = np.interp(ang_all, centers_v, gain_v)
            int_all = int_all / np.maximum(g_pts, 1e-3) * float(np.median(gain_v))
            rospy.loginfo(
                f"AVG aplicado: ganancia {gain_v.min():.1f}-{gain_v.max():.1f} "
                f"(swing {gain_v.max() - gain_v.min():.1f}) sobre {valid_bins.sum()} bins"
            )
        else:
            rospy.logwarn("AVG: pocos bins válidos; mosaico sin corregir.")

    # Point cloud "X Y Z I"
    if save_cloud:
        xyz_file = os.path.join(cloud_dir, "mb_intensity.xyz")
        out = np.column_stack((pts_all, int_all))
        np.savetxt(xyz_file, out, fmt="%.4f %.4f %.4f %.4f")
        rospy.loginfo(f"Intensity cloud saved: {xyz_file}")

    # Georeferenced backscatter mosaic: MEDIANA de intensidad por celda.
    # (Antes se usaba la media, sensible a outliers de backscatter — un solo
    #  retorno especular disparaba la celda. La mediana por celda es robusta y,
    #  combinada con la corrección AVG, da un mosaico con textura de fondo real.)
    margin = 2.0
    x_min = pts_all[:, 0].min() - margin
    x_max = pts_all[:, 0].max() + margin
    y_min = pts_all[:, 1].min() - margin
    y_max = pts_all[:, 1].max() + margin

    width  = int(np.ceil((x_max - x_min) / mosaic_res))
    height = int(np.ceil((y_max - y_min) / mosaic_res))

    c = ((pts_all[:, 0] - x_min) / mosaic_res).astype(np.int64)
    r = ((y_max - pts_all[:, 1]) / mosaic_res).astype(np.int64)

    mask = (c >= 0) & (c < width) & (r >= 0) & (r < height)
    cell = r[mask] * width + c[mask]
    vals = int_all[mask].astype(np.float64)

    # Mediana por celda vía groupby con ordenación (vectorizado, sin bucle por punto).
    order = np.argsort(cell, kind="stable")
    cell_s = cell[order]
    vals_s = vals[order]
    uniq, starts, counts = np.unique(cell_s, return_index=True, return_counts=True)
    img = np.zeros(width * height, dtype=np.float64)
    for u, s, cnt_u in zip(uniq, starts, counts):
        img[u] = np.median(vals_s[s:s + cnt_u])
    img = img.reshape((height, width)).astype(np.float32)

    img8 = enhance_data(img)

    # El CLAHE de enhance_data levanta el fondo vacío de 0 a ~4, así que el 0 deja de
    # significar "sin dato": el histograma del .tif salía con mediana 4 y media 60, que
    # es el falso "mosaico bimodal casi-negro" (la mediana medía el FONDO, no el fondo
    # marino). Reservamos el 0 para nodata: las celdas sin dato vuelven a 0 y las que
    # tienen dato se fuerzan a >=1. mb_sss_mosaic_fusion.py ya usa `mb_g > 0` como
    # máscara de validez, así que esto es justo lo que espera.
    filled = img > 0
    img8[filled] = np.maximum(img8[filled], 1)
    img8[~filled] = 0

    transform = from_origin(x_min, y_max, mosaic_res, mosaic_res)
    tif_file = os.path.join(tif_dir, "mb_intensity.tif")

    with rasterio.open(
        tif_file, 'w', driver='GTiff',
        height=img8.shape[0], width=img8.shape[1],
        count=1, dtype=np.uint8, crs=CRS_UTM,
        transform=transform, compress='deflate',
        photometric='palette', nodata=0
    ) as dst:
        dst.write(img8, 1)
        # Paleta viridis embebida: mantiene los colores del JPG sin dejar de ser
        # 1 banda de backscatter. Ver viridis_colormap().
        dst.write_colormap(1, viridis_colormap())

    rospy.loginfo(f"Backscatter mosaic saved (paleta viridis): {tif_file}")

    # JPG del mosaico (visualización) en results/images/. Se aplica un mapa de
    # color para que la textura del backscatter se lea mejor que en gris plano.
    jpg_file = os.path.join(images_dir, "mb_intensity.jpg")
    color = cv2.applyColorMap(img8, cv2.COLORMAP_VIRIDIS)
    color[~filled] = 0      # nodata en negro, no en el morado oscuro del viridis
    cv2.imwrite(jpg_file, color, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    rospy.loginfo(f"Backscatter JPG saved: {jpg_file}")

    # Pipeline signal
    pub = rospy.Publisher('/pipeline/mb_intensity_done', Bool, queue_size=1, latch=True)
    time.sleep(0.5)
    pub.publish(True)

    rospy.loginfo("===== MULTIBEAM INTENSITY FINISHED =====")


if __name__ == '__main__':
    main()
