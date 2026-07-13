#!/usr/bin/env python3
"""
Sidescan -> top-down georeferenced mosaic (UTM GeoTIFF).
Slant-range corrected, AVG-corrected, accumulated on a north/east grid.

Author: Antoni Martorell (SRV, UIB)
"""

import rospy
import rosbag
import numpy as np
import cv2
import os
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
import tf.transformations as tr
from std_msgs.msg import Bool
import time

# Rango por canal (m). Es SOLO un fallback: el valor real viene en los mensajes
# SSSConfig del bag (topic .../raw_data/<side>/sss_info, campo `range`) y se lee con
# get_sonar_range(). El 30.0 hardcodeado que había aquí era FALSO para los bags de
# Andratx (range real = 50.0 m): colocaba cada muestra al 60% de su rango verdadero,
# comprimiendo el mosaico x0.6 across-track y arrastrando el error a mb_sss_mosaic.tif
# y a la textura de mb_textured_sss.ply.
SONAR_RANGE_FALLBACK = 30.0
MOSAIC_RES = 0.07    # m/pixel
BLIND_ZONE = 0.2     # m, nadir gap to skip

CRS_WGS84 = "EPSG:4326"
CRS_UTM = "EPSG:32631"
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

    raise RuntimeError("Navigation geographic origin not found")


def get_nav_data(bag, nav_topic):
    # Time interpolators for pose + altitude (yaw smoothed).

    ts, north, east, yaw, pitch, roll, alt = [], [], [], [], [], [], []

    for _, msg, _ in bag.read_messages(topics=[nav_topic]):

        ts.append(msg.header.stamp.to_sec())
        north.append(msg.position.north)
        east.append(msg.position.east)

        yaw.append(msg.orientation.yaw)
        pitch.append(msg.orientation.pitch)
        roll.append(msg.orientation.roll)

        alt.append(msg.altitude)

    ts = np.array(ts)
    idx = np.argsort(ts)

    ts = ts[idx]
    north = np.array(north)[idx]
    east = np.array(east)[idx]

    yaw = np.unwrap(np.array(yaw)[idx])
    yaw = gaussian_filter1d(yaw, sigma=2)

    pitch = np.unwrap(np.array(pitch)[idx])
    roll = np.unwrap(np.array(roll)[idx])

    alt = np.array(alt)[idx]

    f_n = interp1d(ts, north, bounds_error=False, fill_value=np.nan)
    f_e = interp1d(ts, east, bounds_error=False, fill_value=np.nan)

    f_y = interp1d(ts, yaw, bounds_error=False, fill_value=np.nan)
    f_p = interp1d(ts, pitch, bounds_error=False, fill_value=np.nan)
    f_r = interp1d(ts, roll, bounds_error=False, fill_value=np.nan)

    f_h = interp1d(ts, alt, bounds_error=False, fill_value=np.nan)

    return (f_n, f_e, f_y, f_p, f_r, f_h), (ts[0], ts[-1])


def get_sonar_range(bag, fallback=SONAR_RANGE_FALLBACK):
    """Rango por canal (m) leído de los SSSConfig del bag; `fallback` si no hay."""
    cfg_topics = [
        t for t in bag.get_type_and_topic_info().topics
        if t.endswith('/sss_info')
    ]

    for _, msg, _ in bag.read_messages(topics=cfg_topics):
        if getattr(msg, 'range', 0.0) > 0.0:
            return float(msg.range)

    rospy.logwarn(f"Sin SSSConfig en el bag; uso SONAR_RANGE={fallback} m (puede ser falso).")

    return float(fallback)


def angle_varying_gain(intensity, theta, is_port, bin_deg=2.0, min_samples=200,
                       max_boost=8.0):
    """
    Corrección AVG (Angle Varying Gain) del backscatter del sidescan.

    El eco del SSS está dominado por el ángulo de incidencia sobre el fondo: brillante
    cerca del nadir, oscuro en rango lejano. Ese patrón es geometría del sonar, no tipo
    de fondo, y domina el contraste del mosaico y de la malla texturizada.

    Perfil = MEDIA de intensidad por bin angular. OJO: la versión del multihaz usa la
    MEDIANA, y aquí NO sirve. El eco crudo del SSS está saturado de ceros (47.5% de las
    muestras son 0 exacto; p75 = 2 sobre 255), así que la mediana por bin vale 0-1 y no
    mide ganancia sino relleno: el "perfil" sale plano y la corrección es un no-op. La
    media sí decae suavemente con el ángulo y es lo que hay que dividir.

    El perfil cae ~700x del nadir al rango lejano, donde la señal es casi todo ceros.
    Dividir por él a pelo amplificaría el ruido de la cola x700, así que la ganancia se
    acota por abajo a `gmax / max_boost`: más allá de ese punto no hay SNR que rescatar
    y la cola conserva algo de oscurecimiento residual, a propósito.

    DOS DIFERENCIAS con el multihaz, ambas deliberadas:
      - Perfil SEPARADO por banda (port/stbd): transductores distintos con ganancias
        distintas; un perfil único dejaría un escalón justo en el nadir.
      - Escala COMÚN g0 para las dos bandas. Normalizar cada lado a su propia mediana
        quitaría el patrón angular pero conservaría el desbalance port/stbd, que
        también es ganancia y no fondo.

    Devuelve (intensidad_corregida, info) o (None, None) si no hay bins suficientes.
    """
    bins = np.arange(0.0, 90.0 + bin_deg, bin_deg)
    centers = (bins[:-1] + bins[1:]) / 2.0
    bin_of = np.clip(np.digitize(theta, bins) - 1, 0, len(centers) - 1)

    sides = {"port": is_port, "stbd": ~is_port}
    gains = {}

    for name, side in sides.items():
        g = np.full(len(centers), np.nan)
        for b in range(len(centers)):
            sel = intensity[side & (bin_of == b)]
            if sel.size >= min_samples:
                g[b] = sel.mean()
        gains[name] = g

    valid = {name: np.isfinite(g) for name, g in gains.items()}

    if min(valid["port"].sum(), valid["stbd"].sum()) < 3:
        return None, None

    g_max = max(float(gains[name][valid[name]].max()) for name in sides)
    g_floor = g_max / max_boost

    clamped = {name: np.maximum(gains[name][valid[name]], g_floor) for name in sides}

    g0 = float(np.median(np.concatenate([clamped[name] for name in sides])))

    out = intensity.astype(np.float32).copy()

    for name, side in sides.items():
        v = valid[name]
        g_pts = np.interp(theta[side], centers[v], clamped[name])
        out[side] = intensity[side] / g_pts * g0

    return out, (gains, valid, centers, g0, g_floor)


def process_mosaic(bag, nav, time_range, T_PORT, T_STBD, sonar_range, apply_avg=True):
    # Accumulate slant-corrected returns onto the UTM-local grid.
    f_n, f_e, f_y, f_p, f_r, f_h = nav
    t0, t1 = time_range

    # Grid extent from the trajectory bounding box + range margin
    ts_samples = np.linspace(t0, t1, 500)

    east_samples = f_e(ts_samples)
    north_samples = f_n(ts_samples)

    valid = ~np.isnan(east_samples) & ~np.isnan(north_samples)

    margin = sonar_range + 5.0

    x_min = np.min(east_samples[valid]) - margin
    x_max = np.max(east_samples[valid]) + margin
    y_min = np.min(north_samples[valid]) - margin
    y_max = np.max(north_samples[valid]) + margin

    width = int(np.ceil((x_max - x_min) / MOSAIC_RES))
    height = int(np.ceil((y_max - y_min) / MOSAIC_RES))

    # Se bufferean las muestras (celda, intensidad, ángulo, banda) en vez de acumularlas
    # al vuelo: el perfil AVG es una mediana GLOBAL por bin angular, así que hay que ver
    # todos los pings antes de corregir. Son ~6 M muestras (~0.1 GB), una sola pasada.
    buf_idx = []
    buf_int = []
    buf_ang = []
    buf_port = []

    def to_idx(x, y):
        c = ((x - x_min) / MOSAIC_RES).astype(np.int32)
        r = ((y_max - y) / MOSAIC_RES).astype(np.int32)
        return c, r

    info = bag.get_type_and_topic_info()

    sss_topics = [
        t for t, v in info.topics.items()
        if "sidescan" in t.lower() and "Image" in v.msg_type
    ]

    for topic, msg, _ in bag.read_messages(topics=sss_topics):

        if not hasattr(msg, 'header') or not hasattr(msg, 'data'):
            continue

        ts = msg.header.stamp.to_sec()

        if ts < t0 or ts > t1:
            continue

        n = float(f_n(ts))
        e = float(f_e(ts))
        yaw = float(f_y(ts))
        h = float(f_h(ts))

        if np.isnan(n) or np.isnan(e) or np.isnan(yaw) or np.isnan(h):
            continue

        if h < 0.2:   # sonar out of water / invalid
            continue

        scan = np.frombuffer(msg.data, dtype=np.uint8).astype(np.float32)

        # Port is reversed so the nadir sits at the inner edge
        is_port = "port" in topic.lower()

        if is_port:
            scan = scan[::-1]
            T_sensor = T_PORT
        else:
            T_sensor = T_STBD

        # Slant-range -> ground-range
        npx = scan.size
        meters_px = sonar_range / npx

        slant = np.arange(npx) * meters_px
        ground = np.sqrt(np.maximum(slant**2 - h**2, 0.0))

        # Ángulo de incidencia sobre el fondo (desde la vertical): 0° en el nadir,
        # ->90° en rango lejano. Es la variable de la que depende el backscatter y con
        # la que se bina el AVG. Con `ground` ya corregido de slant-range, sale directo
        # de la altura sobre el fondo.
        theta = np.degrees(np.arctan2(ground, h))

        valid_mask = ground > BLIND_ZONE

        if not np.any(valid_mask):
            continue

        # Sensor lever-arm rotated into world
        sensor_offset = T_sensor[:3, 3]

        off_n = sensor_offset[0] * np.cos(yaw) - sensor_offset[1] * np.sin(yaw)
        off_e = sensor_offset[0] * np.sin(yaw) + sensor_offset[1] * np.cos(yaw)

        # Across-track direction (opposite sign per side)
        if is_port:
            v_ping_n = np.sin(yaw)
            v_ping_e = -np.cos(yaw)
        else:
            v_ping_n = -np.sin(yaw)
            v_ping_e = np.cos(yaw)

        px_n = (n + off_n) + v_ping_n * ground[valid_mask]
        px_e = (e + off_e) + v_ping_e * ground[valid_mask]

        c, r = to_idx(px_e, px_n)

        mask = (c >= 0) & (c < width) & (r >= 0) & (r < height)

        if not np.any(mask):
            continue

        buf_idx.append((r[mask].astype(np.int64) * width + c[mask]).astype(np.int64))
        buf_int.append(scan[valid_mask][mask])
        buf_ang.append(theta[valid_mask][mask].astype(np.float32))
        buf_port.append(np.full(int(mask.sum()), is_port, dtype=bool))

    if not buf_idx:
        rospy.logerr("No valid sidescan samples")
        return np.zeros((height, width), dtype=np.float32), x_min, y_max

    idx_all  = np.concatenate(buf_idx)
    int_all  = np.concatenate(buf_int).astype(np.float32)
    ang_all  = np.concatenate(buf_ang)
    port_all = np.concatenate(buf_port)

    rospy.loginfo(f"Muestras SSS: {len(int_all)} "
                  f"({port_all.sum()} port / {(~port_all).sum()} stbd)")

    if apply_avg:
        corrected, info = angle_varying_gain(int_all, ang_all, port_all)

        if corrected is None:
            rospy.logwarn("AVG: pocos bins válidos; mosaico sin corregir.")
        else:
            int_all = corrected
            gains, valid, _, g0, g_floor = info

            for name in ("port", "stbd"):
                gv = gains[name][valid[name]]
                rospy.loginfo(
                    f"AVG {name}: ganancia {gv.min():.2f}-{gv.max():.2f} "
                    f"(x{gv.max() / max(gv.min(), 1e-3):.0f}) sobre {valid[name].sum()} bins"
                )

            rospy.loginfo(
                f"AVG: g0={g0:.2f}, suelo de ganancia {g_floor:.2f} "
                f"(boost máx x{max(gains[n][valid[n]].max() for n in ('port','stbd')) / g_floor:.0f})"
            )

    # Media por celda. bincount en vez de np.add.at: mismo resultado, mucho más rápido.
    grid = np.bincount(idx_all, weights=int_all, minlength=width * height)
    cnt  = np.bincount(idx_all, minlength=width * height)

    img = np.zeros(width * height, dtype=np.float32)
    filled = cnt > 0
    img[filled] = (grid[filled] / cnt[filled]).astype(np.float32)

    return img.reshape((height, width)), x_min, y_max

def main():

    rospy.init_node('sss_mosaic_gen', anonymous=True)

    bag_file = rospy.get_param('~bag_file', '')
    output_dir = rospy.get_param('~output_dir', '.')
    nav_topic = rospy.get_param('~nav_topic', '/sparus2/navigator/navigation')
    apply_avg = rospy.get_param('~apply_avg', True)   # corrección AVG por ángulo

    if not bag_file:
        rospy.logerr("ERROR: bag_file not provided")
        return

    tif_dir    = os.path.join(output_dir, 'tif')
    images_dir = os.path.join(output_dir, 'images')

    for d in (tif_dir, images_dir):
        os.makedirs(d, exist_ok=True)

    output_tiff = os.path.join(tif_dir, 'sss_mosaic.tif')
    # PNG y no JPG: el fondo sin dato va TRANSPARENTE y JPEG no tiene canal alfa.
    output_png  = os.path.join(images_dir, 'sss_mosaic.png')

    bag = rosbag.Bag(bag_file)

    lat0, lon0 = get_nav_origin(bag, nav_topic)
    X0_UTM, Y0_UTM = ll_to_utm.transform(lon0, lat0)

    nav, t_range = get_nav_data(bag, nav_topic)

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

    # Rango por canal: del bag (SSSConfig), no hardcodeado. Ver SONAR_RANGE_FALLBACK.
    sonar_range = float(rospy.get_param('~sonar_range', 0.0)) or get_sonar_range(bag)

    rospy.loginfo(f"Rango por canal: {sonar_range:.1f} m "
                  f"({sonar_range / 2000.0 * 1000:.2f} mm/muestra a 2000 muestras)")

    img, x_min, y_max = process_mosaic(
        bag,
        nav,
        t_range,
        T_PORT,
        T_STBD,
        sonar_range,
        apply_avg=apply_avg
    )

    save_geotiff = from_origin(
        X0_UTM + x_min,
        Y0_UTM + y_max,
        MOSAIC_RES,
        MOSAIC_RES
    )

    img8 = enhance_data(img)

    # El CLAHE de enhance_data levanta el fondo vacío por encima de 0, así que el 0
    # deja de significar "sin dato" y contamina el histograma y el JPG. Reservamos el
    # 0 para nodata: fondo a 0, celdas con dato a >=1. mb_sss_mosaic_fusion.py usa
    # `sss_g > 0` como máscara de validez, así que es justo lo que espera.
    filled = img > 0
    img8[filled] = np.maximum(img8[filled], 1)
    img8[~filled] = 0

    with rasterio.open(
        output_tiff,
        'w',
        driver='GTiff',
        height=img8.shape[0],
        width=img8.shape[1],
        count=1,
        dtype=np.uint8,
        crs=CRS_UTM,
        transform=save_geotiff,
        compress='deflate',
        nodata=0
    ) as dst:
        dst.write(img8, 1)

    bag.close()

    rospy.loginfo(f"GeoTIFF generated: {output_tiff}")

    # PNG del mosaico (visualización) en results/images/. Misma rampa viridis que
    # mb_intensity.jpg, para comparar los dos backscatter a ojo. El fondo sin dato va
    # con alfa=0 (transparente), no pintado: así solo se ve la proyección de los datos
    # sobre lo que haya debajo. Por eso es PNG y no JPG — JPEG no tiene canal alfa.
    color = cv2.applyColorMap(img8, cv2.COLORMAP_VIRIDIS)      # BGR
    # El RGB de debajo del alfa también se pone a 0. Si no, queda el morado de
    # viridis(0) ahí escondido y cualquier visor que ignore el canal alfa (o cualquier
    # aplanado sobre fondo) vuelve a pintar el fondo morado.
    color[~filled] = 0
    alpha = np.where(filled, 255, 0).astype(np.uint8)
    bgra = np.dstack([color, alpha])

    cv2.imwrite(output_png, bgra)

    rospy.loginfo(
        f"SSS mosaic PNG saved: {output_png} "
        f"({filled.mean() * 100:.1f}% de píxeles con dato, resto transparente)"
    )

    pub_sss_done = rospy.Publisher('/pipeline/sss_done', Bool, queue_size=1, latch=True)

    time.sleep(0.5)
    pub_sss_done.publish(True)

if __name__ == "__main__":
    main()

