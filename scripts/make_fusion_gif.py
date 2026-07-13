#!/usr/bin/env python3
"""
GIF de la fusión MB + SSS: el relieve 3D del multihaz se tiñe con la intensidad del
sidescan mientras la cámara orbita.

Salida: results/media/07_mb_sss_fusion.gif

Tres actos: la batimetría desnuda, un frente que barre el relieve proyectando la
intensidad del sidescan, y la fusión completa.

POR QUÉ OPEN3D Y NO MATPLOTLIB: matplotlib 3D no tiene z-buffer (Poly3DCollection
ordena las caras con el algoritmo del pintor) y pinta caras traseras encima de las
delanteras. Open3D rasteriza con GL. Se usa el Visualizer legacy con `visible=False`:
el OffscreenRenderer de filament renderiza bien pero deja el proceso colgado al salir.
Solo se puede crear UNA ventana por proceso: tras destroy_window(), GLFW no reinicia.

POR QUÉ EL DEM Y NO mb_mesh.ply: la malla Poisson interpola la nube cruda, y a 20 cm
es ruidosa — sale un fondo erizado, con agujeros donde la malla no es manifold. El DEM
es la MEDIANA de ~100 puntos por celda de 10 cm: misma geometría multihaz, estimador
mucho mejor, y una rejilla regular se triangula sin agujeros. La intensidad del SSS se
muestrea igual que en sss_mb_fusion.py, así que la fusión es la misma.

Uso:  python3 make_fusion_gif.py [results_dir]

Author: Antoni Martorell (SRV, UIB)
"""

import os
import sys

import numpy as np
import open3d as o3d
import rasterio
from scipy.ndimage import binary_erosion, binary_fill_holes, binary_opening, \
    distance_transform_edt, gaussian_filter, median_filter
from PIL import Image, ImageDraw, ImageFont

from make_media import PAGE, INK, INK2, MUTED, CREDIT, CMAP_DEPTH, CMAP_BS, \
    load_raster, crop_to_data

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

W = H = 800
# SIN exageración vertical. El DEM tiene saltos reales de hasta 0.8 m entre celdas
# contiguas de 0.25 m (pendiente p99 de 73 grados) en la mitad este: es el "fanning" por
# actitud del vehículo, documentado, y solo lo corrige el SLAM. No es ruido de alta
# frecuencia: ni una mediana 5x5, ni despiking, ni engordar la celda a 0.30 m lo bajan.
# Exagerar la vertical lo convertía en un bosque de agujas de 2 m que no existe. A x1.0
# el relieve de 6.5 m sobre 55 m ya se lee, y lo que se ve es lo que hay.
VERT_EXAG = 1.0
FPS = 16

ACT1 = 22             # órbita con la batimetría desnuda
ACT2 = 42             # el sidescan barre y se proyecta
ACT3 = 30             # órbita con la fusión completa
TOTAL = ACT1 + ACT2 + ACT3

NO_SSS = np.array([0.16, 0.16, 0.15])   # gris neutro: fuera de la franja del sidescan


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def dem_surface(dem_tif, erode_px=10, smooth=0.8, spike_m=0.35):
    """DEM raster -> (vértices XYZ en UTM, triángulos). Solo celdas con dato."""
    dem, ext = crop_to_data(*load_raster(dem_tif))
    finite = np.isfinite(dem)

    # El flanco del DEM es un fleco de celdas de un solo ping con Z ruidosa: a 1.9x de
    # exageración vertical sale como un bosque de agujas. Erosionar no basta, porque el
    # fleco viene en grumos conectados: primero se APERTURA (mata hebras y motas) y se
    # rellenan los huecos interiores, y luego sí se erosiona la orilla.
    core = binary_fill_holes(binary_opening(finite, iterations=3))
    valid = binary_erosion(core, iterations=erode_px)

    # Relleno por VECINO MÁS CERCANO, no por la mediana global: si no, el suavizado
    # arrastra los bordes hacia la mediana y levanta un labio dentado en la orilla.
    idx = distance_transform_edt(~finite, return_distances=False, return_indices=True)

    # DESPIKING contra la mediana local: el equivalente ráster de surface_relative_filter.
    # La mitad este del DEM tiene celdas de un solo ping (curl de los haces exteriores):
    # medido, el salto de Z entre celdas contiguas de 10 cm llega a 0.84 m en el p99, o
    # sea 83 grados de pendiente. Eso no es relieve, es ruido, y a 1.9x de exageración
    # sale como un bosque de agujas. Ni el gaussiano ni una mediana 5x5 lo quitan (el
    # ruido está correlacionado en más de media celda): hay que sustituir el valor.
    z = dem[tuple(idx)]
    for _ in range(2):
        med = median_filter(z, size=5)
        spike = np.abs(z - med) > spike_m
        z = np.where(spike, med, z)

    z = gaussian_filter(z, smooth)

    zlo, zhi = np.percentile(z[valid], (1, 99))
    z = np.clip(z, zlo, zhi)

    ny, nx = dem.shape
    xs = np.linspace(ext[0], ext[1], nx)
    ys = np.linspace(ext[3], ext[2], ny)          # fila 0 = norte
    X, Y = np.meshgrid(xs, ys)

    idx = -np.ones((ny, nx), np.int64)
    idx[valid] = np.arange(valid.sum())
    verts = np.column_stack([X[valid], Y[valid], z[valid]])

    # Un quad -> dos triángulos, solo si sus cuatro esquinas tienen dato.
    q = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    a, b = idx[:-1, :-1][q], idx[:-1, 1:][q]
    c, d = idx[1:, :-1][q], idx[1:, 1:][q]
    tris = np.vstack([np.column_stack([a, c, b]), np.column_stack([b, c, d])])

    return verts, tris


def sample_sss(verts, sss_tif):
    """Intensidad del sidescan en cada vértice (x, y). Devuelve (intensidad, válido)."""
    with rasterio.open(sss_tif) as src:
        a = src.read(1)
        nodata = src.nodata
        rows, cols = src.index(verts[:, 0], verts[:, 1])

    rows = np.asarray(rows)
    cols = np.asarray(cols)

    inside = (rows >= 0) & (rows < a.shape[0]) & (cols >= 0) & (cols < a.shape[1])

    val = np.zeros(len(verts), np.float32)
    val[inside] = a[rows[inside], cols[inside]]

    if nodata is not None:
        val[val == nodata] = 0.0

    return val, val > 0


def build_colors(verts, sss_tif):
    z = verts[:, 2]
    zlo, zhi = np.percentile(z, (2, 98))
    depth_rgb = CMAP_DEPTH((np.clip(z, zlo, zhi) - zlo) / (zhi - zlo))[:, :3]

    val, valid = sample_sss(verts, sss_tif)

    # La rampa gris llega casi a negro y la iluminación de Open3D la apaga del todo:
    # se comprime al tramo [0.18, 1.0], que en pantalla sigue siendo backscatter bajo.
    ilo, ihi = np.percentile(val[valid], (2, 98))
    norm = np.clip((val - ilo) / max(ihi - ilo, 1e-6), 0, 1)
    sss_rgb = CMAP_BS(0.18 + 0.82 * norm)[:, :3]
    sss_rgb[~valid] = NO_SSS

    return depth_rgb, sss_rgb, valid


def orbit_front(az_deg, elev_deg=30.0):
    a, e = np.radians(az_deg), np.radians(elev_deg)
    return [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]


def caption(img, act, wipe, cover):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)

    try:
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        small = ImageFont.truetype("DejaVuSans.ttf", 16)
        tiny = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        big = small = tiny = ImageFont.load_default()

    d.text((34, 28), "Multibeam + sidescan fusion", font=big, fill=INK)

    sub = {0: "Multibeam bathymetry",
           1: f"Projecting sidescan intensity   {wipe * 100:3.0f}%",
           2: f"Fused  ·  {cover * 100:.0f}% of the seafloor has sidescan coverage"}[act]
    d.text((36, 70), sub, font=small, fill=INK2)

    if act >= 1:
        d.rectangle([36, H - 92, 52, H - 78], fill=tuple((NO_SSS * 255).astype(int)))
        d.text((60, H - 93), "no sidescan coverage", font=tiny, fill=MUTED)

    d.text((36, H - 40), CREDIT, font=tiny, fill=MUTED)
    return np.array(im)


def main():
    res = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PKG_ROOT, "results")
    out = os.path.join(res, "media")
    os.makedirs(out, exist_ok=True)

    verts, tris = dem_surface(os.path.join(res, "tif", "mb_pointcloud.tif"))
    print(f"[fusion] superficie: {len(verts):,} vértices, {len(tris):,} triángulos")

    depth_rgb, sss_rgb, valid = build_colors(verts, os.path.join(res, "tif", "sss_mosaic.tif"))
    cover = float(valid.mean())
    print(f"[fusion] con dato SSS: {valid.sum():,} ({cover * 100:.1f}%)")

    # Centrado: en UTM las coords son ~4.4e6 y el z-buffer pierde precisión.
    c = verts.mean(axis=0)
    v = verts - c
    v[:, 2] *= VERT_EXAG

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    mesh.compute_vertex_normals()

    xlo, xhi = v[:, 0].min(), v[:, 0].max()

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=H, visible=False)
    vis.add_geometry(mesh)

    opt = vis.get_render_option()
    opt.background_color = hex2rgb(PAGE)
    opt.mesh_show_back_face = True

    ctr = vis.get_view_control()
    # FOV al mínimo (5 grados) = casi ortográfico. Con los 60 por defecto y la cámara
    # cerca, el relieve se deformaba y parecía un cañón.
    ctr.change_field_of_view(step=-90)
    frames = []

    for i in range(TOTAL):
        if i < ACT1:
            act, wipe, cols = 0, 0.0, depth_rgb
        elif i < ACT1 + ACT2:
            act = 1
            wipe = (i - ACT1 + 1) / ACT2
            front = xlo + wipe * (xhi - xlo)
            cols = np.where((v[:, 0] < front)[:, None], sss_rgb, depth_rgb)
        else:
            act, wipe, cols = 2, 1.0, sss_rgb

        mesh.vertex_colors = o3d.utility.Vector3dVector(cols)
        vis.update_geometry(mesh)

        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_front(orbit_front(-70 + 360.0 * i / TOTAL))
        ctr.set_zoom(0.48)

        vis.poll_events()
        vis.update_renderer()

        img = (np.asarray(vis.capture_screen_float_buffer(True)) * 255).astype(np.uint8)
        frames.append(caption(img, act, wipe, cover))

        if i % 20 == 0:
            print(f"[fusion] frame {i}/{TOTAL}")

    vis.destroy_window()

    gif = os.path.join(out, "07_mb_sss_fusion.gif")
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save(gif, save_all=True, append_images=pil[1:],
                duration=int(1000 / FPS), loop=0, optimize=True)

    print(f"[fusion] listo -> {gif}  ({len(frames)} frames, {W}x{H})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
