# acoustic_postprocessing

Paquete ROS (catkin, Python puro) del workspace `derelictes_ws`. Genera los
**productos cartográficos directos** de una misión del AUV **Sparus II**: nube
batimétrica, malla, mosaico de backscatter multihaz, mosaico sidescan y las
fusiones entre ellos, **proyectando cada ping con la navegación del INS** (sin
registro entre barridos).

> **No confundir con `multibeam_SLAM`** (paquete hermano). Ese corrige la deriva
> del INS con un grafo de poses; **este no**: proyecta directamente con la
> navegación bruta. Si buscas ICP, loop closure o grafo, estás en el repo
> equivocado. Los dos comparten la convención de ejes, el TF del sensor y la
> corrección AVG, y ese parecido es intencionado — pero los `results/` son
> distintos y no deben mezclarse.

## Qué produce, en una frase

Cinco productos georreferenciados en **UTM 31N (EPSG:32631)**, todos en el mismo
frame, para que se puedan superponer y fusionar: DEM + malla (geometría), dos
mosaicos de backscatter (MB y SSS), y la malla texturizada con el sidescan.

## Entorno

- **ROS Noetic**, **Python 3.8.10**. Paquete catkin de Python (sin C++).
- Versiones probadas: `open3d 0.13.0`, `numpy 1.24.4`, `rasterio 1.3.11`,
  `scipy`, `opencv`, `pyproj`, `pandas 2.0.3` (opcional, solo métricas),
  `matplotlib` (Agg) y `Pillow` (solo scripts de media).
- **`use_sim_time` debe quedarse en `false`.** Los bags se **leen** (no se
  reproducen), así que no hay `/clock`: las esperas de los nodos de fusión se
  colgarían para siempre.
- Sin dependencia de `sparus2_description`: los TF estáticos se leen **del propio
  bag** (`/tf_static`).

## Cómo se ejecuta

```bash
roslaunch acoustic_postprocessing acoustic_pipeline.launch
```

Todo se configura como `<arg>` en
[launch/acoustic_pipeline.launch](launch/acoustic_pipeline.launch) sin tocar
código: bags de entrada, `output_dir`, los seis `enable_*` (uno por producto) y
todos los umbrales.

**Nunca lances dos `roslaunch` de este pipeline a la vez**: los nodos de fusión
son anónimos pero los productores no, y el segundo lanzamiento mata los nodos del
primero a mitad de escritura.

Tópicos esperados en el bag:
- MB scan: `/sparus2/norbit_wbms_multibeam/multibeam_scan` (PointCloud2 con campo
  `intensity`)
- SSS: cualquier tópico con `sidescan` en el nombre y tipo `Image`; la
  configuración (rango por canal) en `.../raw_data/<side>/sss_info` (`SSSConfig`)
- Nav: `/sparus2/navigator/navigation` (north/east/depth, roll/pitch/yaw,
  `altitude` y `origin` lat/lon)

## Los seis nodos y cómo se sincronizan

| Nodo (`enable_*`) | Script | Produce | Espera a |
|---|---|---|---|
| `enable_mb_cloud` | `multibeam_processor.py` | `pointcloud/mb_pointcloud.xyz`, `tif/mb_pointcloud.tif`, `images/mb_pointcloud.jpg`, `mesh/mb_mesh.ply` | — |
| `enable_mb_intensity` | `multibeam_intensity.py` | `tif/mb_intensity.tif`, `images/mb_intensity.jpg`, `pointcloud/mb_intensity.xyz` | — |
| `enable_sss_mosaic` | `sss2mosaic.py` | `tif/sss_mosaic.tif`, `images/sss_mosaic.png` | — |
| `enable_sss_waterfall` | `sss_waterfall.py` | `images/sss_waterfall.png` | — |
| `enable_mosaic_fusion` | `mb_sss_mosaic_fusion.py` | `tif/mb_sss_mosaic.tif` | `mb_intensity_done` + `sss_done` |
| `enable_fusion` | `sss_mb_fusion.py` | `mesh/mb_textured_sss.ply` | `mb_done` + `sss_done` |

La sincronización es por **tópicos latched `std_msgs/Bool`**:
`/pipeline/mb_done`, `/pipeline/mb_intensity_done`, `/pipeline/sss_done`,
`/pipeline/mosaic_fusion_done`. Los dos nodos de fusión esperan `wait_timeout`
segundos (1800 por defecto) y, si vence, **caen a leer lo que haya en disco**.

⚠️ `mb_done` se publica **después del Poisson** (~9 min). Si matas el launch antes,
la señal nunca llega y `fusion_node` se queda esperando hasta el timeout.

## La geometría común (el invariante del paquete)

`multibeam_processor.py` y `multibeam_intensity.py` **deben** aplicar exactamente
la misma cadena de transformaciones, o la nube batimétrica y el mosaico de
intensidad dejan de compartir puntos:

1. **Flip de ejes** `(x, -y, -z)` para casar con el convenio del TF del sensor.
2. **Rotación del sensor** `R_MB` (de `/tf_static`, `base_link → multibeam`).
3. **Rotación del vehículo** `R_veh(roll, pitch, yaw)`, euler `sxyz`.
4. **Recorte angular** (`angle_cutoff`, 55°) medido en el **frame mundo**, no en
   el del sensor: con el vehículo escorado, cortar en el frame del sensor deja
   pasar haces a 55°+roll de la vertical real.
5. **Lever-arm MB→SSS**: `sss_center - mb_offset + [0, mb_sss_extra_offset_y, 0]`.
6. **Traslación a mundo local** (north, east, −depth) y **a UTM**
   (X=easting←east, Y=northing←north).

Cualquier cambio en uno de los dos scripts hay que replicarlo en el otro. Los
parámetros de actitud (`max_roll_deg`, `max_yaw_rate_deg_s`, `roll_bias_deg`,
`angle_cutoff*`, `mb_sss_extra_offset_y`) se pasan a **ambos nodos desde el mismo
`<arg>`** precisamente por eso.

## Decisiones no obvias (no las deshagas sin leer esto)

**Gate por actitud (`max_roll_deg=5`, `max_yaw_rate_deg_s=8`).** En un viraje el
swath se proyecta como un abanico inclinado que no casa con las pasadas vecinas;
no hay arreglo a posteriori sin registrar barridos (eso es el SLAM), así que se
tira **el ping entero**. Se umbraliza la **excursión sobre el sesgo de roll**
(mediana del bag, ~+2° de trim de montaje), no el roll absoluto: un roll constante
lo modela bien la matriz de rotación. Medido en el bag 13_52_22, la fracción de
puntos fuera de la superficie de referencia pasa de 2.1% (|roll−bias|<1°) a 33.7%
(>12°). **No es bala de plata**: quita ~30% de los puntos malos a costa de ~7% de
cobertura; el otro 70% viene de pings con actitud nominal y lo caza
`surface_relative_filter`.

**`surface_relative_filter` en vez de solo SOR.** El SOR estadístico mira la
distancia a los k vecinos y **no ve** los picos verticales del multihaz (haces con
rango erróneo, multipath, curl de borde), que aparecían como "montañas"
fantasma en CloudCompare. El filtro tira los puntos cuya Z se aparta más de
`n_mad·MAD` de la **mediana de su celda XY** — adaptativo, así que el relieve
rocoso real sobrevive. Itera porque un cluster de picos sesga la mediana de su
propia celda en la primera pasada.

**El orden voxel → SOR importa.** Al revés, el SOR construye su KD-tree sobre los
~31 M puntos crudos: lentísimo y pico de RAM. A 1 cm no se pierde nada útil.

**El SOR de Open3D NO es determinista, y eso condiciona cómo se verifica.**
`remove_statistical_outlier` devuelve un conjunto ligeramente distinto en cada
ejecución **con la misma entrada** (medido sobre una nube sintética de 6 M puntos, tres
procesos: 2.077.026 / 2.077.110 / 2.076.492 puntos; `voxel_down_sample`, en cambio, sí
es determinista). Consecuencias:

- `mb_pointcloud.{xyz,tif,jpg}` y `mb_mesh.ply` **no son reproducibles bit a bit**: entre
  dos corridas idénticas varían ~0.01% de los puntos (en el DEM, ~21 de 175.488 celdas,
  con el 99.988% de las Z exactamente iguales). **No es un bug ni una regresión.**
- `mb_intensity.xyz` **sí** sale bit-idéntico, porque no pasa por el SOR.
- Para validar un cambio en `multibeam_processor.py`, **no uses el md5**: compara las
  cuentas por compuerta del log (pings leídos / tirados por roll / por yaw-rate /
  válidos, y raw→voxel→SOR→surface), las celdas con dato del DEM y el p99 de |ΔZ|.

**Poisson `depth=10`, no 11.** Con 31 M puntos, `depth=11` se comió 55 GB y provocó
el OOM-kill. A voxel 0.05 m, 10 sobra.

**La malla se guarda sin color.** Poisson devuelve un array de color por vértice
**todo a cero** cuando la nube de entrada no tiene color, y CloudCompare lo respeta
→ malla completamente negra. Se borra el array para que sombree por normales. El
color pertenece a `mb_textured_sss.ply`.

**AVG (Angle Varying Gain), dos sabores distintos a propósito:**
- **MB** (`multibeam_intensity.py`): perfil = **mediana** por bin angular. Robusto
  a la estructura del fondo y a los outliers especulares.
- **SSS** (`sss2mosaic.py`): perfil = **media**, y con **un perfil por banda**
  (port/stbd son transductores distintos) sobre una **escala común**. La mediana
  **no funciona** aquí: el eco crudo del sidescan está saturado de ceros (47.5% de
  las muestras son exactamente 0), la mediana por bin sale 0–1 y mide *relleno*, no
  ganancia → el perfil sale plano y la corrección es un no-op. Además la ganancia
  se **acota por abajo** (`gmax/max_boost`): el perfil cae ~700× de nadir a rango
  lejano y dividir sin tope amplificaría 700× el ruido de la cola.

**`mb_intensity.tif` es 1 banda con paleta viridis embebida, no RGB.** A propósito:
la banda sigue siendo el **valor** de backscatter, que es lo que consume
`mb_sss_mosaic_fusion.py` (`src.read(1)`). Con 3 bandas RGB leería el canal rojo de
viridis como intensidad. QGIS/GDAL respetan la paleta y lo pintan en color igual.

**`0` está reservado para nodata en los mosaicos.** El CLAHE de `enhance_data()`
sube el fondo vacío de 0 a ~4, con lo que 0 dejaba de significar "sin dato" (el
histograma del .tif salía con mediana 4: el falso "mosaico bimodal casi negro",
donde la mediana medía el FONDO, no el fondo marino). Tras realzar, las celdas
vacías vuelven a 0 y las llenas se fuerzan a ≥1 — que es lo que esperan las
fusiones de su `> 0`.

**Rango del sidescan: se lee del bag, no se hardcodea.** El viejo
`SONAR_RANGE = 30.0` era **falso** para los bags de Andratx (rango real 50 m):
colocaba cada muestra al 60% de su rango verdadero, comprimiendo el mosaico ×0.6
across-track. Ahora sale de `SSSConfig.range` (vía `common.get_sonar_range`); el
parámetro `sss_sonar_range` solo se fija a >0 si el bag no lleva `SSSConfig`.

**El orden de muestras del sidescan está ESPEJADO entre canales.** Medido sobre el
bag 10_44_27 (perfil medio por índice, 400 pings):

| canal | pico del perfil | orden crudo |
|---|---|---|
| starboard | muestra 1/2000 | nadir → lejos |
| **port** | muestra **1998/2000** | **lejos → nadir** |

Por eso **el ping de babor hay que invertirlo antes de cualquier cuenta de rangos**
(`common.slant_to_ground` asume índice 0 = nadir). Si no, el rango 0 cae en la
muestra de rango lejano y el recorte de zona ciega borra el extremo equivocado.
`sss_waterfall.py` no lo hacía, y el bug **se escondía**: como port trae el nadir al
final, su bloque queda pegado a la franja central del `hstack` y la imagen *parecía*
correcta. El waterfall invierte para la geometría y **vuelve a invertir solo para
pintar**.

**`mb_sss_extra_offset_y = -2.0` es deuda técnica, no física.** Es un desplazamiento
**empírico** de 2 m para que la nube MB cuadre con el mosaico SSS. Está expuesto como
parámetro (y no como constante) precisamente porque en un producto georreferenciado
mueve **todo** 2 m. La causa real (lever-arm mal medido, desfase de reloj entre bags,
o error del propio SSS) sigue sin diagnosticar. Ver TODO.md.

## El módulo común (`scripts/common.py`)

**Todos los helpers de bag/TF/nav/ráster viven aquí, y solo aquí.** Antes cada script
llevaba su copia (3× `get_static_transform_from_tf`, 3× `get_nav_origin`, 3×
`enhance_data`, 3× la construcción de interpoladores) y **cada arreglo había que
hacerlo tres veces — en la práctica nunca se hacía**: el rango del sonar se leía del
bag en `sss2mosaic.py` pero seguía hardcodeado en `sss_waterfall.py`, los timestamps
se ordenaban en `sss2mosaic.py` pero no en los dos MB, y la guarda NaN solo existía
en `multibeam_intensity.py`. **Si añades un helper que use más de un script, va aquí.**

El import es de nombre plano y **requiere** esta línea antes, en cada script:

```python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
```

Sin ella `rosrun` falla: `catkin_install_python` genera un wrapper en `devel/lib/` que
hace `exec()` del fuente **sin** su directorio en `sys.path` (aunque sí apunta
`__file__` al fuente, que es lo que hace funcionar el truco). `common.py` **no** se
instala en el `CMakeLists.txt`, y no hace falta. Mismo patrón que `multibeam_SLAM`.

**Dos opciones existen para NO homogeneizar comportamientos, no las quites:**
- `nav_interpolators(smooth_yaw_sigma=...)`: solo el SSS suaviza el yaw (sigma=2), y
  solo los MB piden `with_yaw_rate`. Suavizar el yaw del multihaz cambiaría la nube.
- `enhance_data(nodata_mask=...)`: los mosaicos pasan máscara (0 = sin dato); el
  waterfall **no**, porque allí el 0 es una muestra real del eco. Verificado: con y sin
  máscara difieren en el 23.6% de los píxeles.

## Mapa de scripts (`scripts/`)

**Pipeline (los 6 que instala el `CMakeLists.txt`):**
- **`multibeam_processor.py`** — pings MB → nube UTM → voxel → SOR →
  `surface_relative_filter` → `.xyz` + DEM ráster (mediana de Z por celda) + malla
  Poisson.
- **`multibeam_intensity.py`** — misma geometría, pero se queda con el backscatter:
  AVG → nube `X Y Z I` + mosaico (mediana por celda) con paleta viridis.
- **`sss2mosaic.py`** — sidescan → corrección slant-range (fondo plano a altitud
  `h`) → AVG por banda → acumulación (**media** por celda) en rejilla UTM.
- **`sss_waterfall.py`** — waterfall ping a ping, sin georreferenciar (port=rojo,
  stbd=verde).
- **`mb_sss_mosaic_fusion.py`** — remuestrea los dos mosaicos a una rejilla común y
  los mezcla (`max` | `mean` | `mb` | `sss`).
- **`sss_mb_fusion.py`** — textura la malla MB con la intensidad del mosaico SSS
  (geometría del multihaz, color del sidescan).

**Herramientas (NO instaladas por catkin — se ejecutan con `python3`):**
- **`mb_projection_metrics.py`** — métricas de calidad de la proyección (rugosidad
  por celda, cobertura, histograma de backscatter). Solo numpy/matplotlib.
- **`make_media.py`** — figuras y GIFs de divulgación. Contiene las decisiones de
  color (documentadas en su docstring) y los helpers `load_raster`/`crop_to_data`
  que **importan los otros scripts de media**.
- **`make_nav_cache.py`** — extrae las trayectorias de los **dos** bags a UTM y escribe
  `results/media/.nav_cache.npz` (`tm,xm,ym,zm` / `ts,xs,ys,zs`). **Ejecútalo primero**:
  el resto de la media lo lee y no arranca sin él. Sus bags por defecto son los de la
  media (MB **13_52_22**, SSS 10_44_27), que **no** son los del launch (octógono
  12_52_41) — los rótulos de `make_media.py` dicen "Multibeam 13:52".
- **`make_anim_data.py`**, **`make_pipeline_animation.py`**, **`make_fusion_gif.py`**
  — animación del pipeline (los dos primeros van encadenados) y GIF de fusión 3D
  con Open3D. Todo entra y sale de `results/media/`.

## Salidas (`results/`)

`results/` está **en `.gitignore`**. Layout **por tipo de fichero**, no por sensor
(ver [results/README.md](results/README.md)): `tif/`, `images/`, `pointcloud/`,
`mesh/`, `metrics/`, `media/`.

Todo lo que `results/` contiene es **regenerable**: es seguro borrarlo entero. Las
**entradas** de `make_media.py` viven fuera, en `assets/` (`sparusII.png` versionado;
`assets/mosaics/`, 178 MB de fotomosaicos estéreo, ignorado por git pero a salvo de un
`rm -rf results/`). Si faltan los mosaicos, las dos figuras que los usan degradan solas.

## Convenciones del código

- **Comentarios en inglés, extensos, que justifican el porqué** de cada decisión no
  obvia — con frecuencia citando la medida que la motivó ("el 47.5% de las muestras
  son 0", "55 GB de RAM"). Son la memoria del proyecto sobre qué se probó y por qué
  se descartó: **consérvalos y actualízalos** al editar.
- Docstrings de módulo con **Outputs** y **Usage** explícitos.
- **No copies un helper entre scripts: ponlo en `scripts/common.py`.** Copiarlo es
  exactamente cómo se generaron los bugs que se cerraron el 2026-07-16 (rango
  hardcodeado en un script y leído del bag en otro, nav sin ordenar en dos de tres,
  guarda NaN en uno solo).
- Configuración vía `rospy.get_param('~...', default)`; al añadir un parámetro,
  expón también su `<arg>` en el launch **y pásalo a los dos nodos MB si afecta a la
  geometría**. El default del código debe **coincidir con el `<arg>` del launch**, o
  ejecutar el script suelto da otro resultado.
- `float()` explícito al leer params numéricos: roslaunch entrega `"NaN"` como
  **string** (su conversión automática solo intenta float si el valor lleva un `.`).
- Los scripts de media importan de `make_media.py` (`from make_media import ...`),
  así que se ejecutan **desde `scripts/`** con `python3`, no con `rosrun`.
</content>
</invoke>
