# acoustic_postprocessing

Paquete ROS (catkin) de postproceso acústico submarino del workspace
`derelictes_ws`. Reconstruye un **mapa 3D batimétrico** del fondo marino a partir
de los datos grabados (rosbag) de un AUV **Sparus2** con sonar **multihaz (MBES,
Norbit WBMS)**, corrigiendo la deriva de la navegación inercial mediante un
**SLAM basado en grafo de poses**.

Hay un documento de presentación detallado (problema, motivación, decisiones de
diseño) en [DOCUMENTACION_PRESENTACION.md](DOCUMENTACION_PRESENTACION.md). Léelo
para el "por qué"; este fichero cubre el "qué/cómo" operativo.

## El problema en una frase

Bajo el agua no hay GPS: la pose viene de INS (DVL + IMU) y **deriva** con la
distancia. Proyectar cada barrido con la navegación bruta da un mapa borroso y
duplicado. El SLAM alinea entre sí las nubes del sonar para reconstruir un mapa
coherente, **en postproceso**.

## El reto técnico central: el fondo plano

Gran parte del fondo marino es **plano y sin relieve**. Sobre fondo plano el ICP
geométrico no tiene gradiente en XY (todas las normales apuntan hacia arriba) →
"desliza" y converge a mínimos locales falsos (pasos invertidos 180°,
perpendiculares) que **igualmente pasan** los filtros de fitness/RMSE porque
"plano contra plano" siempre encaja. **Todo el pipeline está diseñado con
salvaguardas (gates) contra este escenario degenerado** — es la aportación
técnica principal y el motivo de casi cada decisión no obvia del código.

Idea de fondo: **confiar en el INS para lo que hace bien** (rotación, longitud
del paso, rumbo) **y en el sonar solo para la corrección fina XY** donde aporta
información fiable. Cuando un registro se rechaza, se cae a la navegación
(`T_init`) con peso de información bajo, sin contaminar el grafo.

## Entorno

- **ROS Noetic**, **Python 3.8**. Es un paquete catkin Python puro (sin C++).
- Dependencias runtime: `open3d`, `numpy`, `scipy`, `tqdm`, `matplotlib` (Agg),
  `rospy`, `rosbag`, `ros_numpy`, `tf`. Opcional `plyfile` (solo para el
  generador de media).
- `use_sim_time` **debe permanecer en `false`** — en `true` las esperas del
  pipeline se cuelgan.
- El launch incluye `sparus2_description` (paquete hermano) para los TF estáticos
  base_link → multibeam.

## Cómo se ejecuta

```bash
roslaunch acoustic_postprocessing acoustic_pipeline.launch
```

El bag de entrada y **todos** los parámetros (algoritmo de registro, gates,
umbrales, monitor) se configuran como `<arg>` en
[launch/acoustic_pipeline.launch](launch/acoustic_pipeline.launch) sin tocar
código. El nodo activo es `multibeam_slam.py`; los nodos de sidescan (mosaico,
waterfall, fusión SSS+MBES) están **comentados** en el launch (trabajo futuro de
fusión multi-sensor; los scripts correspondientes no existen aún en `scripts/`).

Tópicos esperados en el bag:
- scan: `/sparus2/norbit_wbms_multibeam/multibeam_scan` (PointCloud2 con campo
  `intensity` opcional)
- nav: `/sparus2/navigator/navigation` (north/east/depth + roll/pitch/yaw)

## Mapa de los scripts (`scripts/`)

- **`multibeam_slam.py`** — núcleo del pipeline (nodo ROS, `main()`). Contiene
  toda la configuración por defecto como constantes en MAYÚSCULAS (sobrescritas
  por params del launch), `PatchBuilder`, `NavigationInterpolator`, el bucle de
  registro secuencial, el de loop closure, la optimización global, la
  restauración vertical y el guardado de mapas/métricas/plots.
- **`utils.py`** — transformaciones SE(3): `pose_dict_to_matrix`,
  `expected_transform` (prior INS relativo), `ins_rotation_icp_translation`
  (clave: rotación 3D del INS + traslación XY del ICP, con gate de longitud y
  amortiguación cross-track), `constrain_transform`.
- **`robust_icp.py`** — `robust_icp` (GICP multiescala coarse-to-fine + Tukey),
  `robust_colored_icp` (geometría + intensidad acústica), `robust_hybrid_icp`
  (adaptativo: geometría donde hay relieve, intensidad donde el fondo es liso,
  decidido por fracción de normales no verticales). Caché LRU de preprocesado.
- **`robust_ndt.py`** — implementación propia de NDT 2D (least_squares soft_l1).
- **`registration.py`** — registro global RANSAC (FPFH) para sembrar el loop
  closure. Caché LRU de FPFH por nube.
- **`scan_context.py`** — descriptor Scan Context (rejilla polar
  anillos×sectores) + `ScanContextManager` con pre-filtro espacial KD-tree por
  posición INS.
- **`information_matrix.py`** — matriz de información (peso) dinámica por arista
  según fitness/RMSE.
- **`visualization.py`** — `build_final_map`, extracción de trayectoria,
  `PoseGraphMonitor` (ventana Open3D en vivo durante el run).
- **`make_presentation_media.py`** — genera GIF/MP4 de presentación a partir de
  `results/` (solo numpy+matplotlib, no necesita ROS ni Open3D).

## Las cinco etapas (resumen del flujo en `main()`)

1. **PatchBuilder** — agrupa `PATCH_SIZE=100` barridos en una nube local,
   avanzando `PATCH_STRIDE=20` (solape 80%). Proyecta sensor→vehículo→global→
   recentrado, recorta haces > `ANGLE_CUTOFF_DEG=55°`, aplica **corrección AVG**
   del banding de intensidad por ángulo de incidencia, submuestrea a voxel 0.25 m
   y estima normales.
2. **Registro secuencial** — alinea patch `i` con `i-1`. Aquí están casi todos
   los gates anti-fondo-plano (ver abajo).
3. **Loop closure** — Scan Context + pre-filtro KD-tree por proximidad INS +
   RANSAC seed + gates anti-falso-positivo.
4. **Optimización global** — pose graph de Open3D (Levenberg–Marquardt) reparte
   el error de forma coherente.
5. **Restauración vertical** — tras optimizar, se reconstruye roll/pitch/Z desde
   el INS conservando el yaw optimizado (el fondo plano deja el optimizador sin
   restricciones verticales → metía roll/pitch/Z espurios que "engordan" el mapa).

## Gates clave (no tocarlos sin entender el porqué)

Cada gate existe para atajar un fallo concreto del fondo plano, documentado en
extenso en los comentarios de `multibeam_slam.py`. Resumen:

**Secuencial:**
- fitness / RMSE / nº correspondencias (calidad básica).
- **Desviación traslación vs INS** (`MAX_SEQ_ICP_TRANSLATION_DEV=0.5 m`) y
  **dirección vs INS** (`MAX_SEQ_ICP_YAW_DEV=15°`): cazan saltos y pasos
  invertidos/perpendiculares.
- **Banda de ratio de longitud** `[0.85, 1.15]`: el solape 80% sesga el paso ICP
  a ~8% más corto y comprime la trayectoria; fuera de banda se reescala a la
  longitud del INS.
- **Rotación INS + traslación ICP** (`ins_rotation_icp_translation`): la arista
  usa la rotación 3D fiable del INS; la rotación del ICP en fondo plano es ruido.
- **Amortiguación cross-track** (`SEQ_CROSS_TRACK_GAIN=0.0`): anula la componente
  lateral del ICP (sesgo sistemático hacia +East).

**Loop closure:**
- **Proximidad INS** (`MAX_LOOP_INS_DISTANCE=12 m`) — también pre-filtra el Scan
  Context vía KD-tree (de O(N²) a casi lineal).
- **Consistencia ICP–deriva INS** (`MIN_LOOP_ICP_INS_RATIO=0.40`): un cierre real
  corrige la deriva (ratio≈1); un falso positivo sobre fondo plano da ratio≈0.
- Límites Z / XY / yaw del cierre.

## Salidas (`results/`)

`results/` está **en `.gitignore`** (no se versiona). El pipeline genera:
- Mapas: `raw_navigation_map.ply` (solo nav bruta) y `slam_optimized_map.ply`.
- Trayectorias: `raw_trajectory.npy`, `slam_trajectory.npy`.
- `results/metrics/`: `slam_metrics.json` (resumen), `icp_stats.csv`,
  `loop_stats.csv`, `stage_times.json`, y PNGs (`trajectory_comparison.png`,
  `registration_quality.png`, `slam_correction.png`, `rmse_histogram.png`,
  `loop_closure_quality.png`).
- `results/presentation/`: GIF/MP4 generados por `make_presentation_media.py`.

## Convenciones del código

- Estilo muy espaciado (una expresión por línea en muchas llamadas) y
  **comentarios extensos en español** que justifican cada decisión técnica
  (sobre todo los gates). Al editar, **conserva y actualiza esos comentarios** —
  son la memoria del proyecto sobre qué se probó y por qué se descartó.
- Las constantes de configuración viven al principio de `multibeam_slam.py` y se
  sobrescriben con `rospy.get_param("~...", DEFAULT)`; al añadir un parámetro,
  expón también su `<arg>` en el launch.
- `intensity` (backscatter) viaja en el canal de color (gris) de las nubes
  Open3D para que el Colored ICP la use; el voxel_down_sample promedia el color.
