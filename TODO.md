# acoustic_postprocessing — Tareas / mejoras propuestas

Análisis de los scripts del paquete. Mejoras agrupadas por tema y priorizadas.
Cada ítem indica **fichero**, **problema** y **propuesta**.

Prioridad: 🔴 alta (corrección / fiabilidad) · 🟡 media (robustez / calidad de
resultados) · 🟢 baja (mantenibilidad / rendimiento / cosmético).

Estado: ✅ completada · ⬜ pendiente.

---

# 🔬 Línea SOTA (ver background/PROPUESTA_SLAM_SOTA.md)

Plan basado en la revisión del estado del arte (Palomer 2016, Barkby 2009/2011,
Torroba 2020/SVGP, Tan 2022, Teng 2020) para el escenario real: **pasadas paralelas
sin cruces**. Fases R0→R3 (ver §5bis de la propuesta).

- ✅ 🔴 **R0.1 — Métrica de consistency error (Roman 2006)** *(2026-06-26)*.
  Nuevo módulo `scripts/consistency.py` (núcleo unit-testeable + adaptador
  `consistency_error_from_patches`). Cableado en `main()`: calcula el error ANTES
  (nav bruta) y DESPUÉS (grafo optimizado), lo vuelca a `metrics["consistency"]` y lo
  loguea. Param `~consistency_cell_size` (default 1.0 m) expuesto en el launch. Es el
  instrumento de medida primario del SOTA (dispersión vertical en zonas de solape,
  incl. solape adyacente entre franjas). Validado con casos sintéticos. **Falta
  re-lanzar el pipeline** para obtener el primer número real sobre Cabrera.
- ✅ 🔴 **R0.2 — Restricción de solape adyacente entre franjas paralelas** *(2026-06-26)*.
  Nueva fase `cross_track` en `main()` (tras loop closure, antes de la optimización):
  KD-tree espacial sobre las posiciones INS busca pares de franjas VECINAS (banda de
  distancia INS `[xtrack_min/max_ins_dist]`, gap temporal `xtrack_min_temporal_gap`),
  los registra sembrando con el prior INS (`expected_transform`) y añade aristas
  `uncertain=True` si son coherentes (fitness + discrepancia vs prior). Es el corazón
  de Torroba 2020 (corrige deriva entre líneas sin cruces). 7 params en el launch.
  Validado sobre la trayectoria real: selecciona 443 pares (franjas a ~7 m, gap≥40).
- ✅ 🔴 **R0.3 — Registro/grafo 3-DoF gravity-constrained nativo** *(2026-06-26)*.
  Nueva `utils.project_to_3dof()` (yaw+XY del registro, Z+roll/pitch del prior INS).
  La arista cross-track la usa; documentada la restauración vertical como el componente
  3-DoF coherente del back-end. Valida la decisión del usuario (Torroba/Tan: 3-DoF>6-DoF).
- ✅ 🔴 **R1 — pICP: información de arista anisótropa** *(2026-06-26)*. Nuevo módulo
  `scripts/registration_covariance.py` (covarianza 3-DoF tipo Censi 2007 desde el
  Hessiano del coste punto-a-plano). Helper `edge_information()` reemplaza el
  `dynamic_information_matrix` isótropo en las 3 aristas (secuencial/loop/cross-track).
  En fondo plano da información baja cross-track (resuelve el deslizamiento de raíz).
  Validado: ratio cov plano/relieve = 1.4e7. Param `~use_registration_covariance`.
- ✅ 🔴 **R2 — Back-end robusto (line process de Choi 2015 ≈ Switchable Constraints)**
  *(2026-06-26)*. Activado el line process de Open3D vía `preference_loop_closure`
  (0.6, más escéptico con loops/cross-track) + `edge_prune_threshold`. Desactiva
  automáticamente aristas espurias (falsos positivos de franja paralela) sin gates
  manuales. Diagnóstico `metrics["robust_backend"]`. Validado: un loop falso NO
  colapsa el grafo. Params en el launch.
- ✅ 🟡 **R3 — Backscatter en el modelo de incertidumbre (aportación novel)**
  *(2026-06-26)*. `intensity_informativeness()` (entropía×dispersión del canal de
  intensidad) + `fuse_geometry_intensity_cov()`: donde la geometría es plana pero el
  backscatter tiene textura, reduce la incertidumbre cross-track. Cableado en
  `edge_information`. Ni Palomer ni Torroba/Tan usan intensidad en su covarianza →
  es el diferencial publicable. PointNetKL queda como evolución (misma interfaz).
  Validado: texturado 1000→50, uniforme 1000→992. Params en el launch.

- ✅ 🟡 **Script de ablación** *(2026-06-26)*. `scripts/run_ablation.py` lanza el
  pipeline con las combinaciones de flags (R0 → R1 → R1R2 → FULL), recopila el
  consistency error de cada `slam_metrics.json` y produce tabla + `ablation_summary.csv`
  en `results/ablation/`. Vigila la aparición de slam_metrics.json para terminar cada
  roslaunch sin esperar timeout (el nodo no es `required`). Añadido a
  `catkin_install_python`. Uso: `rosrun acoustic_postprocessing run_ablation.py`
  (requiere `catkin_make`/`catkin build` para instalar el nuevo script).

> **Pendiente común a R0–R3: re-lanzar el pipeline sobre Cabrera** para medir el
> consistency error real y hacer la ablación (R0 baseline → +R1 → +R2 → +R3). Toda la
> lógica está validada unitariamente; falta el número real (requiere ROS + bag).
> Lanzar la ablación: `rosrun acoustic_postprocessing run_ablation.py` (o
> `--dry-run` para ver los comandos sin ejecutar).
>
> **Futuro (R4/R5):** PointNetKL (red que aprende la covarianza con backscatter),
> representación SVGP del mapa, y descriptor de lugar aprendido (cuando haya un dataset
> CON cruces de trayectoria).

---

# ✅ Completadas

### Empaquetado y configuración (catkin)

- ✅ 🔴 **`CMakeLists.txt`: `catkin_python_setup()` duplicado** → eliminado el
  duplicado. *(2026-06-25)*
- ✅ 🔴 **Los scripts no se instalaban** → añadido `catkin_install_python` para
  `multibeam_slam.py` y `make_presentation_media.py` (los únicos ejecutables; el
  resto son módulos importados, expuestos vía `catkin_python_setup`). Verificado
  con `cmake` aislado: instala los wrappers en `lib/acoustic_postprocessing`.
  *(2026-06-25)*
  - ⚠️ **Efecto secundario corregido (2026-06-25)**: el wrapper de catkin ejecuta
    el script vía `exec()` desde otra ruta, donde `scripts/` no está en `sys.path`
    → `from utils import *` daba `ModuleNotFoundError`. Solución: el script añade
    su propio directorio a `sys.path` al inicio
    (`sys.path.insert(0, dirname(abspath(__file__)))`), válido en ejecución
    directa y vía wrapper. (A futuro, lo idiomático sería convertir los módulos
    hermanos en un subpaquete Python con imports relativos — ver §8.)
- ✅ 🟡 **`find_package` desalineado con `package.xml`** → quitado `std_msgs`
  (no usado ni declarado), añadido `nav_msgs`, añadido `catkin_package()`.
  Confirmado que los scripts solo importan `rospy`/`ros_numpy`. *(2026-06-25)*
- ✅ 🟢 **`package.xml`: `description`/`license`** → descripción real + licencia
  `BSD` (ajustar si el grupo UIB usa otra). *(2026-06-25)*

### Coherencia de parámetros launch ↔ código

- ✅ 🔴 **Desajuste de `hybrid_min_texture`** → unificado a `0.08` en el launch
  (coincide con `HYBRID_MIN_TEXTURE` del código y el README). *(2026-06-25)*
- ✅ 🟡 **Parámetros no expuestos en el launch** → expuestos como `<arg>` +
  `<param>` + `get_param`: `patch_size`, `patch_stride`, `patch_voxel_size`,
  `final_downsample`, `angle_cutoff_deg`, `fitness_threshold`,
  `seq_rmse_threshold`, `min_correspondences`, `scan_context_threshold`,
  `max_loop_candidates`, `loop_fitness_threshold`, `loop_rmse_threshold`,
  `max_loop_z_translation`, `max_loop_xy_translation`, `max_loop_yaw_deg`,
  `monitor_update_every`. Guardas `max(1,...)` en patch_size/stride/monitor para
  evitar range vacío / división por cero. Log informativo de los valores
  efectivos al arrancar. *(2026-06-25)*
- ✅ 🟢 **`output_dir` ruta absoluta** → ahora
  `$(find acoustic_postprocessing)/results/`. *(2026-06-25)*
- ✅ 🟢 **`FINAL_DOWNSAMPLE=0.2` fijo** → expuesto como `final_downsample`
  (incluido en el lote de parámetros del launch). *(2026-06-25)*

> ⚠️ Nota derivada del cambio: `SEQ_RMSE_THRESHOLD`/`LOOP_RMSE_THRESHOLD` eran
> `VOXEL_SIZE*1.5`. Ahora son params independientes (default 0.375 ≈ 0.25*1.5).
> Si cambias `patch_voxel_size`, ajusta también los RMSE acorde (no se recalculan
> solos). Decidir si se prefiere reacoplar (RMSE derivado del voxel cuando no se
> pasa explícito) — ver §2 pendiente.

---

# ⬜ Pendientes

## 1. Empaquetado y configuración (catkin)

- ⬜ 🟢 **Versión Python en `setup.py`/shebangs.** Todo asume Python 3.8 (ROS
  Noetic). Está bien, solo documentarlo (ya en CLAUDE.md).

## 2. Coherencia de parámetros launch ↔ código

- ⬜ 🟢 **`bag_file` por defecto apunta a rutas absolutas concretas** (varias
  comentadas). Considerar un `args`/`rosparam` de dataset o un README de bags.
- ⬜ 🟢 **RMSE desacoplado del voxel** (ver nota de Completadas §2). Decidir si
  reacoplar `seq/loop_rmse_threshold` a `patch_voxel_size*1.5` cuando el usuario
  no los pasa explícitamente, en vez de defaults fijos.

## 3. Estructura del nodo principal (`multibeam_slam.py`, ~2970 líneas)

- ⬜ 🟡 **`main()` es un monolito de ~1500 líneas.** Hace lectura de params,
  construcción, registro secuencial, loop closure, optimización, restauración,
  guardado y plots. Extraer funciones: `run_sequential_registration(...)`,
  `run_loop_closure(...)`, `restore_vertical(...)`, `save_outputs(...)`. Mejora
  testabilidad y legibilidad sin cambiar comportamiento.
- ⬜ 🟡 **Config global mutada vía `global`.** `main()` reasigna ~25 constantes
  globales desde params. Funciona pero acopla estado. Encapsular la config en un
  `@dataclass SlamConfig` poblado desde params y pasarlo a las funciones.
- ⬜ 🟢 **Constantes "mágicas" repetidas.** `info = np.eye(6) * 0.01` (peso de
  fallback) aparece en 2 sitios; `min_points < 10/50` y `0.05` (umbral de
  longitud) dispersos. Centralizar como constantes nombradas.
- ⬜ 🟢 **El bag se abre dos veces** (`get_static_transform_from_tf` abre y
  cierra su propio `rosbag.Bag`, luego `main` abre otro). Aceptable, pero podría
  reusarse una sola apertura.

## 4. Robustez / correctness del pipeline

- ⬜ 🔴 **`NavigationInterpolator` usa `fill_value='extrapolate'` en TODAS las
  interpolaciones.** Combinado con `has_timestamp` (que recorta al rango) suele
  estar protegido, pero `pose_values` puede extrapolar en los bordes si el
  llamante no comprueba `has_timestamp` (p. ej. el centro del patch sí se
  comprueba, pero conviene revisar todos los usos). Riesgo: poses inventadas en
  los extremos del bag. Considerar `bounds_error` controlado o clip explícito.
- ⬜ 🟡 **`np.unwrap` sobre yaw/pitch/roll asume orden temporal monótono.** Los
  timestamps de navegación se asumen ordenados; si el bag trae mensajes
  desordenados, `interp1d` y `unwrap` darán resultados erróneos sin avisar.
  Ordenar por timestamp y deduplicar antes de construir los interpoladores.
- ⬜ 🟡 **Selección de fuente de timestamp (`_select_scan_time_source`) puede
  elegir un offset espurio.** Si el solape es bajo, hace un *shift* para
  maximizar solape; un bag con relojes muy desfasados podría alinear mal scans y
  navegación silenciosamente (solo un `logwarn`). Añadir un gate duro
  (abortar/avisar fuerte) si el mejor solape < umbral.
- ⬜ 🟡 **Eje de intensidad/AVG: `arctan2(x, |z|)` asume `x`=across-track.** El
  perfil AVG y el recorte por ángulo dependen de esa convención de ejes del
  sensor. Documentar/verificar contra el frame real del Norbit; si el montaje
  cambia, el banding no se corrige. Añadir una aserción o log del rango de
  ángulos observado.
- ⬜ 🟢 **`Patch.center` se calcula en el frame recentrado** (`mean(pts[:,:2])`,
  ≈0 por construcción) pero el nombre sugiere posición global. Renombrar a
  `local_centroid` o calcular el centro global real si se necesita en otro sitio.

## 5. Calidad del mapa / resultados

- ⬜ 🟡 **El PLY de salida guarda XYZ y normales como `double`.** El header del
  `slam_optimized_map.ply` actual usa `property double x/y/z/nx..nz` → ~7 MB para
  140k puntos. Escribir en `float32` (Open3D: pasar por buffers float o
  `write_ascii=False` con conversión) reduce el fichero ~2× sin pérdida
  perceptible para batimetría. Revisar si las normales hacen falta en el PLY
  final (si no, no exportarlas).
- ⬜ 🟡 **0 cierres de bucle aceptados** → analizado a fondo: NO es por RANSAC
  sobre fondo plano. **Ver §9**, la causa raíz es un gate mal formulado y es la
  mejora #1 del proyecto. (Este ítem queda subsumido por §9.)

## 6. Rendimiento

- ⬜ 🟡 **Caches por `id(pcd)` son frágiles.** `_PREP_CACHE` y `_FPFH_CACHE`
  usan `id(pcd)` como clave, válido solo porque los patches son persistentes
  durante el run. Si en el futuro se recrean nubes (deepcopy, reload), el `id`
  puede reciclarse y devolver una entrada de caché de OTRA nube → resultados
  silenciosamente corruptos. Documentado en el código, pero conviene una clave
  más segura (hash de puntos, o atributo `patch.cache_key`) o invalidar el caché
  explícitamente entre fases.
- ⬜ 🟢 **Construcción del grid NDT en Python puro** (`_build_ndt_grid` itera
  punto a punto con `setdefault`). Para `ndt`, vectorizar el agrupado por voxel
  (orden por clave de voxel + `np.split`) aceleraría notablemente. Baja prioridad
  porque `ndt` no es el algoritmo por defecto (`hybrid`).
- ⬜ 🟢 **El monitor Open3D re-añade/quita geometría cada update.** Para
  misiones largas con monitor activo, `remove_geometry`+`add_geometry` por frame
  es costoso. Usar `update_geometry` sobre objetos persistentes donde sea posible.

## 7. Reproducibilidad / observabilidad

- ⬜ 🟡 **No se persiste la configuración usada en cada run.** `slam_metrics.json`
  guarda resultados pero no los parámetros (gates, algoritmo, voxel...). Volcar
  un `run_config.json` con todos los params efectivos junto a las métricas para
  poder comparar runs y reproducirlos. *(Ahora más relevante: con §2 ya hay
  muchos más params configurables que conviene registrar por run.)*
- ⬜ 🟢 **`stage_times.json` no incluye loop-closure si está deshabilitado**
  (se mide 0). Correcto, pero documentar las claves esperadas.
- ⬜ 🟢 **Logging mezclado `loginfo`/`logwarn`.** Cada paso secuencial hace
  `loginfo` por iteración (ruidoso en bags largos: ~1500 líneas). Bajar a
  `logdebug` el log por-paso y dejar `loginfo` para resúmenes por fase.

## 8. Calidad de código / tooling

- ⬜ 🟢 **Sin tests.** Las funciones puras de `utils.py`
  (`ins_rotation_icp_translation`, `constrain_transform`, `expected_transform`,
  `wrap_angle_deg`) y de `scan_context.py` son unit-testeables sin ROS. Añadir
  `tests/` con pytest cubriendo los casos límite de los gates (paso invertido
  180°, ratio fuera de banda, cross-track gain 0/1).
- ⬜ 🟢 **`make_presentation_media.py`: parser PLY propio frágil.** Asume
  layout de propiedades fijo y `red` como color; ya hay fallback a `plyfile`.
  Documentar dependencia opcional `plyfile` en `package.xml`/README de scripts.
- ⬜ 🟢 **Sin `requirements.txt` ni pin de versión de Open3D.** El pipeline
  usa APIs que varían entre versiones de Open3D (`registration_generalized_icp`
  ya se detecta con `hasattr`). Fijar/documentar la versión probada.
- ⬜ 🟢 **Estilo muy espaciado e inconsistente** (una expresión por línea en
  unas zonas, normal en otras). No es un problema funcional; si se adopta un
  formateador (black/autopep8), hacerlo en un commit aislado para no contaminar
  los diffs de lógica.

## 9. Hallazgos del análisis de resultados (results/metrics)

Derivado de [results/metrics/ANALISIS_RESULTADOS.md](results/metrics/ANALISIS_RESULTADOS.md).
Tres ejes: (1) desajuste raw↔SLAM, (2) loop closure, (3) intensidad/hybrid.

### 9.1 — Loop closure: gate `icp_ins_ratio` mal formulado  🔴 (MEJORA #1)

**Causa raíz de los 0 cierres.** De 1929 candidatos, 495 pasan RANSAC con
fitness=1.0 y RMSE≈0.155 m (cierres casi perfectos, revisitas reales de franjas
del lawnmower) y **los 495 se rechazan por `icp_ins_ratio < 0.40`**. 479/495
pasarían todos los demás gates. El gate compara `|T_raw|` (marco LOCAL del patch,
recentrado → ≈0 para un buen cierre) con `ins_distance` (marco GLOBAL, 6–12 m):
magnitudes incomparables, el ratio tiende a 0 para los MEJORES cierres.

- ✅ 🔴 **HECHO + CORREGIDO (2026-06-25)** Gate de loop closure rediseñado tras
  un re-lanzamiento fallido. Historia completa:
  - 1er intento: gate solo-discrepancia (`||T_raw − T_ins_rel|| ≤ 8 m`). Al
    re-lanzar **colapsó el lawnmower** (corrección 1.3 → 16.8 m, ancho 49 → 15 m):
    aceptó 321 falsos positivos de **franja paralela** (el ICP las alinea con
    `|T_raw|≈0` espurio y el optimizador las fusiona). Error de análisis: `|T_raw|≈0`
    NO es buen cierre, es la firma del falso positivo.
  - Corregido: **gate de 3 condiciones AND** — (1) `ins_distance ≤
    MAX_LOOP_REVISIT_INS_DIST` (2.5 m, solo revisitas reales, no franjas
    paralelas), (2) `ratio ≥ MIN_LOOP_ICP_INS_RATIO` (0.40, restaurado del gate
    original; rechaza `|T_raw|≈0`), (3) `discrepancia ≤ MAX_LOOP_INS_DISCREPANCY`
    (8 m, cota superior). Todos configurables por launch. **Validado: acepta
    0/321 falsos positivos** del run fallido (correcto para lawnmower de pasada
    única). **Falta re-lanzar** para confirmar que restaura la corrección sana
    (~1.3 m) + la mejora de 9.2. Ver ANALISIS_RESULTADOS.md §2e/§2f.
- [ ] 🟡 Re-evaluar **sembrar el ICP con prior INS cuando RANSAC falla** (los
  1434 `ransac_failed`). Su descarte previo ("Fix B") se hizo con el gate roto
  activo; repetir el experimento tras 9.1.
- [ ] 🟡 Subir robustez de RANSAC donde haya estructura (iteraciones / fitness
  mínimo) o validar el cierre por solape real (inliers en zona común) en vez de
  por ratio.
- [ ] 🟡 **Scan Context con intensidad** además de `z` (hoy
  `np.maximum.at(desc,...,z)`). En fondo plano `z` no discrimina; la textura de
  intensidad sí → mejor detección de revisitas. (Sinergia con 9.3.)

### 9.2 — Deriva no acotada / gates secuenciales  🟡

La corrección crece con la distancia (corr=0.888), rampa sin retrocesos =
ausencia de cierres (lo arregla 9.1). Además, con relieve real, el 62 % de pasos
secuenciales cae a fallback INS pese a fitness≈1.0.

- ✅ 🟡 **HECHO (2026-06-25)** Gates `MAX_SEQ_ICP_TRANSLATION_DEV`/
  `MAX_SEQ_ICP_YAW_DEV` ahora **modulados por la textura geométrica del par**
  (`_geometric_texture`, min de source/target). Donde NO hay relieve (textura→0)
  el gate queda intacto (protección anti-fondo-plano preservada); donde sí hay
  (textura ≥ `SEQ_GATE_TEXTURE_FULL`=0.30) se relaja hasta
  `SEQ_GATE_TEXTURE_RELAX`=2.0× (0.5m/15° → 1.0m/30°). Configurable por launch;
  `relax=1.0` recupera el umbral fijo. Diagnóstico nuevo en JSON/CSV
  (`geometric_texture`, `icp_translation_dev_m`, `icp_direction_dev_deg`, y los
  umbrales efectivos). **Falta re-lanzar** para medir la subida real de la tasa
  de aceptación secuencial (hoy 38%).
- ✅ 🟢 **HECHO (2026-06-25)** Documentado en código (bloque de restauración
  vertical) que la corrección Z es 0 por diseño (Z=INS) y qué hacer si el INS de
  profundidad no fuera fiable (conservar la Z optimizada en vez de la del INS).

### 9.3 — Proyección de intensidad  🟡 / 🟢

La intensidad proyectada ya es buena (entropía 3.67/4.32 bits, 0.8 % sin
intensidad, sin saturación). En el dataset analizado el fondo tiene relieve
(textura 0.487 ≫ umbral 0.08) → el hybrid usa geometría casi siempre y la
intensidad apenas interviene. Las mejoras rinden **en fondo plano**:

- [ ] 🟡 **Corrección por rango/TVG** además del ángulo (AVG actual). El
  backscatter decae con el rango oblicuo (propagación+absorción); normalizar por
  un perfil ganancia-vs-rango quita gradientes radiales espurios.
- [ ] 🟡 **Ecualización local (CLAHE)** en vez de normalización global por
  percentiles 2–98, para realzar micro-textura del sedimento (la que da gradiente
  XY al Colored ICP en fondo liso).
- [ ] 🟢 AVG con ángulo de incidencia **corregido por pose** (roll/pitch reales),
  en vez de asumir fondo horizontal en `arctan2(across, depth)`. (Relacionado con
  §4 sobre el eje de intensidad.)
- [ ] 🟢 Filtrar outliers de backscatter (specular / nadir bright spot) antes del
  percentil para usar mejor el rango dinámico.

### 9.4 — Coherencia de narrativa  🟢

- [ ] 🟢 Unificar "fondo plano" (README presentación / CLAUDE.md) vs "con relieve"
  (dataset analizado). Aclarar qué dataset se presenta; el run de `results/` actual
  NO coincide con el del README de presentación (647 vs 1456 nodos).

---

## Notas

- Verificado que **NO es un bug**: el parser de media lee el color como
  `red/255.0` y el PLY del pipeline efectivamente guarda `uchar red` (0-255). OK.
- Los *gates* del registro están densamente comentados con el porqué de cada
  valor; cualquier cambio en §5 (loop closure) debe registrarse con su
  experimento, igual que ya se hace en los comentarios (referencias a
  `report.md` / `report_ransac.md` — **localizar o crear esos reports**, no están
  en el repo).
