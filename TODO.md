# acoustic_postprocessing — Tareas / mejoras propuestas

Análisis de los scripts del paquete. Mejoras agrupadas por tema y priorizadas.
Cada ítem indica **fichero**, **problema** y **propuesta**.

Prioridad: 🔴 alta (corrección / fiabilidad) · 🟡 media (robustez / calidad de
resultados) · 🟢 baja (mantenibilidad / rendimiento / cosmético).

Estado: ✅ completada · ⬜ pendiente.

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
- ⬜ 🟡 **0 cierres de bucle aceptados en el dataset actual** (943 candidatos
  rechazados, todos por RANSAC sobre fondo plano, según `loop_stats.csv` /
  README de presentación). El loop closure hoy no aporta corrección. Investigar:
  (a) sembrar el ICP con prior INS cuando RANSAC falla (el código menciona que
  "Fix B" empeoró — documentar el experimento en un report), (b) usar Scan
  Context sobre intensidad además de Z, (c) bajar dependencia de FPFH en fondo
  plano. Es la mayor palanca de mejora de exactitud pendiente.

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

---

## Notas

- Verificado que **NO es un bug**: el parser de media lee el color como
  `red/255.0` y el PLY del pipeline efectivamente guarda `uchar red` (0-255). OK.
- Los *gates* del registro están densamente comentados con el porqué de cada
  valor; cualquier cambio en §5 (loop closure) debe registrarse con su
  experimento, igual que ya se hace en los comentarios (referencias a
  `report.md` / `report_ransac.md` — **localizar o crear esos reports**, no están
  en el repo).
