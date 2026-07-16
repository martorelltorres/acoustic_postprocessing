# acoustic_postprocessing — Tareas / mejoras propuestas

Análisis de los ficheros del paquete (2026-07-14), con la ronda del **2026-07-16**
ya aplicada: se cerraron §1 (bugs), §2 (suite de media) y §3 (módulo común), y quedan
§4 (calidad de los productos), §5 (rendimiento) y §6 (empaquetado y tooling). Mejoras
agrupadas por tema y priorizadas; cada ítem indica **fichero**, **problema** y
**propuesta**.

Prioridad: 🔴 alta (corrección / fiabilidad) · 🟡 media (robustez / calidad de
resultados) · 🟢 baja (mantenibilidad / rendimiento / cosmético).

Estado: ✅ completada · ⬜ pendiente.

**Este fichero es el registro**: al cerrar un ítem, muévelo a §Completadas con la
fecha, qué se cambió y **cómo se verificó**. Si un experimento sale mal, déjalo
escrito igual (el porqué de lo descartado vale tanto como lo que funcionó).

---

# ⬜ Pendientes

## 4. Calidad de los productos

- ⬜ 🟡 **`mb_intensity.xyz` sale sin filtrar y con la intensidad sin acotar.**
  A diferencia de `mb_pointcloud.xyz` (voxel + SOR + filtro de superficie), la nube
  de intensidad solo pasa por `angle_cutoff`: conserva los picos verticales (~5.7%)
  y la cola especular (máx. observado 1212), así que al abrirla en CloudCompare la
  rampa automática la pinta toda azul. **Propuesta:** (a) aplicar la misma cadena de
  filtrado — con el beneficio de que ambas nubes volverían a compartir puntos — y
  (b) acotar la intensidad a p2–p98 al escribir, o guardar además una columna
  normalizada.

- ⬜ 🟡 **El sidescan no gatea por actitud ni corrige el roll.** `sss2mosaic.py`
  proyecta con **solo yaw** ([:317-326](scripts/sss2mosaic.py#L317-L326)): calcula
  `f_p`/`f_r` y no los usa. Con el vehículo escorado, el error across-track es
  ≈ `h·tan(roll)` (a 5° y 10 m de altitud, ~0.9 m). El pipeline MB **sí** tira los
  pings en viraje. **Propuesta:** aplicar a `sss2mosaic.py` el mismo gate
  (`max_roll_deg`, `max_yaw_rate_deg_s`), que ya está medido y parametrizado, y
  evaluar meter el roll en el vector across-track.

- ⬜ 🟡 **La corrección slant-range asume fondo plano** a la altitud `h`
  (`ground = sqrt(slant² − h²)`). Es lo estándar, pero **este paquete ya produce el
  DEM batimétrico** del mismo área: se podría hacer una corrección slant *terrain-
  aware* trazando el rayo contra el DEM. Mejora real en zonas con relieve y es
  publicable. (Depende de que MB y SSS cubran la misma zona — ver la nota de datasets.)

- ⬜ 🟡 **`enhance_data()` filtra a través de los bordes del nodata.** El
  `medianBlur` + CLAHE + kernel de sharpening se aplican sobre el ráster **entero**,
  ceros incluidos, y solo *después* se reimpone `nodata = 0`. El sharpening sangra
  valores a través del borde del swath → halo brillante en los bordes.
  **Propuesta:** enmascarar (o inpaint) antes de filtrar, o filtrar solo dentro de la
  máscara de datos.

- ⬜ 🟡 **La fusión de mosaicos mezcla contrastes, no backscatter.** Los dos `.tif`
  de entrada ya han pasado por `enhance_data()` (normalización por percentiles +
  CLAHE), así que `max`/`mean` combinan dos estiramientos de contraste arbitrarios y
  **distintos**, no dos magnitudes físicas. **Propuesta:** fusionar los valores
  *antes* del realce (guardar un .tif crudo float32 junto al de 8 bits), o al menos
  documentar que la fusión es cualitativa.

- ⬜ 🟢 **`mb_sss_extra_offset_y = -2.0`: diagnosticar la causa.** Es un
  desplazamiento empírico de 2 m aplicado a **todos** los productos MB para que
  cuadren con el SSS. Sospechosos: lever-arm mal medido en el URDF, desfase de reloj
  entre los dos bags (¡son misiones distintas!), o que el error esté en el lado del
  sidescan. Mientras no se sepa, los productos MB están deliberadamente
  desgeorreferenciados 2 m. **Propuesta:** cuantificar el desplazamiento por
  correlación cruzada de los dos mosaicos de backscatter y compararlo con el
  lever-arm del URDF.

## 5. Rendimiento

- ⬜ 🟡 **Bucles Python sobre celdas únicas** (cientos de miles de iteraciones) en
  tres sitios: `_cell_median` ([multibeam_processor.py:79](scripts/multibeam_processor.py#L79)),
  el DEM ([:439](scripts/multibeam_processor.py#L439)) y el mosaico de intensidad
  ([multibeam_intensity.py:402](scripts/multibeam_intensity.py#L402)).
  `mb_projection_metrics._grid_stats` ya enseña el patrón vectorizado (orden +
  sumas acumuladas). La **mediana** también se vectoriza: ordenar por (celda, valor)
  y quedarse con el índice central de cada segmento.

- ⬜ 🟢 **`np.savetxt` para `mb_intensity.xyz`** con ~30 M puntos tarda minutos y
  genera un ASCII enorme; `write_point_cloud(..., write_ascii=True)` igual.
  **Propuesta:** ofrecer salida binaria (o `.laz`), o al menos escribir por bloques.

## 6. Empaquetado y tooling

- ⬜ 🟡 **`CMakeLists.txt` no instala `mb_projection_metrics.py`**, cuyo propio
  docstring anuncia `rosrun acoustic_postprocessing mb_projection_metrics.py`
  ([:14](scripts/mb_projection_metrics.py#L14)) — que hoy **falla**. Decidir: o se
  añade a `catkin_install_python`, o se corrige el docstring. (Los 4 scripts de media
  importan entre sí, así que esos deben seguir ejecutándose con `python3` desde
  `scripts/` mientras no haya módulo común — ver §3.)

- ⬜ 🟢 **`package.xml`: `description` y `license` siguen en `TODO`.** Y faltan
  dependencias declaradas: `python3-matplotlib`, `python3-pil`, `python3-pandas`
  (opcional). Open3D no tiene clave rosdep → documentar la instalación por pip y la
  versión probada (0.13.0).

- ⬜ 🟢 **Sin pin de versiones.** El pipeline usa APIs que varían entre versiones de
  Open3D y rasterio. Probado con: Python 3.8.10, open3d 0.13.0, numpy 1.24.4,
  rasterio 1.3.11, pandas 2.0.3. Añadir un `requirements.txt` informativo.

- ⬜ 🟢 **Tests: `tests/test_common.py` cubre `common.py` (17 tests); falta el resto.**
  Sin cubrir todavía, todas puras y testeables sin ROS ni bags:
  `surface_relative_filter` (pico sintético sobre plano → se elimina; relieve real →
  sobrevive), `angle_varying_gain` (perfil sintético con ganancia conocida → la
  invierte), `viridis_colormap` (0 → transparente), el gridding de los mosaicos y
  `_self_crossings` de `make_media.py`.
  No hay runner: se ejecutan a mano (`cd tests && python3 test_common.py`, o con pytest).
  Engancharlos a `catkin_make run_tests` (`catkin_add_nosetests`) está pendiente.

- ⬜ 🟢 **Constantes de `sss2mosaic.py` no parametrizadas.** `MOSAIC_RES = 0.07` y
  `BLIND_ZONE = 0.2` son constantes de módulo mientras que todo lo demás del pipeline
  se configura por launch. Exponerlas como `<arg>`.

- ⬜ 🟢 **Bags por defecto en rutas absolutas** con las alternativas comentadas en el
  launch. Añadir una tabla de datasets (qué bag cubre qué zona, y que **MB y SSS son
  misiones distintas** con horas distintas — de ahí que la fusión dependa de que se
  solapen espacialmente).

---

# ✅ Completadas

Reconstruido a partir de los comentarios del código (que documentan cada arreglo con
la medida que lo motivó). Fechas aproximadas donde no consta.

---

## Ronda del 2026-07-16 — módulo común + los bugs de §1 + suite de media

**Cómo se verificó todo lo de esta ronda.** La nav de estos bags está limpia (20 Hz,
dt = 0.05 s exacto, 0 retrocesos, 0 duplicados, 0 NaN, 0 huecos) y el refactor se diseñó
neutro, así que el criterio de aceptación fue la **bit-identidad** contra los `md5sum`
guardados antes de tocar nada. Resultado de la corrida completa:

| Producto | Esperado | Resultado |
|---|---|---|
| `mb_intensity.{tif,xyz,jpg}`, `sss_mosaic.{tif,png}` | idénticos | ✅ **bit-idénticos** |
| `sss_waterfall.png` | cambia (es el arreglo) | ✅ cambia |
| `mb_sss_mosaic.tif` | cambia (es el arreglo) | ✅ 0.10→**0.07 m**, 1418×1600→**2025×2286** |
| `mb_pointcloud.{xyz,tif,jpg}`, `mb_mesh.ply` | idénticos | ⚠️ **cambian — y no es una regresión**, ver abajo |

**El susto de `mb_pointcloud`, y lo que enseñó.** Cambió cuando no debía. **No era el
refactor: el SOR de Open3D no es determinista.** La prueba definitiva fue ejecutar el
**mismo código dos veces** sobre el mismo bag:

| | run A | run B | run C |
|---|---|---|---|
| raw | 32.100.582 | 32.100.582 | 32.100.582 |
| voxel(0.05) | 5.105.466 | 5.105.466 | 5.105.466 |
| **SOR** | **4.708.593** | **4.708.338** | **4.707.195** |
| surface filter | 4.606.266 | 4.606.040 | 4.604.908 |

Idénticos hasta el voxel, divergen en el SOR → md5 distintos con el mismo código. (Las
cuentas por ping —147.177/6.181/5.076/135.920— sí salen idénticas las tres veces.)
Reproducido aislado (nube sintética de 6 M puntos, tres procesos): `voxel_down_sample`
da n y md5 idénticos, `remove_statistical_outlier` da 2.077.026 / 2.077.110 / 2.076.492.

Evidencia adicional de que la geometría no se tocó: **las cuentas por compuerta son
idénticas** (147.177 pings leídos, 6.181 tirados por roll, 5.076 por yaw-rate, 135.920
válidos), el DEM tiene **las mismas 175.488 celdas con dato** y el **99.988% de las Z son
exactamente iguales** (21 celdas difieren, p99 de |ΔZ| = 0). Y `mb_intensity.xyz` sale
bit-idéntico justamente porque **no pasa por el SOR**.
→ **El md5 no vale como test para la nube ni la malla**; ver §Notas.

Además, **antes** de reejecutar nada se comprobó que `enhance_data` unificada da
resultados idénticos a las dos versiones viejas, y que el camino de nav da poses
idénticas al viejo sobre los bags reales (incluido el `roll_bias` = 2.153°).

### Módulo común (§3 — cerrada)

- ✅ 🟡 **`scripts/common.py`.** Concentra los helpers de bag/TF/nav/ráster que estaban
  clonados (3× `get_static_transform_from_tf`, 3× `get_nav_origin`, 3× `enhance_data`,
  3× interpoladores). Import de nombre plano + `sys.path.insert(0, dirname(__file__))`,
  como `multibeam_SLAM`; **no** hace falta instalarlo en el `CMakeLists.txt` porque el
  wrapper de catkin apunta `__file__` al fuente (verificado leyendo el wrapper de
  `devel/lib/`). **Era la causa raíz de todo §1**: cada arreglo había que hacerlo tres
  veces y nunca se hacía.
  - **Dos parámetros existen para NO homogeneizar comportamientos**, y hay que
    respetarlos: `nav_interpolators(smooth_yaw_sigma=)` (solo el SSS suaviza el yaw) y
    `enhance_data(nodata_mask=)`.
  - *Corrección al análisis del 2026-07-14*: se dijo que las tres `enhance_data` "habían
    divergido" como si fuera un olvido. **No lo era.** Las de `multibeam_intensity` y
    `sss2mosaic` son idénticas; la del waterfall difiere **con motivo**: allí el 0 es una
    muestra real del eco, no nodata, así que enmascarar `>0` habría cambiado el estirado
    de contraste. Medido: con y sin máscara difieren en el **23.6% de los píxeles**.
    Fusionarlas a ciegas habría roto el waterfall.
  - *Segunda corrección*: `load_raster` **no estaba duplicada**. La de `make_media.py`
    devuelve `(array con NaN en nodata, extent)` y la de `mb_sss_mosaic_fusion.py`
    devuelve `(data, transform, crs, res)`: son **homónimas con contratos distintos**.
    Solo la segunda se movió a `common.py`; la de la media se deja donde está (y sus
    scripts la siguen importando `from make_media import`). Unificarlas por el nombre
    habría roto la media.
- ✅ 🟢 **Borrado `manual_mb_sss_fusion.py`.** Además de fork obsoleto, estaba **roto**:
  llamaba a `src.index` con el dataset ya cerrado, dentro de un `except` que se lo tragaba
  → ponía todo a nodata y moría con "No mesh vertex intersects".

### Bugs (§1 — cerrada)

- ✅ 🔴 **El orden de muestras del sidescan, zanjado con datos.** Perfil medio por índice
  sobre el bag 10_44_27 (400 pings): **starboard pica en la muestra 1/2000** (nadir→lejos)
  y **port en la 1998/2000** (lejos→nadir). Son espejo → **`sss2mosaic.py` acertaba** al
  invertir port y **`sss_waterfall.py` estaba mal**. El bug **se escondía**: como port trae
  el nadir al final, su bloque queda pegado a la franja central y la imagen *parecía* bien,
  pero la corrección slant asignaba rango 0 a la muestra de rango lejano y el recorte de
  zona ciega borraba el extremo lejano. Ahora se invierte para la geometría y se vuelve a
  invertir solo para pintar (el layout no cambia).
- ✅ 🔴 **`sss_waterfall.py`: rango del bag, no hardcodeado.** `SSSConfig.range` = **50.0 m**
  en los dos canales del bag de Andratx, contra el `SONAR_RANGE = 30.0` de módulo.
  **Verificación de los dos arreglos juntos** — media del borde de nadir por canal:

  | | port | stbd |
  |---|---|---|
  | antes | **8.6** | 60.3 |
  | ahora | **61.5** | 79.1 |

  Antes babor daba 8.6 contra 60.3 de estribor: 7× de asimetría entre dos canales que
  deberían parecerse estadísticamente. Ahora son del mismo orden, y babor por fin muestra
  el gradiente nadir-brillante / lejos-oscuro. La imagen pasa de 3200×1561 a 3524×1561 px.
  De paso, `nav_topic` era una **constante de módulo** y ahora es param (con `<arg>`).
- ✅ 🔴 **`mb_sss_mosaic_fusion.py` ya no tira la resolución del SSS.**
  `res = min(mb_res[0], mb_res[1])` → `min(..., sss_res[0], sss_res[1])`. Verificado:
  la fusión pasa de **0.10 m / 1418×1600** a **0.07 m / 2025×2286**, y los píxeles con dato
  de 370.784 a 757.219. También se le puso `nodata=0`, que salía sin declarar mientras sus
  dos entradas sí lo declaran.
- ✅ 🔴 **Interpoladores de nav ordenados y deduplicados** (`common.read_nav`), en los dos
  scripts MB. **Es una guarda, no un arreglo con efecto**: estos bags ya venían ordenados,
  por eso los productos salen bit-idénticos. Se comprobó sobre los bags reales que el
  camino nuevo da poses **idénticas** al viejo (incluido el `roll_bias` = 2.153°).
  - *Corrección al análisis del 2026-07-14*, que decía "`interp1d` y `np.unwrap` dan poses
    erróneas en silencio" con nav desordenada. **`interp1d` no**: su default es
    `assume_sorted=False`, así que **ordena `x` por dentro** y tolera la entrada
    desordenada (verificado). Lo que sí se rompe, y en silencio, es: (1) `np.unwrap(yaw)`,
    que **depende del orden** e inventa saltos de rumbo; (2) `np.gradient(ts)` del
    yaw-rate, que sale **negativo** y hace que el gate de actitud tire los pings
    equivocados; y (3) los timestamps **duplicados**, que sí rompen `interp1d` (medido:
    64.5 donde tocaba 25.0). La guarda hace falta igual, pero por estos tres motivos, no
    por el que se creía. Los tres están fijados en `tests/test_common.py`.
- ✅ 🟡 **Guarda NaN en `multibeam_processor.py`**, replicando la de `multibeam_intensity.py`.
  También defensiva (0 NaN y 0 huecos en estos bags).
- ✅ 🟢 **`angle_cutoff`: default unificado a 55°** en los dos scripts MB (era 60 en código
  y 55 en el launch). Sin efecto vía launch, que ya pasaba 55 explícitamente.

### Rendimiento (§5 — parcial)

- ✅ 🟡 **Una sola pasada de TF.** `common.get_static_transforms(bag, pairs)` resuelve las
  tres TF recorriendo `/tf_static` + `/tf` una vez; `multibeam_processor.py` abría el bag
  4 veces. Cae de propina al deduplicar el helper.

### Suite de media (§2 — cerrada)

- ✅ 🔴 **`make_nav_cache.py`: el productor que faltaba.** `.nav_cache.npz` lo leían dos
  scripts y **no lo escribía nadie** (el de disco venía de una versión de `make_media.py`
  que ya no existe): desde un `results/` limpio la suite entera estaba muerta.
  **Verificado**: borrado el cache y regenerado, sale **bit-exacto** al anterior en las 8
  claves (`tm,xm,ym,zm` / `ts,xs,ys,zs`), lo que confirma que el contrato deducido
  (mapeo de ejes, `z = -depth`, origen UTM propio de cada bag) es el correcto. Después,
  `make_media.py` completa las 8 figuras.
  ⚠️ Sus bags por defecto son los de la **media** (MB **13_52_22** lawnmower + SSS
  10_44_27), **no** los del launch (octógono 12_52_41): los rótulos de `make_media.py`
  dicen "Multibeam 13:52".
- ✅ 🔴 **Assets fuera de `results/`.** `sparusII.png` y `mosaics/` eran **entradas** dentro
  del directorio regenerable; ahora en `assets/`. **Matiz que el análisis anterior no vio**:
  los fotomosaicos pesan **178 MB** (95 + 83) y `assets/` no estaba ignorado, así que
  moverlos tal cual habría metido 178 MB en git. Solución: `assets/sparusII.png` versionado
  (48 KB) y `assets/mosaics/` en `.gitignore` — cumple igual el objetivo (sobrevivir a un
  `rm -rf results/`) y es seguro porque los dos consumidores ya degradan solos.
- ✅ 🟡 **Rutas de la animación unificadas en `results/media/`** (era `anim_data/` +
  `presentation/`, que no casaban ni con el disco ni con el README).
- ✅ 🟢 **`results/README.md` documenta `media/`** (con el orden de ejecución) y `assets/`.

---

### Geometría y proyección
- ✅ 🔴 **Recorte angular en el frame MUNDO, no en el del sensor.** Con el vehículo
  escorado, cortar a 55° en el frame del sensor dejaba pasar haces a 55°+roll de la
  vertical real. Param `angle_cutoff_frame` (default `world`).
- ✅ 🔴 **Gate por actitud (roll de-sesgado + yaw-rate).** Tira el ping entero durante
  los virajes. Se umbraliza la excursión sobre el sesgo de roll (~+2° de trim), no el
  roll absoluto (r=+0.54 vs +0.46). Quita ~30% de los puntos malos con solo ~7% de
  pérdida de cobertura.
- ✅ 🔴 **`surface_relative_filter`.** El SOR k-NN no veía los picos verticales del
  multihaz, que salían como "montañas" fantasma en CloudCompare. Filtro por mediana/MAD
  de la celda XY, adaptativo e iterativo.
- ✅ 🔴 **Orden voxel → SOR** (antes al revés): el SOR construía su KD-tree sobre 31 M
  puntos crudos → lentísimo y pico de RAM.

### Malla
- ✅ 🔴 **OOM del Poisson.** `depth=11` con 31 M puntos consumía 55 GB → OOM-kill.
  Bajado a `depth=10` y voxel 0.01 → 0.05 m. **Verificado OK el 2026-07-08.**
- ✅ 🔴 **Malla completamente negra en CloudCompare.** Poisson devuelve colores por
  vértice **todo a cero** cuando la nube no tiene color, y el PLY los guardaba. Se
  descarta el array para que sombree por normales.

### Intensidad / backscatter
- ✅ 🔴 **Corrección AVG del multihaz** (portada del SLAM): perfil = mediana por bin
  angular, `I_corr = I / gain(θ)`. Sin ella la intensidad salía bimodal (~53% de
  píxeles casi negros). Reduce el banding ~99%.
- ✅ 🔴 **AVG del sidescan, con MEDIA y no mediana.** La mediana por bin no funciona en
  el SSS: el eco crudo está saturado de ceros (47.5% de las muestras son 0), la mediana
  sale 0–1 y mide relleno, no ganancia → perfil plano, corrección nula. Además: perfil
  **por banda** (port/stbd son transductores distintos) sobre **escala común**, y
  ganancia acotada por abajo (`gmax/max_boost`) para no amplificar ×700 el ruido de la
  cola.
- ✅ 🔴 **Mosaico por MEDIANA por celda** (MB): la media era sensible a los outliers
  especulares (un solo retorno reventaba la celda).
- ✅ 🔴 **`0` reservado para nodata tras el CLAHE.** El realce subía el fondo vacío de 0
  a ~4 → el histograma del .tif daba mediana 4 y el "mosaico bimodal casi negro" era en
  realidad la mediana midiendo el FONDO. Ahora: vacías a 0, llenas forzadas a ≥1 (que es
  lo que esperan las fusiones de su `> 0`).
- ✅ 🟡 **`mb_intensity.tif` con paleta viridis embebida en vez de 3 bandas RGB.**
  Mantiene el .tif como 1 banda con el VALOR de backscatter, que es lo que consume
  `mb_sss_mosaic_fusion.py` (con RGB leería el canal rojo de viridis como intensidad).

### Sidescan
- ✅ 🔴 **Rango del sonar leído del bag (`SSSConfig.range`).** El `SONAR_RANGE = 30.0`
  hardcodeado era falso para Andratx (real: 50 m): colocaba cada muestra al 60% de su
  rango, comprimiendo el mosaico ×0.6 across-track. Al corregirlo, la huella pasó de
  72.3×71.5 m a 104.9×92.0 m. *(Aplicado también a `sss_waterfall.py` el 2026-07-16.)*
- ✅ 🟡 **`sss_mosaic.png` en PNG y no JPG**: el fondo sin dato va con alfa=0 y JPEG no
  tiene canal alfa. El RGB bajo el alfa se pone a negro también, para que un visor que
  ignore la transparencia no repinte el morado de `viridis(0)`.

### Fusión
- ✅ 🔴 **`wait_timeout = 0` colgaba la espera para siempre** en los dos nodos de fusión:
  la condición `timeout > 0` vivía **dentro** del `break`, así que 0 desactivaba la
  salida en vez de desactivar la espera. Ahora se comprueba antes del bucle.
- ✅ 🔴 **`fusion_wait_timeout` subido a 1800 s.** `mb_done` se publica **después** del
  Poisson (~9 min); los 600 s anteriores vencían antes.
- ✅ 🟡 **Muestreo del SSS sobre la malla vectorizado** (`src.index` acepta arrays):
  antes era un bucle Python de 1.5 M iteraciones.
- ✅ 🔴 **Normalización solo con los vértices válidos** en `sss_mb_fusion.py`. Normalizar
  el array entero mandaba los vértices sin dato (intensidad 0) por debajo de `imin` → el
  colormap los saturaba a negro, indistinguibles de backscatter bajo real. Ahora los
  vértices fuera del swath llevan un color propio (rojo apagado). *(El fork que arrastraba
  el bug, `manual_mb_sss_fusion.py`, se borró el 2026-07-16.)*

### Layout y empaquetado
- ✅ 🟡 **`results/` reorganizado por tipo de fichero** (`tif/`, `images/`,
  `pointcloud/`, `mesh/`, `metrics/`) en vez de por sensor, documentado en
  `results/README.md`.
- ✅ 🟡 **`mb_projection_metrics.py`**: métricas de calidad de la proyección (rugosidad
  por celda, cobertura, banding del backscatter), solo numpy/matplotlib.

---

## Notas

- **Nunca lances dos `roslaunch` del pipeline a la vez**: el segundo mata los nodos de
  fusión del primero a mitad de escritura.
- **`use_sim_time` debe quedarse en `false`** o las esperas de los nodos de fusión se
  cuelgan (los bags se leen, no se reproducen: no hay `/clock`).
- **MB y SSS vienen de misiones/bags distintos.** Toda fusión depende de que se solapen
  espacialmente; comprobarlo antes de culpar al código.
- **Los helpers compartidos van en `scripts/common.py`, y solo ahí.** Todo §1 fue el mismo
  patrón — *un script lo corrigió y sus clones no* —, que es lo que motivó el módulo común
  del 2026-07-16. Si vuelves a copiar un helper entre scripts, estás reabriendo §1.
- **La nav de los bags de Andratx está limpia** (20 Hz, dt = 0.05 s exacto, sin retrocesos,
  duplicados, NaN ni huecos). Útil: un cambio que no pretenda alterar la geometría debe
  salir **bit-idéntico** en estos bags, así que guardar el `md5sum` de los productos
  **antes** de tocar nada es un test de no-regresión gratis y muy sensible.
- ⚠️ **…pero el md5 NO vale para `mb_pointcloud.*` ni para la malla**:
  `remove_statistical_outlier` de Open3D **no es determinista**. Medido (2026-07-16, misma
  nube sintética de 6 M puntos, tres procesos): `voxel_down_sample` da n y md5 idénticos,
  pero el SOR devuelve 2.077.026 / 2.077.110 / 2.076.492 puntos y md5 distintos (~0.03% de
  variación). En el pipeline real eso son ~21 de 175.488 celdas del DEM (0.012%), con el
  99.988% de las Z **exactamente iguales**.
  Por eso `mb_intensity.xyz` sí sale bit-idéntico: **no pasa por el SOR** (es el producto
  "sin filtrar" de §4). Para validar cambios en `multibeam_processor.py`, compara: las
  **cuentas por compuerta** del log (pings leídos / tirados por roll / por yaw-rate /
  válidos, y raw→voxel→SOR→surface), el **número de celdas con dato** del DEM y el
  **percentil 99 de |ΔZ|** (debe ser 0). El md5 de la nube solo da falsos positivos.
</content>
