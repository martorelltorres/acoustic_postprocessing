# Acoustic Postprocessing — SLAM Multihaz Submarino
## Documento explicativo para presentación

> Proyecto `acoustic_postprocessing` (paquete ROS/catkin del workspace `derelictes_ws`).
> Mantenedor: antoni.martorell@uib.es (UIB). Misiones MINEX / Derelictes con AUV Sparus2.

---

## 1. Qué resuelve el proyecto (el problema)

Un AUV (Sparus2) recorre el fondo marino siguiendo un patrón **lawnmower** (segado en franjas paralelas) llevando un **sonar multihaz (MBES, Norbit WBMS)**. El objetivo es reconstruir un **mapa 3D del fondo** (nube de puntos batimétrica) de alta calidad — por ejemplo para localizar pecios/objetos sumergidos.

El reto central:

- **No hay GPS bajo el agua.** La posición viene de navegación inercial (INS = DVL + IMU). Esta navegación **deriva** (acumula error) con la distancia: la trayectoria estimada se separa progresivamente de la real.
- Si se proyecta cada barrido del sonar usando solo la navegación bruta, el mapa resultante sale **borroso, duplicado y deformado**.

La solución: un **SLAM (Simultaneous Localization and Mapping)** que corrige la trayectoria alineando entre sí las nubes de puntos del sonar, y reconstruye un mapa coherente.

**Mensaje clave para la presentación:** convertimos navegación inercial que deriva + sonar ruidoso en un mapa 3D consistente, sin GPS, en postproceso sobre los datos grabados (rosbag).

---

## 2. El reto específico: el fondo marino plano

Este es **el punto diferenciador del trabajo** y conviene destacarlo.

Los algoritmos clásicos de alineación de nubes (ICP) funcionan asumiendo que hay **relieve geométrico** que restringe el alineamiento. Pero gran parte del fondo marino es **plano y sin estructura**:

- Sobre fondo plano, todas las normales de la superficie apuntan hacia arriba → el ICP **no tiene información para fijar la posición horizontal (XY)**: el coste es plano, hay infinitas soluciones casi igual de buenas.
- Resultado: el ICP "desliza", converge a mínimos locales erróneos (pasos invertidos 180°, perpendiculares...) que aun así pasan los filtros de calidad porque "plano contra plano" siempre encaja bien.

Todo el pipeline está diseñado con **salvaguardas (gates) específicas para este escenario degenerado**. Es la aportación técnica principal.

---

## 3. Arquitectura del pipeline (visión de alto nivel)

```
   rosbag (MBES + navegación INS)
            │
            ▼
   [1] Construcción de PATCHES        ← agrupa barridos en nubes locales
            │
            ▼
   [2] Registro SECUENCIAL (ICP)      ← alinea patch i con patch i-1  → odometría
            │
            ▼
   [3] LOOP CLOSURE (Scan Context)    ← detecta zonas revisitadas
            │
            ▼
   [4] OPTIMIZACIÓN GLOBAL del grafo  ← reparte el error de forma coherente
            │
            ▼
   [5] Restauración vertical + MAPA final (.ply) + métricas
```

El núcleo es un **grafo de poses (pose graph)** de Open3D: cada nodo es la pose de un patch, cada arista es una restricción de alineamiento. Optimizarlo da la trayectoria corregida.

---

## 4. Las cinco etapas, explicadas

### Etapa 1 — Construcción de patches (`PatchBuilder`)

En vez de alinear cada barrido individual (demasiado pobre en puntos), se agrupan **100 barridos consecutivos** en un *patch* (nube local), avanzando de **20 en 20** (solapamiento del 80%).

Operaciones en cada patch:
- Proyección de cada punto: marco del sensor → marco del vehículo (roll/pitch/yaw del INS) → posición global INS → recentrado al centro del patch.
- **Recorte por ángulo de incidencia** (descarta haces > 55° del nadir, los más ruidosos).
- **Corrección AVG (Angle-Varying Gain)** de la intensidad acústica: la intensidad (backscatter) depende fuertemente del ángulo del haz, creando un "banding" artificial. Se construye un perfil de ganancia por ángulo (mediana, robusta) y se normaliza → queda solo la **textura real del fondo**. *(Esto es lo que hace utilizable la intensidad en el paso siguiente.)*
- Submuestreo por vóxel (0.25 m) y estimación de normales.

**Resultado:** ~cientos/miles de patches, cada uno con geometría + intensidad acústica como "color".

### Etapa 2 — Registro secuencial

Alinea cada patch con el anterior. Aquí está la mayoría de la inteligencia anti-fondo-plano.

**Cuatro algoritmos de registro disponibles** (configurable):
- `icp` — ICP geométrico robusto (GICP + pérdida de Tukey, multiescala coarse-to-fine).
- `colored_icp` — combina geometría + intensidad acústica (usa la textura para no deslizar en zonas planas).
- `hybrid` *(el usado por defecto)* — **adaptativo**: usa geometría donde hay relieve, intensidad donde el fondo es liso. Decide por par según la fracción de normales no verticales (textura geométrica).
- `ndt` — Normal Distributions Transform (implementación propia 2D).

**Salvaguardas / "gates" de validación** (lo más importante de contar):

| Gate | Qué hace | Por qué |
|------|----------|---------|
| Fitness / RMSE / nº correspondencias | Filtros básicos de calidad | Descartar alineaciones malas |
| **Desviación de traslación vs INS** | Rechaza si el ICP se aleja > 0.5 m del prior INS | Caza saltos absurdos sobre fondo plano |
| **Desviación de dirección vs INS** | Rechaza si la dirección del paso difiere > 15° del INS | Caza pasos invertidos/perpendiculares |
| **Banda de ratio de longitud [0.85, 1.15]** | Si el paso ICP es mucho más corto/largo que el INS, se reescala a la longitud del INS | El solapamiento del 80% sesga el ICP a pasos ~8% más cortos → comprime la trayectoria |
| **Rotación del INS, traslación del ICP** | La arista usa la rotación 3D fiable del INS y solo la corrección de traslación XY del ICP | La rotación del ICP en fondo plano es ruido (±19°/paso) que produce deriva angular |
| **Amortiguación cross-track** | Anula la componente lateral del ICP (la pone el INS) | El ICP metía un sesgo lateral sistemático hacia el Este |

**Idea de fondo:** *confiar en el INS para lo que el INS hace bien (rotación, longitud del paso, rumbo) y en el sonar solo para la corrección fina donde aporta información fiable.* Cuando el registro se rechaza, se cae a la navegación (T_init) con peso bajo, sin contaminar el grafo.

### Etapa 3 — Loop closure (cierre de bucle)

Detecta cuándo el vehículo **vuelve a pasar por una zona ya vista** (franjas adyacentes del lawnmower). Cerrar el bucle permite corregir la deriva acumulada.

- **Scan Context**: descriptor compacto de cada patch (rejilla polar anillos×sectores), invariante a rotación, comparable por similitud coseno. Permite buscar zonas parecidas de forma eficiente.
- **Pre-filtro espacial por KD-tree** sobre la posición INS → reduce la búsqueda de O(N²) a casi lineal (optimización de rendimiento clave en misiones largas).

**Gates anti-falso-positivo** (de nuevo, contra el fondo plano que "todo se parece"):
- **Proximidad INS** (< 12 m): solo zonas que el INS sitúa cerca.
- **RANSAC global** (FPFH) como semilla del alineamiento.
- Límites de traslación Z / XY / yaw del cierre.
- **Consistencia ICP–deriva INS** (ratio ≥ 0.40): un cierre real corrige la deriva acumulada (ratio ≈ 1); un falso positivo sobre fondo plano da corrección ≈ 0. Este gate descarta los falsos emparejamientos entre franjas paralelas idénticas.

### Etapa 4 — Optimización global del grafo

Una vez construido el grafo (aristas secuenciales + cierres de bucle), se ejecuta la **optimización global de Open3D** (Levenberg–Marquardt). Reparte el error de cierre a lo largo de toda la trayectoria de forma coherente. Cada arista lleva una **matriz de información** (peso) calculada dinámicamente según fitness/RMSE: las alineaciones buenas pesan más.

### Etapa 5 — Restauración vertical y mapa final

El fondo plano deja al optimizador **sin restricciones verticales**, así que tiende a introducir roll/pitch/Z espurios que, aplicados a un patch de 30–50 m, levantan los bordes varios metros → mapa "grueso" y fantasma.

Solución: tras optimizar, se **reconstruye la vertical desde el INS** (roll, pitch y profundidad Z fiables del IMU/INS) **conservando el yaw optimizado** (la corrección SLAM válida es en el plano). Esto adelgaza el mapa drásticamente.

Finalmente se genera y guarda:
- `raw_navigation_map.ply` — mapa solo con navegación bruta (para comparar).
- `slam_optimized_map.ply` — mapa corregido por SLAM.

---

## 5. Salidas y métricas (para mostrar resultados)

El pipeline genera automáticamente en `results/metrics/`:

**Mapas y trayectorias:** los dos `.ply`, más `raw_trajectory.npy` / `slam_trajectory.npy`.

**Gráficas (PNG):**
- `trajectory_comparison.png` — trayectoria bruta vs SLAM (la imagen más vendible).
- `registration_quality.png` — fitness y RMSE por patch (aceptado/rechazado).
- `slam_correction.png` — cuánto corrige el SLAM en cada nodo.
- `rmse_histogram.png` — distribución de error de alineación.
- `loop_closure_quality.png` — diagnóstico de cierres de bucle.

**Datos:** `slam_metrics.json` (resumen completo), `icp_stats.csv`, `loop_stats.csv`, `stage_times.json` (tiempos por etapa).

**Métricas resumen destacables:** corrección media/máxima de la trayectoria, longitud de camino bruto vs SLAM, ratio de aceptación secuencial y de cierres, razones de rechazo desglosadas.

**Monitor en vivo (Open3D):** durante la ejecución se visualiza el grafo de poses construyéndose, con las últimas nubes alineadas — útil para demos en directo.

---

## 6. Stack tecnológico

- **ROS (catkin)** + **Python 3** — orquestación, lectura de rosbags.
- **Open3D** — nubes de puntos, ICP/GICP/Colored ICP, RANSAC, grafo de poses, optimización global, visualización.
- **NumPy / SciPy** — interpolación de navegación, NDT propio, álgebra.
- **tf.transformations** — transformaciones SE(3).
- Entrada: **rosbags** del Sparus2 (tópicos de MBES y de navegación).

Ejecución: `roslaunch acoustic_postprocessing acoustic_pipeline.launch` (todos los parámetros — algoritmo, gates, umbrales — son configurables desde el launch sin tocar código).

> Nota: el launch contiene además nodos comentados de **sidescan sonar** (mosaico, waterfall, fusión SSS+MBES). El proyecto está pensado para crecer hacia fusión multi-sensor; actualmente el nodo activo es el SLAM multihaz.

---

## 7. Puntos clave para los slides (resumen)

1. **Problema:** mapa 3D del fondo marino sin GPS, con navegación inercial que deriva y sonar ruidoso.
2. **Reto técnico único:** el fondo plano degenera el ICP clásico → hace falta SLAM "consciente del fondo plano".
3. **Solución:** SLAM basado en grafo de poses (patches → registro secuencial → loop closure → optimización global).
4. **Aportación principal:** un conjunto de *gates* que fusionan INS (fiable en rotación/escala/rumbo) con sonar (fiable en corrección fina), evitando los fallos típicos del fondo plano.
5. **Innovaciones concretas:** registro híbrido geometría/intensidad, corrección AVG del banding acústico, cierre de bucle con Scan Context + gates anti-falso-positivo, restauración vertical post-optimización.
6. **Resultados:** comparación visual mapa bruto vs SLAM + métricas cuantitativas (corrección de trayectoria, fitness/RMSE, cierres aceptados).
7. **Rendimiento:** optimizaciones para misiones largas (caché de preprocesado, pre-filtro espacial KD-tree, RANSAC acotado).
8. **Futuro:** fusión multi-sensor con sidescan sonar (ya esbozada en el código).
```
