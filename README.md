# 🚗 PathDetector6 – Detección y Seguimiento de Carriles en Video

**PathDetector6** es un sistema avanzado de **detección de carriles y análisis de trayectoria** basado en visión computacional, implementado en **Python y OpenCV**.
El sistema analiza un video, identifica las líneas de la vía (carriles) y calcula métricas de conducción como ángulo de dirección, curvatura, desviación del centro y nivel de confianza.
Ideal para aplicaciones en **ADAS (sistemas avanzados de asistencia a la conducción)**, **robótica móvil**, y **navegación autónoma**.

---

## 🧠 Funcionalidades principales

* Detección automática de carriles mediante **Hough Transform** y **Canny adaptativo**.
* Filtrado de ruido y mejora de contraste con **CLAHE** y **denoising**.
* Seguimiento y suavizado de trayectorias usando **Filtro de Kalman**.
* Cálculo de métricas avanzadas:

  * Ángulo de dirección estimado (steering angle)
  * Ancho del carril
  * Curvatura
  * Desviación respecto al centro
  * Nivel de confianza del modelo
* Visualización en tiempo real con anotaciones gráficas y flecha direccional.
* Exportación de video procesado y métricas a un archivo `.csv`.
* Interfaz gráfica (GUI) para seleccionar el video mediante **Tkinter**.

---

## ⚙️ Requisitos

* **Python 3.10+**
* Librerías necesarias:

  ```bash
  pip install opencv-python numpy
  ```

---

## 🗂️ Estructura del proyecto

```
PathDetector6/
│
├── pathdetector6_OK.py         # Script principal
├── config.json                 # Configuración de parámetros (opcional, se crea si no existe)
├── processed_YYYY_MM_DD_HH_MM.mp4   # Video de salida procesado
├── metrics_YYYY_MM_DD_HH_MM.csv     # Métricas exportadas
└── path_detection_YYYYMMDD.log      # Archivo de log automático
```

---

## ⚙️ Configuración (config.json)

El archivo se genera automáticamente si no existe.
Ejemplo de contenido:

```json
{
    "canny_low": 50,
    "canny_high": 150,
    "hough_threshold": 50,
    "min_line_length": 100,
    "max_line_gap": 50,
    "roi_height_factor": 0.5,
    "confidence_threshold": 0.7,
    "smoothing_window": 5
}
```

**Descripción de parámetros:**

| Parámetro                 | Descripción                                           |
| ------------------------- | ----------------------------------------------------- |
| `canny_low`, `canny_high` | Umbrales para la detección de bordes.                 |
| `hough_threshold`         | Sensibilidad de la detección de líneas.               |
| `min_line_length`         | Longitud mínima de línea para ser considerada carril. |
| `max_line_gap`            | Distancia máxima entre segmentos conectados.          |
| `roi_height_factor`       | Altura relativa de la región de interés (ROI).        |
| `confidence_threshold`    | Nivel mínimo de confianza para visualizar métricas.   |
| `smoothing_window`        | Ventana de suavizado de resultados históricos.        |

---

## 🧮 Principales clases y funciones

### 🔹 `class PathDetector`

Encargada de todo el procesamiento visual y análisis matemático.

**Funciones clave:**

| Método                       | Descripción                                                   |
| ---------------------------- | ------------------------------------------------------------- |
| `preprocess_frame(frame)`    | Mejora contraste y reduce ruido en la imagen.                 |
| `detect_edges(image)`        | Aplica Canny adaptativo según media y desviación.             |
| `get_roi_mask(shape)`        | Define la región trapezoidal donde se buscan carriles.        |
| `detect_lanes(edges, frame)` | Detecta líneas usando HoughLinesP y separa izquierda/derecha. |
| `calculate_path_metrics()`   | Calcula ángulo, curvatura, desviación y confianza.            |
| `draw_visualization()`       | Dibuja líneas de carril, centro y flecha direccional.         |
| `process_frame()`            | Ejecuta el flujo completo de análisis y logging.              |

---

### 🔹 `class VideoProcessor`

Proporciona la interfaz gráfica y coordina el procesamiento del video.

**Flujo general:**

1. Muestra una ventana Tkinter para seleccionar el archivo de video.
2. Procesa cada cuadro con `PathDetector`.
3. Guarda:

   * El video anotado (`processed_YYYY_MM_DD_HH_MM.mp4`)
   * Las métricas en CSV (`metrics_YYYY_MM_DD_HH_MM.csv`)
4. Muestra el avance en porcentaje y permite interrumpir con la tecla `Q`.

---

## ▶️ Ejecución

1. Ejecuta el script principal:

   ```bash
   python pathdetector6_OK.py
   ```
2. Se abrirá una ventana:

   * Haz clic en **“Seleccionar Video”** y elige un archivo `.mp4`.
   * El sistema procesará automáticamente el video.
3. El resultado se mostrará en tiempo real y se guardará en disco.

---

## 📊 Salida

**Archivos generados:**

* `processed_YYYY_MM_DD_HH_MM.mp4` → Video con anotaciones visuales.
* `metrics_YYYY_MM_DD_HH_MM.csv` → Registro de métricas por cuadro:

  ```
  frame,steering_angle,confidence,curvature,center_offset
  0,-1.2,0.88,0.03,15
  1,-1.0,0.89,0.04,17
  ...
  ```
* `path_detection_YYYYMMDD.log` → Archivo de log con eventos relevantes.

---

## 🎥 Visualización

El video procesado incluye:

* Carriles detectados (líneas verdes)
* Línea roja (centro estimado de la trayectoria)
* Línea azul (centro de imagen)
* Flecha direccional indicando el ángulo de dirección
* Texto con:

  * Estado de dirección (Centrado / Girar Izquierda / Girar Derecha)
  * Ángulo, confianza y curvatura.

---

## 🪶 Autor

**Juan Carlos**
Desarrollo de sistemas de visión computacional y análisis de trayectoria.
📧 jcllanosv007@gmail.com




