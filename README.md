# Proyectos de Visión por Computadora 👁️🤖

Este repositorio centraliza mis proyectos y experimentos prácticos en el área de Visión por Computadora, explorando el procesamiento de video en tiempo real, la vectorización de landmarks y el reconocimiento de gestos/posturas.

## 🛠️ Tecnologías Globales
* **Python 3.x**
* **OpenCV** 
* **MediaPipe Tasks**

---

## 📂 Estructura del Repositorio

El proyecto está dividido en módulos independientes según el área de investigación:

### 1. 🖐️ Detección de Manos (`HandTracking`)
Detección y vectorización en tiempo real de los 21 landmarks de la mano.
* **Características:** Identificación de mano izquierda/derecha y reconocimiento dinámico de gestos.
* **Modelo requerido:** `hand_landmarker.task`

### 2. 🤸 Detector de Poses (`PoseDetection`)
Traqueo de cuerpo entero en tiempo real vectorizando las 33 articulaciones principales del esqueleto.
* **Características:** Estructura inicial para el mapeo de coordenadas corporales (hombros, brazos, torso y piernas).
* **Modelo requerido:** `pose_landmarker.task`

### 📦 3. Detección de Objetos con IA (`ObjectDetection`)
Primer acercamiento a arquitecturas avanzadas de Machine Learning para localización y clasificación de elementos mediante cajas de delimitación (*Bounding Boxes*).
* **Características:** Implementación en tiempo real de **YOLOv8** (Ultralytics) utilizando un modelo convolucional pre-entrenado para el reconocimiento simultáneo de múltiples objetos cotidianos.
* **Modelo requerido:** `yolov8n.pt`.

---

## 🚀 Cómo ejecutar cualquiera de los proyectos

1. **Cloná este repositorio:**
   ```bash
   git clone [https://github.com/Benjam1nFares/ComputerVision.git](https://github.com/Benjam1nFares/ComputerVision.git)
   
2. **Descargá los modelos oficiales:**
   - Hand Tracking:
   (https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
   Descargar el archivo [hand_landmarker.task]
   - Pose Detection:
   (https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)
   Descargar el archivo [pose_landmarker_full.task] y Renombrar como [pose_landmarker.task].

3. Instalá las dependencias:
    - HandTracking - PoseDetection:
    `pip install opencv-python mediapipe`
    - 
    - ObjectDetection:
    `pip install opencv-python mediapipe ultralytics`

4. Ejecutá el script principal.

5. Have fun :)
