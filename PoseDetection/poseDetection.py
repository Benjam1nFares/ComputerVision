import cv2
import mediapipe as mp
import time

# --- Configuración de Colores para el Cuerpo (Formato BGR) ---
COLOR_LINEAS = (0, 255, 255)  # Amarillo/Cyan (según lo sintonices)
COLOR_NODOS = (0, 0, 255)  # Rojo para las articulaciones
COLOR_TEXTO = (255, 255, 255)  # Blanco

# 1. Configuración de MediaPipe Tasks para POSE
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions  # La API de Tasks comparte estructuras base
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO
)

# 2. Conexiones principales del esqueleto (Hombros, brazos, torso y piernas)
POSE_CONNECTIONS = [
    # Hombros y Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Brazo Izquierdo
    (11, 13), (13, 15),
    # Brazo Derecho
    (12, 14), (14, 16),
    # Pierna Izquierda
    (23, 25), (25, 27), (27, 31),
    # Pierna Derecho
    (24, 26), (26, 28), (28, 32)
]

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        h, w, _ = frame.shape

        # Preparar la imagen para MediaPipe Tasks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(time.time() * 1000)

        # Procesar frame
        results = landmarker.detect_for_video(mp_image, timestamp)

        # Si detecta un cuerpo en pantalla
        if results.pose_landmarks:
            for pose_landmarks in results.pose_landmarks:

                # ------------------------------------------------------------------
                # A. Dibujar las líneas del esqueleto
                # ------------------------------------------------------------------
                for connection in POSE_CONNECTIONS:
                    idx_start, idx_end = connection
                    # Obtenemos los puntos de inicio y fin
                    pt_start = pose_landmarks[idx_start]
                    pt_end = pose_landmarks[idx_end]

                    # Convertimos a píxeles de pantalla
                    start_coords = (int(pt_start.x * w), int(pt_start.y * h))
                    end_coords = (int(pt_end.x * w), int(pt_end.y * h))

                    cv2.line(frame, start_coords, end_coords, COLOR_LINEAS, 2)

                # ------------------------------------------------------------------
                # B. Dibujar los nodos (Articulaciones)
                # ------------------------------------------------------------------
                # Para no saturar la pantalla con los 33 puntos (como los de la cara),
                # podemos dibujar solo los principales del cuerpo (del 11 al 32)
                for idx in range(11, 33):
                    landmark = pose_landmarks[idx]
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, COLOR_NODOS, cv2.FILLED)

        # Mostrar en pantalla
        cv2.imshow('Detección de Poses v1.0', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()