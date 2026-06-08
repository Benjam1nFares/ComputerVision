import cv2
import mediapipe as mp
import time

# --- Configuración de Colores (Formato BGR) ---
COLOR_LINEAS = (255, 255, 0)  # Cyan
COLOR_NODOS = (0, 0, 255)  # Rojo
COLOR_TEXTO = (255, 255, 255)  # Blanco
COLOR_GESTO = (0, 255, 255)  # Amarillo
# ----------------------------------------------

# 1. Configuración de MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# ------------------------------------------------------------------
# Variables de estado para la memoria del movimiento (Wave)
# ------------------------------------------------------------------
x_anterior = None  # Guarda la posición X de la muñeca del frame anterior
direccion_anterior = 0  # 1 para derecha, -1 para izquierda
cambios_direccion = 0  # Contador de cuántas veces zigzagueó la mano
ultimo_cambio_tiempo = time.time()
tiempo_saludo_activo = 0  # Timer para que el cartel de saludo se quede unos frames en pantalla

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(time.time() * 1000)

        results = landmarker.detect_for_video(mp_image, timestamp)
        texto_gesto_pantalla = ""

        if results.hand_landmarks:
            for hand_landmarks, handedness in zip(results.hand_landmarks, results.shadow_handedness if hasattr(results,
                                                                                                               'shadow_handedness') else results.handedness):

                # Dibujar conexiones
                for connection in HAND_CONNECTIONS:
                    idx_start, idx_end = connection
                    start_pt = (int(hand_landmarks[idx_start].x * w), int(hand_landmarks[idx_start].y * h))
                    end_pt = (int(hand_landmarks[idx_end].x * w), int(hand_landmarks[idx_end].y * h))
                    cv2.line(frame, start_pt, end_pt, COLOR_LINEAS, 2)

                # Estado de los dedos
                indice_levantado = hand_landmarks[8].y < hand_landmarks[6].y
                mayor_levantado = hand_landmarks[12].y < hand_landmarks[10].y
                anular_levantado = hand_landmarks[16].y < hand_landmarks[14].y
                menique_levantado = hand_landmarks[20].y < hand_landmarks[18].y

                # ------------------------------------------------------------------
                # Lógica Dinámica del Saludo (Wave)
                # ------------------------------------------------------------------
                mano_abierta = indice_levantado and mayor_levantado and anular_levantado and menique_levantado

                if mano_abierta:
                    x_actual = int(hand_landmarks[0].x * w)  # X de la muñeca en píxeles

                    if x_anterior is not None:
                        movimiento = x_actual - x_anterior

                        # Definimos un umbral mínimo de movimiento para ignorar el pulso (5 píxeles)
                        if abs(movimiento) > 5:
                            direccion_actual = 1 if movimiento > 0 else -1

                            # Si la dirección actual cambió respecto a la anterior, hay un zigzag
                            if direccion_actual != direccion_anterior and direccion_anterior != 0:
                                cambios_direccion += 1
                                ultimo_cambio_tiempo = time.time()

                            direccion_anterior = direccion_actual

                    x_anterior = x_actual
                else:
                    # Si cierra la mano, reseteamos el rastro
                    x_anterior = None
                    direccion_anterior = 0

                # Si pasa mucho tiempo sin zigzaguear, bajamos el contador gradualmente
                if time.time() - ultimo_cambio_tiempo > 0.5:
                    cambios_direccion = max(0, cambios_direccion - 1)

                # DISPARADOR DEL SALUDO: Si metió al menos 3 cambios de dirección seguidos
                if cambios_direccion >= 3:
                    tiempo_saludo_activo = time.time() + 1.5  # Mantiene el cartel 1.5 segundos
                    cambios_direccion = 0  # Resetea para el próximo saludo

                # Comprobamos si el cartel de saludo debe seguir activo
                if time.time() < tiempo_saludo_activo:
                    texto_gesto_pantalla = "Hola! :D"

                # ------------------------------------------------------------------
                # Condición de Insulto
                # ------------------------------------------------------------------
                elif mayor_levantado and not indice_levantado and not anular_levantado and not menique_levantado:
                    texto_gesto_pantalla = "F U TOO! >:("
                    cambios_direccion = 0  # Evita conflictos

                # Dibujar identificador de mano y nodos
                label = handedness[0].category_name
                score = handedness[0].score
                wrist = hand_landmarks[0]
                text_x, text_y = int(wrist.x * w), int(wrist.y * h) - 20
                mano_texto = "Derecha" if label == "Right" else "Izquierda"

                cv2.putText(frame, f"{mano_texto} ({int(score * 100)}%)", (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXTO, 2, cv2.LINE_AA)

                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, COLOR_NODOS, cv2.FILLED)
        else:
            # Si no hay manos en pantalla, limpiamos la memoria
            x_anterior = None
            direccion_anterior = 0

        # Mostrar el gesto arriba si está activo
        if texto_gesto_pantalla:
            cv2.putText(frame, texto_gesto_pantalla, (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_GESTO, 4, cv2.LINE_AA)

        cv2.imshow('HandTracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()