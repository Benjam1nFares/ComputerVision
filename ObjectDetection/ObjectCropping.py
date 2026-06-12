import cv2
from ultralytics import YOLO

# 1. Cargamos el modelo
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 2. Procesar el frame con YOLO
    results = model(frame, stream=True)

    for r in results:
        # Extraemos las cajas de detección (boxes)
        for box in r.boxes:

            # A. Obtener las coordenadas de la caja en formato (x1, y1, x2, y2)
            # x1, y1 = esquina superior izquierda | x2, y2 = esquina inferior derecha
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convertimos a enteros para OpenCV

            # B. Obtener el índice de la clase detectada (ID numérico) y su nombre
            cls = int(box.cls[0])
            label = model.names[cls]

            # C. Obtener la confianza (porcentaje de certeza)
            conf = float(box.conf[0])

            # FILTRO: Para el lector de dorsales solo nos interesan personas por ahora.
            # Si el objeto es "person" y la confianza es mayor al 50%, operamos:
            if label == "person" and conf > 0.5:

                # ------------------------------------------------------------------
                # [EL TRUCO DE CV]: Recortar la región de interés (ROI)
                # En las matrices de OpenCV/NumPy se indexa primero Filas(Y) y luego Columnas(X)
                # ------------------------------------------------------------------
                recorte_persona = frame[y1:y2, x1:x2]

                # Si el recorte es válido (evita errores si la caja toca el borde de la pantalla)
                if recorte_persona.size > 0:
                    cv2.imshow("Recorte en Tiempo Real", recorte_persona)

                # Dibuja la caja rectangular clásica encima del frame original
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Pone el texto con el nombre y la confianza
                texto = f"{label} {conf:.2f}"
                cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar la cámara general
    cv2.imshow("YOLOv8 - Extrayendo Coordenadas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()