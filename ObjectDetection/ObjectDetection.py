import cv2
from ultralytics import YOLO

# 1. Cargamos el modelo pre-entrenado de YOLOv8 (versión nano, la más rápida)
model = YOLO('yolov8n.pt')

# 2. Captura de la webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Pasar el frame por la Inteligencia Artificial
    # stream=True hace que el procesamiento de video sea súper eficiente en memoria
    results = model(frame, stream=True)

    # 4. Dibujar los resultados en la pantalla
    for r in results:
        annotated_frame = r.plot()

    # 5. Mostrar el frame anotado
    cv2.imshow("YOLOv8 - Detección de Objetos", annotated_frame)

    # Romper con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()