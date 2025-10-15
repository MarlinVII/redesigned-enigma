import cv2
from detector import Detector
from window_capture import WindowCapture

# Используем только одну — нужную модель
model = {
    "model_name": "640_v2",
    "width": 1138,
    "height": 640,
    "shape": (640, 640)
}

# Инициализация окна и детектора
window = WindowCapture("Albion Online Client")
detector = Detector(model["model_name"], model["shape"])

while True:
    screenshot = window.get_screenshot()
    if screenshot is None:
        continue

    # Обнаружение объектов
    boxes, confidences, classes, batch_ids = detector.detect(screenshot)
    result = detector.debug_image(screenshot, boxes, confidences, classes, batch_ids)

    # Отображение изображения
    display = cv2.resize(result, (model["width"], model["height"]))
    cv2.imshow("Computer Vision", display)

    # Завершение по нажатию 'q'
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        exit(0)
