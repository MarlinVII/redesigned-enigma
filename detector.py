import cv2
import numpy as np
import onnxruntime
import random

class Detector:
    def __init__(self, model_name, shape):
        self.labels = self.load_labels(f"./model/{model_name}/labels.txt")
        self.model = self.load_model(f"./model/{model_name}/best.onnx")
        self.colors = [tuple(random.choices(range(256), k=3)) for _ in range(len(self.labels))]

        # Сохраняем размер модели как кортеж
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = shape

    def load_labels(self, label_path):
        with open(label_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def load_model(self, model_path):
        return onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    def preprocess_image(self, image):
        # Принудительное изменение размера изображения к размеру модели (например 640x640)
        image = cv2.resize(image, self.shape, interpolation=cv2.INTER_LINEAR)
        image = image.transpose((2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, 0)    # Add batch dim
        image = np.ascontiguousarray(image).astype(np.float32) / 255.0
        return image

    def detect(self, image, confidence_threshold=0.5, iou_threshold=0.5):
        im = self.preprocess_image(image)
        outputs = self.model.run(None, {'images': im})[0]
        return self.process_output(outputs, confidence_threshold, iou_threshold)

    def process_output(self, output, confidence_threshold=0.5, iou_threshold=0.5):
        # YOLOv7 формат: [batch, boxes, 85], где 85 = [x, y, w, h, conf, class_scores...]
        predictions = output[0]  # [num_boxes, 85]
        boxes = []
        confidences = []
        class_ids = []

        for pred in predictions:
            scores = pred[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                x_center, y_center, width, height = pred[0:4]
                x = int(x_center - width / 2)
                y = int(y_center - height / 2)
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(int(class_id))

        # Non-Max Suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, iou_threshold)

        result_boxes = []
        result_confidences = []
        result_classes = []
        result_batch_ids = []

        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            result_boxes.append(boxes[i])
            result_confidences.append(confidences[i])
            result_classes.append(class_ids[i])
            result_batch_ids.append(0)  # batch_id = 0 since batch size is 1

        return result_boxes, result_confidences, result_classes, result_batch_ids

    def debug_image(self, image, boxes, confidences, classes, batch_ids):
        for box, conf, cls in zip(boxes, confidences, classes):
            x, y, w, h = box
            color = self.colors[cls]
            label = f"{self.labels[cls]} {conf:.2f}"
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return image
