# detection_module.py
import os

import cv2
import numpy as np


class YOLODetector:
    MIN_CONFIDENCE = 0.4
    NMS_THRESHOLD = 0.2
    INPUT_WIDTH = 416
    INPUT_HEIGHT = 416
    FRAME_RESCALE_FACTOR = 0.7
    INTERESTED_CLASS_IDS = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck'}

    def __init__(self, model_base_path):
        self.model_base_path = model_base_path
        self.labels, self.colors, self.net, self.unconnected_layers = self.load_model_and_labels()

    def load_model_and_labels(self):
        label_path = os.path.sep.join([self.model_base_path, 'coco.names'])
        with open(label_path) as f:
            labels_list = f.read().strip().split("\n")

        np.random.seed(42)
        colors_list = np.random.randint(0, 255, size=(len(labels_list), 3), dtype="uint8")

        config_path = os.path.sep.join([self.model_base_path, 'yolov3.cfg'])
        weights_path = os.path.sep.join([self.model_base_path, 'yolov3.weights'])
        read_net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

        layer_names = read_net.getLayerNames()
        unconnected_out_layers = [layer_names[i - 1] for i in read_net.getUnconnectedOutLayers().flatten()]

        return labels_list, colors_list, read_net, unconnected_out_layers

    def detect_objects(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (YOLODetector.INPUT_WIDTH, YOLODetector.INPUT_HEIGHT),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_output = self.net.forward(self.unconnected_layers)
        return layer_output

    @staticmethod
    def process_detections(frame, layer_outputs):
        H, W = frame.shape[:2]
        boxes, confidences, class_ids = [], [], []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id in YOLODetector.INTERESTED_CLASS_IDS and confidence > YOLODetector.MIN_CONFIDENCE:
                    box = detection[0:4] * np.array([W, H, W, H])
                    centerX, centerY, box_width, box_height = box.astype("int")

                    x = int(centerX - (box_width / 2))
                    y = int(centerY - (box_height / 2))

                    boxes.append([x, y, int(box_width), int(box_height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    def draw_boxes(self, frame, boxes, confidences, class_ids):
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, YOLODetector.MIN_CONFIDENCE, YOLODetector.NMS_THRESHOLD)
        counts = {label: 0 for label in YOLODetector.INTERESTED_CLASS_IDS.values()}

        if len(idxs) > 0:
            for i in idxs.flatten():
                if confidences[i] < 0.8:
                    continue
                x, y, w, h = boxes[i]
                color = [int(c) for c in self.colors[class_ids[i]]]
                label = self.labels[class_ids[i]]
                confidence = confidences[i]

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = f"{label}: {confidence:.4f}"
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if label.lower() in counts:
                    counts[label] += 1

        total_count = sum(counts.values())
        if total_count > 0:  # Only print if there is at least one detection
            print(", ".join([f"{key}: {value}" for key, value in counts.items() if value > 0]),
                  f", Total count: {total_count}")
        return counts


class DetectionMonitor:
    def __init__(self, detector):
        self.detector = detector
        self.previous_counts = None

    def update_and_draw(self, frame):
        layer_outputs = self.detector.detect_objects(frame)
        boxes, confidences, class_ids = self.detector.process_detections(frame, layer_outputs)
        current_counts = self.detector.draw_boxes(frame, boxes, confidences, class_ids)

        # Only print the counts if there's a change
        if self.previous_counts != current_counts:
            self.previous_counts = current_counts
