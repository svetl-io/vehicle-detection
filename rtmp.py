import numpy as np
import os
import cv2
import time
from imutils.video import VideoStream
from imutils.video import FPS
import sys

# Model constants
MIN_CONFIDENCE = 0.4
NMS_THRESHOLD = 0.2
MODEL_BASE_PATH = "yolo-rtmp"

# Function to load the model and class labels
def load_model_and_labels():
    # Print status
    print("[+] Loading class labels...")
    # Construct the path to the labels file and load the labels
    label_path = os.path.sep.join([MODEL_BASE_PATH, 'coco.names'])
    with open(label_path) as f:
        labels = f.read().strip().split("\n")

    # Initialize the random seed and generate unique colors for each label
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    # Print status
    print("[+] Loading the YOLO model trained on the COCO dataset...")
    # Construct the paths to the configuration and weights files, and load the network
    config_path = os.path.sep.join([MODEL_BASE_PATH, 'yolov3.cfg'])
    weights_path = os.path.sep.join([MODEL_BASE_PATH, 'yolov3.weights'])
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # Get the names of all layers in the network
    layer_names = net.getLayerNames()
    # Get the indices of the output layers, ensuring compatibility with all OpenCV versions
    unconnected_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # Return the loaded labels, colors, network, and unconnected layer indices
    return labels, colors, net, unconnected_layers


# Parse streaming path from script argument
streaming_path = sys.argv[1]
print(f'Streaming path: {streaming_path}')

# Load model and labels
labels, colors, net, unconnected_layers = load_model_and_labels()

# Start receiving the stream
vs = VideoStream(streaming_path).start()
time.sleep(1.0)  # Allow camera sensor to warm up
fps = FPS().start()
print("[+] Starting stream reception via RTMP...")

interested_classes = {0: 'Person', 1: 'Bicycle', 2: 'Car', 3: 'Motorbike', 5: 'Bus', 7: 'Truck'}

# Iterate over the frames from the stream
while True:
    frame = vs.read()
    frame = cv2.resize(frame, None, fx=0.7, fy=0.7)
    (H, W) = frame.shape[:2]

    # Construct a blob from the frame and perform a forward pass using YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(unconnected_layers)

    boxes, confidences, class_ids = [], [], []
    # After obtaining detections from the YOLO model
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check if the detected class_id is in our list of interest
            if class_id in [0, 1, 2, 3, 5, 7] and confidence > MIN_CONFIDENCE:
                # Scale the bounding box back to the frame's dimensions
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.4f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the current frame
    cv2.imshow('Frame', frame)

    # Exit if the ESC key is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    fps.update()

fps.stop()
print(f"[+] Streaming ended. Total time: {fps.elapsed():.2f} seconds, Approx. FPS: {fps.fps():.2f}")
cv2.destroyAllWindows()
vs.stop()
