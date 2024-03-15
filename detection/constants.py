# detection/constants.py

MIN_CONFIDENCE = 0.4
NMS_THRESHOLD = 0.2
INPUT_WIDTH = 416
INPUT_HEIGHT = 416
FRAME_RESCALE_FACTOR = 0.7
INTERESTED_CLASS_IDS = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorbike', 5: 'bus', 7: 'truck'}

CONFIG_PATH = 'yolov3.cfg'
WEIGHTS_PATH = 'yolov3.weights'
LABELS_PATH = 'coco.names'

MODEL_BASE_PATH = 'yolo-rtmp'
