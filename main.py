# main.py
import sys
import time

import cv2
from imutils.video import VideoStream, FPS

from detection_module import YOLODetector, DetectionMonitor

MODEL_BASE_PATH = "yolo-rtmp"

if __name__ == "__main__":
    streaming_path = sys.argv[1]
    print(f'Streaming path: {streaming_path}')

    detector = YOLODetector(MODEL_BASE_PATH)
    monitor = DetectionMonitor(detector)

    vs = VideoStream(streaming_path).start()
    time.sleep(1.0)  # Allow camera sensor to warm up
    fps = FPS().start()
    print("[+] Starting stream reception via RTMP...")

    while True:
        frame = vs.read()
        frame = cv2.resize(frame, None, fx=YOLODetector.FRAME_RESCALE_FACTOR, fy=YOLODetector.FRAME_RESCALE_FACTOR)

        monitor.update_and_draw(frame)

        # Display the current frame
        cv2.imshow('Frame', frame)

        # Exit if the ESC key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

        fps.update()

    fps.stop()
    print(f"[+] Streaming ended. Total time: {fps.elapsed():.2f} seconds, Approx. FPS: {fps.fps():.2f}")
    cv2.destroyAllWindows()
    vs.stop()
