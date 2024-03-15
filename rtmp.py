import cv2
import subprocess as sp
import numpy as np

rtmp_url = 'rtmp://localhost:1935/live/stream'

ffmpeg_command = [
    'ffmpeg',
    '-i', rtmp_url,
    '-f', 'image2pipe',
    '-pix_fmt', 'bgr24',
    '-vcodec', 'rawvideo',
    '-an',
    '-'
]

process = sp.Popen(ffmpeg_command, stdout=sp.PIPE, bufsize=10**8)

window_name = 'RTMP Stream'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

frame_width = 640
frame_height = 480
frame_size = frame_width * frame_height * 3

while True:

    raw_image = process.stdout.read(frame_size)
    frame = np.frombuffer(raw_image, np.uint8).reshape((frame_height, frame_width, 3))

    if frame.size == frame_size:
        cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    process.stdout.flush()

cv2.destroyAllWindows()
process.terminate()
