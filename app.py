from flask import Flask, Response
from ultralytics import YOLO
from detect_hands import detect_hands_raised
import cv2, threading, time
import numpy as np
import os

# Load model, version 11, nano, with pose
model = YOLO("yolo11n-pose.pt")

app = Flask(__name__)

latest_frame = None
lock = threading.Lock()
stop_flag = False

def capture_frames():
    """Continuously capture the latest frame (drop old ones)."""
    global latest_frame, stop_flag

    # Use the default webcam and use 1280x720
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print('Cannot open camera stream')
        stop_flag = True
        return

    while not stop_flag:
        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame
        else:
            print('Error reading from stream')
        
        time.sleep(1/30)

    cap.release()


def gen_frames():
    """YOLO + stream the annotated frames via Flask."""
    global latest_frame, stop_flag

    while not stop_flag:

        # Fetch a frame
        if latest_frame is None:
            print('Error fetching a frame')
            time.sleep(1/30)
            continue
            
        with lock:
            frame = latest_frame.copy()

        # Run YOLO pose tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
        # results = model.predict(frame)

        # Extract the landmarks
        keypoints = results[0].keypoints.xy.cpu().numpy()

        # Draw keypoints, boxes, etc.
        frame = results[0].plot()
        
        # Annotate raised hands!
        detect_hands_raised(keypoints, frame)

        # Encode to JPEG for browser
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print('Error encoding frame to JPEG')
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start background capture thread
    t = threading.Thread(target=capture_frames, daemon=True)
    t.start()

    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        stop_flag = True
        t.join()
