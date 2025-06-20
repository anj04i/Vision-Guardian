from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import pyttsx3
import threading
import time
from queue import Queue

app = Flask(__name__)

# Initialize components
model = YOLO('yolov8n.pt')
camera = cv2.VideoCapture(0)

# TTS Engine setup
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
last_spoken = ""
last_time = 0

# Thread-safe queue for TTS
tts_queue = Queue()
current_mode = None  # 'object', 'text', None

def tts_worker():
    """Background thread to handle TTS"""
    while True:
        text = tts_queue.get()
        if text == "SHUTDOWN":
            break
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")

# Start TTS thread
tts_thread = threading.Thread(target=tts_worker)
tts_thread.daemon = True
tts_thread.start()

def generate_frames():
    global last_spoken, last_time
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        if current_mode == 'object':
            results = model(frame)
            names = model.names
            detected_labels = set()

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = names[cls_id]
                    detected_labels.add(label)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Announce objects
            if detected_labels:
                description = "I see " + ", ".join(detected_labels)
                if description != last_spoken or time.time() - last_time > 3:
                    tts_queue.put(description)
                    last_spoken = description
                    last_time = time.time()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<mode>')
def set_mode(mode):
    global current_mode
    current_mode = mode if mode in ['object', 'text'] else None
    return {"status": "success", "mode": current_mode}

def cleanup():
    tts_queue.put("SHUTDOWN")
    camera.release()
    tts_thread.join()

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        cleanup()