import cv2
from ultralytics import YOLO
import pyttsx3
import time

# Initialize TTS engine
tts = pyttsx3.init()
tts.setProperty('rate', 150)

# Load YOLOv5 model (small and fast)
model = YOLO('yolov8n.pt')  # You can use yolov5n.pt or yolov8n.pt

# Open webcam
cap = cv2.VideoCapture(0)
last_spoken = ""
last_time = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame)
        names = model.names if hasattr(model, 'names') else results[0].names
        detected_labels = set()

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                detected_labels.add(label)
                # Draw rectangle and label
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Describe objects every 3 seconds or if different
        if detected_labels:
            description = "I see " + ", ".join(detected_labels)
            if description != last_spoken or time.time() - last_time > 3:
                tts.say(description)
                tts.runAndWait()
                last_spoken = description
                last_time = time.time()

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()