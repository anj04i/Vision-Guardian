import cv2
import time
import json
import pyttsx3
import threading
import speech_recognition as sr
from ultralytics import YOLO
from utils.voice_commands import voice_command_loop  # ✅ voice command logic from utils

# === Init TTS ===
tts = pyttsx3.init()
tts.setProperty('rate', 150)

# === Load danger labels + messages ===
with open("danger_objects.json", "r") as f:
    danger_objects = json.load(f)

# === Load YOLOv8 ===
model = YOLO('yolov8n.pt')

# === Load Haar Cascade for Face Detection ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Start Voice Command in Background ===
threading.Thread(target=voice_command_loop, daemon=True).start()

# === Open Webcam ===
cap = cv2.VideoCapture(0)
last_spoken = ""
last_time = 0
spoken_labels = set()
last_clear_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_dangers = set()

        # === YOLO Object Detection ===
        results = model(frame)
        names = model.names if hasattr(model, 'names') else results[0].names

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id].lower()

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Check if it's a dangerous object
                if label in danger_objects:
                    current_dangers.add(label)

        # === Speak Dangerous Object Alerts ===
        for danger in current_dangers:
            if (danger not in spoken_labels) or (time.time() - last_time > 5):
                alert = danger_objects[danger]
                tts.say(alert)
                tts.runAndWait()
                last_spoken = alert
                last_time = time.time()
                spoken_labels.add(danger)

        # === Face Detection (Haar Cascade) ===
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if "face" not in spoken_labels:
                tts.say("Face detected nearby.")
                tts.runAndWait()
                spoken_labels.add("face")
                last_time = time.time()

        # === Clear spoken labels every 30 seconds ===
        if time.time() - last_clear_time > 30:
            spoken_labels.clear()
            last_clear_time = time.time()

        # === Show Feed ===
        cv2.imshow("Vision Guardian", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

finally:
    cap.release()
    cv2.destroyAllWindows()






# import cv2
# from ultralytics import YOLO
# import pyttsx3
# import time
# import json
# import threading
# import speech_recognition as sr

# # === Init TTS ===
# tts = pyttsx3.init()
# tts.setProperty('rate', 150)

# # === Load danger labels + messages ===
# with open("danger_objects.json", "r") as f:
#     danger_objects = json.load(f)

# # === Load YOLOv8 ===
# model = YOLO('yolov8n.pt')

# # === Load Haar Cascade for Face Detection ===
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # === Voice Command Function ===
# def listen_for_commands():
#     recognizer = sr.Recognizer()
#     mic = sr.Microphone()
#     with mic as source:
#         recognizer.adjust_for_ambient_noise(source)
#     while True:
#         with mic as source:
#             try:
#                 print("🎙️ Listening for command...")
#                 audio = recognizer.listen(source, timeout=5)
#                 command = recognizer.recognize_google(audio).lower()
#                 print("🗣️ You said:", command)

#                 # === Add your voice command actions here ===
#                 if "who are you" in command:
#                     tts.say("I am Vision Guardian. Your AI assistant.")
#                     tts.runAndWait()
#                 elif "stop detection" in command:
#                     tts.say("Stopping detection. Goodbye.")
#                     tts.runAndWait()
#                     break
#                 elif "hello" in command:
#                     tts.say("Hello! How can I help you?")
#                     tts.runAndWait()
#                 # Add more commands here
#             except sr.WaitTimeoutError:
#                 continue
#             except sr.UnknownValueError:
#                 continue
#             except Exception as e:
#                 print("Voice error:", e)

# # === Start voice command in background ===
# command_thread = threading.Thread(target=listen_for_commands, daemon=True)
# command_thread.start()

# # === Open Webcam ===
# cap = cv2.VideoCapture(0)
# last_spoken = ""
# last_time = 0
# spoken_labels = set()
# last_clear_time = time.time()

# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         current_dangers = set()

#         # === YOLO Object Detection ===
#         results = model(frame)
#         names = model.names if hasattr(model, 'names') else results[0].names

#         for result in results:
#             for box in result.boxes:
#                 cls_id = int(box.cls[0])
#                 label = names[cls_id].lower()

#                 # Draw bounding box
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#                 # Check if it's a dangerous object
#                 if label in danger_objects:
#                     current_dangers.add(label)

#         # === Speak Dangerous Object Alerts ===
#         for danger in current_dangers:
#             if (danger not in spoken_labels) or (time.time() - last_time > 5):
#                 alert = danger_objects[danger]
#                 tts.say(alert)
#                 tts.runAndWait()
#                 last_spoken = alert
#                 last_time = time.time()
#                 spoken_labels.add(danger)

#         # === Face Detection (Haar Cascade) ===
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         if len(faces) > 0:
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#             if "face" not in spoken_labels:
#                 tts.say("Face detected nearby.")
#                 tts.runAndWait()
#                 spoken_labels.add("face")
#                 last_time = time.time()

#         # === Clear spoken labels every 30 seconds ===
#         if time.time() - last_clear_time > 30:
#             spoken_labels.clear()
#             last_clear_time = time.time()

#         # === Show Feed ===
#         cv2.imshow("Vision Guardian", frame)
#         if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
#             break

# finally:
#     cap.release()
#     cv2.destroyAllWindows()