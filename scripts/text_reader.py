import cv2
import pytesseract
import pyttsx3
import speech_recognition as sr
import threading

# Setup Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    print("🔊", text)
    engine.say(text)
    engine.runAndWait()

# Global flag
reading_active = False

# Speech command listening
def listen_for_stop():
    global reading_active
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("🎤 Listening for 'stop reading'...")
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio).lower()
        print("✅ You said:", command)
        speak(f"You said {command}")
        if "stop reading" in command:
            reading_active = False
    except:
        pass  # Ignore errors quietly during live reading

def read_text_from_camera():
    global reading_active
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera not working.")
        return

    reading_active = True
    speak("Reading started. Say 'stop reading' to stop.")
    
    while reading_active:
        ret, frame = cap.read()
        if not ret:
            speak("Camera failed.")
            break

        cv2.imshow("📷 Vision Guardian", frame)

        # OCR text
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        if text.strip():
            print("📝", text.strip())
            speak(text.strip())

        # Start background listener
        t = threading.Thread(target=listen_for_stop)
        t.start()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            reading_active = False
            break

    cap.release()
    cv2.destroyAllWindows()
    speak("Reading stopped.")

# Wait for 'start reading'
def main():
    speak("Welcome to Vision Guardian.")
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        speak("Say 'start reading' to begin or 'exit' to quit.")
        with mic as source:
            print("🎤 Waiting for command...")
            audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio).lower()
            print("✅ You said:", command)
            speak(f"You said {command}")
            if "start reading" in command:
                read_text_from_camera()
            elif "exit" in command:
                speak("Goodbye.")
                break
        except:
            speak("Sorry, I didn’t catch that.")

if __name__ == "__main__":
    main()
