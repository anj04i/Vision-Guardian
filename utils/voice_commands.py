import speech_recognition as sr
import pyttsx3
import threading

# Global stop event to signal shutdown
stop_event = threading.Event()

# === Init TTS engine ===
engine = pyttsx3.init()
engine.setProperty("rate", 150)

def speak(text):
    print(f"🔈 {text}")
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"❌ TTS error: {e}")

def listen_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening for command...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
        except sr.WaitTimeoutError:
            print("⌛ Listening timed out.")
            return ""
    try:
        command = recognizer.recognize_google(audio)
        print("✅ You said:", command)
        return command.lower()
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
        return ""
    except sr.RequestError:
        print("🚫 Could not request results from Google.")
        return ""

def handle_command(command):
    if "who are you" in command:
        speak("I am Vision Guardian. Your AI assistant.")
    elif "hello" in command:
        speak("Hello! How can I help you?")
    elif "stop detection" in command or "exit" in command:
        speak("Goodbye! Exiting now.")
        stop_event.set()
    else:
        speak("Sorry, I didn't understand that command.")

def voice_command_loop():
    while not stop_event.is_set():
        command = listen_command()
        if command:
            handle_command(command)