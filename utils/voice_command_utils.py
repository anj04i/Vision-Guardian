import speech_recognition as sr
import pyttsx3
import threading

engine = pyttsx3.init()
engine.setProperty("rate", 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

def listen_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening for command...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

    try:
        command = r.recognize_google(audio)
        print("🔊 You said:", command)
        return command.lower()
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
        return ""
    except sr.RequestError:
        print("🚫 Speech recognition service error")
        return ""

def handle_command(command):
    if "read text" in command:
        speak("Starting text reading now.")
        # TODO: trigger OCR function here
    elif "detect object" in command:
        speak("Object detection activated.")
        # TODO: trigger object detection
    elif "exit" in command or "stop" in command:
        speak("Goodbye!")
        exit()
    else:
        speak("Sorry, I didn't understand that command.")

def voice_command_loop():
    while True:
        command = listen_command()
        if command:
            handle_command(command)

# To run in background: threading.Thread(target=voice_command_loop, daemon=True).start()
