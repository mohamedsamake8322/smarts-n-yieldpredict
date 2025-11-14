import speech_recognition as sr

def get_voice_input() -> str:
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio, language='en-US')
    except sr.UnknownValueError:
        return "Sorry, I didn't understand."
    except sr.RequestError:
        return "Connection error with the voice service."
