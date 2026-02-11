import speech_recognition as sr

def record_answer(timeout=120):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source, phrase_time_limit=timeout)

    try:
        text = r.recognize_google(audio)
        print("📝 Answer:", text)
        return text
    except sr.UnknownValueError:
        return "[No clear answer detected]"
    except sr.RequestError:
        return "[Speech service error]"
