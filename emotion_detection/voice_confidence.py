import sounddevice as sd
import numpy as np
from collections import deque
import threading
import time

SAMPLE_RATE = 16000
WINDOW_DURATION = 1.0
ENERGY_THRESHOLD = 0.01

voice_conf_buffer = deque(maxlen=20)
current_voice_confidence = 50
running = True


def audio_loop():
    global current_voice_confidence

    while running:
        audio = sd.rec(
            int(WINDOW_DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        energy = np.mean(audio ** 2)

        if energy < ENERGY_THRESHOLD:
            score = 0.3
        else:
            score = 0.8

        voice_conf_buffer.append(score)
        current_voice_confidence = int(np.mean(voice_conf_buffer) * 100)


def start_audio_thread():
    thread = threading.Thread(target=audio_loop, daemon=True)
    thread.start()


def get_voice_confidence():
    return current_voice_confidence
