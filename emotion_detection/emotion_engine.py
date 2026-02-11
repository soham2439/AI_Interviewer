import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

class EmotionEngine:
    def __init__(self, model_path):
        self.model = load_model(model_path)

        self.emotion_labels = [
            'angry', 'disgust', 'fear',
            'happy', 'sad', 'surprise', 'neutral'
        ]

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.emotion_buffer = deque(maxlen=15)

        self.confidence_map = {
            "happy": 1.0,
            "neutral": 0.9,
            "surprise": 0.6,
            "sad": 0.4,
            "fear": 0.3,
            "angry": 0.5,
            "disgust": 0.4
        }

        self.confidence_buffer = deque(maxlen=30)

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        final_emotion = None
        confidence = None

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.reshape(face, (1, 48, 48, 1))

            preds = self.model.predict(face, verbose=0)
            emotion = self.emotion_labels[np.argmax(preds)]

            self.emotion_buffer.append(emotion)

            if len(self.emotion_buffer) == self.emotion_buffer.maxlen:
                final_emotion = max(
                    set(self.emotion_buffer),
                    key=self.emotion_buffer.count
                )
            else:
                final_emotion = emotion

            self.confidence_buffer.append(
                self.confidence_map.get(final_emotion, 0.5)
            )

            confidence = int(np.mean(self.confidence_buffer) * 100)

            break  # Only one face needed

        return final_emotion, confidence
