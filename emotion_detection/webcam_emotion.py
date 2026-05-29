import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque
from questions import QUESTIONS
from voice_confidence import get_voice_confidence, start_audio_thread

# Load trained model
model = load_model("models/emotion_model.h5")

# Emotion labels (must match training order)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

emotion_buffer = deque(maxlen=15)

confidence_map = {
    "happy": 1.0,
    "neutral": 0.9,
    "surprise": 0.5,
    "sad": 0.3,
    "fear": 0.2,
    "angry": 0.4,
    "disgust": 0.3
}

confidence_buffer = deque(maxlen=30)

baseline_buffer = deque(maxlen=50)
baseline_emotion = None

questions = QUESTIONS
current_question = 0
question_start_time = cv2.getTickCount()
QUESTION_TIME = 15  # seconds

start_audio_thread()

last_emotion = "Detecting..."
last_confidence = 0




while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    voice_confidence = get_voice_confidence()

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        prediction = model.predict(face, verbose=0)
        emotion = emotion_labels[np.argmax(prediction)]

        emotion_buffer.append(emotion)

        if len(emotion_buffer) == emotion_buffer.maxlen:
            final_emotion = max(set(emotion_buffer), key=emotion_buffer.count)
        else:
            final_emotion = emotion

        # Baseline calibration (first few seconds)
        if baseline_emotion is None:
            baseline_buffer.append(final_emotion)

            if len(baseline_buffer) == baseline_buffer.maxlen:
                baseline_emotion = max(set(baseline_buffer), key=baseline_buffer.count)

        # Confidence calculation
        if baseline_emotion and final_emotion == baseline_emotion:
            confidence_buffer.append(0.9)
        else:
            confidence_buffer.append(confidence_map[final_emotion])

        confidence_score = int(np.mean(confidence_buffer) * 100)

        last_emotion = final_emotion
        last_confidence = confidence_score


        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        


        cv2.putText(
            frame,
            f"Voice Confidence: {voice_confidence}%",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 165, 0),
            2
        )


        if baseline_emotion:
            cv2.putText(
                frame,
                f"Baseline: {baseline_emotion}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

    elapsed_time = (cv2.getTickCount() - question_start_time) / cv2.getTickFrequency()

    if elapsed_time > QUESTION_TIME:
        current_question = (current_question + 1) % len(questions)
        question_start_time = cv2.getTickCount()
    

    cv2.putText(
        frame,
        f"Confidence: {last_confidence}%",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

cv2.putText(
        frame,
        f"Emotion: {last_emotion}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.putText(
    frame,
    f"Q: {questions[current_question]}",
    (20, frame.shape[0] - 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 255, 255),
    2
    )

    cv2.imshow("AI Interviewer - Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
