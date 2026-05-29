import datetime
import threading
import time
import tkinter as tk
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as PDFImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from emotion_engine import EmotionEngine
from questions import QUESTIONS
from voice_input import record_answer


# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "emotion_model.h5"
REPORT_PATH = BASE_DIR / "AI_Interview_Report.pdf"
CONFIDENCE_CHART_PATH = BASE_DIR / "confidence_chart.png"
ANSWER_CHART_PATH = BASE_DIR / "answer_score_chart.png"

QUESTION_TIME = 120
FRAME_DELAY_MS = 30

questions = QUESTIONS
current_question = 0
question_start_time = None

question_log = []
recording_thread = None
current_answer = ""
recording_active = False
finishing = False

current_emotion_scores = []
current_emotions = []

emotion_engine = EmotionEngine(str(MODEL_PATH))


# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam is not accessible. Check camera permissions and try again.")


# =========================
# GUI SETUP
# =========================
root = tk.Tk()
root.title("AI Interviewer")
root.geometry("960x760")
root.configure(bg="#f6f7fb")

header = tk.Frame(root, bg="#f6f7fb")
header.pack(fill="x", padx=24, pady=(18, 8))

title_label = tk.Label(
    header,
    text="AI Interviewer",
    font=("Helvetica", 22, "bold"),
    bg="#f6f7fb",
    fg="#101828",
)
title_label.pack(anchor="w")

progress_label = tk.Label(
    header,
    text="",
    font=("Helvetica", 11),
    bg="#f6f7fb",
    fg="#667085",
)
progress_label.pack(anchor="w", pady=(2, 0))

question_label = tk.Label(
    root,
    text="",
    font=("Helvetica", 18, "bold"),
    wraplength=850,
    justify="center",
    bg="#f6f7fb",
    fg="#101828",
)
question_label.pack(pady=(10, 10))

timer_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#f6f7fb", fg="#344054")
timer_label.pack()

video_label = tk.Label(root, bg="#101828")
video_label.pack(pady=12)

metrics_label = tk.Label(
    root,
    text="Emotion: Detecting... | Confidence: --%",
    font=("Helvetica", 12),
    bg="#f6f7fb",
    fg="#344054",
)
metrics_label.pack(pady=(0, 4))

status_label = tk.Label(root, text="", font=("Helvetica", 12), bg="#f6f7fb", fg="#027a48")
status_label.pack(pady=5)


# =========================
# SCORING
# =========================
def clamp(value, minimum=0, maximum=100):
    return max(minimum, min(maximum, int(round(value))))


def score_answer(answer_text):
    cleaned = answer_text.strip()
    if not cleaned or cleaned.startswith("["):
        return 20

    lowered = cleaned.lower()
    words = [word.strip(".,!?;:()[]{}\"'").lower() for word in cleaned.split()]
    words = [word for word in words if word]
    unique_ratio = len(set(words)) / max(len(words), 1)

    length_score = min(len(words) / 70, 1.0) * 45
    detail_score = min(cleaned.count(".") + cleaned.count(",") + lowered.count(" because "), 6) * 4
    clarity_score = unique_ratio * 25
    filler_penalty = sum(words.count(word) for word in ("um", "uh", "like", "actually")) * 2

    return clamp(25 + length_score + detail_score + clarity_score - filler_penalty)


def summarize_emotions(emotions):
    if not emotions:
        return "Not detected"

    return max(set(emotions), key=emotions.count).title()


def average_confidence(scores):
    if not scores:
        return 50

    return clamp(sum(scores) / len(scores))


# =========================
# LOGGING
# =========================
def log_question(answer_text):
    global question_start_time, current_emotion_scores, current_emotions

    if question_start_time is None:
        return

    end_time = time.time()
    emotion_confidence = average_confidence(current_emotion_scores)

    question_log.append(
        {
            "question": questions[current_question],
            "answer": answer_text or "[No answer captured]",
            "start_time": datetime.datetime.fromtimestamp(question_start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
            "emotion": summarize_emotions(current_emotions),
            "emotion_confidence": emotion_confidence,
            "answer_score": score_answer(answer_text or ""),
        }
    )

    current_emotion_scores = []
    current_emotions = []


# =========================
# RECORDING
# =========================
def start_recording():
    global current_answer, recording_active

    recording_active = True
    try:
        current_answer = record_answer(timeout=QUESTION_TIME)
    except Exception as exc:
        current_answer = f"[Recording error: {exc}]"
    finally:
        recording_active = False


def begin_question():
    global question_start_time, current_answer, recording_thread

    question_start_time = time.time()
    current_answer = ""
    status_label.config(text="Recording answer...")
    recording_thread = threading.Thread(target=start_recording, daemon=True)
    recording_thread.start()


# =========================
# NEXT QUESTION
# =========================
def next_question():
    global current_question, question_start_time

    if finishing or question_start_time is None:
        return

    next_btn.config(state="disabled")
    status_label.config(text="Saving answer...")
    root.update_idletasks()

    if recording_active and recording_thread is not None:
        recording_thread.join(timeout=1.5)

    log_question(current_answer)
    current_question += 1

    if current_question >= len(questions):
        finish_interview()
        return

    question_start_time = None
    next_btn.config(state="normal")
    status_label.config(text="")
    update_question_text()


def update_question_text():
    progress_label.config(text=f"Question {current_question + 1} of {len(questions)}")
    question_label.config(text=questions[current_question])


# =========================
# UPDATE FRAME
# =========================
def update_frame():
    global question_start_time

    if finishing:
        return

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        emotion, confidence = emotion_engine.process_frame(frame)

        if confidence is not None:
            current_emotion_scores.append(confidence)
        if emotion:
            current_emotions.append(emotion)

        label_emotion = emotion.title() if emotion else "Detecting..."
        label_confidence = f"{confidence}%" if confidence is not None else "--%"
        metrics_label.config(text=f"Emotion: {label_emotion} | Confidence: {label_confidence}")

        cv2.putText(
            frame,
            f"{label_emotion} {label_confidence}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (51, 204, 153),
            2,
        )

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img.thumbnail((860, 480))
        imgtk = ImageTk.PhotoImage(img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    if question_start_time is None:
        begin_question()

    elapsed = int(time.time() - question_start_time)
    remaining = QUESTION_TIME - elapsed

    if remaining <= 0:
        next_question()
        return

    mins, secs = divmod(max(remaining, 0), 60)
    timer_label.config(text=f"Time Remaining: {mins:02d}:{secs:02d}")
    update_question_text()

    root.after(FRAME_DELAY_MS, update_frame)


# =========================
# PDF REPORT
# =========================
def add_chart(path, title, values, color):
    idx = list(range(1, len(values) + 1))
    plt.figure(figsize=(8, 3))
    plt.bar(idx, values, color=color)
    plt.ylim(0, 100)
    plt.xticks(idx)
    plt.title(title)
    plt.xlabel("Question")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def pdf_text(value):
    return escape(str(value)).replace("\n", "<br/>")


def generate_pdf_report():
    if not question_log:
        return

    doc = SimpleDocTemplate(str(REPORT_PATH), pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>AI Interview - Final Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(
        Paragraph(
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<b>Question-wise Analysis</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    for i, item in enumerate(question_log, 1):
        text = f"""
        <b>Q{i}:</b> {pdf_text(item['question'])}<br/>
        <b>Answer:</b> {pdf_text(item['answer'])}<br/>
        Dominant Emotion: {pdf_text(item['emotion'])}<br/>
        Confidence: {item['emotion_confidence']}%<br/>
        Answer Score: {item['answer_score']}%
        """
        elements.append(Paragraph(text, styles["Normal"]))
        elements.append(Spacer(1, 10))

    conf = [item["emotion_confidence"] for item in question_log]
    score = [item["answer_score"] for item in question_log]

    add_chart(CONFIDENCE_CHART_PATH, "Confidence per Question", conf, "#2e90fa")
    add_chart(ANSWER_CHART_PATH, "Answer Score per Question", score, "#12b76a")

    elements.append(PDFImage(str(CONFIDENCE_CHART_PATH), 6 * inch, 3 * inch))
    elements.append(Spacer(1, 20))
    elements.append(PDFImage(str(ANSWER_CHART_PATH), 6 * inch, 3 * inch))

    avg_conf = clamp(sum(conf) / len(conf))
    avg_score = clamp(sum(score) / len(score))
    overall = clamp((avg_conf * 0.45) + (avg_score * 0.55))
    verdict = "Excellent" if overall >= 80 else "Good" if overall >= 65 else "Needs Improvement"

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<b>Final Evaluation</b>", styles["Heading2"]))
    elements.append(Paragraph(f"Overall Confidence: {avg_conf}%", styles["Normal"]))
    elements.append(Paragraph(f"Overall Answer Accuracy: {avg_score}%", styles["Normal"]))
    elements.append(Paragraph(f"Overall Interview Score: {overall}%", styles["Normal"]))
    elements.append(Paragraph(f"<b>Verdict:</b> {verdict}", styles["Normal"]))

    doc.build(elements)


# =========================
# FINISH
# =========================
def finish_interview():
    global finishing

    if finishing:
        return

    finishing = True
    next_btn.config(state="disabled")
    status_label.config(text="Generating report...")
    root.update_idletasks()

    try:
        if question_start_time is not None and current_question < len(questions):
            log_question(current_answer)
        generate_pdf_report()
    finally:
        cap.release()
        root.destroy()

    print(f"Interview finished. PDF generated: {REPORT_PATH}")


# =========================
# BUTTON
# =========================
next_btn = tk.Button(
    root,
    text="Next Question",
    font=("Helvetica", 13, "bold"),
    bg="#1d4ed8",
    fg="white",
    activebackground="#1e40af",
    activeforeground="white",
    padx=20,
    pady=8,
    borderwidth=0,
    command=next_question,
)
next_btn.pack(pady=15)


# =========================
# CLOSE HANDLER
# =========================
root.protocol("WM_DELETE_WINDOW", finish_interview)


# =========================
# START
# =========================
root.after(0, update_frame)
root.mainloop()
