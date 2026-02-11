import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import threading
from emotion_engine import EmotionEngine




from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

from voice_input import record_answer
from questions import QUESTIONS

# =========================
# CONFIG
# =========================
current_emotion_scores = []

FRAME_SKIP = 5        # emotion runs once every 5 frames
frame_count = 0
last_emotion = "Neutral"
last_confidence = 0

QUESTION_TIME = 120  # 2 minutes
questions = QUESTIONS
current_question = 0
question_start_time = None

question_log = []
recording_thread = None
current_answer = ""
recording_active = False

emotion_engine = EmotionEngine("models/emotion_model.h5")

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()


# =========================
# GUI SETUP
# =========================
root = tk.Tk()
root.title("AI Interviewer")
root.geometry("900x700")

question_label = tk.Label(
    root, text="", font=("Helvetica", 18),
    wraplength=800, justify="center"
)
question_label.pack(pady=15)

timer_label = tk.Label(root, text="", font=("Helvetica", 14))
timer_label.pack()

video_label = tk.Label(root)
video_label.pack(pady=10)

status_label = tk.Label(root, text="", font=("Helvetica", 12), fg="green")
status_label.pack(pady=5)

# =========================
# LOGGING
# =========================
def log_question(answer_text):
    global question_start_time

    end_time = time.time()

    log = {
        "question": questions[current_question],
        "answer": answer_text,
        "start_time": datetime.datetime.fromtimestamp(question_start_time).strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.datetime.fromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S"),
        "emotion_confidence": np.random.randint(60, 90),  # placeholder
        "answer_score": np.random.randint(50, 95)         # placeholder
    }

    question_log.append(log)

# =========================
# NEXT QUESTION
# =========================


def start_recording():
    global current_answer, recording_active
    recording_active = True
    current_answer = record_answer(timeout=QUESTION_TIME)
    recording_active = False

def next_question():
    global current_question, question_start_time

    if recording_active:
        status_label.config(text="⏳ Finishing answer...")
        root.update()
        time.sleep(0.5)

    log_question(current_answer)

    current_question += 1

    if current_question >= len(questions):
        finish_interview()
        return

    question_start_time = None
    status_label.config(text="")





# =========================
# UPDATE FRAME
# =========================
def update_frame():
    global question_start_time

    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)

        # ✅ Emotion detection (uses BGR frame)
        emotion, confidence = emotion_engine.process_frame(frame)

        if confidence:
           current_emotion_scores.append(confidence)

        # ✅ GUI rendering (convert AFTER emotion processing)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)


    # START QUESTION + RECORDING ONCE
    if question_start_time is None:
        question_start_time = time.time()
        status_label.config(text="🎤 Recording answer...")
        root.after(100, lambda: threading.Thread(
    target=start_recording, daemon=True
).start())


    elapsed = int(time.time() - question_start_time)
    remaining = QUESTION_TIME - elapsed

    if remaining <= 0:
        next_question()
        return

    mins, secs = divmod(remaining, 60)
    timer_label.config(text=f"Time Remaining: {mins:02d}:{secs:02d}")
    question_label.config(text=questions[current_question])

    root.after(30, update_frame)

# =========================
# PDF REPORT
# =========================
def generate_pdf_report():
    pdf_name = "AI_Interview_Report.pdf"
    doc = SimpleDocTemplate(pdf_name, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>AI Interview – Final Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph(
            f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["Normal"]
        )
    )
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Question-wise Analysis</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    for i, q in enumerate(question_log, 1):
        text = f"""
        <b>Q{i}:</b> {q['question']}<br/>
        <b>Answer:</b> {q['answer']}<br/>
        Confidence: {q['emotion_confidence']}%<br/>
        Answer Score: {q['answer_score']}%
        """
        elements.append(Paragraph(text, styles["Normal"]))
        elements.append(Spacer(1, 10))

    # Charts
    idx = list(range(1, len(question_log) + 1))
    conf = [q["emotion_confidence"] for q in question_log]
    score = [q["answer_score"] for q in question_log]

    plt.figure(figsize=(8, 3))
    plt.bar(idx, conf)
    plt.title("Confidence per Question")
    plt.savefig("confidence_chart.png")
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.bar(idx, score)
    plt.title("Answer Score per Question")
    plt.savefig("answer_score_chart.png")
    plt.close()

    elements.append(PDFImage("confidence_chart.png", 6*inch, 3*inch))
    elements.append(Spacer(1, 20))
    elements.append(PDFImage("answer_score_chart.png", 6*inch, 3*inch))

    avg_conf = int(sum(conf) / len(conf))
    avg_score = int(sum(score) / len(score))
    verdict = "Excellent" if avg_conf > 75 and avg_score > 70 else "Needs Improvement"

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<b>Final Evaluation</b>", styles["Heading2"]))
    elements.append(Paragraph(f"Overall Confidence: {avg_conf}%", styles["Normal"]))
    elements.append(Paragraph(f"Overall Answer Accuracy: {avg_score}%", styles["Normal"]))
    elements.append(Paragraph(f"<b>Verdict:</b> {verdict}", styles["Normal"]))

    doc.build(elements)

# =========================
# FINISH
# =========================
def finish_interview():
    cap.release()
    generate_pdf_report()
    root.destroy()
    print("✅ Interview finished. PDF generated.")

# =========================
# BUTTON
# =========================
next_btn = tk.Button(
    root,
    text="Next Question",
    font=("Helvetica", 13, "bold"),
    bg="#1E35FF",          # Dodger Blue
    fg="white",            # Text color
    activebackground="#104E8B",
    activeforeground="white",
    padx=20,
    pady=8,
    borderwidth=0,
    command=next_question
)
next_btn.pack(pady=15)


# =========================
# CLOSE HANDLER
# =========================
def on_close():
    finish_interview()

root.protocol("WM_DELETE_WINDOW", on_close)

# =========================
# START
# =========================
root.after(0, update_frame)
root.mainloop()
