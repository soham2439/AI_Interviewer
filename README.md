# AI Interviewer

AI Interviewer is a Python desktop application that runs a mock interview using webcam-based emotion detection and microphone-based speech capture. It asks interview questions, records spoken answers, estimates confidence from facial emotion signals, scores answers with a simple heuristic, and generates a PDF interview report with charts.

## Features

- Tkinter desktop interview interface
- Webcam feed with live emotion and confidence display
- Speech-to-text answer capture using `SpeechRecognition`
- Question-by-question interview flow with countdown timer
- Dominant emotion and confidence tracking per question
- Heuristic answer scoring based on length, clarity, detail, and filler words
- Final PDF report with confidence and answer score charts
- FER2013 emotion model training script included

## Project Structure

```text
Ai_interviewer/
+-- emotion_detection/
|   +-- interview_gui.py        # Main GUI application
|   +-- emotion_engine.py       # Emotion model loading and frame processing
|   +-- voice_input.py          # Microphone recording and speech recognition
|   +-- voice_confidence.py     # Optional voice confidence helper
|   +-- questions.py            # Interview question list
|   +-- webcam_emotion.py       # Standalone webcam emotion demo
|   +-- train_emotion_model.py  # Emotion model training script
+-- models/
|   +-- emotion_model.h5        # Trained emotion detection model
+-- data/
|   +-- fer2013/                # FER2013 dataset used for training
+-- README.md
```

## Requirements

- Python 3.12
- Webcam
- Microphone
- Internet connection for Google speech recognition
- Trained model file at `models/emotion_model.h5`

Main Python packages used by the project:

- `opencv-python`
- `tensorflow`
- `pillow`
- `matplotlib`
- `reportlab`
- `SpeechRecognition`
- `PyAudio`
- `numpy`

## Setup

Open PowerShell in the project folder:

```powershell
cd "C:\Users\GRISH PATIL\OneDrive\Desktop\Ai_interviewer"
```

Activate the existing virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Install missing packages if needed:

```powershell
python -m pip install opencv-python tensorflow pillow matplotlib reportlab SpeechRecognition PyAudio numpy
```

## Run the App

From the project root:

```powershell
python emotion_detection\interview_gui.py
```

Or run directly with the virtualenv Python:

```powershell
.\venv\Scripts\python.exe emotion_detection\interview_gui.py
```

The app will open a desktop window, start the webcam, display each interview question, and record your spoken answer.

## Output Files

After the interview finishes, these files are generated in the project root:

```text
AI_Interview_Report.pdf
confidence_chart.png
answer_score_chart.png
```

The PDF report contains:

- Each question and captured answer
- Dominant detected emotion
- Confidence score
- Answer score
- Overall interview score
- Final verdict

## Editing Questions

Questions are stored in:

```text
emotion_detection/questions.py
```

Edit the `QUESTIONS` list to customize the interview.

## Training the Emotion Model

The training script is:

```powershell
python emotion_detection\train_emotion_model.py
```

It expects the FER2013-style dataset under:

```text
data/fer2013/
```

The trained model should be saved as:

```text
models/emotion_model.h5
```

## Troubleshooting

If you see `No module named speech_recognition`, install:

```powershell
python -m pip install SpeechRecognition
```

If microphone access fails, install `PyAudio`:

```powershell
python -m pip install PyAudio
```

If the webcam is not accessible:

- Close other apps using the camera.
- Check Windows camera permissions.
- Make sure a webcam is connected.

If speech recognition returns service errors:

- Check your internet connection.
- Try speaking clearly and close to the microphone.
- Run the app in a quieter environment.

If the model cannot be loaded:

- Confirm `models/emotion_model.h5` exists.
- Run the app from the project root folder.

## Notes

The answer score is a lightweight heuristic, not a full semantic evaluation. For stronger interview evaluation, the next improvement would be to add an NLP-based scoring module that compares answers against expected concepts for each question.
