import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tkinter import *
from PIL import Image, ImageTk
from datetime import datetime
import pandas as pd
import csv
import os
from playsound import playsound
import threading

face_detector = YOLO("yolov8n-face.pt")

age_model = load_model("age_predictor_model.h5", compile=False)
emotion_model = load_model("emotion_detector_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

csv_path = "entry_log.csv"
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Visitor ID", "Time", "Age", "Emotion", "Status", "Snapshot"])

if not os.path.exists("snapshots"):
    os.makedirs("snapshots")

def alert_sound():
    threading.Thread(target=lambda: playsound("alert.mp3")).start()

root = Tk()
root.title("🎬 Theatre Entry System")
root.geometry("950x700")
root.configure(bg="#222")

video_label = Label(root)
video_label.pack()

info_label = Label(root, text="Status: Waiting for detection...", font=("Arial", 14), fg="white", bg="#222")
info_label.pack(pady=10)

stats_label = Label(root, text="", font=("Arial", 12), fg="lightgreen", bg="#222")
stats_label.pack()

cap = cv2.VideoCapture(0)
allowed_count = 0
blocked_count = 0
visitor_id = 0

def detect():
    global allowed_count, blocked_count, visitor_id

    ret, frame = cap.read()
    if not ret:
        return

    results = face_detector(frame, verbose=False)[0]
    detections = results.boxes.xyxy.cpu().numpy().astype(int)

    for (x1, y1, x2, y2) in detections:
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        age_input = cv2.resize(face, (64, 64)) / 255.0
        age_input = np.expand_dims(age_input, axis=0)
        age = int(age_model.predict(age_input)[0][0])

        if age < 13 or age > 60:
            status = "Not Allowed"
            color = (0, 0, 255)
            blocked_count += 1
            alert_sound()
        else:
            status = "Allowed"
            color = (0, 255, 0)
            allowed_count += 1

        emotion_text = "-"
        if status == "Allowed":
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray_face = cv2.resize(gray_face, (48, 48)) / 255.0
            gray_face = np.expand_dims(gray_face, axis=(0, -1))
            emotion_pred = emotion_model.predict(gray_face)
            emotion_idx = np.argmax(emotion_pred)
            emotion_text = emotion_labels[emotion_idx]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"Age: {age} | {status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if status == "Allowed":
            cv2.putText(frame, emotion_text, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        visitor_id += 1
        snapshot_filename = f"snapshots/visitor_{visitor_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(snapshot_filename, face)

        with open(csv_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([visitor_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), age, emotion_text, status, snapshot_filename])

    stats_label.config(text=f"✅ Allowed: {allowed_count}   ❌ Blocked: {blocked_count}")

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, detect)

detect()

root.mainloop()
cap.release()
cv2.destroyAllWindows()
