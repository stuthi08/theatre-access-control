# 🎬 Age and Emotion Detector for Movie Theatre Access Control

A real-time computer vision tool that uses machine learning to detect a person's age and emotion using webcam input. If the person is **under 13** or **over 60**, access is denied, an alarm is triggered, and a red box is shown. If the person is within the valid age range, their **emotion** is predicted and all data is **logged into a CSV file**.

---

## 📌 Objective

To automate age-based access control at horror movie theatres using AI, ensuring safety and compliance with age restrictions by integrating facial analysis and real-time alerts.

---

## 🎯 Features

- 📷 Real-time webcam face detection using OpenCV
- 🔢 Age prediction using CNN (`age_predictor_model.h5`)
- 😐 Emotion recognition using CNN trained on FER-2013
- ❌ Deny access if:
  - Age < 13
  - Age > 60
- 🚨 Alarm sound (from `alert.mp3`) for denied entry
- ✅ Show emotion for users aged 13–60
- 🧾 Log: Age, Emotion, Timestamp, Access Status to CSV
- 📁 All logs and evidence saved for review
- 🧠 Accuracy: ~70% for both age and emotion

---

## 🗂️ Project Structure

| File / Folder                     | Description                                               |
|----------------------------------|-----------------------------------------------------------|
| `age_predictor_model.h5`         | Trained CNN model for age prediction                     |
| `emotion_detector_model.h5`      | Trained CNN model for emotion detection                  |
| `alert.mp3`                      | Alarm sound file for denied entries                      |
| `age-det.ipynb`                  | Jupyter notebook for age detection testing               |
| `emotion-detection3.ipynb`       | Jupyter notebook for emotion detection                   |
| `gui.py`                         | Python GUI script (optional interface)                   |
| `emotion_detection3/` (if folder)| Supporting code for emotion detection (if applicable)    |
| `haarcascade_frontalface_default.xml` | Haar Cascade for face detection                     |
| `requirement.txt`               | List of Python dependencies                              |
| `report.pdf`                     | Final internship report                                  |
| `yolov8n-face.pt`                | YOLOv8 face detection weights (optional for torso/dress) |

---

## 🧠 Model Architecture

- **Age Model**: Custom CNN trained on facial age dataset
- **Emotion Model**: FER-2013 based emotion classification
- **Face Detection**: OpenCV Haar cascade or YOLOv8
- **Audio Alert**: Played using `pygame` or `playsound`

---

## ▶️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirement.txt
