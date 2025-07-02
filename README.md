# 🎬 Age and Emotion Detector for Movie Theatre Access Control

A real-time machine learning application that predicts a person's **age** and **emotion** from live webcam input and restricts access to horror movie theatres based on age limits. Individuals under 13 or above 60 are denied entry and marked with a red rectangle, while others are allowed, with their emotion displayed.

---

## 📌 Project Objective

To build an intelligent, real-time tool that automates age restriction enforcement in movie theatres using machine learning and computer vision. The tool also logs each detected individual's age, emotion (if eligible), and entry timestamp into a CSV file for reporting.

---

## 🎯 Features

- 📷 Real-time face detection via webcam
- 🔢 Age prediction using a CNN regression model
- 😐 Emotion classification using CNN trained on FER-2013
- ❌ Restrict access for:
  - Age < 13
  - Age > 60
- ✅ Allow and display emotion for age 13–60
- 🧾 Log:
  - Age
  - Emotion
  - Entry time
- 📁 Save logs in `CSV` format
- 🧠 Model accuracy: ~70% (for both age and emotion)

---

## 🗂️ Project Structure

| Folder / File             | Description                                                        |
|---------------------------|--------------------------------------------------------------------|
| `models/`                 | Trained models: `age_model.h5`, `emotion_model.h5`                |
| `outputs/`                | Screenshots and logs: `above60.jpg`, `under10.jpg`, `log.csv`     |
|                
|  `main.py`             | Main entry script with webcam detection                           |
| `age_predictor.py`    | Module for age prediction                                          |
|  `emotion_detector.py` | Module for emotion classification                                  |
| `logger.py`           | Handles CSV logging                                                |
| `utils.py`            | Utilities (face detection, preprocessing, etc.)                    |
| `requirements.txt`        | List of required Python packages                                   |
| `README.md`               | This documentation file                                            |

---

## 🧠 Model Architecture

- **Age Detection**: Custom CNN regression model
- **Emotion Detection**: CNN classifier trained on FER-2013
- **Face Detection**: OpenCV Haar cascade or DNN

---

## 📸 Sample Outputs

| Age > 60 (Denied) | Age < 13 (Denied) |
|-------------------|-------------------|
| ![Above 60](outputs/above60.jpg) | ![Under 10](outputs/under10.jpg) |

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/age-emotion-detector.git
cd age-emotion-detector
pip install -r requirements.txt
