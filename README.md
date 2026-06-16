# 🎯 Study Guardian

## Overview

Study Guardian is an AI-powered real-time productivity monitoring and focus enhancement system designed for students and professionals. The application uses Computer Vision, Machine Learning, and Human Attention Analytics to monitor user focus levels, detect distractions, identify drowsiness, and provide instant alerts to improve productivity.

The system integrates MediaPipe Face Mesh, YOLOv8 Object Detection, and a Machine Learning-based gaze prediction model to create an intelligent study environment that encourages focused learning sessions.

---

## Features

### 👁️ Real-Time Gaze Tracking

* Tracks eye movements using MediaPipe Face Mesh.
* Identifies whether the user is:

  * Looking at the screen
  * Looking away
  * Looking downward
  * Distracted

### 😴 Drowsiness Detection

* Monitors eye closure duration.
* Detects prolonged eye closure indicating sleepiness.
* Generates warning alerts when drowsiness is detected.

### 📱 Mobile Phone Detection

* Uses YOLOv8 object detection.
* Identifies mobile phone usage during study sessions.
* Tracks distraction caused by phone usage.

### 👥 Multi-Face Detection

* Detects multiple faces appearing in the frame.
* Flags possible distractions during study sessions.

### 🔔 Smart Alert System

* Audio beep warnings.
* Voice-based focus reminders.
* Escalating alert mechanism for repeated distractions.

### 📊 Productivity Analytics Dashboard

* Real-time focus score tracking.
* Session performance statistics.
* Focus timeline visualization.
* Distraction category analysis.

### 👤 User Profile Management

* Facial profile registration.
* User-specific analytics.
* Personalized productivity reports.

---

## Technologies Used

### Programming Language

* Python

### Libraries and Frameworks

* Streamlit
* OpenCV
* MediaPipe
* Ultralytics YOLOv8
* Scikit-learn
* Pandas
* NumPy
* Joblib
* SciPy

### Machine Learning

* Supervised Learning for gaze classification
* Facial landmark feature extraction
* Object detection using YOLOv8

---

## Project Workflow

### Step 1: User Authentication

* Existing users can access previous profiles.
* New users can register their facial profile.

### Step 2: Profile Analytics

* View historical productivity reports.
* Analyze focus performance metrics.

### Step 3: Session Configuration

* Configure distraction thresholds.
* Set focus monitoring parameters.
* Customize alert settings.

### Step 4: Active Monitoring

* Webcam-based face tracking.
* Gaze prediction.
* Drowsiness monitoring.
* Mobile phone detection.
* Real-time analytics generation.

---

## Dataset

The gaze prediction model was trained using custom-collected facial landmark features extracted from MediaPipe Face Mesh.

### Input Features

* Iris Position
* Eye Width
* Nose Relative Position
* Face Symmetry

### Output Classes

* Center
* Left
* Right
* Distracted

---

## Installation

### Clone Repository

```bash
git clone https://github.com/vharinandana/Study-Guardian.git
cd Study-Guardian
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Project Structure

```text
Study-Guardian/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── packages.txt
├── README.md
│
├── 1.data_collection.ipynb
├── 2.model_train.ipynb
├── 3.Main.ipynb
│
├── gaze_model.pkl
├── yolov8n.pt
├── face_recognizer_model.yml
├── user_map.pkl
│
├── gaze_data.csv
└── face_data/
```

---

## Future Enhancements

* Cloud-based user authentication
* Session history database integration
* Emotion recognition
* Study habit recommendations
* Mobile application support
* Advanced productivity analytics
* Multi-user dashboard

---

## Applications

* Online Learning
* Competitive Exam Preparation
* Remote Education
* Employee Productivity Monitoring
* Training and Certification Programs
* Research in Human Attention Analysis

---

## Author

**Harinandana V**

M.Sc. Statistics

Data Science & Machine Learning Enthusiast

GitHub: https://github.com/vharinandana

---

## License

This project is developed for educational and research purposes.

