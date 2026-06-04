import cv2
import mediapipe as mp
import joblib
import numpy as np
import pandas as pd 
import time
import os
from plyer import notification
import pygetwindow as gw
from ultralytics import YOLO
import threading
import pyttsx3
import streamlit as st
from scipy.io import wavfile
import io

# --- STREAMLIT CONFIG & INITIAL PAGE SETUP ---
st.set_page_config(page_title="Study Guardian Workspace", layout="wide", page_icon="🎯")

st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { color: #1E3A8A; font-weight: 800; }
    .stButton>button { border-radius: 8px; transition: all 0.3s ease; width: 100%; }
    .report-card { background-color: #F3F4F6; padding: 25px; border-radius: 12px; border-left: 6px solid #1E3A8A; margin-bottom: 20px; }
    .step-box { background-color: #EFF6FF; padding: 15px; border-radius: 8px; border: 1px solid #BFDBFE; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 Study Guardian Ultimate Workspace")
st.caption("Advanced Computer Vision & Real-Time Analytics Dashboard for Focused Productivity")
st.markdown("---")

# --- FILE DIRECTORY SETUP FOR PERSISTENCE ---
FACE_DATA_DIR = "face_data"
MODEL_SAVE_PATH = "face_recognizer_model.yml"
USER_MAP_PATH = "user_map.pkl"

if not os.path.exists(FACE_DATA_DIR):
    os.makedirs(FACE_DATA_DIR)

# --- CORE RESOURCE OPTIMIZATION (CACHED MODEL LOADING) ---
@st.cache_resource
def load_core_models():
    mp_face_mesh = mp.solutions.face_mesh
    mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=2) 
    gaze_mdl = joblib.load('gaze_model.pkl')
    obj_mdl = YOLO("yolov8n.pt")
    return mesh, gaze_mdl, obj_mdl

face_mesh, model, object_model = load_core_models()

def get_face_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    user_map = {}
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(USER_MAP_PATH):
        try:
            recognizer.read(MODEL_SAVE_PATH)
            user_map = joblib.load(USER_MAP_PATH)
        except Exception: pass
    return recognizer, user_map

face_recognizer, user_mapping = get_face_recognizer()

# --- STEP-BY-STEP APPLICATION CONTROL FLOW STATE MACHINE ---
if "app_step" not in st.session_state:
    st.session_state.app_step = "QUESTION_GATE"  # Steps: QUESTION_GATE -> VIEW_REPORT -> READY_TO_START -> ACTIVE_SESSION
if "registered_users" not in st.session_state:
    st.session_state.registered_users = list(user_mapping.values()) if user_mapping else ["Default_Student"]
if "active_user" not in st.session_state:
    st.session_state.active_user = "Unknown"
if "voice_active" not in st.session_state:
    st.session_state.voice_active = False

# Analytical Dynamic Tracking Framework Data Structures
if "start_time" not in st.session_state: st.session_state.start_time = time.time()
if "total_frames" not in st.session_state: st.session_state.total_frames = 0
if "center_frames" not in st.session_state: st.session_state.center_frames = 0
if "distracted_frames" not in st.session_state: st.session_state.distracted_frames = 0
if "sleep_frames" not in st.session_state: st.session_state.sleep_frames = 0
if "phone_frames" not in st.session_state: st.session_state.phone_frames = 0
if "downward_frames" not in st.session_state: st.session_state.downward_frames = 0
if "multiface_frames" not in st.session_state: st.session_state.multiface_frames = 0
if "alert_streak" not in st.session_state: st.session_state.alert_streak = 0
if "timeline_history" not in st.session_state:
    st.session_state.timeline_history = pd.DataFrame(columns=["Elapsed Time", "Focus Score"])

# --- AUDIO WORKER BACKENDS ---
def play_cross_platform_beep(frequency, duration_ms):
    sample_rate = 44100
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    t = np.linspace(0, duration_ms / 1000.0, num_samples, endpoint=False)
    audio_data = np.sin(2 * np.pi * frequency * t) * 32767
    audio_encoded = audio_data.astype(np.int16)
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sample_rate, audio_encoded)
    st.audio(byte_io.getvalue(), format="audio/wav", autoplay=True, key=f"bp_{time.time_ns()}")

def _execute_voice_worker(message):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        engine.say(message)
        engine.runAndWait()
        engine.stop()
        del engine  
    except Exception: pass
    finally: st.session_state.voice_active = False

def speak_warning_async(message):
    if not st.session_state.voice_active:
        st.session_state.voice_active = True
        threading.Thread(target=_execute_voice_worker, args=(message,), daemon=True).start()

def train_face_recognizer():
    images, labels, current_mapping = [], [], {}
    user_id_counter = 0
    for filename in os.listdir(FACE_DATA_DIR):
        if filename.endswith(".jpg"):
            username = filename.split("_face_")[0]
            if username not in current_mapping.values():
                current_mapping[user_id_counter] = username
                target_id = user_id_counter
                user_id_counter += 1
            else:
                target_id = [k for k, v in current_mapping.items() if v == username][0]
            img_path = os.path.join(FACE_DATA_DIR, filename)
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is not None:
                images.append(gray_img)
                labels.append(target_id)
    if len(images) > 0:
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.train(images, np.array(labels))
        rec.write(MODEL_SAVE_PATH)
        joblib.dump(current_mapping, USER_MAP_PATH)
        return True
    return False


# ==============================================================================
# STAGE 1: THE INITIAL QUESTION GATE
# ==============================================================================
if st.session_state.app_step == "QUESTION_GATE":
    st.markdown("<div class='step-box'><h3>Step 1: System Authentication Check</h3></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("👋 Welcome back! Select this option to quickly jump into your loaded analytics dashboard and profiles.")
        if st.button("🙋‍♂️ I am an Existing Registered User", type="primary"):
            st.session_state.app_step = "VIEW_REPORT"
            st.rerun()
            
    with col2:
        st.warning("📸 Choose this option to map your unique facial features into the local database models before starting.")
        with st.expander("🆕 Register/Enroll New Face Profile Context"):
            new_username = st.text_input("Enter a Unique New Username ID").strip()
            if st.button("Start Camera Face Capture Session"):
                if new_username:
                    temp_cap = cv2.VideoCapture(0)
                    time.sleep(1.0)
                    ret, capture_frame = temp_cap.read()
                    temp_cap.release()
                    if ret:
                        rgb_cb = cv2.cvtColor(capture_frame, cv2.COLOR_BGR2RGB)
                        res = face_mesh.process(rgb_cb)
                        if res.multi_face_landmarks:
                            h, w, _ = capture_frame.shape
                            landmarks = res.multi_face_landmarks[0]
                            x_coords = [int(lm.x * w) for lm in landmarks.landmark]
                            y_coords = [int(lm.y * h) for lm in landmarks.landmark]
                            xmin, xmax = max(0, min(x_coords)), min(w, max(x_coords))
                            ymin, ymax = max(0, min(y_coords)), min(h, max(y_coords))
                            
                            cropped_face = capture_frame[ymin:ymax, xmin:xmax]
                            gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
                            resized_face = cv2.resize(gray_face, (200, 200))
                            
                            for idx in range(5):
                                filename = f"{new_username}_face_{idx}_{int(time.time())}.jpg"
                                cv2.imwrite(os.path.join(FACE_DATA_DIR, filename), resized_face)
                            
                            if train_face_recognizer():
                                face_recognizer, user_mapping = get_face_recognizer()
                                if new_username not in st.session_state.registered_users:
                                    st.session_state.registered_users.append(new_username)
                                st.session_state.active_user = new_username
                                st.success(f"Profile context created for '{new_username}'!")
                                time.sleep(1.5)
                                st.session_state.app_step = "VIEW_REPORT"
                                st.rerun()
                        else: st.error("No clear face targets identified.")
                    else: st.error("Camera drivers unresponsive.")


# ==============================================================================
# STAGE 2: VIEW REPORT CARD & PERFORMANCE GRAPH
# ==============================================================================
elif st.session_state.app_step == "VIEW_REPORT":
    st.markdown("<div class='step-box'><h3>Step 2: Profile Selection & Historical Report Performance Hub</h3></div>", unsafe_allow_html=True)
    
    # Let user pick which profile report card they want to see
    selected_user = st.selectbox("Select Profile To Pull Report Cards For:", options=st.session_state.registered_users)
    st.session_state.active_user = selected_user
    
    # Calculate performance data points securely
    tf_count = st.session_state.total_frames if st.session_state.total_frames > 0 else 1
    focus_pct = ((st.session_state.center_frames / tf_count) * 100)
    
    # RENDER THE COMPREHENSIVE PERFORMANCE CARD REPORT
    st.markdown("<div class='report-card'>", unsafe_allow_html=True)
    st.markdown(f"### 📑 Focus Analysis Report Card: Profile - {st.session_state.active_user}")
    rc_col1, rc_col2, rc_col3 = st.columns(3)
    with rc_col1:
        st.metric("Productivity Score", f"{focus_pct:.1f}%")
        st.metric("Drowsiness Counter", f"{st.session_state.sleep_frames} Frames")
    with rc_col2:
        st.metric("Total Processing Window", f"{st.session_state.total_frames} Frames")
        st.metric("Sideways Look-Aways", f"{st.session_state.distracted_frames} Frames")
    with rc_col3:
        st.metric("Steady Focused Interval", f"{st.session_state.center_frames} Frames")
        st.metric("Mobile Handset Detections", f"{st.session_state.phone_frames} Frames")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # GRAPH MATRIX VISUAL SYSTEM
    g_col1, g_col2 = st.columns([2, 1])
    with g_col1:
        st.markdown("#### Focus Timeline Performance Curve")
        if not st.session_state.timeline_history.empty:
            st.line_chart(st.session_state.timeline_history.set_index("Elapsed Time"))
        else:
            st.info("No timeline matrix tracked yet for this individual session run.")
    with g_col2:
        st.markdown("#### Distraction Profiler Category Distribution")
        analytics_data = pd.DataFrame({
            "Distraction Class Category": ["Focused", "Drowsiness", "Looking Sideways", "Looking Downward", "Device Usage"],
            "Frames Captured": [st.session_state.center_frames, st.session_state.sleep_frames, st.session_state.distracted_frames, st.session_state.downward_frames, st.session_state.phone_frames]
        })
        st.bar_chart(analytics_data.set_index("Distraction Class Category"))
        
    st.markdown("---")
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("⬅️ Change Profile"):
            st.session_state.app_step = "QUESTION_GATE"
            st.rerun()
    with col_btn2:
        if st.button("Proceed To Setup Sessions Constraints ➡️", type="primary"):
            st.session_state.app_step = "READY_TO_START"
            st.rerun()


# ==============================================================================
# STAGE 3: THE START SESSION CONFIGURATION HUB
# ==============================================================================
elif st.session_state.app_step == "READY_TO_START":
    st.markdown("<div class='step-box'><h3>Step 3: Setup Guardrails Configuration Parameters</h3></div>", unsafe_allow_html=True)
    st.success(f"🔓 Selected Target Profile Locked in Session Layer: **{st.session_state.active_user}**")
    
    col1, col2 = st.columns(2)
    with col1:
        look_away_options = {"1.5 Seconds": 1.5, "3.0 Seconds (Default)": 3.0, "5.0 Seconds": 5.0, "7.0 Seconds": 7.0}
        selected_look_away = st.selectbox("Look Away Timeout Limit", options=list(look_away_options.keys()), index=1)
        st.session_state.look_away_limit_val = look_away_options[selected_look_away]
        
        drowsy_options = {"1.0 Second": 1.0, "2.0 Seconds (Default)": 2.0, "3.0 Seconds": 3.0}
        selected_drowsy = st.selectbox("Drowsiness Trigger Threshold", options=list(drowsy_options.keys()), index=1)
        st.session_state.sleep_threshold_val = drowsy_options[selected_drowsy]

    with col2:
        st.session_state.break_time_mins_val = st.number_input("Pomodoro Target Work Window (mins)", min_value=1, max_value=120, value=25)
        st.session_state.target_study_window_val = st.text_input("Target Workspace IDE Window Title", value="Visual Studio Code")

    st.markdown("<br><br>", unsafe_allow_html=True)
    btn_c1, btn_c2 = st.columns([1, 2])
    with btn_c1:
        if st.button("⬅️ Back To Reports"):
            st.session_state.app_step = "VIEW_REPORT"
            st.rerun()
    with btn_c2:
        # THE CLEAR START SESSION BUTTON REQUESTED
        if st.button("🚀 LAUNCH CAMERA AND START ACTIVE SESSION NOW", type="primary"):
            # Flush previous run parameters cleanly
            st.session_state.start_time = time.time()
            st.session_state.total_frames = 0
            st.session_state.center_frames = 0
            st.session_state.distracted_frames = 0
            st.session_state.sleep_frames = 0
            st.session_state.phone_frames = 0
            st.session_state.downward_frames = 0
            st.session_state.multiface_frames = 0
            st.session_state.alert_streak = 0
            st.session_state.timeline_history = pd.DataFrame(columns=["Elapsed Time", "Focus Score"])
            st.session_state.app_step = "ACTIVE_SESSION"
            st.rerun()


# ==============================================================================
# STAGE 4: LIVE WEBCAM ACTIVE MONITORING PROCESSING LOOP
# ==============================================================================
elif st.session_state.app_step == "ACTIVE_SESSION":
    st.markdown("<div class='step-box'><h3>🔴 Step 4: Active Computer Vision Security Stream Active</h3></div>", unsafe_allow_html=True)
    
    # Stop Session button available during live loop runs
    if st.button("🛑 STOP AND TERMINATE SESSION RUN"):
        st.session_state.app_step = "VIEW_REPORT"
        st.rerun()
        
    main_view_col, side_telemetry_col = st.columns([2, 1])
    with main_view_col:
        view_frame = st.empty()
    with side_telemetry_col:
        st.markdown("#### Live Telemetry Elements")
        metric_status = st.empty()
        metric_confidence = st.empty()
        metric_faces = st.empty()
        metric_phone = st.empty()

    cap = cv2.VideoCapture(0)
    away_start_time = None
    
    while cap.isOpened() and st.session_state.app_step == "ACTIVE_SESSION":
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        current_time = time.time()
        elapsed = current_time - st.session_state.start_time
        st.session_state.total_frames += 1
        
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prediction = "center"
        num_faces_detected = 0

        if results.multi_face_landmarks:
            num_faces_detected = len(results.multi_face_landmarks)
            face_landmarks = results.multi_face_landmarks[0]
            
            left_iris = face_landmarks.landmark[468]
            left_outer = face_landmarks.landmark[33]
            left_inner = face_landmarks.landmark[133]
            right_outer = face_landmarks.landmark[263]
            nose = face_landmarks.landmark[1]        
            left_top_eyelid = face_landmarks.landmark[159]
            left_bottom_eyelid = face_landmarks.landmark[145]
            
            eye_height = left_bottom_eyelid.y - left_top_eyelid.y
            left_edge = face_landmarks.landmark[234]  
            right_edge = face_landmarks.landmark[454] 
            top_head = face_landmarks.landmark[10]    
            bottom_chin = face_landmarks.landmark[152] 

            eye_width = left_inner.x - left_outer.x
            iris_pos = left_iris.x - left_outer.x
            face_width = right_edge.x - left_edge.x
            nose_relative_pos = (nose.x - left_edge.x) / face_width
            
            dist_to_left = abs(left_outer.x - nose.x)
            dist_to_right = abs(right_outer.x - nose.x)
            face_symmetry = dist_to_left / (dist_to_right + 1e-6)
            face_total_height = bottom_chin.y - top_head.y
            nose_height_ratio = (nose.y - top_head.y) / (face_total_height + 1e-6)

            if eye_height < 0.008: prediction = "closed"
            elif nose_height_ratio > 0.68: prediction = "downward"
            else:
                features_df = pd.DataFrame([[iris_pos, eye_width, nose_relative_pos, face_symmetry]], columns=['iris_pos', 'eye_width', 'nose_relative_pos', 'face_symmetry'])
                try:
                    raw_prediction = model.predict(features_df)[0]
                    probabilities = model.predict_proba(features_df)[0]
                    prediction = raw_prediction if (probabilities[list(model.classes_).index(raw_prediction)] >= 0.75) else "center"
                except Exception: prediction = "center"

        phone_in_view = False
        if st.session_state.total_frames % 6 == 0:
            obj_results = object_model(frame, verbose=False)[0]
            for box in obj_results.boxes:
                if object_model.names[int(box.cls[0])] in ["cell phone"]: phone_in_view = True

        if num_faces_detected > 1:
            prediction = "multi-face distraction"
            st.session_state.multiface_frames += 1

        if prediction == "center" and not phone_in_view: 
            st.session_state.center_frames += 1
            if st.session_state.total_frames % 30 == 0: st.session_state.alert_streak = max(0, st.session_state.alert_streak - 1)
        elif prediction == "closed": st.session_state.sleep_frames += 1
        elif prediction in ["Distracted", "Distracted"]: st.session_state.distracted_frames += 1
        elif prediction == "downward": st.session_state.downward_frames += 1
        if phone_in_view: st.session_state.phone_frames += 1

        if st.session_state.total_frames % 20 == 0:
            current_score = (st.session_state.center_frames / st.session_state.total_frames) * 100
            new_entry = pd.DataFrame([{"Elapsed Time": round(elapsed, 1), "Focus Score": round(current_score, 2)}])
            st.session_state.timeline_history = pd.concat([st.session_state.timeline_history, new_entry], ignore_index=True)

        is_violating = (prediction in ["Distracted", "Distracted", "downward", "multi-face distraction"] or prediction == "closed" or phone_in_view)
        if is_violating:
            if away_start_time is None: away_start_time = current_time
            trigger_limit = st.session_state.sleep_threshold_val if prediction == "closed" else st.session_state.look_away_limit_val
            if (current_time - away_start_time) >= trigger_limit:
                st.session_state.alert_streak += 1
                away_start_time = current_time
                
                if st.session_state.alert_streak == 1: 
                    play_cross_platform_beep(600, 150)
                elif st.session_state.alert_streak == 2:
                    if not st.session_state.voice_active:
                        phrase = "Focus on your workspace."
                        if prediction == "closed": phrase = "Wake up! Don't sleep during study hours."
                        elif phone_in_view: phrase = "Put your mobile phone away."
                        speak_warning_async(phrase)
                elif st.session_state.alert_streak >= 3: 
                    play_cross_platform_beep(1500, 400)
                    
                try:
                    target_windows = gw.getWindowsWithTitle(st.session_state.target_study_window_val)
                    if target_windows:
                        study_win = target_windows[0]
                        if not study_win.isActive:
                            if study_win.isMinimized: study_win.restore()
                            study_win.activate()
                except Exception: pass
        else: away_start_time = None

        display_gaze = prediction.replace("-", " ").upper()
        cv2.rectangle(frame, (0, 0), (w, 45), (0, 0, 0), -1)
        cv2.putText(frame, f"GAZE: {display_gaze}", (20, 30), 0, 0.6, (0, 0, 255) if is_violating else (0, 255, 0), 2)
        
        view_frame.image(frame, channels="BGR", width="stretch")
        metric_status.metric("Gaze Status", display_gaze)
        metric_confidence.metric("Confidence Layer", "100%")
        metric_faces.metric("Faces Detected", num_faces_detected)
        metric_phone.metric("Mobile Exposure", "MOBILE DETECTED" if phone_in_view else "CLEAR")
        time.sleep(0.01)
        
    cap.release()