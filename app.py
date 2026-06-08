import streamlit as st

st.title("Deployment Test")

try:
    import cv2
    st.success(f"OpenCV loaded: {cv2.__version__}")
except Exception as e:
    st.error(f"OpenCV error: {type(e).__name__}: {e}")

try:
    import mediapipe as mp
    st.success("MediaPipe loaded")
except Exception as e:
    st.error(f"MediaPipe error: {type(e).__name__}: {e}")