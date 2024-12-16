import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# Defined constants
IMG_SIZE = 224
CATEGORIES = ["mask_weared_incorrect", "with_mask", "without_mask"]

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    locs = []
    preds = []
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
                
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)
            
            preds.append(maskNet.predict(face)[0])
            locs.append((startX, startY, endX, endY))
    
    return (locs, preds)

# Load models once
@st.cache_resource
def load_models():
    prototxtPath = "models/deploy.prototxt"
    weightsPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = tf.keras.models.load_model("mask_detector.h5")
    return faceNet, maskNet

faceNet, maskNet = load_models()

# Initialize variables
st.title("Real-Time Mask Detection")
run = st.checkbox("Start Camera")

if "camera" not in st.session_state:
    st.session_state.camera = None

if run:
    if st.session_state.camera is None:
        st.session_state.camera = cv2.VideoCapture(0)  # Start camera

    FRAME_WINDOW = st.image([])  # Placeholder for video stream
    
    while run:
        ret, frame = st.session_state.camera.read()
        if not ret or frame is None:
            st.warning("Unable to access the camera feed. Check your permissions.")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            class_id = np.argmax(pred)
            label = CATEGORIES[class_id]
            color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)
            label = f"{label}: {pred[class_id] * 100:.2f}%"
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        FRAME_WINDOW.image(frame)

else:
    # Stop the camera and release resources
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
        st.success("Camera feed stopped.")
    cv2.destroyAllWindows()
