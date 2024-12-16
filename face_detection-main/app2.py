import time
import sqlite3
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Load face detection model
prototxtPath = "models/deploy.prototxt"
weightsPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load mask detection model
maskNet = tf.keras.models.load_model("mask_detector.h5")

IMG_SIZE = 224
CATEGORIES = ["mask_weared_incorrect", "with_mask", "without_mask"]

# Timer variables
MASK_DETECTION_THRESHOLD = 2  # Seconds
mask_detected_start_time = None  # Initialize the timer globally
tasks_displayed = False  # State flag to show tasks only once

# Function to detect mask
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []
    
    if detections is not None:  # Add a safeguard for None detections
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

# Function to get worker schedule from the database
def get_worker_schedule(worker_id):
    conn = sqlite3.connect("tasks.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM tasks WHERE worker_id = ?", (worker_id,))
    result = cursor.fetchone()
    conn.close()
    return result

# Streamlit app
st.title("Worker Face Mask Detection")
camera_checkbox = st.checkbox("Turn on Camera")

if camera_checkbox:
    stframe = st.empty()
    video_stream = cv2.VideoCapture(0)

    mask_detected_start_time, tasks_displayed  

    while True:
        ret, frame = video_stream.read()
        
        # Check if frame is valid, otherwise skip processing
        if not ret or frame is None:
            st.error("Failed to capture video stream or camera turned off.")
            break
        
        frame = cv2.resize(frame, (640, 480))
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        
        mask_present = False

        # Analyze predictions
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            class_id = np.argmax(pred)
            label = CATEGORIES[class_id]
            
            if label == "with_mask":
                mask_present = True
                color = (0, 255, 0)
            else:
                mask_present = False
                mask_detected_start_time = None  # Reset the timer
                color = (0, 0, 255)
            
            cv2.putText(frame, f"{label}: {pred[class_id] * 100:.2f}%", (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # Timer logic
        if mask_present and not tasks_displayed:
            if mask_detected_start_time is None:
                mask_detected_start_time = time.time()
            elif time.time() - mask_detected_start_time >= MASK_DETECTION_THRESHOLD:
                st.success("Mask detected! Fetching tasks...")
                
                # Show schedule and tasks for worker ID = 1
                worker_id = 1  # Hardcoded for now, replace with dynamic ID if required
                worker_data = get_worker_schedule(worker_id)

                if worker_data:
                    st.write(f"**Name:** {worker_data[1]}")
                    st.write(f"**Schedule:** {worker_data[2]}")
                    st.write(f"**Tasks:** {worker_data[3]}")
                else:
                    st.warning("No worker data found.")
                
                # Mark tasks as displayed
                tasks_displayed = True
        
        stframe.image(frame, channels="BGR")

        # Exit the loop if checkbox is turned off
        if not camera_checkbox:
            break

    video_stream.release()
