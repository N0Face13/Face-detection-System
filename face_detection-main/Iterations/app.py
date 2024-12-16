import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import sqlite3

# Define constants
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

# Database setup
@st.cache_resource
def init_db():
    conn = sqlite3.connect("tasks.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            worker_id INTEGER PRIMARY KEY,
            name TEXT,
            schedule TEXT,
            tasks TEXT
        )
    """)
    conn.commit()
    return conn

def get_worker_schedule(worker_id):
    # Create a new SQLite connection
    conn = sqlite3.connect("tasks.db")
    cursor = conn.cursor()
    # Query the worker's schedule and tasks
    cursor.execute("SELECT * FROM tasks WHERE worker_id = ?", (worker_id,))
    result = cursor.fetchone()
    conn.close()
    return result

# Load models
@st.cache_resource
def load_models():
    prototxtPath = "models/deploy.prototxt"
    weightsPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    maskNet = tf.keras.models.load_model("mask_detector.h5")
    return faceNet, maskNet

faceNet, maskNet = load_models()
conn = init_db()

# Initialize UI
st.title("Worker Safety and Task Management System")
run_camera = st.checkbox("Start Camera")

if "camera" not in st.session_state:
    st.session_state.camera = None

if run_camera:
    if st.session_state.camera is None:
        st.session_state.camera = cv2.VideoCapture(0)  # Start the camera

    FRAME_WINDOW = st.image([])  # Placeholder for video stream
    
    while run_camera:
        ret, frame = st.session_state.camera.read()
        if not ret or frame is None:
            st.warning("Unable to access the camera feed.")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            class_id = np.argmax(pred)
            label = CATEGORIES[class_id]
            
            if label == "with_mask":
                st.success("Mask detected! Fetching schedule...")
                
                # Display worker-specific schedule
                worker_id = 1  # Assume we identify the worker here (could be enhanced with ID systems)
                worker_data = get_worker_schedule(worker_id)
                if worker_data:
                    name, schedule, tasks = worker_data[1:]
                    st.subheader(f"Worker: {name}")
                    st.write(f"**Schedule**: {schedule}")
                    st.write(f"**Tasks**: {tasks}")
                else:
                    st.error("No worker data found.")

                st.stop()  # Stop further execution to show tasks

            color = (0, 255, 0) if label == "with_mask" else (0, 0, 255)
            label = f"{label}: {pred[class_id] * 100:.2f}%"
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        FRAME_WINDOW.image(frame)

else:
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
    cv2.destroyAllWindows()
    st.success("Camera feed stopped.")
