import cv2
import numpy as np
import tensorflow as tf

# Define constants
IMG_SIZE = 224
CATEGORIES = ["mask_weared_incorrect", "with_mask", "without_mask"]

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    faces = []
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

# Load face detection model
print("[INFO] loading face detector model...")
prototxtPath = "models/deploy.prototxt"
weightsPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

if faceNet.empty():
    raise ValueError("Error: Face detection model not loaded properly.")

# Load mask detector model
print("[INFO] loading mask detector model...")
maskNet = tf.keras.models.load_model("mask_detector.h5")

# Initialize video stream
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)

while True:
    ret, frame = vs.read()
    if not ret:
        break
        
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
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
