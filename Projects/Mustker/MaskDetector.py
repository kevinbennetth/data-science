import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

import cv2

import os
import numpy as np

from imutils.video import VideoStream
import imutils
import time

# Face Detector
print("[LOAD] Loading Face Detector model...")
prototxt = "deploy.prototxt.txt"
caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"

face_detect = cv2.dnn.readNetFromCaffe(prototxt,caffemodel)
print("[INFO] Successfully loaded Face Detector model")

# Mask Detector
print("[LOAD] Loading Mask Detector model...")
mask_detect = load_model('mask_detector')
print("[INFO] Successfully loaded mask detector model")

# Function to detect faces' locations along with individual predictions
def live_detect(frame):
    h,w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),(104.0, 177.0, 123.0))

    face_detect.setInput(blob)
    detections = face_detect.forward()

    faces = []
    locs = []
    preds = []

    # For each faces
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        # If we are > 50% confident it's a face
        if confidence > 0.5:
            # Save coordinates of the 4 points of the bounding box
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            startX, startY, endX, endY = box.astype("int")

            startX, startY = (max(0, startX), max(0, startY))
            endX, endY = (min(w - 1, endX), min(h - 1, endY))
        
            # Select the face region, preprocess then append it to face list
            face = frame[startY:endY, startX:endX, :]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            faces.append(face)
            locs.append((startX,startY,endX,endY))

    # If there's any faces
    if len(faces) > 0:
        # Predict whether this face has a mask on or not
        faces = np.array(faces, dtype="float32")
        preds = mask_detect.predict(faces, batch_size=32)

    return (locs, preds)

# For video demonstration through webcam
vs = VideoStream(src=0).start()
time.sleep(2.0)

print("[LOAD] Begin video streaming...")
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=720)

    (locs,preds) = live_detect(frame)

    # For all bounding box's coordinates and predictions
    for (box,pred) in zip(locs, preds):
        # Unpack
        (startX, startY, endX, endY) = box
        (noMask, mask) = pred

        label = "Mask" if mask>noMask else "No Mask"
        color = (0,255,0) if label == "Mask" else (0,0,255)

        label = "{}: {:.2f}%".format(label, max(mask,noMask)*100)

        # Draw the bounding box and place label for the face on top of the bounding box
        cv2.putText(frame, label, (startX, startY-10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Creates new window displaying your live webcam
    cv2.imshow("Press q to exit", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
print("[INFO] Video stream has ended")

cv2.destroyAllWindows()
vs.stop()