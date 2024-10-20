import cv2
import pickle
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

# Load the pre-trained KNN model and data
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load saved faces and names data
with open('data/names.pkl', 'rb') as f:
    names = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    faces = pickle.load(f)

# Reshape the faces data for KNN
faces = faces.reshape(faces.shape[0], -1)

# Initialize and fit KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces, names)

# Start capturing video feed
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_detected:
        # Capture and resize the detected face
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)

        # Predict the name of the detected face
        output = knn.predict(resized_img)

        # Draw a rectangle around the face and display the name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.putText(frame, str(output[0]), (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Face Recognition", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release video feed and close windows
video.release()
cv2.destroyAllWindows()
