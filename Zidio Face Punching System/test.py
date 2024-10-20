from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)
print('Length of LABELS array --> ', len(LABELS))

# Check for mismatch in the number of samples
if FACES.shape[0] != len(LABELS):
    print(f"Mismatch: FACES has {FACES.shape[0]} samples, but LABELS has {len(LABELS)} labels.")
    
    # Trim LABELS to match FACES if LABELS has more elements
    if len(LABELS) > FACES.shape[0]:
        LABELS = LABELS[:FACES.shape[0]]
        print(f"LABELS array trimmed to {len(LABELS)} elements.")
    else:
        raise ValueError("LABELS has fewer elements than FACES. Check your data sources.")

# Fit the KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Initialize attendance tracking
attendance = {}

imgBackground = cv2.imread("background.png")
COL_NAMES = ['NAME', 'CHECK-IN', 'CHECK-OUT']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
        # Update attendance tracking
        employee_name = str(output[0])
        if employee_name not in attendance:
            # Check-in
            attendance[employee_name] = {'check_in': timestamp, 'check_out': None}
        else:
            # Check-out
            if attendance[employee_name]['check_out'] is None:
                attendance[employee_name]['check_out'] = timestamp

    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)
    
    k = cv2.waitKey(1)
    
    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)

        # Save attendance to CSV
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                for name, times in attendance.items():
                    writer.writerow([name, times['check_in'], times['check_out'] if times['check_out'] else 'N/A'])
        else:
            with open("Attendance/Attendance_" + date + ".csv", "w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)  # Write header
                for name, times in attendance.items():
                    writer.writerow([name, times['check_in'], times['check_out'] if times['check_out'] else 'N/A'])

    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
