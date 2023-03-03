import cv2
import socket
import numpy as np

# Set up socket connection
host = 'YOUR-SERVER-IP'
port = 8000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((host, port))

# Load the trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle around the face and extract the face region of interest
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Convert face region of interest to bytes and send it to the server
        img_bytes = cv2.imencode('.jpg', roi_gray)[1].tobytes()
        s.sendall(img_bytes)

    # Display the webcam feed
    cv2.imshow('Webcam Feed', frame)

    # Exit the loop on pressing the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources and close the socket connection
cap.release()
cv2.destroyAllWindows()
s.close()
