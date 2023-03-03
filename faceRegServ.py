import cv2
import socket
import numpy as np

# Set up socket connection
host = 'YOUR-SERVER-IP'
port = 8000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))
s.listen(1)

# Load the trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Receive and process images from client
while True:
    conn, addr = s.accept()
    img_bytes = b''

    # Receive the bytes of the image in chunks
    while True:
        data = conn.recv(4096)
        if not data:
            break
        img_bytes += data

    # Convert the received bytes to an image and detect faces
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    # Display the image with detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Received Image', img)

    # Exit the loop on pressing the 'q' key
    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources and close the socket connection
cv2.destroyAllWindows()
s.close()
