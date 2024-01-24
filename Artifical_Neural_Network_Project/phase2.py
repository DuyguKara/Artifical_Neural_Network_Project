import cv2
import numpy as np
from tensorflow.keras.models import load_model

cap = cv2.VideoCapture("Test_Video_Jimmy Kimmel-Jodie Foster.mp4")
model = load_model("trained_model.h5")
img_size = (128, 128)
class_names = ["Guest Large Face", "Guest Small Face", "Kim Large Face", "Kim Small Face", "Other"]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:

        face_image = frame[y:y+h, x:x+w]

        face_image = cv2.resize(face_image, img_size)
        face_image = face_image / 255.0 

        predictions = model.predict(np.expand_dims(face_image, axis=0))
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_names[predicted_class_index]

        text = f"{predicted_class} ({x}, {y})"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

