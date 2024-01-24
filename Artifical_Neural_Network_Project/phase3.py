import cv2
import os
import hashlib
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

video_path = 'Test_Video_Jimmy Kimmel-Jodie Foster.mp4'
cap = cv2.VideoCapture(video_path)

model_path = 'trained_model.h5'
face_detection_model = load_model(model_path)

others_folder = 'others'
os.makedirs(others_folder, exist_ok=True)

seen_faces = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (128, 128))
    img_array = image.img_to_array(resized_frame)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = face_detection_model.predict(img_array)
    if predictions[0][0] > 0.5: 
        face_image = resized_frame

        face_bytes = cv2.imencode('.png', face_image)[1].tobytes()

        face_hash = hashlib.sha256(face_bytes).hexdigest()

        if face_hash not in seen_faces:
            seen_faces.append(face_hash)
            face_number = len(seen_faces)
            cv2.imwrite(os.path.join(others_folder, f"face_{face_number}.png"), face_image)

    cv2.imshow(frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Number of Unique Faces: {len(seen_faces)}")