import os
import cv2

video_filename = 'Training_Video_Jimmy Kimmel-Jake Johnson.mp4'

video_path = os.path.join(os.getcwd(), video_filename)

cap = cv2.VideoCapture(video_path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

data_folder = 'Dataset1/Training'
small_faces_folder = os.path.join(data_folder, 'small_faces')
large_faces_folder = os.path.join(data_folder, 'large_faces')

os.makedirs(small_faces_folder, exist_ok=True)
os.makedirs(large_faces_folder, exist_ok=True)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
    
        face_img = cv2.resize(frame[y:y+h, x:x+w], (64, 64))
        
        label = "large_faces" if w > 100 else "small_faces" if w > 50 else "no_face"
        
        if label == "large_faces":
            cv2.imwrite(os.path.join(large_faces_folder, f"{len(os.listdir(large_faces_folder)) + 1}.png"), face_img)
        elif label == "small_faces":
            cv2.imwrite(os.path.join(small_faces_folder, f"{len(os.listdir(small_faces_folder)) + 1}.png"), face_img)

cap.release()
cv2.destroyAllWindows()

