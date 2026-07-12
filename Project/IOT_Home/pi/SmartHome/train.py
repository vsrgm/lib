import cv2
import numpy as np
import os
from PIL import Image

# Path for face image database
path = 'dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created. Add folders like '1_Ananth' with images and run again.")
        return [], []

    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    faceSamples = []
    ids = []

    for folder in imagePaths:
        # Expecting folder name format "ID_Name" (e.g., "1_Ananth")
        folder_name = os.path.basename(folder)
        try:
            user_id = int(folder_name.split('_')[0])
        except:
            continue

        for imagePath in [os.path.join(folder, f) for f in os.listdir(folder)]:
            try:
                PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
                img_numpy = np.array(PIL_img, 'uint8')

                faces = detector.detectMultiScale(img_numpy)
                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y+h, x:x+w])
                    ids.append(user_id)
            except Exception as e:
                print(f"Error processing {imagePath}: {e}")

    return faceSamples, ids

print("\nTraining faces. This will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)

if len(faces) > 0:
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer.yml
    recognizer.write('trainer.yml')
    print(f"\n{len(np.unique(ids))} faces trained. trainer.yml created successfully.")
else:
    print("\nNo faces found in dataset folder. trainer.yml not created.")
