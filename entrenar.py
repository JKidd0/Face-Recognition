import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def getImagesAndLabels(path):
    faceSamples = []
    ids = []
    id_mapping = {}
    current_id = 0

    for root, dirs, files in os.walk(path):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            image_files = os.listdir(dir_path)
            num_images = len(image_files)

            if num_images > 0:
                if dir_name not in id_mapping:
                    id_mapping[dir_name] = current_id
                    current_id += 1

                id = id_mapping[dir_name]
                print(f"Entrenando con {dir_name}, {num_images} fotos, ID: {id}")

                for file_name in image_files:
                    if file_name.endswith(".jpg"):
                        image_path = os.path.join(dir_path, file_name)
                        PIL_img = Image.open(image_path).convert('L')
                        img_numpy = np.array(PIL_img, 'uint8')
                        faces = detector.detectMultiScale(img_numpy)

                        for (x, y, w, h) in faces:
                            faceSamples.append(img_numpy[y:y + h, x:x + w])
                            ids.append(id)

    return faceSamples, ids

faces, ids = getImagesAndLabels('dataset')
recognizer.train(faces, np.array(ids))
recognizer.write('trainer/modelo.yml')
