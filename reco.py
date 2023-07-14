import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/modelo.yml')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_SIMPLEX

# Obtiene los nombres de las carpetas en el directorio de entrenamiento
names = [name for name in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', name))]

# Inicializa el contador de ID
id = 0

# Inicializa la captura de video en tiempo real
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Ancho del video
cam.set(4, 480)  # Alto del video

# Define el tamaño mínimo de ventana para ser reconocido como un rostro
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Si la confianza es menor que 100, se considera una coincidencia perfecta
        if confidence < 100:
            name = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            name = "Desconocido"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(
            img,
            name,
            (x + 5, y - 5),
            font,
            1,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img,
            confidence,
            (x + 5, y + h - 5),
            font,
            1,
            (255, 255, 0),
            1
        )

    cv2.imshow('CamIA/Rec/RO', img)
    k = cv2.waitKey(10) & 0xff  # Presiona 'ESC' para salir del video
    if k == 27:
        break

# Limpieza
print("\n [INFO] Saliendo del programa y liberando recursos")
cam.release()
cv2.destroyAllWindows()
