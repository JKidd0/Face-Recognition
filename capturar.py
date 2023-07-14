import cv2
import os
import numpy as np
from tkinter import Tk, Label, Entry, Button
from PIL import ImageTk, Image

# Inicializar la cámara
vid_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Variables globales
face_id = ''
count = 0

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Función para capturar y guardar la imagen
def capture_image():
    global count
    ret, image_frame = vid_cam.read()
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        count += 1
        image_path = f"./dataset/{face_id}/{count}.jpg"
        cv2.imwrite(image_path, gray[y:y + h, x:x + w])
        print(f"Ruta de la foto {count}: {image_path}")
    
    show_frame(image_frame)
    
    if count >= 500:
        print("fotos listas")

def train_model():
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
    print("Entrenamiento completado")

def update_people():
    train_model()
    
# Función para mostrar el marco actualizado en la ventana de Tkinter
def show_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = frame.resize((400, 300), Image.ANTIALIAS)
    frame = ImageTk.PhotoImage(frame)
    video_label.config(image=frame)
    video_label.image = frame

# Función para registrar el nombre ingresado por el usuario
def register_name():
    global face_id
    face_id = name_entry.get()
    name_label.config(text=f"Nombre: {face_id}")
    
    # Crear la carpeta dataset si no existe
    if not os.path.exists("./dataset"):
        os.makedirs("./dataset")
    
    # Crear la carpeta con el nombre del registro
    if not os.path.exists(f"./dataset/{face_id}"):
        os.makedirs(f"./dataset/{face_id}")

# Crear la interfaz de usuario con Tkinter
root = Tk()
root.title("Registro de Cara IA / DevByRO")
root.geometry("400x550")

video_label = Label(root)
video_label.pack()

name_label = Label(root, text="Ingrese su nombre:")
name_label.pack()

name_entry = Entry(root)
name_entry.pack()

register_button = Button(root, text="Registrar", command=register_name)
register_button.pack()

space_label = Label(root, text=" ")
space_label.pack()

update_button = Button(root, text="Actualizar Personas", command=update_people)
update_button.pack()

# Función para actualizar el video de la cámara en bucle
def update_video():
    global count
    ret, frame = vid_cam.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        show_frame(frame)
        
        if face_id and count < 500:
            capture_image()
            
    video_label.after(10, update_video)

update_video()
root.mainloop()
