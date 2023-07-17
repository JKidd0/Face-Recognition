afrom flask import Flask, render_template, Response, request, session, redirect
from flaskext.mysql import MySQL
import cv2
import datetime
import os
import numpy as np
from threading import Thread
from flask_bcrypt import Bcrypt
import json
import time
import zmq
import concurrent.futures


global capture, rec_frame, grey, switch, neg, face, rec, out
capture = 0
grey = 0
neg = 0
face = 0
switch = 1
rec = 0
url = "http://192.168.8.77:8081/stream"

# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

mysql = MySQL()
app = Flask(__name__)
bcrypt = Bcrypt(app)

# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'pyweb'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)

app.secret_key = 'why would I tell you my secret key?'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/api/login', methods=['POST'])
def apiLogin():
    try:
        _email = request.form['email']
        _password = request.form['password']
        conn = mysql.connect()
        cursor = conn.cursor()
        cursor.callproc('sp_checkUserByEmail', (_email,))
        data = cursor.fetchall()

        if len(data) > 0:

            if bcrypt.check_password_hash(str(data[0][4]), _password):
                session['user'] = data[0][1]
                return redirect('/user/index')
            else:
                return render_template('index.html', error="Incorrect password")
        else:
            return render_template('index.html', error="Email not registered")

    except Exception as e:
        return render_template('index.html', error=str(e))
    finally:
        cursor.close()
        conn.close()

@app.route('/api/register', methods=['POST'])
def apiRegister():
    try:
        _name = request.form['name']
        _username = request.form['username']
        _email = request.form['email']
        _password = request.form['password']

        if _name and _username and _email and _password:
            conn = mysql.connect()
            cursor = conn.cursor()
            _hashed_password = bcrypt.generate_password_hash(_password).decode('utf-8')

            cursor.callproc('sp_createUser', (_name, _username, _email, _hashed_password))
            data = cursor.fetchall()

            if len(data) == 0:
                conn.commit()
                return render_template('index.html', error='Registered successfully!')
            else:
                return render_template('register.html', error='Email already exists in the database!')

        else:
            return redirect('/register')

    except Exception as e:
        return json.dumps({'error': str(e)})
    finally:
        cursor.close()
        conn.close()

@app.route('/user/index')
def userIndex():
    if session.get('user'):
        return render_template('/user/index.html')
    else:
        return render_template('index.html', error='You must log in')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/')

camera = cv2.VideoCapture(0)
start_time = time.time()
detected_person = ''

def detect_face(frame):
    global start_time, detected_person

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/modelo.yml')
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    font = cv2.FONT_HERSHEY_SIMPLEX

    names = []
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset')

    for dir_name in os.listdir(dataset_path):
        if os.path.isdir(os.path.join(dataset_path, dir_name)):
            names.append(dir_name)

    minW = 100
    minH = 100

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(int(minW), int(minH)))

    recognized_name = "Desconocido"
    current_time = time.time()
    time_diff = current_time - start_time

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(frame_gray[y:y + h, x:x + w])
        
        if id >= 0 and id < len(names):
            if confidence <= 60:
                recognized_name = names[id]
                if recognized_name != detected_person:
                    start_time = time.time()
                    detected_person = recognized_name
                if time_diff >= 10.0:
                    cv2.putText(frame, "USUARIO CONFIRMADO", (x, y + h + 30), font, 1, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, recognized_name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            else:
                recognized_name = "Desconocido"
                if recognized_name != detected_person:
                    start_time = time.time()
                    detected_person = recognized_name
                    time_diff = current_time - start_time

        confidence_text = "  {0}%".format(round(100 - confidence))
        text_size, _ = cv2.getTextSize(recognized_name, font, 1, 2)
        text_x = x + (w - text_size[0]) // 2
        text_y = y + h // 2

        cv2.putText(frame, recognized_name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, confidence_text, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    current_time = time.time()
    time_diff = current_time - start_time

    if time_diff >= 10.0 and detected_person == "Desconocido":
        cv2.putText(frame, "INTRUSO", (10, 30), font, 1, (0, 0, 255), 3)

    cv2.putText(frame, "Tiempo transcurrido: {:.2f} segundos".format(time_diff), (10, 60), font, 1, (0, 0, 255), 2)

    return cv2.flip(frame, 1)


def gen_frames():
    global out, capture, rec_frame
    with concurrent.futures.ThreadPoolExecutor() as executor:
        while True:
            success, frame = camera.read()
            if success:
                frame = cv2.resize(frame, (640, 480))
                if capture:
                    frame = detect_face(frame)
                try:
                    ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                except Exception as e:
                    pass


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    

@app.route('/user/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Activar':
            global capture
            capture = 1
        elif request.form.get('stop') == 'Stop/Start':
            if switch == 1:
                switch = 0
                camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch = 1
    elif request.method == 'GET':
        return render_template('/user/index.html')
    return render_template('/user/index.html')

if __name__ == "__main__":
    app.run(debug=True)
    app.run()

camera.release()
cv2.destroyAllWindows()

